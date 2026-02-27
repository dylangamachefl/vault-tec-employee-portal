"""
Stage 4: Persistence — Qdrant upsert + JSON backup.

Embedding models:
  Dense:  sentence-transformers/all-MiniLM-L6-v2 (NOT Google generative AI client)
  Sparse: fastembed SparseTextEmbedding (prithivida/Splade_PP_en_v1)

Collection schema (Task 5D-b): named vectors "dense" + "sparse" for hybrid RRF search.

Accepts list[DocumentChunk] (Phase 1 refactor) while remaining backward compatible
with list[tuple[str, ChunkMetadata]] via an isinstance check.
"""

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings
from src.pipelines.models import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)

# Embedding dimension for sentence-transformers/all-MiniLM-L6-v2
VECTOR_SIZE = 384

# Module-level sparse model — shared across all upsert calls in this process.
_sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client using host/port from settings."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def reset_collection_hybrid(
    client: QdrantClient,
    collection_name: str,
    dense_vector_size: int = VECTOR_SIZE,
) -> None:
    """
    Delete (if exists) and recreate a Qdrant collection with named dense + sparse vectors.

    Named vectors:
      "dense"  — cosine similarity, size=dense_vector_size
      "sparse" — SPLADE sparse vectors (no size needed)

    This is the canonical collection setup for Task 5D-b hybrid search.
    Must be called before upsert_chunks_hybrid when rebuilding from scratch.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info("Deleted Qdrant collection '%s'.", collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=dense_vector_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    logger.info(
        "Created hybrid Qdrant collection '%s' (dense_size=%d, sparse=SPLADE).",
        collection_name,
        dense_vector_size,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = VECTOR_SIZE,
) -> None:
    """Create the hybrid Qdrant collection if it does not already exist (idempotent)."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        reset_collection_hybrid(client, collection_name, vector_size)
    else:
        logger.info("Qdrant collection '%s' already exists — skipping create.", collection_name)


def reset_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = VECTOR_SIZE,
) -> None:
    """Delete and recreate a Qdrant collection for a clean re-ingestion run."""
    reset_collection_hybrid(client, collection_name, vector_size)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using sentence-transformers/all-MiniLM-L6-v2.
    Returns a list of float vectors.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def upsert_chunks_hybrid(
    client: QdrantClient,
    collection_name: str,
    chunks: list[DocumentChunk],
) -> None:
    """
    Compute dense + sparse embeddings for each chunk and upsert into Qdrant
    using named vectors ("dense" and "sparse").

    Sparse embeddings are computed in a single batch for efficiency.
    Dense embeddings are computed via sentence-transformers/all-MiniLM-L6-v2.
    """
    if not chunks:
        logger.warning("upsert_chunks_hybrid called with empty list — nothing to do.")
        return

    texts = [chunk.text for chunk in chunks]

    # Dense embeddings
    logger.info("Computing dense embeddings for %d chunks …", len(texts))
    dense_embeddings = embed_texts(texts)

    # Sparse embeddings — batch call for efficiency
    logger.info("Computing sparse (SPLADE) embeddings for %d chunks …", len(texts))
    sparse_embeddings = list(_sparse_model.embed(texts))

    points = []
    for chunk, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source_document,
                    "date": chunk.effective_date.isoformat() if chunk.effective_date else "",
                    "access_level": chunk.access_level,
                    "department": chunk.department,
                    "doc_type": chunk.content_type,
                    "version_date": chunk.effective_date.isoformat()
                    if chunk.effective_date
                    else "",
                    "section_header": chunk.section_header,
                    "status": chunk.document_status,
                    # Retrieval chain compatibility fields
                    "source_document": chunk.source_document,
                    "section_title": chunk.section_header,
                    "doc_date": chunk.effective_date.isoformat() if chunk.effective_date else "",
                    "doc_status": chunk.document_status,
                    "content_type": chunk.content_type,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "token_count": chunk.token_count,
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Upserted %d chunks into '%s' (dense+sparse).", len(points), collection_name)


def upsert_chunks(
    chunks: "list[DocumentChunk] | list[tuple[str, ChunkMetadata]]",
    collection_name: str = "vault_documents_256",
) -> None:
    """
    Embed chunks (dense + sparse) and upsert into Qdrant as named vectors.

    Accepts both new-style list[DocumentChunk] and legacy-style
    list[tuple[str, ChunkMetadata]] for backward compatibility.
    """
    if not chunks:
        logger.warning("upsert_chunks called with empty chunk list — nothing to do.")
        return

    client = get_qdrant_client()
    ensure_collection(client, collection_name)

    # Normalise legacy tuples → DocumentChunk-like objects are handled below
    doc_chunks: list[DocumentChunk] = []
    legacy_items: list[tuple[str, ChunkMetadata]] = []

    for item in chunks:
        if isinstance(item, DocumentChunk):
            doc_chunks.append(item)
        else:
            legacy_items.append(item)

    if doc_chunks:
        upsert_chunks_hybrid(client, collection_name, doc_chunks)

    # Legacy path: tuple[str, ChunkMetadata] — dense-only upsert for backward compat
    if legacy_items:
        logger.warning(
            "upsert_chunks received %d legacy tuple items — using dense-only upsert.",
            len(legacy_items),
        )
        texts = [text for text, _ in legacy_items]
        dense_embeddings = embed_texts(texts)
        sparse_embeddings = list(_sparse_model.embed(texts))

        points = []
        for (text, meta), dense_emb, sparse_emb in zip(
            legacy_items, dense_embeddings, sparse_embeddings
        ):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        "dense": dense_emb,
                        "sparse": SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                    },
                    payload={
                        "chunk_id": meta.chunk_id,
                        "text": text,
                        "source": meta.source_document,
                        "access_level": meta.access_level,
                        "department": meta.department,
                        "doc_type": meta.doc_format,
                        "version_date": meta.doc_date,
                        "section_header": meta.section_title,
                        "status": meta.doc_status,
                        "source_document": meta.source_document,
                        "section_title": meta.section_title,
                        "doc_date": meta.doc_date,
                        "doc_status": meta.doc_status,
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points)
        logger.info("Upserted %d legacy chunks into '%s'.", len(points), collection_name)


def write_json_backup(
    chunks: "list[DocumentChunk] | list[tuple[str, ChunkMetadata]]",
    output_dir: str = "data/processed",
) -> None:
    """
    Write each chunk to data/processed/{source_document}_{chunk_index:04d}.json.

    Accepts both DocumentChunk and legacy tuple[str, ChunkMetadata].
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for item in chunks:
        if isinstance(item, DocumentChunk):
            chunk_index = item.chunk_index
            source = item.source_document
            record = {
                "text": item.text,
                "metadata": item.to_qdrant_payload(),
            }
        else:
            text, meta = item
            chunk_index = meta.chunk_index
            source = meta.source_document
            record = {
                "text": text,
                "metadata": meta.model_dump(),
            }

        filename = f"{source}_{chunk_index:04d}.json"
        filepath = out_path / filename
        filepath.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    logger.info("Wrote %d JSON backup files to '%s'.", len(chunks), output_dir)
