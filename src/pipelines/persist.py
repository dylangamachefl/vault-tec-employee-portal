"""
Stage 4: Persistence — ChromaDB upsert + JSON backup.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (NOT the Google
generative AI client, which is reserved for generation only).

Accepts list[DocumentChunk] (Phase 1 refactor) while remaining backward
compatible with list[tuple[str, ChunkMetadata]] via an isinstance check.
"""

import json
import logging
from pathlib import Path

import chromadb

from src.config import settings
from src.pipelines.models import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.HttpClient:
    """Return a ChromaDB HTTP client using host/port from settings."""
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using sentence-transformers/all-MiniLM-L6-v2.
    Returns a list of float vectors.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def persist_to_chroma(
    chunks: "list[DocumentChunk] | list[tuple[str, ChunkMetadata]]",
    collection_name: str = "vault_documents",
) -> None:
    """
    Embed chunks and upsert into ChromaDB.

    Accepts both new-style list[DocumentChunk] and legacy-style
    list[tuple[str, ChunkMetadata]] for backward compatibility.

    The collection uses cosine similarity.
    """
    if not chunks:
        logger.warning("persist_to_chroma called with empty chunk list — nothing to do.")
        return

    # Normalise to (text, id, metadata_dict) triples
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for item in chunks:
        if isinstance(item, DocumentChunk):
            texts.append(item.text)
            ids.append(item.chunk_id)
            metadatas.append(item.to_chroma_metadata())
        else:
            # Legacy tuple[str, ChunkMetadata]
            text, meta = item
            texts.append(text)
            ids.append(meta.chunk_id)
            metadatas.append(meta.model_dump())

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Embedding %d chunks for collection '%s' …", len(texts), collection_name)
    embeddings = embed_texts(texts)

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info("Upserted %d chunks into ChromaDB collection '%s'.", len(texts), collection_name)


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
                "metadata": item.to_chroma_metadata(),
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
