"""
Hybrid retrieval using Qdrant native sparse+dense vectors with built-in RRF fusion.

Dense vectors:  sentence-transformers/all-MiniLM-L6-v2 (must match ingestion model)
Sparse vectors: SPLADE via fastembed SparseTextEmbedding (prithivida/Splade_PP_en_v1)
Fusion:         Qdrant built-in RRF via prefetch + FusionQuery pattern

Named vector selection in Prefetch uses the `using` parameter (qdrant-client >= 1.9).
Phase 2 RBAC scaffold: access_filter dict {"field": ..., "value": ...} is wired in
but defaults to None — filtering is applied at both prefetch steps simultaneously.
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from src.pipelines.models import RetrievedChunk

# Module-level sparse model — shared across all calls in this process.
# fastembed downloads the model on first instantiation; subsequent calls reuse it.
_sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")


def embed_query_sparse(query: str) -> SparseVector:
    """Encode a query string into a SPLADE sparse vector for Qdrant."""
    result = list(_sparse_model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def hybrid_retrieve(
    client: QdrantClient,
    collection_name: str,
    query: str,
    dense_embedding: list[float],
    k: int = 5,
    access_filter: dict | None = None,
) -> list[RetrievedChunk]:
    """
    Run Qdrant native hybrid search (dense + sparse) with built-in RRF fusion.

    Args:
        client:           Live QdrantClient instance.
        collection_name:  Target collection (must have both "dense" and "sparse" named vectors).
        query:            Raw query string — encoded to a sparse vector here.
        dense_embedding:  Pre-computed dense vector for the query (list[float]).
        k:                Number of results to return (applied after RRF fusion).
        access_filter:    Optional Phase 2 RBAC filter: {"field": str, "value": str}.
                          Applied identically to both prefetch steps.

    Returns:
        List of RetrievedChunk models.
    """
    sparse_embedding = embed_query_sparse(query)

    query_filter: Filter | None = None
    if access_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=access_filter["field"],
                    match=MatchValue(value=access_filter["value"]),
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Dense ANN prefetch — uses "dense" named vector via `using`
            Prefetch(
                query=dense_embedding,
                using="dense",
                limit=k,
                filter=query_filter,
            ),
            # Sparse SPLADE prefetch — uses "sparse" named vector via `using`
            Prefetch(
                query=sparse_embedding,
                using="sparse",
                limit=k,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
        with_vectors=True,
    )

    chunks = []
    for result in results.points:
        metadata = {key: val for key, val in result.payload.items() if key != "text"}
        # Extract the dense vector which is stored under the "dense" key
        dense_vec = result.vector.get("dense") if isinstance(result.vector, dict) else result.vector
        if dense_vec is None:
            dense_vec = []

        chunks.append(
            RetrievedChunk(
                chunk_id=str(result.id),
                content=result.payload["text"],
                embedding=dense_vec,
                metadata=metadata,
                relevance_score=result.score,
            )
        )

    return chunks
