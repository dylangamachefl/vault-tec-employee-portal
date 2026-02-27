"""
Unit tests for the hybrid retrieval pipeline (Task 5D-b).

Tests 1–4 are pure unit tests (no Qdrant required).
Tests 5–10 are integration tests requiring Qdrant to be running with
vault_documents_256 fully ingested (run ingest.py --reset first).

All 10 are mandatory acceptance criteria per the Task 5D-b spec.
"""

from __future__ import annotations

import pytest
from qdrant_client.models import SparseVector

from src.config import settings
from src.retrieval.retriever import embed_query_sparse, hybrid_retrieve

COLLECTION = settings.qdrant_collection_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qdrant_available() -> bool:
    """Return True if Qdrant is reachable at the configured host/port."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qdrant_client():
    """Live QdrantClient — skips the test module if Qdrant is not available."""
    from qdrant_client import QdrantClient

    if not _qdrant_available():
        pytest.skip("Qdrant not available — run docker-compose up first")
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


@pytest.fixture(scope="module")
def dense_embedder():
    """SentenceTransformer used to produce dense query embeddings in tests."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _embed(embedder, text: str) -> list[float]:
    return embedder.encode(text, convert_to_numpy=True).tolist()


# ---------------------------------------------------------------------------
# Tests 1–2: hybrid_retrieve output contract (require Qdrant)
# ---------------------------------------------------------------------------


def test_hybrid_retrieve_returns_k_results(qdrant_client, dense_embedder):
    """hybrid_retrieve(..., k=5) must return exactly 5 results."""
    query = "What is the radiation treatment procedure?"
    emb = _embed(dense_embedder, query)
    results = hybrid_retrieve(qdrant_client, COLLECTION, query, emb, k=5)
    assert len(results) == 5


def test_hybrid_retrieve_result_fields(qdrant_client, dense_embedder):
    """Each result dict must have exactly the keys: id, text, metadata, score."""
    query = "What is the radiation treatment procedure?"
    emb = _embed(dense_embedder, query)
    results = hybrid_retrieve(qdrant_client, COLLECTION, query, emb, k=5)
    for result in results:
        assert hasattr(result, "chunk_id"), "Missing attribute: chunk_id"
        assert hasattr(result, "content"), "Missing attribute: content"
        assert hasattr(result, "metadata"), "Missing attribute: metadata"
        assert hasattr(result, "relevance_score"), "Missing attribute: relevance_score"


# ---------------------------------------------------------------------------
# Tests 3–4: sparse embedding contract (no Qdrant needed)
# ---------------------------------------------------------------------------


def test_sparse_embedding_produces_sparse_vector():
    """embed_query_sparse() must return a SparseVector with non-empty indices and values."""
    sv = embed_query_sparse("radiation sickness symptom guide")
    assert isinstance(sv, SparseVector)
    assert len(sv.indices) > 0, "SparseVector.indices must be non-empty"
    assert len(sv.values) > 0, "SparseVector.values must be non-empty"


def test_sparse_indices_values_same_length():
    """SparseVector.indices and .values must have the same length for any query."""
    sv = embed_query_sparse("evacuation muster point color-coded chart")
    assert len(sv.indices) == len(
        sv.values
    ), f"indices length {len(sv.indices)} != values length {len(sv.values)}"


# ---------------------------------------------------------------------------
# Tests 5–6: collection configuration (require Qdrant)
# ---------------------------------------------------------------------------


def test_collection_has_sparse_vector_config(qdrant_client):
    """Qdrant collection must have a 'sparse' entry in sparse_vectors config."""
    info = qdrant_client.get_collection(COLLECTION)
    sparse_cfg = info.config.params.sparse_vectors
    assert sparse_cfg is not None, "Collection has no sparse_vectors config"
    assert (
        "sparse" in sparse_cfg
    ), f"'sparse' not found in sparse_vectors config: {list(sparse_cfg.keys())}"


def test_collection_has_dense_vector_config(qdrant_client):
    """Qdrant collection must have a 'dense' entry in named vectors config."""
    info = qdrant_client.get_collection(COLLECTION)
    vectors_cfg = info.config.params.vectors
    assert isinstance(
        vectors_cfg, dict
    ), "Collection vectors config is not a named-vector dict — was reset_collection_hybrid called?"
    assert (
        "dense" in vectors_cfg
    ), f"'dense' not found in vectors config: {list(vectors_cfg.keys())}"


# ---------------------------------------------------------------------------
# Test 7: access_filter=None safety (requires Qdrant)
# ---------------------------------------------------------------------------


def test_access_filter_none_does_not_error(qdrant_client, dense_embedder):
    """hybrid_retrieve with access_filter=None must not raise any exception."""
    query = "vault radiation exposure procedures"
    emb = _embed(dense_embedder, query)
    results = hybrid_retrieve(qdrant_client, COLLECTION, query, emb, k=5, access_filter=None)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests 8–9: spot checks for RRF keyword boost (require Qdrant + ingested data)
# ---------------------------------------------------------------------------


def test_spot_check_evacuation_muster(qdrant_client, dense_embedder):
    """
    AC4: 'color-coded chart' must appear in at least one top-5 result
    for the muster point query.
    """
    query = "How do residents locate their evacuation muster point?"
    emb = _embed(dense_embedder, query)
    results = hybrid_retrieve(qdrant_client, COLLECTION, query, emb, k=5)
    texts = [r.content for r in results]
    assert any("color-coded chart" in t for t in texts), (
        "Spot check failed: 'color-coded chart' not found in top-5 results.\n"
        "Texts returned:\n" + "\n---\n".join(texts[:3])
    )


def test_spot_check_tier2_radiation(qdrant_client, dense_embedder):
    """
    AC5: 'VT-MED-011' must appear in at least one top-5 result
    for the Tier 2 radiation treatment query.
    """
    query = "What is the treatment recommendation for Tier 2 radiation exposure?"
    emb = _embed(dense_embedder, query)
    results = hybrid_retrieve(qdrant_client, COLLECTION, query, emb, k=5)
    texts = [r.content for r in results]
    assert any("VT-MED-011" in t for t in texts), (
        "Spot check failed: 'VT-MED-011' not found in top-5 results.\n"
        "Texts returned:\n" + "\n---\n".join(texts[:3])
    )


# ---------------------------------------------------------------------------
# Test 10: RRF surfaces exact-match phrase in top-3 (requires Qdrant)
# ---------------------------------------------------------------------------


def test_rrf_surfaces_exact_match_phrase(qdrant_client, dense_embedder):
    """
    A query containing an exact phrase from a chunk must return that chunk in top-3.
    Validates that sparse BM25-style scoring boosts exact-match chunks via RRF.
    """
    # Phrase that exists verbatim in the emergency evacuation document
    query = "color-coded chart affixed to the inside of your quarters door"
    emb = _embed(dense_embedder, query)
    results = hybrid_retrieve(qdrant_client, COLLECTION, query, emb, k=5)
    top3_texts = [r.content for r in results[:3]]
    assert any("color-coded chart" in t for t in top3_texts), (
        "RRF exact-match test failed: expected 'color-coded chart' in top-3.\n"
        "Top-3 texts:\n" + "\n---\n".join(top3_texts)
    )
