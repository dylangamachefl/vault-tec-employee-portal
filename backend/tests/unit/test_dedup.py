from src.pipelines.models import RetrievedChunk
from src.retrieval.dedup import deduplicate_chunks


def create_mock_chunk(
    chunk_id: str, embedding: list[float], content: str = "mock content", score: float = 0.9
) -> RetrievedChunk:
    metadata = {
        "chunk_id": chunk_id,
        "source_document": f"doc_{chunk_id}.txt",
        "doc_format": "txt",
        "access_level": "general",
        "department": "general",
        "doc_date": "2076-08-01",
        "doc_status": "active",
        "chunk_index": 0,
        "total_chunks": 1,
        "section_title": "mock section",
    }
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        embedding=embedding,
        metadata=metadata,
        relevance_score=score,
    )


def test_all_unique_inputs():
    chunk1 = create_mock_chunk("1", [1.0, 0.0, 0.0])
    chunk2 = create_mock_chunk("2", [0.0, 1.0, 0.0])
    chunk3 = create_mock_chunk("3", [0.0, 0.0, 1.0])

    chunks = [chunk1, chunk2, chunk3]
    result = deduplicate_chunks(chunks, similarity_threshold=0.95)

    assert len(result) == 3
    assert result == chunks


def test_all_identical_inputs():
    chunk1 = create_mock_chunk("1", [1.0, 0.0, 0.0])
    chunk2 = create_mock_chunk("2", [1.0, 0.0, 0.0])
    chunk3 = create_mock_chunk("3", [1.0, 0.0, 0.0])

    chunks = [chunk1, chunk2, chunk3]
    result = deduplicate_chunks(chunks, similarity_threshold=0.95)

    assert len(result) == 1
    assert result[0] == chunk1


def test_mixed_inputs_criterion_1():
    # Given a list of 5 chunks where chunks 1 and 3 are near-identical
    # (cosine similarity > 0.95), the function returns 4 chunks.
    # Chunk 1 is kept. Chunk 3 is discarded. The order of all other chunks is unchanged.
    chunk1 = create_mock_chunk("1", [1.0, 0.0, 0.0])
    chunk2 = create_mock_chunk("2", [0.0, 1.0, 0.0])
    chunk3 = create_mock_chunk("3", [0.99, 0.1, 0.0])  # cos sim > 0.95
    chunk4 = create_mock_chunk("4", [0.0, 0.0, 1.0])
    chunk5 = create_mock_chunk("5", [0.5, 0.5, 0.5])

    chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]
    result = deduplicate_chunks(chunks, similarity_threshold=0.95)

    assert len(result) == 4
    assert result[0] == chunk1
    assert result[1] == chunk2
    assert result[2] == chunk4
    assert result[3] == chunk5


def test_threshold_respected_high():
    # Running the same input with similarity_threshold=1.01 returns all 5 chunks unchanged.
    chunk1 = create_mock_chunk("1", [1.0, 0.0, 0.0])
    chunk2 = create_mock_chunk("2", [0.0, 1.0, 0.0])
    chunk3 = create_mock_chunk("3", [0.99, 0.1, 0.0])
    chunk4 = create_mock_chunk("4", [0.0, 0.0, 1.0])
    chunk5 = create_mock_chunk("5", [0.5, 0.5, 0.5])

    chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]
    result = deduplicate_chunks(chunks, similarity_threshold=1.01)

    assert len(result) == 5
    assert result == chunks


def test_threshold_respected_low():
    # Running with similarity_threshold=0.0 returns only 1 chunk (the first one)
    # Provide vectors that all have positive correlation to chunk1
    chunk1 = create_mock_chunk("1", [1.0, 0.0, 0.0])
    chunk2 = create_mock_chunk("2", [0.5, 0.5, 0.0])
    chunk3 = create_mock_chunk("3", [0.99, 0.1, 0.0])
    chunk4 = create_mock_chunk("4", [0.5, 0.0, 0.5])
    chunk5 = create_mock_chunk("5", [0.5, 0.5, 0.5])

    chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]
    result = deduplicate_chunks(chunks, similarity_threshold=0.0)

    assert len(result) == 1
    assert result[0] == chunk1


def test_no_embedding_kept():
    chunk1 = create_mock_chunk("1", [])
    chunk2 = create_mock_chunk("2", [])

    chunks = [chunk1, chunk2]
    result = deduplicate_chunks(chunks, similarity_threshold=0.95)
    assert len(result) == 2
