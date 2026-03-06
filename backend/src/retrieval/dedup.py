import math

from src.pipelines.models import RetrievedChunk


def deduplicate_chunks(
    chunks: list[RetrievedChunk], similarity_threshold: float = 0.95
) -> list[RetrievedChunk]:
    """
    Deduplicate a list of RetrievedChunk objects based on dense embedding cosine similarity.
    Preserves the order of chunks (higher-ranked chunks are kept).
    """
    kept_chunks: list[RetrievedChunk] = []

    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    for chunk in chunks:
        if not kept_chunks:
            kept_chunks.append(chunk)
            continue

        if not chunk.embedding:
            # If a chunk somehow has no embedding, keep it to be safe
            kept_chunks.append(chunk)
            continue

        is_duplicate = False
        for kept in kept_chunks:
            if not kept.embedding:
                continue
            sim = cosine_similarity(chunk.embedding, kept.embedding)
            if sim > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_chunks.append(chunk)

    return kept_chunks
