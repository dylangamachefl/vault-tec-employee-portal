"""
Smoke-test for VaultRetriever (Task 4).

NOT a pytest suite — run directly:
    python tests/test_retrieval_chain.py

Exercises:
  Queries 1-5 : The five canonical Task 4 queries (AC-7)
  Query  6    : access_level_filter="hr" path (AC-6)
"""

import logging
import sys

# Ensure project root is in path when running as a script
sys.path.insert(0, ".")

from src.pipelines.retrieval_chain import CitedResponse, QueryInput, VaultRetriever

logging.basicConfig(level=logging.WARNING)  # suppress noisy INFO during smoke run

# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------

CANONICAL_QUERIES: list[dict] = [
    {
        "id": 1,
        "query": "What are the symptoms of radiation exposure at Tier 2?",
    },
    {
        "id": 2,
        "query": "What is the Overseer's base ration allocation per week?",
    },
    {
        "id": 3,
        "query": "What is the G.O.A.T. exam and how are results determined?",
    },
    {
        "id": 4,
        "query": "What procedures must be followed to cycle the vault door?",
    },
    {
        "id": 5,
        "query": "What happens to residents who are flagged for NVDR?",
    },
]

HR_FILTER_QUERY = {
    "id": "6-AC6",
    "query": "What are the compensation details for Overseer-level personnel?",
    "access_level_filter": "hr",
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _divider(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _print_response(qid: str | int, response: CitedResponse) -> None:
    _divider("═")
    print(f"  QUERY {qid}: {response.query}")
    _divider()
    print(f"  ANSWER ({response.retrieved_chunk_count} chunks retrieved):\n")
    # Indent the answer for readability
    for line in response.answer.splitlines():
        print(f"    {line}")
    print()
    print(f"  SOURCES ({len(response.sources)} unique document(s)):")
    if not response.sources:
        print("    ⚠  No sources returned.")
    for i, src in enumerate(response.sources, start=1):
        print(f"    [{i}] {src.source_document}")
        print(f"         Section : {src.section_title or 'N/A'}")
        print(f"         Dept    : {src.department}")
        print(f"         Access  : {src.access_level}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    _divider("═")
    print("  VAULT-TEC INTERNAL KNOWLEDGE ASSISTANT — RETRIEVAL CHAIN SMOKE TEST")
    _divider("═")
    print()

    # Instantiate once — shared across all queries
    retriever = VaultRetriever(collection_name="vault_documents_512")

    errors: list[str] = []

    # ── Queries 1-5 (AC-7) ───────────────────────────────────────────────────
    for spec in CANONICAL_QUERIES:
        qid = spec["id"]
        try:
            response = retriever.query(
                QueryInput(query=spec["query"])
            )
            _print_response(qid, response)

            # Inline AC assertions
            assert response.answer, f"Q{qid}: answer is empty"
            assert response.sources, f"Q{qid}: sources list is empty"
            assert response.retrieved_chunk_count > 0, f"Q{qid}: no chunks retrieved"

            # Deduplication check (AC-3)
            seen_docs = [s.source_document for s in response.sources]
            assert len(seen_docs) == len(set(seen_docs)), (
                f"Q{qid}: duplicate source_document values in sources: {seen_docs}"
            )

        except AssertionError as exc:
            errors.append(str(exc))
            print(f"  ⚠  ASSERTION FAILED: {exc}\n")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Q{qid} raised {type(exc).__name__}: {exc}")
            print(f"  ✗  EXCEPTION on Query {qid}: {exc}\n")

    # ── Query 6 — AC-6: access_level_filter="hr" path ────────────────────────
    _divider("═")
    print(f"  QUERY {HR_FILTER_QUERY['id']} (AC-6 — access_level_filter=\"hr\"): "
          f"{HR_FILTER_QUERY['query']}")
    _divider()
    try:
        response = retriever.query(
            QueryInput(
                query=HR_FILTER_QUERY["query"],
                access_level_filter=HR_FILTER_QUERY["access_level_filter"],
            )
        )
        _print_response(HR_FILTER_QUERY["id"], response)
        print(f"  ✓  AC-6 PASSED — hr filter executed without exception "
              f"({response.retrieved_chunk_count} chunk(s) retrieved)\n")
    except Exception as exc:  # noqa: BLE001
        msg = f"AC-6 raised {type(exc).__name__}: {exc}"
        errors.append(msg)
        print(f"  ✗  EXCEPTION: {msg}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    _divider("═")
    if errors:
        print(f"  RESULT: {len(errors)} failure(s) detected:\n")
        for err in errors:
            print(f"    • {err}")
        print()
        sys.exit(1)
    else:
        print("  RESULT: All smoke-test queries completed without exceptions. ✓")
        print("  Ready for Task 5 RAGAS evaluation harness.\n")


if __name__ == "__main__":
    main()
