"""
Task 4 — Retrieval Chain with Source Citations.

Standalone, importable inference engine for the Vault-Tec Employee Portal.
No Streamlit, no FastAPI — pure Python, callable from test scripts and the
RAGAS eval harness (Task 5).

Architecture:
  Qdrant (vault_documents_256) → VaultRetriever → gemma-3-27b-it → CitedResponse

RBAC note: access_level_filter is wired into the schema but defaults to None
for Task 4. Phase 2 will populate it before calling .query(). This module has
NO knowledge of users or roles.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

# Suppress FutureWarning from google.generativeai — package still works;
# migration to google-genai is tracked separately (not in Task 4 scope).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="google.generativeai",
)

import google.generativeai as genai  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.config import settings  # noqa: E402
from src.pipelines.models import RetrievedChunk  # noqa: E402
from src.retrieval.dedup import deduplicate_chunks  # noqa: E402
from src.retrieval.retriever import hybrid_retrieve  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models — AC-9: exact field names per spec
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    """One deduplicated source document referenced in a CitedResponse."""

    source_document: str
    section_title: str | None = None
    access_level: str
    department: str
    doc_date: str | None = None
    doc_status: str | None = None


class QueryInput(BaseModel):
    """Input contract for VaultRetriever.query()."""

    query: str
    collection_name: str = "vault_documents_256"
    top_k: int = 5
    # RESERVED — Phase 2 will populate this. Task 4 always passes None.
    access_level_filter: str | None = None


class CitedResponse(BaseModel):
    """Structured output from the retrieval chain."""

    answer: str
    sources: list[SourceCitation]
    retrieved_chunk_count: int
    query: str  # Echo of the original query — used by RAGAS eval harness
    # AC-6 (Task 5): raw chunk texts passed to the LLM, in retrieval order.
    # Required by RAGAS harness to build the 'contexts' column.
    retrieved_chunks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt constants — do NOT deviate from spec
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Vault-Tec Internal Knowledge Assistant. Answer employee questions using the provided document excerpts.  # noqa: E501

Rules you must follow:

1. Base your answer exclusively on the provided context.
   Do not use outside knowledge.

2. Always attempt to answer. Read every excerpt carefully,
   including all table rows, before concluding information is
   absent. Only say "This information is not available in the
   provided documents" if you have read all excerpts and the
   answer is genuinely not present.

3. Lead with the direct answer. No preamble such as "Based on
   the provided context..." or "The documents state..."

4. For table data: read the full row. The answer is in the cell
   that corresponds to the column header matching the question.
   Report that specific value directly.

5. Be concise. One to three sentences unless the question
   requires listing multiple items.

6. Cite the section name or document title when it helps
   locate the answer.
"""

_USER_TEMPLATE = """\
Answer the following question using only the document excerpts provided below.

QUESTION: {query}

DOCUMENT EXCERPTS:
{context_block}

Answer:
"""


def _format_chunk(doc_text: str, metadata: dict[str, Any]) -> str:
    """Render one chunk in the required context-block format."""
    source = metadata.get("source_document", "Unknown")
    section = metadata.get("section_title") or "N/A"
    dept = metadata.get("department", "Unknown")
    access = metadata.get("access_level", "Unknown")
    return (
        f"[Source: {source} | Section: {section} | Dept: {dept} | Access: {access}]\n"
        f"{doc_text}\n"
        "---"
    )


# ---------------------------------------------------------------------------
# VaultRetriever
# ---------------------------------------------------------------------------


class VaultRetriever:
    """
    Core inference engine: retrieves chunks from Qdrant, grounds an LLM
    prompt, and returns a CitedResponse with deduplicated source citations.

    Data lives in the Docker-hosted Qdrant container — uses QdrantClient
    connecting via REST (port 6333).
    """

    def __init__(self, collection_name: str = "vault_documents_256") -> None:
        logger.info("Initialising VaultRetriever (collection: %s)", collection_name)

        self._collection_name = collection_name

        # Qdrant — data is in Docker volume
        self._qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        collection_info = self._qdrant.get_collection(collection_name)
        logger.info(
            "Connected to Qdrant collection '%s' (%d points).",
            collection_name,
            collection_info.points_count,
        )

        # Embedding model — MUST match ingestion (all-MiniLM-L6-v2)
        logger.info("Loading SentenceTransformer (all-MiniLM-L6-v2) …")
        self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Google Generative AI client
        genai.configure(api_key=settings.google_api_key)
        self._llm = genai.GenerativeModel(model_name=settings.llm_model)
        logger.info("LLM model '%s' configured.", settings.llm_model)

    # ------------------------------------------------------------------
    # retrieve()
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int,
        access_level_filter: str | None,
        access_filter: dict | None = None,  # Phase 2 RBAC scaffold (field/value dict)
    ) -> list[RetrievedChunk]:
        """
        Embed query, run hybrid search (dense + sparse RRF), return raw results.

        Each dict has keys: 'document' (str) and 'metadata' (dict).

        access_level_filter: Legacy string filter (kept for backward compat).
                             Maps to {"field": "access_level", "value": value}.
        access_filter:       Phase 2 RBAC scaffold — {"field": ..., "value": ...}.
                             Both default to None — no filtering applied.
        """
        # Dense embedding — returns np.ndarray
        embedding = self._embedder.encode(query, convert_to_numpy=True)

        # Resolve access filter: Phase 2 dict takes precedence over legacy string
        resolved_filter: dict | None = None
        if access_filter:
            resolved_filter = access_filter
        elif access_level_filter is not None:
            resolved_filter = {"field": "access_level", "value": access_level_filter}

        raw_results = hybrid_retrieve(
            client=self._qdrant,
            collection_name=self._collection_name,
            query=query,
            dense_embedding=embedding.tolist(),
            k=top_k,
            access_filter=resolved_filter,
        )

        chunks = deduplicate_chunks(
            raw_results, similarity_threshold=settings.retrieval_dedup_threshold
        )

        logger.debug("Retrieved %d chunks for query: %.80s…", len(chunks), query)
        return chunks

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> CitedResponse:
        """
        Build a grounded prompt from retrieved chunks, call the LLM, and
        return a CitedResponse with deduplicated source citations.

        Only chunks passed in here are ever sent to the LLM — no external
        knowledge, no injected content.
        """
        # Build context block
        context_block = "\n".join(_format_chunk(chunk.content, chunk.metadata) for chunk in chunks)

        # Assemble full prompt (system + user roles via combined string for gemma)
        prompt = f"{_SYSTEM_PROMPT}\n\n" + _USER_TEMPLATE.format(
            query=query, context_block=context_block
        )

        logger.info("Calling LLM with %d chunk(s) …", len(chunks))
        response = self._llm.generate_content(prompt)
        answer = response.text.strip()

        # Build deduplicated SourceCitation list — one entry per source_document
        seen: set[str] = set()
        sources: list[SourceCitation] = []
        for chunk in chunks:
            meta = chunk.metadata
            src_doc: str = meta.get("source_document", "Unknown")
            if src_doc in seen:
                continue
            seen.add(src_doc)
            sources.append(
                SourceCitation(
                    source_document=src_doc,
                    section_title=meta.get("section_title") or None,
                    access_level=meta.get("access_level", "Unknown"),
                    department=meta.get("department", "Unknown"),
                    doc_date=meta.get("doc_date") or None,
                    doc_status=meta.get("doc_status") or None,
                )
            )

        return CitedResponse(
            answer=answer,
            sources=sources,
            retrieved_chunk_count=len(chunks),
            retrieved_chunks=[chunk.content for chunk in chunks],
            query=query,
        )

    # ------------------------------------------------------------------
    # query() — public entry point
    # ------------------------------------------------------------------

    def query(self, input: QueryInput) -> CitedResponse:
        """
        Public entry point consumed by the test harness and (Task 5) RAGAS eval.

        Calls retrieve() then generate(). Returns a validated CitedResponse.
        """
        chunks = self.retrieve(
            query=input.query,
            top_k=input.top_k,
            access_level_filter=input.access_level_filter,
        )
        return self.generate(query=input.query, chunks=chunks)
