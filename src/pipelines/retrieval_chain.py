"""
Task 4 — Retrieval Chain with Source Citations.

Standalone, importable inference engine for the Vault-Tec Employee Portal.
No Streamlit, no FastAPI — pure Python, callable from test scripts and the
RAGAS eval harness (Task 5).

Architecture:
  ChromaDB (vault_documents_512) → VaultRetriever → gemma-3-27b-it → CitedResponse

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

import chromadb
import google.generativeai as genai  # noqa: E402
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.config import settings

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
    collection_name: str = "vault_documents_512"
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
You are the Vault-Tec Internal Knowledge Assistant. Your role is to answer employee questions accurately and exclusively based on the provided document excerpts. 

Rules you must follow:
1. Only use information from the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say: "The provided documents do not contain sufficient information to answer this question."
3. Be concise and professional.
4. Do not speculate, infer beyond the text, or mention information that is not present in the retrieved excerpts.
"""

_USER_TEMPLATE = """\
Answer the following question using only the document excerpts provided below.

QUESTION: {query}

DOCUMENT EXCERPTS:
{context_block}

Provide your answer below:
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
    Core inference engine: retrieves chunks from ChromaDB, grounds an LLM
    prompt, and returns a CitedResponse with deduplicated source citations.

    Data lives in the Docker-hosted ChromaDB container — uses HttpClient,
    NOT PersistentClient (confirmed: local chroma_db/ directory is empty).
    """

    def __init__(self, collection_name: str = "vault_documents_512") -> None:
        logger.info("Initialising VaultRetriever (collection: %s)", collection_name)

        # ChromaDB — data is in Docker volume, always use HttpClient
        self._chroma = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._collection = self._chroma.get_collection(name=collection_name)
        logger.info(
            "Connected to ChromaDB collection '%s' (%d items).",
            collection_name,
            self._collection.count(),
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
    ) -> list[dict[str, Any]]:
        """
        Embed query, query ChromaDB, return raw results as list of dicts.

        Each dict has keys: 'document' (str) and 'metadata' (dict).

        If access_level_filter is not None, a ChromaDB where-filter is applied.
        The caller (Phase 2) is responsible for choosing a sensible filter value.
        Task 4 always passes None — no filtering.
        """
        # Embed — returns np.ndarray; ChromaDB requires list-of-lists
        embedding = self._embedder.encode(query, convert_to_numpy=True)

        where_filter: dict[str, Any] | None = None
        if access_level_filter is not None:
            where_filter = {"access_level": {"$eq": access_level_filter}}

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        # ChromaDB returns parallel lists; zip into dicts for ergonomic access
        documents: list[str] = results["documents"][0]
        metadatas: list[dict[str, Any]] = results["metadatas"][0]

        chunks = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(documents, metadatas)
        ]
        logger.debug("Retrieved %d chunks for query: %.80s…", len(chunks), query)
        return chunks

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> CitedResponse:
        """
        Build a grounded prompt from retrieved chunks, call the LLM, and
        return a CitedResponse with deduplicated source citations.

        Only chunks passed in here are ever sent to the LLM — no external
        knowledge, no injected content.
        """
        # Build context block
        context_block = "\n".join(
            _format_chunk(chunk["document"], chunk["metadata"])
            for chunk in chunks
        )

        # Assemble full prompt (system + user roles via combined string for gemma)
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            + _USER_TEMPLATE.format(query=query, context_block=context_block)
        )

        logger.info("Calling LLM with %d chunk(s) …", len(chunks))
        response = self._llm.generate_content(prompt)
        answer = response.text.strip()

        # Build deduplicated SourceCitation list — one entry per source_document
        seen: set[str] = set()
        sources: list[SourceCitation] = []
        for chunk in chunks:
            meta = chunk["metadata"]
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
            retrieved_chunks=[chunk["document"] for chunk in chunks],
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
