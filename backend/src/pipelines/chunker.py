"""
Stage 3 (Phase 1 Refactor — Docling): Structure-aware chunking via IBM Docling.

Architecture:
    DocumentConverter  →  DoclingDocument  →  HybridChunker  →  list[DocumentChunk]

Docling's DocumentConverter uses ML-based layout analysis (TableFormer) to detect
tables, code blocks, section hierarchies, and reading order as first-class structural
elements. The HybridChunker then produces token-bounded chunks that respect those
structural boundaries — tables and code blocks are NEVER split.

Public API:
    chunk_document_docling(doc_path, max_tokens, doc_slug, base_metadata)
        → list[DocumentChunk]          (primary API used by ingest.py)

    chunk_document(...)                 ← legacy wrapper for backward compatibility
                                          with existing unit tests
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import tiktoken

from src.config import settings
from src.pipelines.models import DocumentChunk

if TYPE_CHECKING:
    from docling.chunking import DocChunk

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Content-type classifier from Docling DocItem labels
# ---------------------------------------------------------------------------

# Docling label strings (from docling_core.types.doc.labels.DocItemLabel)
_TABLE_LABELS = {"table"}
_CODE_LABELS = {"code"}
_LIST_LABELS = {"list_item"}
_TITLE_LABELS = {"title", "section_header"}


def _classify_content_type(chunk: DocChunk) -> str:
    """
    Map Docling's structural labels to our content_type vocabulary.
    Priority: table > code_block > list > procedure (numbered) > narrative.
    """
    try:
        labels = {str(item.label).lower() for item in chunk.meta.doc_items}
    except AttributeError:
        labels = set()

    if labels & _TABLE_LABELS:
        return "table"
    if labels & _CODE_LABELS:
        return "code_block"
    if labels & _LIST_LABELS:
        # Distinguish numbered procedure lists from bullet lists
        text = chunk.text or ""
        import re

        numbered = len(re.findall(r"^\s*\d+[\.\)]\s+", text, re.MULTILINE))
        if numbered >= 3:
            return "procedure"
        return "list"
    return "narrative"


# ---------------------------------------------------------------------------
# Section header extractor from Docling chunk metadata
# ---------------------------------------------------------------------------


def _extract_section_header(chunk: DocChunk) -> str:
    """
    Extract the most-specific section heading from Docling's chunk metadata.
    Docling populates chunk.meta.headings as a list of heading strings,
    outermost first (e.g. ["SECTION 2", "Tier Table"]).
    Returns the last (most specific) heading, or empty string if none.
    """
    try:
        headings = chunk.meta.headings
        if headings:
            return headings[-1]
    except AttributeError:
        pass
    # Fallback: check for a doc_item with a section_header / title label
    try:
        for item in chunk.meta.doc_items:
            label = str(item.label).lower()
            if label in _TITLE_LABELS:
                try:
                    return item.text
                except AttributeError:
                    pass
    except AttributeError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Markdown serializer helpers
# ---------------------------------------------------------------------------


def _serialize_chunk_to_markdown(chunk: DocChunk) -> str:
    """
    Serialize a Docling chunk to a markdown string.
    Falls back to chunk.text if the Markdown serializer isn't available.
    """
    try:
        text = chunk.text or ""
    except ImportError:
        text = chunk.text or ""

    # Docling already serializes tables as markdown in chunk.text when using
    # the default markdown serializer on the DocumentConverter.
    return text.strip()


def _get_table_markdown(raw_chunk: DocChunk, docling_doc: object) -> str | None:
    """
    Fix 1: If this chunk corresponds to a table, return its full pipe-markdown
    string via export_to_markdown(doc=docling_doc).

    IMPORTANT: chunk.meta.doc_items contains generic DocItem objects (not
    TableItem subclasses), so isinstance(item, TableItem) always returns False.
    We must match on item.label string instead, then look up the canonical live
    TableItem from docling_doc.tables (which are full TableItem objects with
    the export_to_markdown method) by self_ref.
    """
    _TABLE_LABEL_STRINGS = {"table", "document_index"}

    try:
        for item in raw_chunk.meta.doc_items:
            label = str(getattr(item, "label", "")).lower()
            if label not in _TABLE_LABEL_STRINGS:
                continue

            item_ref = getattr(item, "self_ref", None)
            if item_ref is None:
                continue

            # Look up the live TableItem in docling_doc.tables
            for tbl in docling_doc.tables:
                if getattr(tbl, "self_ref", None) == item_ref:
                    try:
                        md = tbl.export_to_markdown(doc=docling_doc)
                        if md and "|" in md:
                            return md.strip()
                    except Exception:
                        pass

    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Primary API
# ---------------------------------------------------------------------------


def chunk_document_docling(
    doc_path: str | Path,
    max_tokens: int,
    doc_slug: str,
    base_metadata: DocumentChunk,
    export_dir: Path | None = None,
) -> list[DocumentChunk]:
    """
    Convert and chunk a single document using Docling.

    Args:
        doc_path:      Path to the source document (PDF, DOCX, or MD).
        max_tokens:    HybridChunker token limit per chunk.
        doc_slug:      Slug for chunk_id prefix (e.g. 'doc02_radiation_sickness').
        base_metadata: DocumentChunk with all fixed fields populated (department,
                       access_level, document_status, effective_date, source_document).
                       Per-chunk fields (chunk_id, text, chunk_index, etc.) are
                       filled in by this function.
        export_dir:    If provided, export the parsed DoclingDocument to markdown
                       here (for audit / AC9 verification).

    Returns:
        list[DocumentChunk] ready for Qdrant upsert.
    """
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    doc_path = Path(doc_path)
    logger.info("[Docling] Converting: %s", doc_path.name)

    # ── Stage 1: Parse ──────────────────────────────────────────────────────
    converter = DocumentConverter()
    result = converter.convert(str(doc_path))
    docling_doc = result.document

    # ── AC9: Export to markdown for audit ───────────────────────────────────
    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        md_text = docling_doc.export_to_markdown()
        out_file = export_dir / f"{doc_path.stem}.md"
        out_file.write_text(md_text, encoding="utf-8")
        logger.info("[Docling] Exported markdown: %s", out_file)

    # ── Stage 2: Chunk ──────────────────────────────────────────────────────
    chunker = HybridChunker(
        tokenizer="BAAI/bge-small-en-v1.5",  # fast, consistent with our stack
        max_tokens=max_tokens,
        merge_peers=True,
    )

    raw_chunks = list(chunker.chunk(docling_doc))
    logger.info(
        "[Docling] %s → %d raw chunks (max_tokens=%d)", doc_path.name, len(raw_chunks), max_tokens
    )

    # ── Stage 3: Enrich ─────────────────────────────────────────────────────
    result_chunks: list[DocumentChunk] = []

    for i, raw_chunk in enumerate(raw_chunks):
        section_header = _extract_section_header(raw_chunk)
        content_type = _classify_content_type(raw_chunk)

        # Fix 1: For table chunks, use pipe markdown instead of the HybridChunker's
        # key=value serialization (e.g. **Col** = Value), which embedders score poorly.
        if content_type == "table":
            table_md = _get_table_markdown(raw_chunk, docling_doc)
            if table_md:
                text = table_md
            else:
                logger.warning(
                    "[Docling] Table chunk %d in %s: export_to_markdown() lookup failed or "
                    "returned no pipe chars — falling back to chunk.text. "
                    "The key=value serialization bug may survive in this chunk.",
                    i,
                    doc_path.name,
                )
                text = _serialize_chunk_to_markdown(raw_chunk)
        else:
            text = _serialize_chunk_to_markdown(raw_chunk)

        if not text:
            continue  # skip empty chunks

        # Fix 2: Prepend section heading as a text prefix for context-free chunks.
        # Embedders score poorly on chunks with no topical anchor.
        if section_header and not text.startswith(section_header):
            text = f"{section_header}\n\n{text}"

        token_count = _count_tokens(text)

        # Warn (but don't truncate) oversized chunks — usually a large table
        if token_count > max_tokens * 1.5:
            logger.warning(
                "[Docling] Oversized chunk in %s (chunk %d, %d tokens) — "
                "likely a large table; keeping intact.",
                doc_path.name,
                i,
                token_count,
            )

        chunk = base_metadata.model_copy(
            update={
                "chunk_id": f"{doc_slug}_{i:03d}",
                "text": text,
                "section_header": section_header,
                "content_type": content_type,
                "chunk_index": i,
                "total_chunks": 0,  # back-filled below
                "token_count": token_count,
            }
        )
        result_chunks.append(chunk)

    # Back-fill total_chunks
    total = len(result_chunks)
    result_chunks = [c.model_copy(update={"total_chunks": total}) for c in result_chunks]

    logger.info("[Docling] %s → %d final chunks", doc_path.name, total)
    return result_chunks


# ---------------------------------------------------------------------------
# Legacy backward-compatible wrapper
# (existing unit tests import chunk_document from this module)
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    metadata_template: ChunkMetadata,  # type: ignore[name-defined]  # noqa: F821
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    doc_format: str = "txt",
    filepath: str | None = None,
) -> list[tuple[str, ChunkMetadata]]:  # type: ignore[name-defined]  # noqa: F821
    """
    Legacy API — kept for backward compatibility with test_pipeline.py.

    If a filepath is provided and points to an existing file, delegates to
    chunk_document_docling. Otherwise falls back to a simple RecursiveTextSplitter
    so synthetic-text tests still work without file I/O.
    """
    from src.pipelines.models import ChunkMetadata
    from src.pipelines.models import DocumentChunk as DC

    _AL_MAP = {
        "general": "General Employee",
        "hr": "HR Restricted",
        "marketing": "Marketing Eyes Only",
        "admin": "Admin Eyes Only",
    }
    _DEPT_MAP = {
        "general": "General",
        "hr": "HR",
        "marketing": "Marketing",
        "admin": "Admin",
    }

    if filepath and Path(filepath).exists():
        doc_slug = Path(filepath).stem.lower().replace(" ", "_").replace("-", "_")
        base = DC(
            chunk_id="placeholder",
            text="",
            source_document=metadata_template.source_document,
            section_header="",
            department=_DEPT_MAP.get(metadata_template.department, "General"),  # type: ignore[arg-type]
            access_level=_AL_MAP.get(metadata_template.access_level, "General Employee"),  # type: ignore[arg-type]
            effective_date=None,
            document_status="ARCHIVED" if metadata_template.doc_status == "archived" else "ACTIVE",
            content_type="narrative",
            chunk_index=0,
            total_chunks=1,
            token_count=0,
        )
        dc_chunks = chunk_document_docling(filepath, chunk_size, doc_slug, base)
        return [
            (
                dc.text,
                metadata_template.model_copy(
                    update={
                        "chunk_id": dc.chunk_id,
                        "chunk_index": dc.chunk_index,
                        "total_chunks": dc.total_chunks,
                        "section_title": dc.section_header,
                    }
                ),
            )
            for dc in dc_chunks
        ]

    # Fallback: simple token-aware splitter for synthetic text tests
    import re as _re
    from uuid import uuid4

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Pre-build a list of (char_offset, heading_text) from the full text
    _heading_re = _re.compile(r"^(#{1,6}\s+.+)$", _re.MULTILINE)
    _headings_index: list[tuple[int, str]] = [
        (m.start(), m.group(1).strip()) for m in _heading_re.finditer(text)
    ]

    def _nearest_heading(pos: int) -> str:
        nearest = ""
        for offset, title in _headings_index:
            if offset <= pos:
                nearest = title
            else:
                break
        return nearest

    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = splitter.split_text(text)
    result: list[tuple[str, ChunkMetadata]] = []
    search_pos = 0
    for i, chunk_text in enumerate(text_chunks):
        # Find approximate position of this chunk in the original text
        anchor = chunk_text[:60]
        idx = text.find(anchor, search_pos)
        heading = _nearest_heading(idx if idx >= 0 else search_pos)
        if idx >= 0:
            search_pos = idx + max(1, len(chunk_text) // 2)

        meta = metadata_template.model_copy(
            update={
                "chunk_id": str(uuid4()),
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "section_title": heading,
            }
        )
        result.append((chunk_text, meta))
    return result
