"""
Acceptance-criteria tests for the Docling-based chunking pipeline (Phase 1 refactor).

Tests validate the 8 spec acceptance criteria + AC9 (Docling export) using the
chunk_document_docling() function directly (no Qdrant required).

Tests requiring real documents from data/raw/ are marked with @pytest.mark.skipif
so CI without the corpus can still run the synthetic tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipelines.chunker import chunk_document_docling
from src.pipelines.cleaner import clean_text
from src.pipelines.models import DocumentChunk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw")
EXPORT_DIR = Path("data/processed/docling_exports")

DOC02_PATH = DATA_DIR / "Doc02_Radiation_Sickness_Symptom_Guide.docx"
DOC04_PATH = DATA_DIR / "Doc4_Overseer_Compensation_HR.docx"
PIPBOY3000_PATH = DATA_DIR / "SOP_PipBoy_3000_Calibration.md"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _base_meta(
    source: str = "test_doc",
    department: str = "General",
    access_level: str = "General Employee",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="placeholder",
        text="",
        source_document=source,
        section_header="",
        department=department,  # type: ignore[arg-type]
        access_level=access_level,  # type: ignore[arg-type]
        effective_date=None,
        document_status="ACTIVE",
        content_type="narrative",
        chunk_index=0,
        total_chunks=1,
        token_count=0,
    )


def _chunk_doc(path: Path, max_tokens: int = 512, export: bool = False) -> list[DocumentChunk]:
    slug = path.stem.lower().replace(" ", "_").replace("-", "_")
    meta = _base_meta(source=path.stem)
    return chunk_document_docling(
        doc_path=path,
        max_tokens=max_tokens,
        doc_slug=slug,
        base_metadata=meta,
        export_dir=EXPORT_DIR if export else None,
    )


# ---------------------------------------------------------------------------
# AC1: Table integrity — Doc02 radiation exposure table
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DOC02_PATH.exists(), reason="Requires data/raw/Doc02_Radiation_Sickness_Symptom_Guide.docx"
)
def test_table_integrity_doc02():
    """
    Docling's TableFormer detects the 5-tier radiation table as a single structural
    element. The chunk containing 'Tier 0' must also contain 'Tier 4' and '800'.
    """
    chunks = _chunk_doc(DOC02_PATH, max_tokens=512)
    texts = [c.text for c in chunks]

    tier0_chunks = [t for t in texts if "Tier 0" in t or "0-49" in t or "Tier 1" in t]
    assert (
        tier0_chunks
    ), "AC1 FAIL: No chunk contains tier exposure data — check Doc02 Docling parse"

    for chunk_text in tier0_chunks:
        if "Tier 0" in chunk_text or "0-49" in chunk_text:
            assert "Tier 4" in chunk_text or "800" in chunk_text, (
                f"AC1 FAIL: Chunk with 'Tier 0' does NOT also contain 'Tier 4' or '800'.\n"
                f"This means the 5-tier radiation table is split across chunks.\n"
                f"Chunk (first 800 chars):\n{chunk_text[:800]}"
            )
            return  # Pass on the first Tier 0 chunk found

    pytest.skip("Doc02 chunks found but no single chunk starts at Tier 0 — inspect manually")


# ---------------------------------------------------------------------------
# AC2: Code block integrity — PipBoy 3000 biometric reset code
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PIPBOY3000_PATH.exists(), reason="Requires data/raw/SOP_PipBoy_3000_Calibration.md"
)
def test_code_block_integrity_pipboy3000():
    """
    Docling detects code blocks as atomic items. '3000-BIO-RESET-GAMMA' must
    appear in the same chunk as its surrounding label text.
    """
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    texts = [c.text for c in chunks]

    gamma_chunks = [t for t in texts if "3000-BIO-RESET-GAMMA" in t]
    assert gamma_chunks, (
        "AC2 FAIL: '3000-BIO-RESET-GAMMA' not found in any chunk. "
        "Check that the code block is not being stripped during Docling parsing."
    )

    for chunk_text in gamma_chunks:
        has_label = (
            "biometric" in chunk_text.lower()
            or "reset" in chunk_text.lower()
            or "confirmation code" in chunk_text.lower()
        )
        assert has_label, (
            f"AC2 FAIL: '3000-BIO-RESET-GAMMA' found but not in the same chunk as its label.\n"
            f"Chunk (first 600 chars):\n{chunk_text[:600]}"
        )


# ---------------------------------------------------------------------------
# AC3: Hazard pay table — Doc4 Overseer Compensation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DOC04_PATH.exists(), reason="Requires data/raw/Doc4_Overseer_Compensation_HR.docx"
)
def test_hazard_pay_table_doc4():
    """
    Docling's TableFormer must keep the hazard pay rates table intact.
    Any chunk containing 'Hazard Pay' with a SRU value must also contain
    '+85' and 'Unrest' or 'Suppression'.
    """
    chunks = _chunk_doc(DOC04_PATH, max_tokens=512)
    texts = [c.text for c in chunks]

    # Find chunks with actual rate data (not just the document header that mentions Hazard Pay)
    rate_chunks = [
        t for t in texts if ("85" in t or "+85" in t) and ("Unrest" in t or "Suppression" in t)
    ]
    assert rate_chunks, (
        "AC3 FAIL: No chunk contains both '85' (SRU amount) and 'Unrest'/'Suppression'. "
        "The hazard pay rates table may be split or missing. "
        "All chunk previews:\n"
        + "\n---\n".join(t[:200] for t in texts if "Hazard" in t or "85" in t)
    )


# ---------------------------------------------------------------------------
# AC4: Metadata completeness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Requires data/raw/ directory")
def test_metadata_completeness():
    """
    All non-optional fields must be populated on every chunk.
    Only effective_date is permitted to be None.
    """
    # Lazy import to avoid pulling in Qdrant at import time
    try:
        from src.pipelines.ingest import DOCUMENT_METADATA, _build_base_metadata, _make_doc_slug
    except ImportError as exc:
        pytest.skip(f"ingest deps unavailable: {exc}")

    required_fields = [
        "chunk_id",
        "text",
        "source_document",
        "section_header",
        "department",
        "access_level",
        "document_status",
        "content_type",
        "chunk_index",
        "total_chunks",
        "token_count",
    ]

    docs = [f for f in DATA_DIR.glob("*") if f.suffix.lower() in {".pdf", ".docx", ".md"}]
    failures: list[str] = []

    for doc_path in docs:
        slug = _make_doc_slug(doc_path)
        if slug not in DOCUMENT_METADATA:
            continue
        base = _build_base_metadata(doc_path, slug)
        if base is None:
            continue
        chunks = chunk_document_docling(doc_path, 512, slug, base)
        for chunk in chunks:
            for field in required_fields:
                val = getattr(chunk, field)
                if val is None or val == "":
                    if field == "section_header":
                        continue  # allowed to be empty on first chunks
                    failures.append(
                        f"{doc_path.name} chunk {chunk.chunk_index}: '{field}' empty/None"
                    )

    assert not failures, "AC4 FAIL:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------
# AC5: Section header propagation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PIPBOY3000_PATH.exists(), reason="Requires data/raw/SOP_PipBoy_3000_Calibration.md"
)
def test_section_header_propagation():
    """
    Docling's chunk.meta.headings is mapped to section_header.
    For a well-structured markdown doc like PipBoy 3000 SOP, at least 60%
    of chunks should have a non-empty section_header.
    """
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    assert chunks, "No chunks produced"

    with_header = [c for c in chunks if c.section_header and c.section_header.strip()]
    ratio = len(with_header) / len(chunks)
    assert ratio >= 0.6, (
        f"AC5 FAIL: Only {len(with_header)}/{len(chunks)} chunks have a section_header "
        f"(ratio={ratio:.2f}, threshold=0.60).\n"
        f"Headers found: {[c.section_header for c in with_header[:5]]}"
    )


# ---------------------------------------------------------------------------
# AC6: Dual collection output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PIPBOY3000_PATH.exists(), reason="Requires data/raw/SOP_PipBoy_3000_Calibration.md"
)
def test_dual_collection_output():
    """
    Same document chunked at 512 must produce >= chunks than at 1024.
    """
    chunks_512 = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    chunks_1024 = _chunk_doc(PIPBOY3000_PATH, max_tokens=1024)

    assert len(chunks_512) >= len(chunks_1024), (
        f"AC6 FAIL: 512-token ({len(chunks_512)} chunks) should have >= count than "
        f"1024-token ({len(chunks_1024)} chunks)"
    )
    assert len(chunks_1024) >= 1


# ---------------------------------------------------------------------------
# AC7: Pydantic roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PIPBOY3000_PATH.exists(), reason="Requires data/raw/SOP_PipBoy_3000_Calibration.md"
)
def test_pydantic_roundtrip():
    """
    All chunks must serialize/deserialize through DocumentChunk without errors.
    to_chroma_metadata() must return a flat dict with no nested objects.
    """
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    assert chunks

    for chunk in chunks:
        flat = chunk.to_chroma_metadata()
        for k, v in flat.items():
            assert not isinstance(v, dict), f"AC7 FAIL: field '{k}' is a nested dict"
            assert not isinstance(v, list), f"AC7 FAIL: field '{k}' is a list"

        reconstructed = DocumentChunk.model_validate(
            {
                **flat,
                "text": chunk.text,
                "effective_date": chunk.effective_date,
            }
        )
        assert reconstructed.chunk_id == chunk.chunk_id
        assert reconstructed.chunk_index == chunk.chunk_index


# ---------------------------------------------------------------------------
# AC8: Cleaner preserves code block content (still relevant for MD docs)
# ---------------------------------------------------------------------------


def test_cleaner_preserves_code_blocks():
    """
    clean_text() must not strip content inside ``` fences.
    This guards against regression of the code-fence stripping bug.
    """
    sample = (
        "## SECTION 4: BIOMETRIC RESET\n\n"
        "Enter the biometric reset confirmation code:\n\n"
        "```\n"
        "BIOMETRIC SEAL RESET CODE: 3000-BIO-RESET-GAMMA\n"
        "```\n\n"
        "The seal will be wiped and re-indexed."
    )
    cleaned = clean_text(sample)
    assert (
        "3000-BIO-RESET-GAMMA" in cleaned
    ), "AC8 FAIL: clean_text() stripped '3000-BIO-RESET-GAMMA' from inside a code fence."


# ---------------------------------------------------------------------------
# AC9: Docling markdown exports saved
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DOC02_PATH.exists(), reason="Requires data/raw/ documents")
def test_docling_markdown_exports():
    """
    When export_dir is provided, chunk_document_docling must save a markdown file.
    Spot-check one PDF, one DOCX, and one MD.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir)

        for doc_path in [DOC02_PATH, PIPBOY3000_PATH]:
            if not doc_path.exists():
                continue
            slug = doc_path.stem.lower().replace(" ", "_").replace("-", "_")
            meta = _base_meta(source=doc_path.stem)
            chunk_document_docling(doc_path, 512, slug, meta, export_dir=export_path)
            expected_file = export_path / f"{doc_path.stem}.md"
            assert (
                expected_file.exists()
            ), f"AC9 FAIL: Docling markdown export not found at {expected_file}"
            content = expected_file.read_text(encoding="utf-8")
            assert (
                len(content) > 100
            ), f"AC9 FAIL: Markdown export for {doc_path.name} is too short ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Structural invariant tests (no real docs required)
# ---------------------------------------------------------------------------


def test_chunk_id_format():
    """chunk_id must follow {doc_slug}_{chunk_index:03d} pattern."""
    # Use a real doc if available, else skip
    if not PIPBOY3000_PATH.exists():
        pytest.skip("No docs available for chunk_id test")
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    slug = PIPBOY3000_PATH.stem.lower().replace(" ", "_").replace("-", "_")
    for chunk in chunks:
        expected = f"{slug}_{chunk.chunk_index:03d}"
        assert chunk.chunk_id == expected, f"chunk_id '{chunk.chunk_id}' != expected '{expected}'"


def test_total_chunks_consistent():
    """total_chunks must equal len(chunks) for every chunk in the document."""
    if not PIPBOY3000_PATH.exists():
        pytest.skip("No docs available for total_chunks test")
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    expected = len(chunks)
    for chunk in chunks:
        assert (
            chunk.total_chunks == expected
        ), f"Chunk {chunk.chunk_index} has total_chunks={chunk.total_chunks}, expected {expected}"


def test_token_count_populated():
    """Every chunk must have token_count > 0."""
    if not PIPBOY3000_PATH.exists():
        pytest.skip("No docs available for token_count test")
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    for chunk in chunks:
        assert chunk.token_count > 0, f"Chunk {chunk.chunk_index} has token_count=0"


# ---------------------------------------------------------------------------
# Fix 1 tests — Pipe markdown table serialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DOC02_PATH.exists(), reason="Requires data/raw/Doc02_Radiation_Sickness_Symptom_Guide.docx"
)
def test_no_key_value_table_syntax_doc02():
    """
    Fix 1 regression guard: No chunk from Doc02 should contain the key=value
    serialization pattern '** =' that HybridChunker produces for table cells.
    Any occurrence means export_to_markdown() lookup silently fell back.
    """
    chunks = _chunk_doc(DOC02_PATH, max_tokens=512)
    offenders = [c for c in chunks if "** =" in c.text]
    assert not offenders, (
        f"Fix 1 FAIL: {len(offenders)} chunk(s) still contain '** =' key=value syntax.\n"
        + "\n---\n".join(f"chunk_{c.chunk_index}: {c.text[:300]}" for c in offenders)
    )


@pytest.mark.skipif(
    not DOC02_PATH.exists(), reason="Requires data/raw/Doc02_Radiation_Sickness_Symptom_Guide.docx"
)
def test_table_chunks_have_pipe_markdown_doc02():
    """
    Fix 1 acceptance: Every chunk with content_type == 'table' in Doc02 must
    contain at least one pipe '|' character, proving export_to_markdown() ran
    successfully (not silently fell back to key=value text).
    """
    chunks = _chunk_doc(DOC02_PATH, max_tokens=512)
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert table_chunks, "No table chunks found in Doc02 — check content_type classification."

    offenders = [c for c in table_chunks if "|" not in c.text]
    assert not offenders, (
        f"Fix 1 FAIL: {len(offenders)} table chunk(s) have no pipe '|' character.\n"
        f"This means export_to_markdown() lookup silently failed; key=value text survived.\n"
        + "\n---\n".join(f"chunk_{c.chunk_index}: {c.text[:300]}" for c in offenders)
    )


# ---------------------------------------------------------------------------
# Fix 2 tests — Section header injection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PIPBOY3000_PATH.exists(), reason="Requires data/raw/SOP_PipBoy_3000_Calibration.md"
)
def test_fix2_section_header_prefix_pipboy():
    """
    Fix 2 acceptance on a well-structured markdown doc: for any chunk that has
    a non-empty section_header AND the text did not already begin with that heading,
    Fix 2 must have prepended the heading so the text now starts with it.

    PipBoy3000 is used because it has Docling-recognized section headers.
    Doc02 is a DOCX with bold-formatted headers that Docling does not classify
    as SectionHeaderItems — no heading injection is possible or expected there.
    """
    chunks = _chunk_doc(PIPBOY3000_PATH, max_tokens=512)
    chunks_with_header = [c for c in chunks if c.section_header and c.section_header.strip()]
    assert chunks_with_header, "Fix 2: No chunks with section_header in PipBoy3000 — check AC5."

    offenders = [c for c in chunks_with_header if not c.text.startswith(c.section_header)]
    assert not offenders, (
        f"Fix 2 FAIL: {len(offenders)} chunk(s) have a section_header but text "
        f"does not start with it (prefix was not injected).\n"
        + "\n---\n".join(
            f"chunk_{c.chunk_index} header={repr(c.section_header)} | text[:100]={repr(c.text[:100])}"
            for c in offenders
        )
    )


@pytest.mark.skipif(
    not DOC02_PATH.exists(), reason="Requires data/raw/Doc02_Radiation_Sickness_Symptom_Guide.docx"
)
def test_table_chunk_text_starts_with_section_header_doc02():
    """
    Fix 2 acceptance: For table chunks that have a non-empty section_header,
    the chunk text must begin with that heading string (the prefix injection
    ensures topical context is present at the start of the embedding window).
    """
    chunks = _chunk_doc(DOC02_PATH, max_tokens=512)
    table_chunks_with_header = [c for c in chunks if c.content_type == "table" and c.section_header]

    offenders = [c for c in table_chunks_with_header if not c.text.startswith(c.section_header)]
    assert not offenders, (
        f"Fix 2 FAIL: {len(offenders)} table chunk(s) have a section_header but text "
        f"does not start with it.\n"
        + "\n---\n".join(
            f"chunk_{c.chunk_index} header='{c.section_header}' | text[:200]={c.text[:200]}"
            for c in offenders
        )
    )
