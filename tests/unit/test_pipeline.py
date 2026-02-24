"""
Unit tests for the Vault-Tec preprocessing pipeline.

Covers:
  - ChunkMetadata Pydantic model validation
  - clean_text() cleaning steps
  - chunk_document() chunking and metadata stamping
"""

from __future__ import annotations

import pytest

from src.pipelines.chunker import chunk_document
from src.pipelines.cleaner import clean_text
from src.pipelines.models import ChunkMetadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_metadata() -> ChunkMetadata:
    """A fully-populated, valid ChunkMetadata instance for use in tests."""
    return ChunkMetadata(
        chunk_id="00000000-0000-0000-0000-000000000001",
        source_document="Doc01_Vault_Dweller_Code_of_Conduct.pdf",
        doc_format="pdf",
        access_level="general",
        department="general",
        doc_date="2076-08-01",
        doc_status="active",
        chunk_index=0,
        total_chunks=1,
        section_title="",
    )


# ---------------------------------------------------------------------------
# ChunkMetadata validation
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_valid_metadata_passes(self, base_metadata: ChunkMetadata):
        """A fully-populated metadata instance should pass model_validate."""
        dumped = base_metadata.model_dump()
        validated = ChunkMetadata.model_validate(dumped)
        assert validated.chunk_id == base_metadata.chunk_id

    def test_invalid_access_level_raises(self):
        """access_level must be one of the four approved literals."""
        with pytest.raises(Exception):  # pydantic ValidationError
            ChunkMetadata(
                chunk_id="abc",
                source_document="foo.pdf",
                doc_format="pdf",
                access_level="superadmin",  # invalid
                department="general",
                doc_date="2076-01-01",
                doc_status="active",
                chunk_index=0,
                total_chunks=1,
            )

    def test_invalid_doc_status_raises(self):
        """doc_status must be 'active', 'archived', or 'unknown'."""
        with pytest.raises(Exception):
            ChunkMetadata(
                chunk_id="abc",
                source_document="foo.pdf",
                doc_format="pdf",
                access_level="general",
                department="general",
                doc_date="2076-01-01",
                doc_status="deleted",  # invalid
                chunk_index=0,
                total_chunks=1,
            )

    def test_invalid_doc_format_raises(self):
        """doc_format must be one of 'pdf', 'docx', 'md', 'txt'."""
        with pytest.raises(Exception):
            ChunkMetadata(
                chunk_id="abc",
                source_document="foo.xlsx",
                doc_format="xlsx",  # invalid
                access_level="general",
                department="general",
                doc_date="1970-01-01",
                doc_status="active",
                chunk_index=0,
                total_chunks=1,
            )

    def test_all_access_levels_accepted(self):
        """Each valid access level should be accepted without error."""
        for level in ("general", "hr", "marketing", "admin"):
            meta = ChunkMetadata(
                chunk_id="abc",
                source_document="test.pdf",
                doc_format="pdf",
                access_level=level,  # type: ignore[arg-type]
                department=level,  # type: ignore[arg-type]
                doc_date="1970-01-01",
                doc_status="active",
                chunk_index=0,
                total_chunks=1,
            )
            assert meta.access_level == level

    def test_default_doc_status_is_active(self):
        """doc_status should default to 'active' when not provided."""
        meta = ChunkMetadata(
            chunk_id="abc",
            source_document="test.pdf",
            doc_format="pdf",
            access_level="general",
            department="general",
            doc_date="1970-01-01",
            chunk_index=0,
            total_chunks=1,
        )
        assert meta.doc_status == "active"

    def test_model_dump_is_flat(self, base_metadata: ChunkMetadata):
        """model_dump() must return a flat dict (ChromaDB requirement)."""
        dumped = base_metadata.model_dump()
        for v in dumped.values():
            assert not isinstance(v, dict), "metadata must be a flat dict"
            assert not isinstance(v, list), "metadata must be a flat dict"

    def test_archived_doc_status(self):
        """doc_status='archived' should be accepted (used for doc12)."""
        meta = ChunkMetadata(
            chunk_id="abc",
            source_document="doc12_surface_exploration_archived.docx",
            doc_format="docx",
            access_level="general",
            department="general",
            doc_date="1970-01-01",
            doc_status="archived",
            chunk_index=0,
            total_chunks=1,
        )
        assert meta.doc_status == "archived"


# ---------------------------------------------------------------------------
# clean_text() tests
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_smart_quotes_normalised(self):
        raw = "\u201cVault-Tec\u201d is \u2018great\u2019"
        result = clean_text(raw)
        assert '"Vault-Tec"' in result or "'great'" in result
        assert "\u201c" not in result
        assert "\u2019" not in result

    def test_em_dash_replaced(self):
        raw = "Overseer\u2014approved policy"
        result = clean_text(raw)
        assert "--" in result
        assert "\u2014" not in result

    def test_excess_newlines_collapsed(self):
        raw = "Line 1\n\n\n\n\nLine 2"
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_uppercase_header_removed(self):
        """Short all-caps lines (< 60 chars) are treated as page headers and removed."""
        raw = "VAULT-TEC CORPORATION\n\nThis is the actual content of the document."
        result = clean_text(raw)
        assert "VAULT-TEC CORPORATION" not in result
        assert "actual content" in result

    def test_uppercase_header_long_line_kept(self):
        """Long uppercase lines (>= 60 chars) should NOT be stripped."""
        long_header = "A" * 60
        raw = f"{long_header}\n\nBody text here."
        result = clean_text(raw)
        assert long_header in result

    def test_form_fill_fields_removed(self):
        """Lines of underscores (form fields) should be stripped."""
        raw = "Name: ______________________\n\nSignature required above."
        result = clean_text(raw)
        assert "______" not in result
        assert "Signature required above" in result

    def test_boilerplate_footer_removed(self):
        """Vault-Tec legal boilerplate should be stripped."""
        raw = "Policy content here.\n\nVAULT-TEC CORPORATION accepts no liability for any radiation exposure."
        result = clean_text(raw)
        assert "VAULT-TEC CORPORATION accepts no liability" not in result
        assert "Policy content here" in result

    def test_returns_string(self):
        assert isinstance(clean_text("hello"), str)

    def test_empty_input(self):
        assert clean_text("") == ""

    def test_non_breaking_space_removed(self):
        raw = "Hello\u00a0World"
        result = clean_text(raw)
        assert "\u00a0" not in result


# ---------------------------------------------------------------------------
# chunk_document() tests
# ---------------------------------------------------------------------------


class TestChunkDocument:
    _SAMPLE_TEXT = (
        "# Introduction\n\n"
        + "The Vault-Tec Employee Portal is the authoritative source of all HR, "
        "Administrative, and General information for Vault residents. "
        "All employees are required to read and comply with all policies contained herein. "
        "Failure to comply may result in disciplinary action up to and including "
        "reassignment to Sector 7-G. "
        * 30
        + "\n\n# Safety Procedures\n\n"
        + "In the event of a radroach infestation, residents must immediately proceed "
        "to the nearest security checkpoint and await further instructions from the Overseer. "
        "Do not attempt to engage radroaches with improvised weaponry. " * 30
    )

    def _make_template(
        self,
        access_level: str = "general",
        doc_status: str = "active",
        doc_format: str = "md",
    ) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_id="placeholder",
            source_document="test_doc.md",
            doc_format=doc_format,  # type: ignore[arg-type]
            access_level=access_level,  # type: ignore[arg-type]
            department=access_level,  # type: ignore[arg-type]
            doc_date="1970-01-01",
            doc_status=doc_status,  # type: ignore[arg-type]
            chunk_index=0,
            total_chunks=1,
            section_title="",
        )

    def test_returns_list_of_tuples(self):
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        assert isinstance(result, list)
        assert len(result) > 0
        for text, meta in result:
            assert isinstance(text, str)
            assert isinstance(meta, ChunkMetadata)

    def test_chunk_index_is_sequential(self):
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        for expected_i, (_, meta) in enumerate(result):
            assert meta.chunk_index == expected_i

    def test_total_chunks_consistent(self):
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        total = len(result)
        for _, meta in result:
            assert meta.total_chunks == total

    def test_every_chunk_has_unique_chunk_id(self):
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        ids = [meta.chunk_id for _, meta in result]
        assert len(ids) == len(set(ids)), "All chunk_ids must be unique UUIDs"

    def test_access_level_preserved(self):
        template = self._make_template(access_level="admin")
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        for _, meta in result:
            assert meta.access_level == "admin"

    def test_doc_status_preserved(self):
        template = self._make_template(doc_status="archived")
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        for _, meta in result:
            assert meta.doc_status == "archived"

    def test_markdown_section_title_extracted(self):
        """At least one chunk should have a non-empty section_title from the # heading."""
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        section_titles = {meta.section_title for _, meta in result}
        # Should detect "Introduction" or "Safety Procedures"
        assert any(
            t != "" for t in section_titles
        ), "Expected at least one chunk to have a section_title from Markdown headings"

    def test_larger_chunk_size_produces_fewer_chunks(self):
        template = self._make_template()
        result_256 = chunk_document(self._SAMPLE_TEXT, 256, 32, template, doc_format="md")
        result_1024 = chunk_document(self._SAMPLE_TEXT, 1024, 128, template, doc_format="md")
        assert len(result_256) >= len(
            result_1024
        ), "Smaller chunk_size should produce more (or equal) chunks"

    def test_each_chunk_passes_pydantic_validation(self):
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 512, 64, template, doc_format="md")
        for _, meta in result:
            # This will raise if any field is invalid
            ChunkMetadata.model_validate(meta.model_dump())

    def test_template_chunk_id_not_used(self):
        """The 'placeholder' chunk_id on the template must be replaced in every chunk."""
        template = self._make_template()
        result = chunk_document(self._SAMPLE_TEXT, 512, 64, template, doc_format="md")
        for _, meta in result:
            assert meta.chunk_id != "placeholder"
