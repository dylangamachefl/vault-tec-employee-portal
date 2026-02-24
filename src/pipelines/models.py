"""
Pydantic schemas for chunk metadata.

ChunkMetadata — legacy schema (backward-compat with existing tests and ChromaDB).
DocumentChunk  — new schema introduced in Phase 1 refactor to support structure-aware
                 chunking and Phase 2 RBAC filtering.

Every chunk stored in ChromaDB must pass DocumentChunk.model_validate().
This is the single source of truth for the metadata contract used across
RBAC filtering (Phase 2), staleness detection (Phase 3), and contradiction
flagging (Phase 3).
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Legacy types — kept for backward compatibility with existing tests
# ---------------------------------------------------------------------------

AccessLevel = Literal["general", "hr", "marketing", "admin"]
Department = Literal["general", "hr", "marketing", "admin"]
DocFormat = Literal["pdf", "docx", "md", "txt"]
DocStatus = Literal["active", "archived", "unknown"]


class ChunkMetadata(BaseModel):
    # Identity
    chunk_id: str = Field(..., description="UUID4 — unique per chunk")
    source_document: str = Field(
        ..., description="Original filename, e.g. 'Doc02_Radiation_Sickness_Symptom_Guide.docx'"
    )
    doc_format: DocFormat

    # Access Control — used by RBAC layer in Phase 2
    access_level: AccessLevel = Field(
        ..., description="The minimum role required to retrieve this chunk"
    )
    department: Department

    # Temporal — used by staleness detector in Phase 3
    doc_date: str = Field(
        ...,
        description="ISO date string extracted from document, e.g. '2076-08-01'. Use '1970-01-01' if unknown.",
    )
    doc_status: DocStatus = Field(
        default="active", description="'archived' if document is explicitly superseded"
    )

    # Chunking context
    chunk_index: int = Field(
        ..., description="0-based position of this chunk within its source document"
    )
    total_chunks: int = Field(..., description="Total number of chunks from this source document")

    # Content summary — used by contradiction detector in Phase 3
    section_title: str = Field(
        default="", description="Nearest heading above this chunk, if detectable"
    )


# ---------------------------------------------------------------------------
# New schema — Phase 1 refactor (structure-aware chunker)
# ---------------------------------------------------------------------------

DepartmentV2 = Literal["General", "HR", "Marketing", "Admin"]
AccessLevelV2 = Literal[
    "General Employee", "HR Restricted", "Marketing Eyes Only", "Admin Eyes Only"
]
ContentType = Literal["narrative", "table", "procedure", "code_block", "list"]
DocumentStatus = Literal["ACTIVE", "ARCHIVED"]


class DocumentChunk(BaseModel):
    """
    Output schema for the structure-aware chunker.

    chunk_id format:  "{doc_slug}_{chunk_index:03d}"
    doc_slug:        lowercase filename without extension, spaces → underscores
    """

    chunk_id: str = Field(..., description='Format: "{doc_slug}_{chunk_index:03d}"')
    text: str = Field(..., description="The chunk content")
    source_document: str = Field(
        ...,
        description='Canonical document filename, e.g. "Doc02_Radiation_Sickness_Symptom_Guide"',
    )
    section_header: str = Field(
        ..., description="Nearest preceding heading (e.g. '## Section 2: Symptom Identification')"
    )
    department: DepartmentV2
    access_level: AccessLevelV2
    effective_date: date | None = Field(
        None, description="Document effective/issue date; None if genuinely absent"
    )
    document_status: DocumentStatus = Field(default="ACTIVE")
    content_type: ContentType = Field(..., description="Structural category of the chunk content")
    chunk_index: int = Field(..., description="0-indexed position within document")
    total_chunks: int = Field(..., description="Total chunks for this document")
    token_count: int = Field(..., description="Actual tiktoken cl100k_base token count")

    def to_chroma_metadata(self) -> dict:
        """
        Return a flat dict safe for ChromaDB (no nested objects or None values).
        effective_date is serialised as ISO string or empty string.
        """
        return {
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "section_header": self.section_header,
            "department": self.department,
            "access_level": self.access_level,
            "effective_date": self.effective_date.isoformat() if self.effective_date else "",
            "document_status": self.document_status,
            "content_type": self.content_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
        }
