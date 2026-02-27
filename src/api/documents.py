"""
Document registry for the API layer.

Derived from src/pipelines/ingest.py DOCUMENT_METADATA. This module
provides the document catalogue for GET /api/documents without importing
the full ingestion pipeline (which pulls heavy ML dependencies).
"""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Access level hierarchy — determines which docs each clearance can see
# ---------------------------------------------------------------------------

_ACCESS_HIERARCHY: dict[str, list[str]] = {
    "General": ["General Employee"],
    "HR": ["General Employee", "HR Restricted"],
    "Marketing": ["General Employee", "Marketing Eyes Only"],
    "Admin": ["General Employee", "HR Restricted", "Marketing Eyes Only", "Admin Eyes Only"],
}


# ---------------------------------------------------------------------------
# Document catalogue
# ---------------------------------------------------------------------------


class DocumentRecord(BaseModel):
    id: str
    title: str
    department: str
    accessLevel: str
    status: str
    effectiveDate: str


_CATALOGUE: list[DocumentRecord] = [
    DocumentRecord(
        id="doc01",
        title="Vault Dweller's Code of Conduct",
        department="General",
        accessLevel="General Employee",
        status="ACTIVE",
        effectiveDate="2076-05-01",
    ),
    DocumentRecord(
        id="doc02",
        title="Radiation Sickness Symptom Guide",
        department="General",
        accessLevel="General Employee",
        status="ACTIVE",
        effectiveDate="2076-08-01",
    ),
    DocumentRecord(
        id="doc03",
        title="Emergency Evacuation Procedures",
        department="General",
        accessLevel="General Employee",
        status="ACTIVE",
        effectiveDate="2076-09-01",
    ),
    DocumentRecord(
        id="doc12",
        title="Surface Exploration Guide (Archived)",
        department="General",
        accessLevel="General Employee",
        status="ARCHIVED",
        effectiveDate="2077-01-01",
    ),
    DocumentRecord(
        id="doc13",
        title="Surface Exploration Operations Manual",
        department="General",
        accessLevel="General Employee",
        status="ACTIVE",
        effectiveDate="2077-11-01",
    ),
    DocumentRecord(
        id="doc04",
        title="Overseer Compensation Schedule",
        department="HR",
        accessLevel="HR Restricted",
        status="ACTIVE",
        effectiveDate="2076-01-01",
    ),
    DocumentRecord(
        id="doc05",
        title="Non-Violent Disciplinary Report (NVDR)",
        department="HR",
        accessLevel="HR Restricted",
        status="ACTIVE",
        effectiveDate="2076-03-01",
    ),
    DocumentRecord(
        id="doc06",
        title="G.O.A.T. Exam Administration Manual",
        department="HR",
        accessLevel="HR Restricted",
        status="ACTIVE",
        effectiveDate="2076-06-01",
    ),
    DocumentRecord(
        id="doc07",
        title="Vault 76 Tricentennial Promotional Strategy",
        department="Marketing",
        accessLevel="Marketing Eyes Only",
        status="ACTIVE",
        effectiveDate="2076-02-01",
    ),
    DocumentRecord(
        id="doc08",
        title="G.E.C.K. Advertising Guidelines",
        department="Marketing",
        accessLevel="Marketing Eyes Only",
        status="ACTIVE",
        effectiveDate="2076-04-01",
    ),
    DocumentRecord(
        id="doc09",
        title="End-of-World Crisis Response Messaging",
        department="Marketing",
        accessLevel="Marketing Eyes Only",
        status="ACTIVE",
        effectiveDate="2077-10-01",
    ),
    DocumentRecord(
        id="doc10",
        title="Vault Door Override Protocol",
        department="Admin",
        accessLevel="Admin Eyes Only",
        status="ACTIVE",
        effectiveDate="2076-12-01",
    ),
    DocumentRecord(
        id="doc11",
        title="ZAX Mainframe Root Access Manual",
        department="Admin",
        accessLevel="Admin Eyes Only",
        status="ACTIVE",
        effectiveDate="2076-07-01",
    ),
    DocumentRecord(
        id="sop01",
        title="PipBoy 2000 Calibration SOP",
        department="Admin",
        accessLevel="Admin Eyes Only",
        status="ACTIVE",
        effectiveDate="2075-03-01",
    ),
    DocumentRecord(
        id="sop02",
        title="PipBoy 3000 Calibration SOP",
        department="Admin",
        accessLevel="Admin Eyes Only",
        status="ACTIVE",
        effectiveDate="2076-11-01",
    ),
]


def get_accessible_documents(access_level: str) -> list[DocumentRecord]:
    """Return documents visible to the given access level."""
    allowed = _ACCESS_HIERARCHY.get(access_level, ["General Employee"])
    return [doc for doc in _CATALOGUE if doc.accessLevel in allowed]
