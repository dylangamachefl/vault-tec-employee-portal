"""Unit tests for the audit trail logger."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.audit.logger import log_query_event
from src.audit.models import AuditLog
from src.auth.database import Base
from src.pipelines.retrieval_chain import CitedResponse, SourceCitation

# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    import src.audit.models  # noqa: F401
    import src.auth.models  # noqa: F401

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cited_response(query: str, sources: list[str]) -> CitedResponse:
    return CitedResponse(
        answer="Test answer.",
        sources=[
            SourceCitation(
                source_document=doc,
                access_level="General Employee",
                department="General",
            )
            for doc in sources
        ],
        retrieved_chunk_count=len(sources),
        retrieved_chunks=["chunk text"] * len(sources),
        query=query,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_log_query_event_creates_row(db_session):
    """log_query_event() must persist exactly one row to the DB."""
    before = db_session.query(AuditLog).count()
    response = _make_cited_response("What is radiation?", ["Doc02_Radiation"])
    log_query_event(db_session, "u1", "Dweller-101", "What is radiation?", response)
    after = db_session.query(AuditLog).count()
    assert after == before + 1


def test_audit_log_user_id(db_session):
    """The committed row must record the correct user_id."""
    response = _make_cited_response("Test query A", ["DocA"])
    entry = log_query_event(db_session, "u2", "Barnsworth B.", "Test query A", response)
    assert entry.user_id == "u2"
    assert entry.username == "Barnsworth B."


def test_audit_log_query_text(db_session):
    """The committed row must store the exact query string."""
    query = "What are the evacuation procedures?"
    response = _make_cited_response(query, ["Doc03_Evacuation"])
    entry = log_query_event(db_session, "u1", "Dweller-101", query, response)
    assert entry.query == query


def test_audit_log_accessed_documents(db_session):
    """accessed_documents must list the source_document values from CitedResponse.sources."""
    docs = ["Doc04_Overseer_Compensation", "Doc05_NVDR"]
    response = _make_cited_response("Overseer pay?", docs)
    entry = log_query_event(db_session, "u2", "Barnsworth B.", "Overseer pay?", response)
    assert set(entry.accessed_documents) == set(docs)


def test_audit_log_chunk_count(db_session):
    """chunk_count must equal CitedResponse.retrieved_chunk_count."""
    response = _make_cited_response("Admin query", ["Doc10", "Doc11"])
    entry = log_query_event(db_session, "u4", "Carmichael J.", "Admin query", response)
    assert entry.chunk_count == response.retrieved_chunk_count


def test_audit_log_has_timestamp(db_session):
    """The committed row must have a non-None timestamp."""
    response = _make_cited_response("Another query", ["DocX"])
    entry = log_query_event(db_session, "u1", "Dweller-101", "Another query", response)
    assert entry.timestamp is not None


def test_audit_log_id_auto_increments(db_session):
    """Multiple log entries must receive unique, increasing IDs."""
    r1 = _make_cited_response("Query 1", ["DocA"])
    r2 = _make_cited_response("Query 2", ["DocB"])
    e1 = log_query_event(db_session, "u3", "Gable M.", "Query 1", r1)
    e2 = log_query_event(db_session, "u3", "Gable M.", "Query 2", r2)
    assert e2.id > e1.id
