"""Audit logging function — called after every /api/query invocation."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.audit.models import AuditLog
from src.pipelines.retrieval_chain import CitedResponse


def log_query_event(
    db: Session,
    user_id: str,
    username: str,
    query: str,
    response: CitedResponse,
) -> AuditLog:
    """Create and commit one AuditLog row.

    Args:
        db:        Active SQLAlchemy session.
        user_id:   Authenticated user's ID (from JWT sub).
        username:  Human-readable username (from JWT payload).
        query:     The original query string.
        response:  CitedResponse returned by the retrieval chain.

    Returns:
        The committed AuditLog instance (with `id` populated).
    """
    accessed_docs = [src.source_document for src in response.sources]
    entry = AuditLog(
        user_id=user_id,
        username=username,
        query=query,
        accessed_documents=accessed_docs,
        chunk_count=response.retrieved_chunk_count,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
