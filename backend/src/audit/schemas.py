"""Pydantic schemas for audit log API responses."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AuditLogOut(BaseModel):
    """Serialised audit log entry returned by GET /api/audit-logs."""

    model_config = {"from_attributes": True}

    id: int
    user_id: str
    username: str
    timestamp: datetime
    query: str
    accessed_documents: list[str]
    chunk_count: int
