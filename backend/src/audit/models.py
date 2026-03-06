"""SQLAlchemy ORM model for the AuditLog table."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.auth.database import Base


class AuditLog(Base):
    """One row per user query — written after every successful /api/query call."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    # UTC timestamp — stored as a timezone-aware datetime
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(tz=UTC),
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    # Flat list of source_document strings from CitedResponse.sources
    accessed_documents: Mapped[list] = mapped_column(JSON, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} user={self.user_id!r} "
            f"docs={self.accessed_documents} ts={self.timestamp}>"
        )
