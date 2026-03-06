"""SQLAlchemy engine + session factory for the auth/audit SQLite database."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.auth_db_url,
    connect_args={"check_same_thread": False},  # Required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency — yields a SQLAlchemy session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """Create all ORM tables. Called by the seed script and app startup."""
    # Import models here so their table definitions are registered on Base
    import src.audit.models  # noqa: F401
    import src.auth.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
