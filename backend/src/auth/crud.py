"""CRUD helpers for the auth layer."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.auth.models import User


def get_user(db: Session, user_id: str) -> User | None:
    """Return the User ORM object for the given id, or None if not found."""
    return db.get(User, user_id)


def get_allowed_levels(db: Session, user_id: str) -> list[str] | None:
    """Return the list of Qdrant access_level values the user's role permits.

    Returns None if the user_id does not exist in the database.
    """
    user = get_user(db, user_id)
    if user is None:
        return None
    return user.role.allowed_access_levels
