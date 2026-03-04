"""Unit tests for the RBAC auth layer (no Qdrant, in-memory SQLite)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.auth.crud import get_allowed_levels, get_user
from src.auth.database import Base
from src.auth.models import Role, User

# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_session():
    """Create an in-memory SQLite DB seeded with the 4 demo roles and users."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    # Import audit models so the shared Base includes audit_logs table
    import src.audit.models  # noqa: F401
    import src.auth.models  # noqa: F401

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Seed roles
    roles = [
        Role(
            id="role_general",
            name="general",
            display_name="General Employee",
            allowed_access_levels=["General Employee"],
        ),
        Role(
            id="role_hr",
            name="hr",
            display_name="HR Specialist",
            allowed_access_levels=["General Employee", "HR Restricted"],
        ),
        Role(
            id="role_marketing",
            name="marketing",
            display_name="Marketing Associate",
            allowed_access_levels=["General Employee", "Marketing Eyes Only"],
        ),
        Role(
            id="role_admin",
            name="admin",
            display_name="IT Administrator",
            allowed_access_levels=[
                "General Employee",
                "HR Restricted",
                "Marketing Eyes Only",
                "Admin Eyes Only",
            ],
        ),
    ]
    db.add_all(roles)
    db.flush()

    users = [
        User(id="u1", username="Dweller-101", role_id="role_general"),
        User(id="u2", username="Barnsworth B.", role_id="role_hr"),
        User(id="u3", username="Gable M.", role_id="role_marketing"),
        User(id="u4", username="Carmichael J.", role_id="role_admin"),
    ]
    db.add_all(users)
    db.commit()

    yield db
    db.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_user_returns_user(db_session):
    """get_user() with a valid id returns the User ORM object."""
    user = get_user(db_session, "u1")
    assert user is not None
    assert user.id == "u1"
    assert user.username == "Dweller-101"


def test_unknown_user_returns_none(db_session):
    """get_user() with an unknown id returns None."""
    assert get_user(db_session, "not-a-real-id") is None


def test_get_allowed_levels_general(db_session):
    """u1 (General) may only retrieve 'General Employee' chunks."""
    levels = get_allowed_levels(db_session, "u1")
    assert levels == ["General Employee"]


def test_get_allowed_levels_hr(db_session):
    """u2 (HR) may retrieve General Employee and HR Restricted chunks."""
    levels = get_allowed_levels(db_session, "u2")
    assert "General Employee" in levels
    assert "HR Restricted" in levels
    assert "Admin Eyes Only" not in levels


def test_get_allowed_levels_marketing(db_session):
    """u3 (Marketing) may retrieve General Employee and Marketing Eyes Only chunks."""
    levels = get_allowed_levels(db_session, "u3")
    assert "General Employee" in levels
    assert "Marketing Eyes Only" in levels
    assert "HR Restricted" not in levels
    assert "Admin Eyes Only" not in levels


def test_get_allowed_levels_admin(db_session):
    """u4 (Admin) may retrieve all four access levels."""
    levels = get_allowed_levels(db_session, "u4")
    assert set(levels) == {
        "General Employee",
        "HR Restricted",
        "Marketing Eyes Only",
        "Admin Eyes Only",
    }


def test_general_cannot_see_hr_docs(db_session):
    """'HR Restricted' must NOT be in u1's allowed levels."""
    levels = get_allowed_levels(db_session, "u1")
    assert "HR Restricted" not in levels


def test_general_cannot_see_admin_docs(db_session):
    """'Admin Eyes Only' must NOT be in u1's allowed levels."""
    levels = get_allowed_levels(db_session, "u1")
    assert "Admin Eyes Only" not in levels


def test_get_allowed_levels_unknown_user(db_session):
    """get_allowed_levels() returns None for an unknown user."""
    assert get_allowed_levels(db_session, "ghost") is None
