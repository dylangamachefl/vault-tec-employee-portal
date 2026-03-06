"""Idempotent seed script for the auth/audit database.

Run directly:
    python -m src.auth.seed

Or call run_seed() programmatically (e.g. from the FastAPI startup event).
The script is safe to run multiple times — existing rows are not duplicated.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from src.auth.database import SessionLocal, create_all_tables
from src.auth.models import Role, User

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role definitions — mirror the AccessLevelV2 values used in Qdrant payloads
# ---------------------------------------------------------------------------

_ROLES: list[dict] = [
    {
        "id": "role_general",
        "name": "general",
        "display_name": "General Employee",
        "allowed_access_levels": ["General Employee"],
    },
    {
        "id": "role_hr",
        "name": "hr",
        "display_name": "HR Specialist",
        "allowed_access_levels": ["General Employee", "HR Restricted"],
    },
    {
        "id": "role_marketing",
        "name": "marketing",
        "display_name": "Marketing Associate",
        "allowed_access_levels": ["General Employee", "Marketing Eyes Only"],
    },
    {
        "id": "role_admin",
        "name": "admin",
        "display_name": "IT Administrator",
        "allowed_access_levels": [
            "General Employee",
            "HR Restricted",
            "Marketing Eyes Only",
            "Admin Eyes Only",
        ],
    },
]

# ---------------------------------------------------------------------------
# Demo users — IDs match the LoginView.tsx DEMO_USERS and api/main.py
# ---------------------------------------------------------------------------

_USERS: list[dict] = [
    {"id": "u1", "username": "Dweller-101", "role_id": "role_general"},
    {"id": "u2", "username": "Barnsworth B.", "role_id": "role_hr"},
    {"id": "u3", "username": "Gable M.", "role_id": "role_marketing"},
    {"id": "u4", "username": "Carmichael J.", "role_id": "role_admin"},
]


def _seed_roles(db: Session) -> None:
    for role_data in _ROLES:
        if db.get(Role, role_data["id"]) is None:
            db.add(Role(**role_data))
            logger.info("Seeded role: %s", role_data["name"])


def _seed_users(db: Session) -> None:
    for user_data in _USERS:
        if db.get(User, user_data["id"]) is None:
            db.add(User(**user_data))
            logger.info("Seeded user: %s", user_data["id"])


def run_seed() -> None:
    """Create tables and upsert seed data. Safe to call multiple times."""
    create_all_tables()
    db = SessionLocal()
    try:
        _seed_roles(db)
        db.flush()  # Ensure roles exist before users (FK constraint)
        _seed_users(db)
        db.commit()
        logger.info("Seed complete.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_seed()
    print("✓ Auth database seeded successfully.")
