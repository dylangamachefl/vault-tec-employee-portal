"""Integration tests for Phase 2 RBAC API endpoints.

Uses FastAPI TestClient with dependency overrides:
  - get_db    → in-memory SQLite using StaticPool (single shared connection)
  - VaultRetriever.query is patched to avoid Qdrant/LLM calls.

All tests run without Docker or external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.audit.models  # noqa: F401
import src.auth.models  # noqa: F401
from src.api.main import app
from src.auth.database import Base, get_db
from src.auth.models import Role, User
from src.pipelines.retrieval_chain import CitedResponse, SourceCitation

# ---------------------------------------------------------------------------
# Shared in-memory engine (StaticPool keeps a single connection for all
# sessions; crucial for in-memory SQLite to work across test calls).
# ---------------------------------------------------------------------------

TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Import ORM models so all tables are registered on Base before create_all
# (Models imported above)

Base.metadata.create_all(TEST_ENGINE)

TestSession = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def _seed(db) -> None:
    """Seed roles and users into the in-memory DB."""
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


# Seed once at module import time
with TestSession() as _seed_db:
    _seed(_seed_db)


def override_get_db():
    """Dependency override — yields a session backed by the shared in-memory engine."""
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[get_db] = override_get_db
    # Suppress startup seed (DB already has data)
    with patch("src.api.main.on_startup"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_cited_response(access_levels: list[str]) -> CitedResponse:
    return CitedResponse(
        answer="Vault-Tec answer.",
        sources=[
            SourceCitation(
                source_document=f"Doc_{lvl.replace(' ', '_')}",
                access_level=lvl,
                department="General",
            )
            for lvl in access_levels
        ],
        retrieved_chunk_count=len(access_levels),
        retrieved_chunks=["chunk"] * len(access_levels),
        query="test query",
    )


def _login(client, user_id: str) -> str:
    resp = client.post("/api/login", json={"user_id": user_id})
    assert resp.status_code == 200, f"Login failed for {user_id}: {resp.text}"
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# POST /api/login tests
# ---------------------------------------------------------------------------


def test_login_returns_jwt(client):
    """Valid user_id → 200 with access_token and token_type fields."""
    resp = client.post("/api/login", json={"user_id": "u1"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert len(body["access_token"]) > 20


def test_login_unknown_user_404(client):
    """Unknown user_id → 404."""
    resp = client.post("/api/login", json={"user_id": "ghost"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/query auth tests
# ---------------------------------------------------------------------------


def test_query_without_token_401(client):
    """Omitting Authorization header must return 401."""
    resp = client.post("/api/query", json={"query": "test"})
    assert resp.status_code == 401


def test_query_with_invalid_token_401(client):
    """Sending a garbage token must return 401."""
    resp = client.post(
        "/api/query",
        json={"query": "test"},
        headers={"Authorization": "Bearer this.is.garbage"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/query RBAC tests
# ---------------------------------------------------------------------------


def test_query_as_general_only_sees_general_levels(client):
    """u1 (General) — access_filter passed to query() must contain only General Employee."""
    token = _login(client, "u1")

    with patch("src.api.main.VaultRetriever") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.query.return_value = _fake_cited_response(["General Employee"])

        resp = client.post(
            "/api/query",
            json={"query": "What is radiation?", "top_k": 5},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    call_args = mock_instance.query.call_args
    # access_filter is passed as a keyword argument
    access_filter = call_args.kwargs.get("access_filter")
    assert access_filter is not None
    assert access_filter["field"] == "access_level"
    assert access_filter["values"] == ["General Employee"]


def test_query_as_admin_passes_all_levels(client):
    """u4 (Admin) — access_filter must include all 4 access levels."""
    token = _login(client, "u4")

    with patch("src.api.main.VaultRetriever") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.query.return_value = _fake_cited_response(
            ["General Employee", "HR Restricted", "Marketing Eyes Only", "Admin Eyes Only"]
        )

        resp = client.post(
            "/api/query",
            json={"query": "Vault door override?", "top_k": 5},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    call_args = mock_instance.query.call_args
    access_filter = call_args.kwargs.get("access_filter")
    assert set(access_filter["values"]) == {
        "General Employee",
        "HR Restricted",
        "Marketing Eyes Only",
        "Admin Eyes Only",
    }


# ---------------------------------------------------------------------------
# GET /api/audit-logs tests
# ---------------------------------------------------------------------------


def test_audit_log_populated_after_query(client):
    """After a query, GET /api/audit-logs (Admin token) must return ≥1 row."""
    admin_token = _login(client, "u4")

    with patch("src.api.main.VaultRetriever") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.query.return_value = _fake_cited_response(["General Employee"])

        client.post(
            "/api/query",
            json={"query": "Any query for audit", "top_k": 3},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    resp = client.get("/api/audit-logs", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    logs = resp.json()
    assert len(logs) >= 1

    entry = logs[0]
    assert "user_id" in entry
    assert "query" in entry
    assert "accessed_documents" in entry
    assert "timestamp" in entry
    assert "chunk_count" in entry


def test_audit_log_forbidden_for_non_admin(client):
    """GET /api/audit-logs with a General-role token must return 403."""
    token = _login(client, "u1")
    resp = client.get("/api/audit-logs", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_audit_log_records_correct_user(client):
    """The audit log entry for u3's query must contain user_id='u3'."""
    token_u3 = _login(client, "u3")
    admin_token = _login(client, "u4")

    with patch("src.api.main.VaultRetriever") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.query.return_value = _fake_cited_response(["Marketing Eyes Only"])

        client.post(
            "/api/query",
            json={"query": "Marketing campaign details", "top_k": 3},
            headers={"Authorization": f"Bearer {token_u3}"},
        )

    resp = client.get("/api/audit-logs", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    user_ids = [log["user_id"] for log in resp.json()]
    assert "u3" in user_ids
