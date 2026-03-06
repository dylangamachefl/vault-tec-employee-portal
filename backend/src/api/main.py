"""FastAPI application — Phase 3: Client-side embeddings, JWT Auth, RBAC, Audit Trail."""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.documents import (
    DOC_ID_TO_FILENAME,
    DocumentRecord,
    get_accessible_doc_ids,
    get_accessible_documents,
)
from src.audit.logger import log_query_event
from src.audit.schemas import AuditLogOut
from src.auth.crud import get_allowed_levels, get_user
from src.auth.database import get_db
from src.auth.jwt import create_access_token, get_current_user
from src.auth.schemas import TokenPayload, TokenResponse
from src.pipelines.retrieval_chain import CitedResponse, QueryInput, VaultRetriever

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vault-Tec Employee Portal API",
    version="0.4.0",
    description="Phase 3: Client-side embeddings (Transformers.js), JWT auth, RBAC, audit trail.",
)

# ---------------------------------------------------------------------------
# CORS — driven by ALLOWED_ORIGINS env var for production flexibility.
# Default covers Vite dev server and Docker nginx.
# Production: ALLOWED_ORIGINS=https://vault-tec.vercel.app,https://vault-tec.netlify.app
# ---------------------------------------------------------------------------

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:80,http://localhost"
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup — seed the auth/audit database
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    """Ensure auth.db tables exist and demo data is seeded."""
    from src.auth.seed import run_seed

    run_seed()
    logger.info("Auth database ready.")


# ---------------------------------------------------------------------------
# Type alias for the injected current user
# ---------------------------------------------------------------------------

CurrentUser = Annotated[TokenPayload, Depends(get_current_user)]
DBSession = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    user_id: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    # Pre-computed embedding from the browser (Phase 3).
    # When present, the backend skips in-process embedding entirely.
    vector: list[float] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "message": "Vault-Tec systems nominal."}


# ------------------------------------------------------------------
# POST /api/login — issue JWT
# ------------------------------------------------------------------


@app.post("/api/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: DBSession) -> TokenResponse:
    """Validate user_id against the auth DB and return a signed JWT."""
    user = get_user(db, req.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown user_id: {req.user_id!r}",
        )

    allowed_levels = get_allowed_levels(db, req.user_id) or []
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role.name,
        allowed_levels=allowed_levels,
    )
    return TokenResponse(access_token=token)


# ------------------------------------------------------------------
# GET /api/documents — accessible documents for the current user
# ------------------------------------------------------------------

# Role slug → legacy access level key used by get_accessible_documents()
_ROLE_TO_ACCESS_LEVEL: dict[str, str] = {
    "general": "General",
    "hr": "HR",
    "marketing": "Marketing",
    "admin": "Admin",
}


@app.get("/api/documents", response_model=list[DocumentRecord])
async def list_documents(
    current_user: CurrentUser,
) -> list[DocumentRecord]:
    """Return documents the authenticated user is allowed to see."""
    access_level = _ROLE_TO_ACCESS_LEVEL.get(current_user.role, "General")
    return get_accessible_documents(access_level)


# ------------------------------------------------------------------
# GET /api/documents/{doc_id}/content — raw markdown content
# ------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
_EXPORTS_DIR = _PROJECT_ROOT / "data" / "processed" / "docling_exports"


class DocumentContentResponse(BaseModel):
    content: str


@app.get("/api/documents/{doc_id}/content", response_model=DocumentContentResponse)
async def get_document_content(
    doc_id: str,
    current_user: CurrentUser,
) -> DocumentContentResponse:
    """Return the raw markdown content of a document.

    Access is enforced: the doc must be within the user's permitted levels.
    """
    access_level = _ROLE_TO_ACCESS_LEVEL.get(current_user.role, "General")
    allowed_ids = get_accessible_doc_ids(access_level)

    if doc_id not in allowed_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Clearance level insufficient for document {doc_id!r}.",
        )

    filename = DOC_ID_TO_FILENAME.get(doc_id)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No source file mapped for document {doc_id!r}.",
        )

    filepath = _EXPORTS_DIR / filename
    if not filepath.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source file not found on disk: {filename!r}.",
        )

    content = filepath.read_text(encoding="utf-8")
    return DocumentContentResponse(content=content)


# ------------------------------------------------------------------
# POST /api/query — RBAC-enforced retrieval + audit
# ------------------------------------------------------------------


@app.post("/api/query", response_model=CitedResponse)
async def query_knowledge_base(
    req: QueryRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> CitedResponse:
    """Run a retrieval query filtered to the user's permitted access levels.

    - Identity and allowed_levels come from the verified JWT — no client-supplied
      role override is possible.
    - Every query is persisted to the audit log.
    """
    retriever = VaultRetriever()
    result = retriever.query(
        QueryInput(
            query=req.query,
            top_k=req.top_k,
            access_level_filter=None,
            vector=req.vector,
        ),
        access_filter={
            "field": "access_level",
            "values": current_user.allowed_levels,
        },
    )

    # Persist audit trail entry
    log_query_event(
        db=db,
        user_id=current_user.sub,
        username=current_user.username,
        query=req.query,
        response=result,
    )

    return result


# ------------------------------------------------------------------
# GET /api/audit-logs — Admin-only audit trail viewer
# ------------------------------------------------------------------


@app.get("/api/audit-logs", response_model=list[AuditLogOut])
async def get_audit_logs(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AuditLogOut]:
    """Return audit log entries. Restricted to Admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Audit log access requires Admin role.",
        )

    from src.audit.models import AuditLog

    rows = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
    return [AuditLogOut.model_validate(row) for row in rows]
