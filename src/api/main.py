"""FastAPI application — Phase 1 REST API for the React frontend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.documents import DocumentRecord, get_accessible_documents
from src.pipelines.retrieval_chain import CitedResponse, QueryInput, VaultRetriever

app = FastAPI(title="Vault-Tec Employee Portal API", version="0.2.0")

# ---------------------------------------------------------------------------
# CORS — allow Vite dev server (port 3000) and Docker nginx (port 80/3000)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:80", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Demo credential profiles — match LoginView.tsx DEMO_USERS
# ---------------------------------------------------------------------------

_DEMO_USERS: dict[str, dict] = {
    "u1": {
        "id": "u1",
        "username": "Dweller-101",
        "role": "General Employee",
        "accessLevel": "General",
    },
    "u2": {
        "id": "u2",
        "username": "Barnsworth B.",
        "role": "HR Specialist",
        "accessLevel": "HR",
    },
    "u3": {
        "id": "u3",
        "username": "Gable M.",
        "role": "Marketing Associate",
        "accessLevel": "Marketing",
    },
    "u4": {
        "id": "u4",
        "username": "Carmichael J.",
        "role": "IT Administrator",
        "accessLevel": "Admin",
    },
}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    user_id: str


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    accessLevel: str


class QueryRequest(BaseModel):
    query: str
    access_level: str
    top_k: int = 5


# ---------------------------------------------------------------------------
# Access level → Qdrant filter mapping
# Phase 2 will enforce hard RBAC; Phase 1 uses soft filter (no restriction).
# ---------------------------------------------------------------------------

_ACCESS_FILTER: dict[str, str | None] = {
    "General": None,  # Phase 1: no filter enforced
    "HR": None,
    "Marketing": None,
    "Admin": None,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "message": "Vault-Tec systems nominal."}


@app.post("/api/login", response_model=UserResponse)
async def login(req: LoginRequest) -> UserResponse:
    user = _DEMO_USERS.get(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {req.user_id}")
    return UserResponse(**user)


@app.get("/api/documents", response_model=list[DocumentRecord])
async def list_documents(
    access_level: str = Query(..., description="Caller's clearance level"),
) -> list[DocumentRecord]:
    return get_accessible_documents(access_level)


@app.post("/api/query", response_model=CitedResponse)
async def query_knowledge_base(req: QueryRequest) -> CitedResponse:
    retriever = VaultRetriever()
    result = retriever.query(
        QueryInput(
            query=req.query,
            top_k=req.top_k,
            access_level_filter=_ACCESS_FILTER.get(req.access_level),
        )
    )
    return result
