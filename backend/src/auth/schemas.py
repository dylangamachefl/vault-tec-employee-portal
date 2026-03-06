"""Pydantic schemas for the auth layer (API request/response models)."""

from __future__ import annotations

from pydantic import BaseModel


class TokenPayload(BaseModel):
    """Decoded JWT payload — returned by get_current_user dependency."""

    sub: str  # user_id e.g. "u1"
    username: str
    role: str  # role slug e.g. "admin"
    allowed_levels: list[str]  # Qdrant access_level values


class TokenResponse(BaseModel):
    """Returned by POST /api/login."""

    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    """Public user profile (no sensitive data)."""

    id: str
    username: str
    role: str  # role slug
    display_name: str  # role display name
    allowed_levels: list[str]
