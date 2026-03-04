"""JWT creation, verification, and FastAPI dependency for the auth layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.schemas import TokenPayload
from src.config import settings

# FastAPI will look for Bearer <token> in the Authorization header.
# tokenUrl points to our login endpoint (informational — not OAuth2 password flow).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

_ALGORITHM = "HS256"
_EXPIRY_HOURS = 24


def create_access_token(
    user_id: str,
    username: str,
    role: str,
    allowed_levels: list[str],
) -> str:
    """Sign and return an HS256 JWT for the given user."""
    now = datetime.now(tz=UTC)
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "allowed_levels": allowed_levels,
        "iat": now,
        "exp": now + timedelta(hours=_EXPIRY_HOURS),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify an access token. Raises HTTP 401 on failure."""
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    """FastAPI dependency — decode Bearer token and return the TokenPayload.

    Inject into any endpoint that requires authentication:
        current_user: TokenPayload = Depends(get_current_user)
    """
    payload = decode_access_token(token)
    return TokenPayload(
        sub=payload["sub"],
        username=payload["username"],
        role=payload["role"],
        allowed_levels=payload["allowed_levels"],
    )
