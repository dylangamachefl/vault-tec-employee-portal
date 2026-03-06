"""Unit tests for JWT creation, decoding, and error handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt as pyjwt
import pytest

from src.auth.jwt import create_access_token, decode_access_token
from src.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEST_PAYLOAD = {
    "user_id": "u1",
    "username": "Dweller-101",
    "role": "general",
    "allowed_levels": ["General Employee"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_and_decode_token():
    """create_access_token() produces a token that decode_access_token() can verify."""
    token = create_access_token(**_TEST_PAYLOAD)
    assert isinstance(token, str)
    assert len(token) > 20

    payload = decode_access_token(token)
    assert payload["sub"] == "u1"
    assert payload["username"] == "Dweller-101"
    assert payload["role"] == "general"
    assert payload["allowed_levels"] == ["General Employee"]


def test_token_has_expiry():
    """Decoded token must include an 'exp' claim in the future."""
    token = create_access_token(**_TEST_PAYLOAD)
    payload = decode_access_token(token)
    exp = payload["exp"]
    now = datetime.now(tz=UTC).timestamp()
    assert exp > now


def test_expired_token_raises_401():
    """A token with exp in the past must raise HTTP 401."""
    from fastapi import HTTPException

    past = datetime.now(tz=UTC) - timedelta(hours=1)
    expired_payload = {
        "sub": "u1",
        "username": "Dweller-101",
        "role": "general",
        "allowed_levels": ["General Employee"],
        "iat": past - timedelta(hours=25),
        "exp": past,
    }
    expired_token = pyjwt.encode(expired_payload, settings.jwt_secret_key, algorithm="HS256")

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(expired_token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_tampered_token_raises_401():
    """A token whose signature has been altered must raise HTTP 401."""
    from fastapi import HTTPException

    token = create_access_token(**_TEST_PAYLOAD)
    # Flip the last character to corrupt the signature
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(tampered)
    assert exc_info.value.status_code == 401


def test_completely_invalid_token_raises_401():
    """Garbage input must raise HTTP 401."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token("this.is.not.a.jwt")
    assert exc_info.value.status_code == 401


def test_wrong_secret_raises_401():
    """A token signed with a different secret must raise HTTP 401."""
    from fastapi import HTTPException

    bad_token = pyjwt.encode(
        {"sub": "u1", "username": "x", "role": "general", "allowed_levels": []},
        "wrong-secret",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(bad_token)
    assert exc_info.value.status_code == 401
