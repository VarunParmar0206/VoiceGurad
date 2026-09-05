"""Tests for voiceguard.security.tokens (JWT access + opaque refresh/legacy).

Covers signature, expiry, issuer, audience, and type-claim validation, plus
refresh/login token generation and hashing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from voiceguard.config import settings
from voiceguard.security.tokens import (
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
    InvalidTokenError,
    TokenError,
    create_access_token,
    decode_access_token,
    generate_login_token,
    generate_refresh_token,
    hash_login_token,
    hash_refresh_token,
)


def _make_token(**overrides) -> str:
    """Create an access token with arbitrary claim overrides for testing."""
    payload = {
        "sub": str(uuid.uuid4()),
        "type": ACCESS_TOKEN_TYPE,
        "jti": str(uuid.uuid4()),
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "iat": datetime.now(UTC),
        "nbf": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(minutes=15),
    }
    payload.update(overrides)
    # Remove any overridden "iat"/"exp" passed as timestamps handled by jwt.
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


class TestCreateAccessToken:
    def test_creates_compact_token(self) -> None:
        user_id = uuid.uuid4()
        token = create_access_token(user_id)
        assert isinstance(token, str)
        assert token.count(".") == 2

    def test_embeds_expected_claims(self) -> None:
        user_id = uuid.uuid4()
        token = create_access_token(user_id)
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
        )
        assert payload["sub"] == str(user_id)
        assert payload["type"] == ACCESS_TOKEN_TYPE
        assert payload["iss"] == settings.JWT_ISSUER
        assert payload["aud"] == settings.JWT_AUDIENCE
        assert "jti" in payload
        assert "iat" in payload
        assert "exp" in payload


class TestDecodeAccessToken:
    def test_valid_token_decodes(self) -> None:
        user_id = uuid.uuid4()
        token = create_access_token(user_id)
        claims = decode_access_token(token)
        assert claims.user_id == user_id
        assert isinstance(claims.jti, str)
        assert isinstance(claims.exp, datetime)

    def test_wrong_signature_rejected(self) -> None:
        user_id = uuid.uuid4()
        forged = jwt.encode(
            {
                "sub": str(user_id),
                "type": ACCESS_TOKEN_TYPE,
                "jti": str(uuid.uuid4()),
                "iss": settings.JWT_ISSUER,
                "aud": settings.JWT_AUDIENCE,
                "iat": datetime.now(UTC),
                "exp": datetime.now(UTC) + timedelta(minutes=15),
            },
            "a-different-wrong-secret-key-is-very-long-0000000000",
            algorithm="HS256",
        )
        with pytest.raises(InvalidTokenError):
            decode_access_token(forged)

    def test_expired_token_rejected(self) -> None:
        token = _make_token(exp=datetime.now(UTC) - timedelta(minutes=1))
        with pytest.raises(InvalidTokenError):
            decode_access_token(token)

    def test_wrong_issuer_rejected(self) -> None:
        token = _make_token(iss="evil-issuer")
        with pytest.raises(InvalidTokenError):
            decode_access_token(token)

    def test_wrong_audience_rejected(self) -> None:
        token = _make_token(aud="evil-audience")
        with pytest.raises(InvalidTokenError):
            decode_access_token(token)

    def test_refresh_type_rejected_as_access(self) -> None:
        token = _make_token(type=REFRESH_TOKEN_TYPE)
        with pytest.raises(InvalidTokenError):
            decode_access_token(token)

    def test_missing_required_claims_rejected(self) -> None:
        token = _make_token()
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
        )
        del payload["jti"]
        stripped = jwt.encode(
            payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
        with pytest.raises(InvalidTokenError):
            decode_access_token(stripped)

    def test_malformed_token_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_access_token("not.a.jwt")

    def test_garbage_string_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_access_token("garbage")

    def test_token_error_is_base_class(self) -> None:
        assert issubclass(InvalidTokenError, TokenError)

    def test_sub_not_a_uuid_rejected(self) -> None:
        token = _make_token(sub="not-a-uuid")
        with pytest.raises(InvalidTokenError):
            decode_access_token(token)


class TestRefreshToken:
    def test_generate_is_high_entropy(self) -> None:
        a = generate_refresh_token()
        b = generate_refresh_token()
        assert a != b
        assert len(a) >= 40

    def test_hash_is_sha256_hex(self) -> None:
        t = generate_refresh_token()
        h = hash_refresh_token(t)
        assert len(h) == 64
        # Same input maps to the same hash.
        assert hash_refresh_token(t) == h

    def test_hash_is_not_reversible_in_plain(self) -> None:
        t = generate_refresh_token()
        assert t not in hash_refresh_token(t)


class TestLoginToken:
    def test_generate_is_random(self) -> None:
        a = generate_login_token()
        b = generate_login_token()
        assert a != b
        assert len(a) >= 40

    def test_hash(self) -> None:
        t = generate_login_token()
        h = hash_login_token(t)
        assert len(h) == 64
        assert hash_login_token(t) == h
