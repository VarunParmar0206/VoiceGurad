"""VoiceGuard V2 — FastAPI dependency injection.

Provides reusable dependencies for:
- Database sessions
- Repository instances
- Rate limiting per scope
- Auth context stubs (Phase 4 will implement JWT verification)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.config import settings
from voiceguard.db.session import get_async_session
from voiceguard.repositories import (
    AuditLogRepository,
    AuthAttemptRepository,
    ChallengeRepository,
    SessionRepository,
    TransactionRepository,
    UserRepository,
    VoiceModelRepository,
    VoiceTemplateRepository,
)
from voiceguard.security.rate_limit import (
    RateLimitResult,
    check_rate_limit,
    make_rate_limit_key,
)
from voiceguard.security.tokens import InvalidTokenError, decode_access_token

# ── Database session ─────────────────────────────────────────────────────


async def get_db(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> AsyncSession:
    """Yield an ``AsyncSession`` for use in route handlers and services."""
    return session


# ── Repository dependencies ──────────────────────────────────────────────


def get_user_repo(session: Annotated[AsyncSession, Depends(get_db)]) -> UserRepository:
    return UserRepository(session)


def get_voice_template_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> VoiceTemplateRepository:
    return VoiceTemplateRepository(session)


def get_voice_model_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> VoiceModelRepository:
    return VoiceModelRepository(session)


def get_session_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SessionRepository:
    return SessionRepository(session)


def get_transaction_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> TransactionRepository:
    return TransactionRepository(session)


def get_audit_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AuditLogRepository:
    return AuditLogRepository(session)


def get_challenge_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ChallengeRepository:
    return ChallengeRepository(session)


def get_auth_attempt_repo(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AuthAttemptRepository:
    return AuthAttemptRepository(session)


# ── Rate-limiting dependencies ───────────────────────────────────────────


async def rate_limit_login(
    request: Request,
) -> RateLimitResult:
    """Enforce per-IP login rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    key = make_rate_limit_key("login", client_ip)
    result = await check_rate_limit(
        key,
        limit=settings.RATE_LIMIT_AUTH_PER_MINUTE,
        window_seconds=60,
    )
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(result.retry_after_seconds or 60),
            },
        )
    return result


async def rate_limit_auth(
    user_id: UUID,
) -> RateLimitResult:
    """Enforce per-user auth rate limiting (voice, MFA)."""
    key = make_rate_limit_key("auth", str(user_id))
    result = await check_rate_limit(
        key,
        limit=settings.RATE_LIMIT_AUTH_PER_MINUTE,
        window_seconds=60,
    )
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later.",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(result.retry_after_seconds or 60),
            },
        )
    return result


async def rate_limit_voice(
    user_id: UUID,
) -> RateLimitResult:
    """Enforce per-user voice verification rate limiting."""
    key = make_rate_limit_key("voice", str(user_id))
    result = await check_rate_limit(
        key,
        limit=settings.RATE_LIMIT_VOICE_PER_MINUTE,
        window_seconds=60,
    )
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many voice verification requests.",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(result.retry_after_seconds or 60),
            },
        )
    return result


async def rate_limit_transaction(
    user_id: UUID,
) -> RateLimitResult:
    """Enforce per-user transaction rate limiting."""
    key = make_rate_limit_key("transaction", str(user_id))
    result = await check_rate_limit(
        key,
        limit=settings.RATE_LIMIT_TRANSACTION_PER_HOUR,
        window_seconds=3600,
    )
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many transactions. Please try again later.",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(result.retry_after_seconds or 3600),
            },
        )
    return result


# ── Auth context (JWT verification) ──────────────────────────────────────


@dataclass
class AuthContext:
    """Authenticated user context derived from a JWT access token.

    ``user_id`` is the authenticated user's UUID; ``is_authenticated`` is
    ``True`` only when a valid, unexpired access token was presented.
    """

    user_id: UUID | None = None
    is_authenticated: bool = False


def _bearer_token(request: Request) -> str | None:
    """Extract the bearer token from the Authorization header.

    Returns the raw token, or ``None`` if the header is absent or not
    scheme ``Bearer``.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


async def get_auth_context(
    request: Request,
) -> AuthContext:
    """Resolve the auth context by verifying the JWT access token.

    A missing, malformed, or expired token yields an unauthenticated
    context (does not raise) so downstream logic can choose its behaviour.
    """
    token = _bearer_token(request)
    if token is None:
        return AuthContext(user_id=None, is_authenticated=False)
    try:
        claims = decode_access_token(token)
    except InvalidTokenError:
        return AuthContext(user_id=None, is_authenticated=False)
    return AuthContext(user_id=claims.user_id, is_authenticated=True)


async def require_auth(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> AuthContext:
    """Dependency that requires a valid, authenticated access token.

    Raises:
        HTTPException 401: If the token is missing, invalid, or expired.
            No internal error details are exposed.
    """
    if not auth.is_authenticated or auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth
