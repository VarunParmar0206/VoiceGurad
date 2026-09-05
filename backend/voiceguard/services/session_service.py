"""VoiceGuard V2 — Session service.

Encapsulates session lifecycle: creation, rotation (refresh), revocation,
and concurrent-session limiting (Architecture §15, Roadmap Phase 4 ¶5).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.config import settings
from voiceguard.models.session import Session
from voiceguard.repositories import SessionRepository
from voiceguard.security.tokens import (
    generate_refresh_token,
    hash_refresh_token,
)


class SessionError(Exception):
    """Base exception for session failures (never exposes internals)."""


class InvalidRefreshTokenError(SessionError):
    """Raised when a refresh token is unknown, revoked, or expired."""


class ConcurrentSessionLimitError(SessionError):
    """Raised when a user exceeds the concurrent session limit."""


class SessionService:
    """Owns the lifecycle of user sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = SessionRepository(session)

    async def create_session(
        self,
        user_id: uuid.UUID,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> Session:
        """Create a new active session, enforcing the concurrent limit.

        Returns the newly created ``Session`` with its plaintext refresh
        token available via :attr:`Session.new_refresh_token`.

        Attributes
        ----------
        On the returned instance, ``new_refresh_token`` holds the plaintext
        refresh token (only provided here at creation time) while
        ``refresh_token`` holds its stored SHA-256 hash.

        Raises:
            ConcurrentSessionLimitError: If the user already has the maximum
                number of active sessions.
        """
        await self._enforce_concurrent_limit(user_id)

        plaintext = generate_refresh_token()
        token_hash = hash_refresh_token(plaintext)
        expires_at = datetime.now(UTC) + timedelta(
            days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        session = Session(
            id=uuid.uuid4(),
            user_id=user_id,
            refresh_token=token_hash,
            user_agent=user_agent,
            ip_address=ip_address,
            expires_at=expires_at,
        )
        await self._repo.add(session)
        # Expose the one-time plaintext on the ORM instance for the caller.
        session.new_refresh_token = plaintext  # type: ignore[attr-defined]
        return session

    async def _enforce_concurrent_limit(self, user_id: uuid.UUID) -> None:
        active = await self._repo.list_active_for_user(user_id)
        limit = settings.CONCURRENT_SESSION_LIMIT
        if len(active) >= limit:
            raise ConcurrentSessionLimitError(
                "Maximum number of concurrent sessions reached."
            )

    async def get_by_refresh_token(self, refresh_token: str) -> Session | None:
        """Look up a session by its plaintext refresh token."""
        token_hash = hash_refresh_token(refresh_token)
        return await self._repo.get_by_refresh_token(token_hash)

    async def revoke(self, refresh_token: str) -> bool:
        """Revoke the session identified by *refresh_token*.

        Returns ``True`` if a session was revoked.
        """
        session = await self.get_by_refresh_token(refresh_token)
        if session is None:
            return False
        if session.revoked_at is None:
            await self._repo.revoke(session)
        return True

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> int:
        """Revoke every active session for a user; returns count revoked."""
        return await self._repo.revoke_all_for_user(user_id)

    async def rotate(
        self,
        refresh_token: str,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> Session:
        """Rotate a refresh token: revoke the old session, create a new one.

        Implements refresh-token rotation so a replayed (stolen) refresh
        token cannot be reused after its first consumption.

        Raises:
            InvalidRefreshTokenError: If the token is unknown, revoked, or
                expired.
            ConcurrentSessionLimitError: If the concurrent limit is exceeded.
        """
        session = await self.get_by_refresh_token(refresh_token)
        if session is None:
            raise InvalidRefreshTokenError("Refresh token is invalid.")
        if not session.is_active:
            raise InvalidRefreshTokenError("Refresh token is no longer valid.")

        old_user_agent = session.user_agent or user_agent
        old_ip = session.ip_address or ip_address

        # Revoke the old session before creating the replacement.
        await self._repo.revoke(session)
        return await self.create_session(
            session.user_id,
            user_agent=old_user_agent,
            ip_address=old_ip,
        )
