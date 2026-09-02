"""VoiceGuard V2 — Auth service.

Business logic for authentication flows.  Phase 3 provides stubs
for the registration and login endpoints.  Actual password hashing
(Argon2id), JWT token generation, and session management belong to
Phase 4.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.repositories import (
    AuthAttemptRepository,
    ChallengeRepository,
    SessionRepository,
    UserRepository,
)
from voiceguard.services.user_service import UserService


class AuthService:
    """Encapsulates authentication-related business logic."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._user_repo = UserRepository(session)
        self._session_repo = SessionRepository(session)
        self._challenge_repo = ChallengeRepository(session)
        self._attempt_repo = AuthAttemptRepository(session)
        self._user_service = UserService(session)

    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        display_name: str | None = None,
    ) -> uuid.UUID:
        """Register a new user.

        Phase 3 stub: accepts a plaintext password and stores it as-is.
        Phase 4 will hash with Argon2id before storage.
        """
        if await self._user_service.username_exists(username):
            raise ValueError("Username already taken")
        if await self._user_service.email_exists(email):
            raise ValueError("Email already registered")

        user = await self._user_service.create_user(
            username=username,
            email=email,
            password_hash=password,
            display_name=display_name,
        )
        return user.id

    async def verify_password(
        self,
        username: str,
        password: str,
    ) -> uuid.UUID | None:
        """Verify user credentials.

        Phase 3 stub: compares plaintext passwords.
        Phase 4 will use Argon2id verification.
        """
        user = await self._user_repo.get_by_username(username)
        if user is None:
            return None
        if not user.is_active or user.is_locked:
            return None
        # Phase 4: verify against Argon2id hash
        if user.password_hash != password:
            return None
        return user.id

    async def issue_tokens(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, str | int]:
        """Issue access + refresh tokens.

        Phase 3 stub: returns placeholder tokens.
        Phase 4 will generate real JWTs.
        """
        # Phase 4: generate real JWT access + refresh tokens
        return {
            "access_token": f"stub-access-{user_id}",
            "refresh_token": f"stub-refresh-{user_id}",
            "token_type": "bearer",
            "expires_in": 900,
        }

    async def generate_challenge(
        self,
        user_id: uuid.UUID,
    ) -> uuid.UUID:
        """Generate a challenge for voice authentication.

        Phase 3 stub: creates a minimal challenge record.
        Phase 10 will implement full challenge vocabulary + STT validation.
        """
        from datetime import UTC, datetime, timedelta

        from voiceguard.config import settings
        from voiceguard.models.challenge import Challenge

        challenge = Challenge(
            id=uuid.uuid4(),
            user_id=user_id,
            challenge_text="Verify your identity",
            challenge_type="phrase",
            is_used=False,
            expires_at=datetime.now(UTC)
            + timedelta(seconds=settings.CHALLENGE_EXPIRY_SECONDS),
        )
        self._session.add(challenge)
        await self._session.flush()
        return challenge.id

    async def record_attempt(
        self,
        user_id: uuid.UUID | None,
        attempt_type: str,
        success: bool,
        failure_reason: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """Record an authentication attempt for auditing."""
        from voiceguard.models.auth_attempt import AuthAttempt

        attempt = AuthAttempt(
            user_id=user_id,
            attempt_type=attempt_type,
            success=success,
            failure_reason=failure_reason,
            ip_address=ip_address,
        )
        self._session.add(attempt)
        await self._session.flush()
