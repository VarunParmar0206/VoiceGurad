"""VoiceGuard V2 — Auth attempt repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, select

from voiceguard.models.auth_attempt import AuthAttempt
from voiceguard.repositories.base import RepositoryBase


class AuthAttemptRepository(RepositoryBase[AuthAttempt]):
    """Data-access operations for the ``auth_attempts`` table."""

    model = AuthAttempt

    async def count_failures_since(
        self,
        user_id: uuid.UUID,
        since: datetime,
        attempt_type: str | None = None,
    ) -> int:
        """Count failed attempts for a user since *since*.

        Used by account-lockout logic.
        """
        stmt = select(func.count(AuthAttempt.id)).where(
            AuthAttempt.user_id == user_id,
            AuthAttempt.success.is_(False),
            AuthAttempt.created_at >= since,
        )
        if attempt_type is not None:
            stmt = stmt.where(AuthAttempt.attempt_type == attempt_type)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def count_failures_for_ip_since(
        self, ip_address: str, since: datetime
    ) -> int:
        """Count failed attempts from an IP address (per-IP rate limit)."""
        result = await self.session.execute(
            select(func.count(AuthAttempt.id)).where(
                AuthAttempt.ip_address == ip_address,
                AuthAttempt.success.is_(False),
                AuthAttempt.created_at >= since,
            )
        )
        return result.scalar_one()

    async def latest_for_user(
        self, user_id: uuid.UUID, limit: int = 10
    ) -> list[AuthAttempt]:
        result = await self.session.execute(
            select(AuthAttempt)
            .where(AuthAttempt.user_id == user_id)
            .order_by(AuthAttempt.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_attempts_since(
        self, user_id: uuid.UUID, since: timedelta
    ) -> int:
        return await self.count_failures_since(user_id, datetime.now() - since)
