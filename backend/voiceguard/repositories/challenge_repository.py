"""VoiceGuard V2 — Challenge repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from voiceguard.models.challenge import Challenge
from voiceguard.repositories.base import RepositoryBase


class ChallengeRepository(RepositoryBase[Challenge]):
    """Data-access operations for the ``challenges`` table."""

    model = Challenge

    async def get_valid(self, challenge_id: uuid.UUID) -> Challenge | None:
        """Fetch an unused, unexpired challenge by ID; ``None`` otherwise."""
        now = datetime.now(UTC)
        result = await self.session.execute(
            select(Challenge).where(
                Challenge.id == challenge_id,
                Challenge.is_used.is_(False),
                Challenge.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def mark_used(self, challenge: Challenge) -> None:
        """Mark a challenge as consumed (one-time use)."""
        challenge.is_used = True
        await self.session.flush()

    async def count_for_user_since(
        self, user_id: uuid.UUID, since: datetime
    ) -> int:
        """Count challenges issued to a user since *since* (for rate limits)."""
        result = await self.session.execute(
            select(Challenge.id)
            .where(Challenge.user_id == user_id, Challenge.created_at >= since)
        )
        return len(result.scalars().all())

    async def delete_expired(self) -> int:
        """Remove expired challenges; returns number removed (housekeeping)."""
        now = datetime.now(UTC)
        result = await self.session.execute(
            select(Challenge).where(Challenge.expires_at <= now)
        )
        expired = list(result.scalars().all())
        for c in expired:
            await self.session.delete(c)
        await self.session.flush()
        return len(expired)
