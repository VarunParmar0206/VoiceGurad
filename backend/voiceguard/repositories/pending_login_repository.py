"""VoiceGuard V2 — Pending login repository."""

from __future__ import annotations

from sqlalchemy import select

from voiceguard.models.pending_login import PendingLogin
from voiceguard.repositories.base import RepositoryBase


class PendingLoginRepository(RepositoryBase[PendingLogin]):
    """Data-access operations for the ``pending_logins`` table."""

    model = PendingLogin

    async def get_by_token_hash(self, token_hash: str) -> PendingLogin | None:
        result = await self.session.execute(
            select(PendingLogin).where(PendingLogin.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def mark_used(self, pending: PendingLogin) -> None:
        """Consume a pending login (one-time use)."""
        from datetime import UTC, datetime

        pending.used_at = datetime.now(UTC)
        await self.session.flush()
