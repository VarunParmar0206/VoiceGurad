"""VoiceGuard V2 — Session repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult

from voiceguard.models.session import Session
from voiceguard.repositories.base import RepositoryBase


class SessionRepository(RepositoryBase[Session]):
    """Data-access operations for the ``sessions`` table."""

    model = Session

    async def get_by_refresh_token(self, token: str) -> Session | None:
        result = await self.session.execute(
            select(Session).where(Session.refresh_token == token)
        )
        return result.scalar_one_or_none()

    async def list_active_for_user(self, user_id: uuid.UUID) -> list[Session]:
        now = datetime.now(UTC)
        result = await self.session.execute(
            select(Session).where(
                Session.user_id == user_id,
                Session.revoked_at.is_(None),
                Session.expires_at > now,
            )
        )
        return list(result.scalars().all())

    async def revoke(self, session: Session) -> None:
        """Mark a session as revoked."""
        session.revoked_at = datetime.now(UTC)
        await self.session.flush()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> int:
        """Revoke all active sessions for a user; returns count revoked."""
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await self.session.execute(
                update(Session)
                .where(
                    Session.user_id == user_id,
                    Session.revoked_at.is_(None),
                    Session.expires_at > now,
                )
                .values(revoked_at=now)
            ),
        )
        await self.session.flush()
        return result.rowcount or 0

    async def count_active_for_user(self, user_id: uuid.UUID) -> int:
        return len(await self.list_active_for_user(user_id))
