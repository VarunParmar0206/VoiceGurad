"""VoiceGuard V2 — Audit log repository."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select

from voiceguard.models.audit_log import AuditLog
from voiceguard.repositories.base import RepositoryBase


class AuditLogRepository(RepositoryBase[AuditLog]):
    """Data-access operations for the append-only ``audit_log`` table.

    Application code should only **add** audit rows — never update or
    delete them.  This repository deliberately exposes no ``update`` or
    ``delete`` helpers.
    """

    model = AuditLog

    async def list_by_user(
        self, user_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[AuditLog]:
        result = await self.session.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def list_by_event_type(
        self,
        event_type: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        stmt = (
            select(AuditLog)
            .where(AuditLog.event_type == event_type)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        if since is not None:
            stmt = stmt.where(AuditLog.created_at >= since)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
