"""VoiceGuard V2 — Transaction repository."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import func, select

from voiceguard.models.transaction import Transaction
from voiceguard.repositories.base import RepositoryBase


class TransactionRepository(RepositoryBase[Transaction]):
    """Data-access operations for the ``transactions`` table.

    Financial safety is the responsibility of the service layer, which
    wraps debit + credit + insert in a single DB transaction.  This
    repository provides the query surface only.
    """

    model = Transaction

    async def get_by_request_id(self, request_id: uuid.UUID) -> Transaction | None:
        """Idempotency check — return the existing transaction if the
        ``request_id`` was already processed."""
        result = await self.session.execute(
            select(Transaction).where(Transaction.request_id == request_id)
        )
        return result.scalar_one_or_none()

    async def list_for_sender(
        self, sender_id: uuid.UUID, limit: int = 20, offset: int = 0
    ) -> list[Transaction]:
        result = await self.session.execute(
            select(Transaction)
            .where(Transaction.sender_id == sender_id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_for_sender(
        self, sender_id: uuid.UUID, since: datetime | None = None
    ) -> int:
        stmt = select(func.count(Transaction.id)).where(
            Transaction.sender_id == sender_id
        )
        if since is not None:
            stmt = stmt.where(Transaction.created_at >= since)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def sum_sent_since(
        self, sender_id: uuid.UUID, since: datetime
    ) -> Decimal:
        """Total amount sent by a user since *since* (for daily limits)."""
        result = await self.session.execute(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.sender_id == sender_id,
                Transaction.created_at >= since,
                Transaction.status == "completed",
            )
        )
        return Decimal(str(result.scalar_one()))
