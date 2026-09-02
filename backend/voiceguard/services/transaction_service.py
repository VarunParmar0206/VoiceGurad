"""VoiceGuard V2 — Transaction service.

Business logic for payment transactions.  Phase 3 provides the
transaction creation and query workflow.  Voice authorization
before transaction execution is a Phase 5-6 concern.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.config import settings
from voiceguard.models.transaction import Transaction
from voiceguard.repositories import TransactionRepository, UserRepository
from voiceguard.services.user_service import UserService


class TransactionService:
    """Encapsulates transaction-related business logic."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._tx_repo = TransactionRepository(session)
        self._user_repo = UserRepository(session)
        self._user_service = UserService(session)

    async def create_transaction(
        self,
        sender_id: uuid.UUID,
        recipient_name: str,
        amount: Decimal,
        description: str | None = None,
        recipient_id: uuid.UUID | None = None,
        request_id: uuid.UUID | None = None,
        voice_score: float | None = None,
        challenge_id: uuid.UUID | None = None,
    ) -> Transaction:
        """Create and persist a new transaction.

        Performs idempotency check, balance validation, and atomic
        balance update within the current session.
        """
        # Idempotency check
        if request_id:
            existing = await self._tx_repo.get_by_request_id(request_id)
            if existing:
                return existing

        sender = await self._user_repo.get(sender_id)
        if sender is None:
            raise ValueError("Sender not found")
        if not sender.is_active:
            raise ValueError("Sender account is inactive")

        # Daily limit check
        today_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        daily_spent = await self._tx_repo.sum_sent_since(sender_id, today_start)
        if daily_spent + amount > Decimal(str(settings.DAILY_TRANSACTION_LIMIT)):
            raise ValueError("Daily transaction limit exceeded")

        # Balance check and debit
        await self._user_service.debit_balance(sender, amount)

        # Credit recipient if specified
        if recipient_id and recipient_id != sender_id:
            recipient = await self._user_repo.get(recipient_id)
            if recipient and recipient.is_active:
                await self._user_service.credit_balance(recipient, amount)

        tx = Transaction(
            id=uuid.uuid4(),
            sender_id=sender_id,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            amount=amount,
            currency="INR",
            description=description,
            status="completed",
            voice_score=Decimal(str(voice_score)) if voice_score else None,
            challenge_id=challenge_id,
            request_id=request_id,
        )
        return await self._tx_repo.add(tx)

    async def get_transaction(
        self,
        transaction_id: uuid.UUID,
    ) -> Transaction | None:
        return await self._tx_repo.get(transaction_id)

    async def list_transactions(
        self,
        sender_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Transaction], int]:
        """List transactions for a sender with total count."""
        txs = await self._tx_repo.list_for_sender(
            sender_id, limit=limit, offset=offset
        )
        total = await self._tx_repo.count_for_sender(sender_id)
        return txs, total

    async def get_balance_info(
        self,
        user_id: uuid.UUID,
    ) -> dict[str, Decimal]:
        """Get balance and daily limit information."""
        user = await self._user_repo.get(user_id)
        if user is None:
            raise ValueError("User not found")

        today_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        daily_spent = await self._tx_repo.sum_sent_since(user_id, today_start)
        daily_limit = Decimal(str(settings.DAILY_TRANSACTION_LIMIT))

        return {
            "balance": user.balance or Decimal("0.00"),
            "daily_limit": daily_limit,
            "daily_spent": daily_spent,
            "daily_remaining": max(Decimal("0.00"), daily_limit - daily_spent),
        }
