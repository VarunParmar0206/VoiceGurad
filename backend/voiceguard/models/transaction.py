"""VoiceGuard V2 — Transaction model."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    UUIDPrimaryKeyMixin,
    _utcnow,
    _uuid_type,
)


class Transaction(UUIDPrimaryKeyMixin, Base):
    """Financial transaction record.

    Designed for atomic balance operations: the application layer will
    ``BEGIN`` a transaction, debit the sender (``UPDATE users SET balance
    = balance - amount WHERE id = :sender AND balance >= amount``), credit
    the recipient, and ``INSERT`` this record — all within a single
    database transaction.

    ``request_id`` provides idempotency: duplicate ``request_id`` values
    are rejected at the database level (``UNIQUE`` constraint) to prevent
    double-submission of the same payment.
    """

    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_transactions_amount_positive"),
        Index("ix_transactions_sender_created", "sender_id", "created_at"),
    )

    sender_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    recipient_id: Mapped[uuid.UUID | None] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    recipient_name: Mapped[str] = mapped_column(String(128), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), default="INR", nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )
    voice_score: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    challenge_id: Mapped[uuid.UUID | None] = mapped_column(
        _uuid_type(), nullable=True
    )
    # Idempotency key — each POST /transactions request must include a
    # client-generated UUID.  Duplicate request_id values are rejected.
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        _uuid_type(), unique=True, nullable=True
    )
    decline_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction {self.id} amount={self.amount} "
            f"status={self.status}>"
        )
