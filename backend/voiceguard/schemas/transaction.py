"""VoiceGuard V2 — Transaction request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

# ── Request schemas ──────────────────────────────────────────────────────


class TransactionCreateRequest(BaseModel):
    """POST /api/v1/transactions request body.

    Voice authorization audio is sent as multipart/form-data alongside
    this JSON body in the actual endpoint implementation.
    """

    recipient_name: str = Field(
        ..., min_length=1, max_length=128, description="Recipient name"
    )
    recipient_id: uuid.UUID | None = Field(
        None, description="Recipient user UUID (if registered user)"
    )
    amount: Decimal = Field(
        ..., gt=0, decimal_places=2, description="Transaction amount (INR)"
    )
    description: str | None = Field(
        None, max_length=256, description="Optional description"
    )
    request_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Idempotency key (client-generated UUID v4)",
    )


# ── Response schemas ─────────────────────────────────────────────────────


class TransactionResponse(BaseModel):
    """Single transaction response body."""

    transaction_id: uuid.UUID = Field(..., alias="id")
    sender_id: uuid.UUID
    recipient_name: str
    recipient_id: uuid.UUID | None = None
    amount: Decimal
    currency: str = "INR"
    description: str | None = None
    status: str
    voice_score: Decimal | None = None
    decline_reason: str | None = None
    created_at: datetime

    model_config = {"populate_by_name": True}


class TransactionListResponse(BaseModel):
    """Paginated transaction list response body."""

    transactions: list[TransactionResponse] = Field(default_factory=list)
    total_count: int = Field(0, ge=0)
    page: int = Field(1, ge=1)
    limit: int = Field(20, ge=1)
    has_more: bool = Field(False)


class BalanceResponse(BaseModel):
    """GET /api/v1/transactions/balance response body."""

    balance: Decimal
    daily_limit: Decimal
    daily_spent: Decimal = Field(
        Decimal("0.00"), description="Amount spent today"
    )
    daily_remaining: Decimal = Field(
        Decimal("0.00"), description="Remaining daily limit"
    )
