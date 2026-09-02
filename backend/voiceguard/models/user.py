"""VoiceGuard V2 — User model."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import Boolean, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Registered user of the VoiceGuard system.

    Stores identity and account metadata.  Biometric templates live in
    separate tables (``VoiceTemplate``, ``VoiceModel``) linked by user_id.
    """

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    balance: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), default=Decimal("10000.00"), nullable=False
    )
    daily_limit: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), default=Decimal("50000.00"), nullable=False
    )

    def __repr__(self) -> str:
        return f"<User {self.username} active={self.is_active}>"
