"""VoiceGuard V2 — Challenge model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import Base, UUIDPrimaryKeyMixin, _utcnow, _uuid_type


class Challenge(UUIDPrimaryKeyMixin, Base):
    """Server-generated challenge for voice liveness verification.

    Each challenge has a text that the user must speak, an expiry time,
    and a one-time-use flag.  The challenge ID is sent to the client;
    the expected text is **never** sent to the client.
    """

    __tablename__ = "challenges"

    user_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    challenge_text: Mapped[str] = mapped_column(String(128), nullable=False)
    challenge_type: Mapped[str] = mapped_column(String(32), nullable=False)
    is_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    @property
    def is_expired(self) -> bool:
        return self.expires_at <= datetime.now(UTC)

    @property
    def is_valid(self) -> bool:
        """True if the challenge is unused and not expired."""
        return not self.is_used and not self.is_expired

    def __repr__(self) -> str:
        return (
            f"<Challenge {self.id} type={self.challenge_type} "
            f"valid={self.is_valid}>"
        )
