"""VoiceGuard V2 — Session model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    UUIDPrimaryKeyMixin,
    _inet_type,
    _utcnow,
    _uuid_type,
)


class Session(UUIDPrimaryKeyMixin, Base):
    """User session created after successful authentication.

    Each session stores a refresh token.  The corresponding JWT access
    token is short-lived and held only in client memory.
    """

    __tablename__ = "sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    refresh_token: Mapped[str] = mapped_column(
        String(512), unique=True, nullable=False, index=True
    )
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(_inet_type(), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @property
    def is_active(self) -> bool:
        """True if the session has not been revoked and has not expired.

        ``expires_at`` is compared in UTC; SQLite returns naive datetimes, so
        a naive value is normalized to UTC before comparison.
        """
        if self.revoked_at is not None:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires > datetime.now(UTC)

    def __repr__(self) -> str:
        return f"<Session user={self.user_id} active={self.is_active}>"
