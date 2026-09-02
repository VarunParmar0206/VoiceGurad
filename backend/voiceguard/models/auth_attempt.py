"""VoiceGuard V2 — Auth attempt model."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    _bigint_pk_type,
    _inet_type,
    _utcnow,
    _uuid_type,
)


class AuthAttempt(Base):
    """Record of every authentication attempt (success or failure).

    Used for:
    - Rate limiting (count recent failures per user/IP)
    - Account lockout decisions
    - Audit trail
    - Forensic analysis of brute-force attacks

    ``attempt_type`` is one of: ``password``, ``voice``, ``mfa``.
    """

    __tablename__ = "auth_attempts"

    id: Mapped[int] = mapped_column(
        _bigint_pk_type(), primary_key=True, autoincrement=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    attempt_type: Mapped[str] = mapped_column(String(16), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(_inet_type(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<AuthAttempt user={self.user_id} type={self.attempt_type} "
            f"success={self.success}>"
        )
