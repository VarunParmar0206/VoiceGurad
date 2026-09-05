"""VoiceGuard V2 — Audit log model."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    _bigint_pk_type,
    _inet_type,
    _jsonb_type,
    _utcnow,
    _uuid_type,
)


class AuditLog(Base):
    """Append-only audit trail for security-relevant events.

    Rows are never updated or deleted by application code.  The ``id``
    column is ``BIGSERIAL`` (auto-incrementing) rather than UUID because
    audit records are ordered and accessed primarily by ``created_at``.

    Event types follow the pattern ``<domain>.<action>[.<detail>]``, e.g.:
    - ``auth.login.success``
    - ``auth.login.failure``
    - ``auth.voice.enroll``
    - ``auth.voice.verify.success``
    - ``auth.voice.verify.failure``
    - ``auth.anti_spoof.replay_detected``
    - ``auth.anti_spoof.deepfake_detected``
    - ``auth.lockout.activated``
    - ``transaction.created``
    - ``transaction.declined``
    - ``security.suspicious_activity``
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(
        _bigint_pk_type(), primary_key=True, autoincrement=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_detail: Mapped[dict[str, object] | None] = mapped_column(_jsonb_type(), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(_inet_type(), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.event_type} user={self.user_id}>"
