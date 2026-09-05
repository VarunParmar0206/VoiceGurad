"""VoiceGuard V2 — Pending login model.

A server-side, one-time login state created after a successful password
verification (step 1) and consumed by the secondary-factor step (TOTP).

Why this exists
***************
The two-step login flow must not trust a client-supplied ``user_id`` for the
secondary factor.  If ``login-totp`` accepted an arbitrary user UUID, a caller
could pair that UUID with any valid TOTP code without ever proving possession
of the account's password.  Instead, ``login-password`` issues an opaque,
one-time ``login_token`` that is **server-side bound** to the authenticated
user.  ``login-totp`` consumes that token to derive the target account, so the
secondary factor can only complete for the same user whose password was just
verified.

Properties
**********
- Opaque, high-entropy one-time token; only its SHA-256 hash is stored.
- Bound to ``user_id``.
- One-time use (``used_at`` set on consumption) and short-lived (``expires_at``),
  so an intercepted token cannot be replayed or used later.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    UUIDPrimaryKeyMixin,
    _inet_type,
    _utcnow,
    _uuid_type,
)


class PendingLogin(UUIDPrimaryKeyMixin, Base):
    """A one-time server-side login state issued after password verification."""

    __tablename__ = "pending_logins"

    user_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # SHA-256 hash of the opaque one-time login token (never stored plaintext).
    token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    ip_address: Mapped[str | None] = mapped_column(_inet_type(), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @property
    def is_expired(self) -> bool:
        now = datetime.now(UTC)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires <= now

    @property
    def is_used(self) -> bool:
        return self.used_at is not None

    @property
    def is_valid(self) -> bool:
        """True if the pending login is unused and not expired."""
        return not self.is_used and not self.is_expired

    def __repr__(self) -> str:
        return (
            f"<PendingLogin user={self.user_id} "
            f"valid={self.is_valid}>"
        )
