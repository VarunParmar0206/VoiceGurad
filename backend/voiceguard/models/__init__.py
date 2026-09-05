"""VoiceGuard V2 — SQLAlchemy model registry.

Importing this package ensures all ORM models are registered with the
``DeclarativeBase`` and that string-based relationship declarations are
resolved by ``configure_mappers()``.

Usage::

    from voiceguard.models import Base, User, VoiceTemplate, ...

    # ``Base.metadata`` now contains all table definitions and can be
    # used for Alembic auto-generation or ``create_all()``.
"""

from __future__ import annotations

# ── Wire up back-references ──────────────────────────────────────────────
# Each model declares its forward-references as strings.  After all models
# are imported we tell SQLAlchemy to resolve them.
from sqlalchemy.orm import configure_mappers, relationship  # noqa: E402

from voiceguard.models.audit_log import AuditLog
from voiceguard.models.auth_attempt import AuthAttempt
from voiceguard.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from voiceguard.models.challenge import Challenge
from voiceguard.models.pending_login import PendingLogin
from voiceguard.models.session import Session
from voiceguard.models.transaction import Transaction
from voiceguard.models.user import User
from voiceguard.models.voice_model import VoiceModel, VoiceTemplate

# Relationship back-references that were omitted from individual model
# files to avoid circular imports.
User.voice_templates = relationship(
    "VoiceTemplate", back_populates="user", cascade="all, delete-orphan"
)
User.voice_models = relationship(
    "VoiceModel", back_populates="user", cascade="all, delete-orphan"
)
User.sessions = relationship(
    "Session", back_populates="user", cascade="all, delete-orphan"
)
User.sent_transactions = relationship(
    "Transaction",
    foreign_keys=[Transaction.sender_id],
    back_populates="sender",
    cascade="all, delete-orphan",
)
User.challenges = relationship(
    "Challenge", back_populates="user", cascade="all, delete-orphan"
)
User.pending_logins = relationship(
    "PendingLogin", back_populates="user", cascade="all, delete-orphan"
)

VoiceTemplate.user = relationship("User", back_populates="voice_templates")
VoiceTemplate.voice_models = relationship(
    "VoiceModel", back_populates="template", cascade="all, delete-orphan"
)

VoiceModel.user = relationship("User", back_populates="voice_models")
VoiceModel.template = relationship(
    "VoiceTemplate", back_populates="voice_models"
)

Session.user = relationship("User", back_populates="sessions")

Transaction.sender = relationship(
    "User", foreign_keys=[Transaction.sender_id], back_populates="sent_transactions"
)
Transaction.recipient = relationship("User", foreign_keys=[Transaction.recipient_id])

Challenge.user = relationship("User", back_populates="challenges")

PendingLogin.user = relationship("User", back_populates="pending_logins")

configure_mappers()

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "User",
    "VoiceTemplate",
    "VoiceModel",
    "Session",
    "Transaction",
    "AuditLog",
    "Challenge",
    "AuthAttempt",
    "PendingLogin",
]
