"""VoiceGuard V2 — Voice template and voice model SQLAlchemy models.

These tables store biometric data encrypted at rest using AES-256-GCM
(see ``voiceguard.security.crypto.BiometricEncryptor``).

What is encrypted
*****************
- ``VoiceTemplate.template_data`` — the cancelable-transformed speaker
  embedding(s).  Raw audio and raw embeddings are **never** stored.
- ``VoiceModel.model_data``       — GMM parameters or other per-user
  model artifacts.

What remains plaintext
**********************
- ``user_id``, ``model_version``, ``enrollment_samples``, ``quality_scores``
- ``salt`` (per-user random value used for cancelable transform derivation)
- ``model_type``, ``parameters`` (JSONB of non-secret hyperparameters)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from voiceguard.models.base import (
    Base,
    UUIDPrimaryKeyMixin,
    _jsonb_type,
    _utcnow,
    _uuid_type,
)


class VoiceTemplate(UUIDPrimaryKeyMixin, Base):
    """Cancelable biometric template produced during enrollment.

    One user may have multiple templates over time (e.g., re-enrollment).
    Only the most recent ``is_active = True`` template is used for
    verification.
    """

    __tablename__ = "voice_templates"

    user_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    template_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    enrollment_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_scores: Mapped[dict | None] = mapped_column(_jsonb_type(), nullable=True)
    salt: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<VoiceTemplate user={self.user_id} "
            f"version={self.model_version} active={self.is_active}>"
        )


class VoiceModel(UUIDPrimaryKeyMixin, Base):
    """Per-user voice model (e.g., GMM parameters) stored encrypted at rest.

    Linked to exactly one ``VoiceTemplate`` and one ``User``.
    """

    __tablename__ = "voice_models"

    user_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    template_id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        ForeignKey("voice_templates.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    parameters: Mapped[dict | None] = mapped_column(_jsonb_type(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<VoiceModel user={self.user_id} type={self.model_type}>"
