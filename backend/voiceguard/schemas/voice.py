"""VoiceGuard V2 — Voice request/response schemas.

Phase 3 provides schema definitions only. Voice processing pipeline
(preprocessing, feature extraction, ML inference) belongs to Phases 5-6.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

# ── Request schemas ──────────────────────────────────────────────────────


class VoiceEnrollRequest(BaseModel):
    """Voice enrollment request metadata.

    Audio files are sent as multipart/form-data.
    """

    user_id: uuid.UUID = Field(..., description="User to enroll")


class VoiceVerifyRequest(BaseModel):
    """Voice verification request metadata.

    Audio file is sent as multipart/form-data.
    """

    user_id: uuid.UUID = Field(..., description="User to verify against")
    challenge_id: uuid.UUID | None = Field(
        None, description="Associated challenge ID"
    )


class VoiceReEnrollRequest(BaseModel):
    """Voice re-enrollment request metadata."""

    user_id: uuid.UUID = Field(..., description="User to re-enroll")


# ── Response schemas ─────────────────────────────────────────────────────


class VoiceEnrollResponse(BaseModel):
    """POST /api/v1/voice/enroll response body."""

    status: str = Field(
        "enrolled", description="Enrollment status"
    )
    user_id: uuid.UUID
    template_id: uuid.UUID = Field(
        ..., description="ID of the created voice template"
    )
    enrollment_samples: int = Field(
        ..., description="Number of samples accepted"
    )


class VoiceStatusResponse(BaseModel):
    """GET /api/v1/voice/status response body."""

    user_id: uuid.UUID
    is_enrolled: bool
    enrollment_samples: int = Field(0)
    model_version: str | None = None
    created_at: str | None = None


class VoiceVerifyResponse(BaseModel):
    """POST /api/v1/voice/verify response body."""

    status: str = Field(..., description="verified | rejected")
    voice_score: float = Field(
        ..., description="Speaker verification score"
    )
    anti_spoof_score: float | None = Field(
        None, description="Anti-spoofing composite score"
    )
    threshold: float = Field(
        ..., description="Threshold used for decision"
    )
