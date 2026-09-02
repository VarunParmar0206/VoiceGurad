"""VoiceGuard V2 — Auth request/response schemas.

Phase 3 provides schema definitions only. Authentication logic
(password hashing, JWT tokens) belongs to Phase 4.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, EmailStr, Field

# ── Request schemas ──────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    """POST /api/v1/auth/register request body."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=32,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Unique username (alphanumeric + underscore)",
    )
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(
        ..., min_length=8, max_length=128, description="User password"
    )
    display_name: str | None = Field(
        None, max_length=64, description="Optional display name"
    )


class LoginPasswordRequest(BaseModel):
    """POST /api/v1/auth/login-password request body."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Account password")


class LoginVoiceRequest(BaseModel):
    """POST /api/v1/auth/login-voice request body (multipart handled by route)."""

    user_id: str = Field(
        ..., description="Username or user UUID"
    )
    challenge_id: uuid.UUID = Field(
        ..., description="ID of the challenge being responded to"
    )
    device_id: str | None = Field(
        None, description="Optional device identifier"
    )


class TokenRefreshRequest(BaseModel):
    """POST /api/v1/auth/refresh request body."""

    refresh_token: str = Field(..., description="Refresh token to rotate")


class LogoutRequest(BaseModel):
    """POST /api/v1/auth/logout request body."""

    refresh_token: str | None = Field(
        None, description="Specific refresh token to revoke; omit for all"
    )


# ── Response schemas ─────────────────────────────────────────────────────


class TokenPairResponse(BaseModel):
    """Token pair returned after successful authentication."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(
        ..., description="Access token lifetime in seconds"
    )


class RegisterResponse(BaseModel):
    """POST /api/v1/auth/register response body."""

    user_id: uuid.UUID = Field(..., description="Newly created user ID")
    username: str = Field(..., description="Registered username")


class LoginPasswordResponse(BaseModel):
    """POST /api/v1/auth/login-password response body (step 1 success)."""

    status: str = Field(
        "password_verified",
        description="Status after password verification",
    )
    user_id: uuid.UUID = Field(..., description="Authenticated user ID")


class LoginVoiceResponse(BaseModel):
    """POST /api/v1/auth/login-voice response body."""

    status: str = Field(
        "authenticated", description="Authentication status"
    )
    access_token: str | None = Field(None, description="JWT access token")
    refresh_token: str | None = Field(None, description="Refresh token")
    expires_in: int | None = Field(
        None, description="Access token lifetime in seconds"
    )
    voice_score: float | None = Field(
        None, description="Voice verification score"
    )
    anti_spoof_score: float | None = Field(
        None, description="Anti-spoofing composite score"
    )


class LoginVoiceRejectedResponse(BaseModel):
    """POST /api/v1/auth/login-voice response body (rejection)."""

    status: str = Field("rejected", description="Authentication rejected")
    reason: str = Field(..., description="Machine-readable rejection reason")
    detail: str = Field(..., description="Human-readable rejection detail")
    voice_score: float | None = Field(None)
    attempts_remaining: int | None = Field(None)
