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


class TOTPSetupRequest(BaseModel):
    """POST /api/v1/auth/totp/setup request body.

    The user is derived from the authenticated JWT — the target is never
    taken from a client-supplied ``user_id`` (prevents configuring another
    account's TOTP).
    """


class TOTPSetupResponse(BaseModel):
    """Response containing the one-time TOTP secret + provisioning URI."""

    secret: str = Field(..., description="Base32 TOTP secret (shown once)")
    otpauth_uri: str = Field(..., description="otpauth:// provisioning URI")


class TOTPConfirmRequest(BaseModel):
    """POST /api/v1/auth/totp/confirm request body.

    The user is derived from the authenticated JWT — the target is never
    taken from a client-supplied ``user_id`` (prevents enabling another
    account's TOTP).
    """

    code: str = Field(
        ...,
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
        description="6-digit TOTP code from the authenticator app",
    )


class TOTPLoginRequest(BaseModel):
    """POST /api/v1/auth/login-totp request body (secondary factor).

    The target user is derived **server-side** from the one-time
    ``login_token`` issued by the password step — never from a
    client-supplied ``user_id``.  This prevents pairing an arbitrary user's
    UUID with a valid code without first completing that user's password step.
    """

    login_token: str = Field(
        ..., description="One-time login state issued by the password step"
    )
    code: str = Field(
        ...,
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
        description="6-digit TOTP code",
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
    """POST /api/v1/auth/login-password response body (step 1 success).

    Returns the authenticated ``user_id`` (informational) plus a short-lived,
    one-time ``login_token`` that the secondary-factor step (``login-totp``)
    must present to complete authentication.  The token is server-side bound
    to this user, so the secondary step never trusts a client-supplied UUID.
    """

    status: str = Field(
        "password_verified",
        description="Status after password verification",
    )
    user_id: uuid.UUID = Field(..., description="Authenticated user ID")
    login_token: str = Field(
        ...,
        description="One-time server-side login state consumed by the "
        "secondary-factor step (expires shortly)",
    )
    requires_secondary: bool = Field(
        True,
        description="True if a secondary factor (TOTP/voice) is required "
        "before tokens are issued",
    )


class LoginVoiceResponse(BaseModel):
    """POST /api/v1/auth/login-voice response body.

    **Phase 4:** Voice biometric verification is NOT implemented until the
    later ML phases.  This endpoint therefore never issues access/refresh
    tokens.  ``status`` is one of ``not_implemented`` (primary), and this
    schema is the envelope used to convey that the secondary factor is
    unavailable.
    """

    status: str = Field(
        "not_implemented",
        description="Authentication status (voice is not yet implemented)",
    )
    reason: str | None = Field(
        None, description="Machine-readable reason (e.g. not_implemented)"
    )
    detail: str | None = Field(
        None, description="Human-readable explanation"
    )


class LoginVoiceRejectedResponse(BaseModel):
    """POST /api/v1/auth/login-voice response body (rejection)."""

    status: str = Field("rejected", description="Authentication rejected")
    reason: str = Field(..., description="Machine-readable rejection reason")
    detail: str = Field(..., description="Human-readable rejection detail")
    voice_score: float | None = Field(None)
    attempts_remaining: int | None = Field(None)
