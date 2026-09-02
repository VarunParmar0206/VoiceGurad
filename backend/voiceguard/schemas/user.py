"""VoiceGuard V2 — User request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, EmailStr, Field

# ── Request schemas ──────────────────────────────────────────────────────


class UserUpdateRequest(BaseModel):
    """PUT /api/v1/users/me request body."""

    display_name: str | None = Field(
        None, max_length=64, description="Display name"
    )
    email: EmailStr | None = Field(None, description="New email address")


class PasswordChangeRequest(BaseModel):
    """PUT /api/v1/users/me/password request body."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ..., min_length=8, max_length=128, description="New password"
    )


# ── Response schemas ─────────────────────────────────────────────────────


class UserResponse(BaseModel):
    """User profile response body."""

    user_id: uuid.UUID = Field(..., alias="id")
    username: str
    email: str
    display_name: str | None = None
    balance: Decimal
    daily_limit: Decimal
    is_active: bool
    created_at: datetime

    model_config = {"populate_by_name": True}


class UserBriefResponse(BaseModel):
    """Minimal user info for embedded responses."""

    user_id: uuid.UUID = Field(..., alias="id")
    username: str

    model_config = {"populate_by_name": True}
