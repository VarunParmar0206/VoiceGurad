"""VoiceGuard V2 — Pydantic request/response schemas."""

from __future__ import annotations

from voiceguard.schemas.auth import (
    LoginPasswordRequest,
    LoginPasswordResponse,
    LoginVoiceRejectedResponse,
    LoginVoiceRequest,
    LoginVoiceResponse,
    LogoutRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPairResponse,
    TokenRefreshRequest,
)
from voiceguard.schemas.common import (
    ErrorResponse,
    PaginatedResponse,
    PaginationParams,
)
from voiceguard.schemas.health import HealthResponse, ReadyResponse
from voiceguard.schemas.transaction import (
    BalanceResponse,
    TransactionCreateRequest,
    TransactionListResponse,
    TransactionResponse,
)
from voiceguard.schemas.user import (
    PasswordChangeRequest,
    UserBriefResponse,
    UserResponse,
    UserUpdateRequest,
)
from voiceguard.schemas.voice import (
    VoiceEnrollRequest,
    VoiceEnrollResponse,
    VoiceReEnrollRequest,
    VoiceStatusResponse,
    VoiceVerifyRequest,
    VoiceVerifyResponse,
)

__all__ = [
    # Auth
    "LoginPasswordRequest",
    "LoginPasswordResponse",
    "LoginVoiceRejectedResponse",
    "LoginVoiceRequest",
    "LoginVoiceResponse",
    "LogoutRequest",
    "RegisterRequest",
    "RegisterResponse",
    "TokenPairResponse",
    "TokenRefreshRequest",
    # Common
    "ErrorResponse",
    "PaginatedResponse",
    "PaginationParams",
    # Health
    "HealthResponse",
    "ReadyResponse",
    # Transaction
    "BalanceResponse",
    "TransactionCreateRequest",
    "TransactionListResponse",
    "TransactionResponse",
    # User
    "PasswordChangeRequest",
    "UserBriefResponse",
    "UserResponse",
    "UserUpdateRequest",
    # Voice
    "VoiceEnrollRequest",
    "VoiceEnrollResponse",
    "VoiceReEnrollRequest",
    "VoiceStatusResponse",
    "VoiceVerifyRequest",
    "VoiceVerifyResponse",
]
