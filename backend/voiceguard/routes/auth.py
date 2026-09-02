"""VoiceGuard V2 — Auth routes.

POST /api/v1/auth/register        — Register new user
POST /api/v1/auth/login-password   — Password login (step 1)
POST /api/v1/auth/login-voice      — Voice auth (step 2)
POST /api/v1/auth/refresh          — Refresh access token
POST /api/v1/auth/logout           — Invalidate session

Phase 3: Registration accepts plaintext password; login issues stub
tokens.  Phase 4 will add Argon2id hashing and real JWT generation.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.dependencies import (
    AuthContext,
    get_auth_context,
    get_db,
    rate_limit_login,
)
from voiceguard.schemas.auth import (
    LoginPasswordRequest,
    LoginPasswordResponse,
    LoginVoiceRequest,
    LoginVoiceResponse,
    RegisterRequest,
    RegisterResponse,
    TokenPairResponse,
    TokenRefreshRequest,
)
from voiceguard.schemas.common import ErrorResponse
from voiceguard.services.auth_service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def register(
    body: RegisterRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> RegisterResponse:
    """Register a new user account."""
    svc = AuthService(session)
    try:
        user_id = await svc.register_user(
            username=body.username,
            email=body.email,
            password=body.password,
            display_name=body.display_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from None

    return RegisterResponse(user_id=user_id, username=body.username)


@router.post(
    "/login-password",
    response_model=LoginPasswordResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        423: {"model": ErrorResponse, "description": "Account locked"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def login_password(
    body: LoginPasswordRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    _rate: Annotated[None, Depends(rate_limit_login)] = None,
) -> LoginPasswordResponse:
    """Authenticate with username + password (step 1 of 2)."""
    svc = AuthService(session)
    user_id = await svc.verify_password(body.username, body.password)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return LoginPasswordResponse(
        status="password_verified",
        user_id=user_id,
    )


@router.post(
    "/login-voice",
    response_model=LoginVoiceResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Voice verification failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def login_voice(
    body: LoginVoiceRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> LoginVoiceResponse:
    """Authenticate with voice (step 2 of 2).

    Phase 3 stub: always returns placeholder tokens.
    Phase 4-6 will wire up the full voice verification pipeline.
    """
    import uuid

    try:
        user_id = uuid.UUID(body.user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id must be a valid UUID",
        ) from None

    svc = AuthService(session)
    tokens = await svc.issue_tokens(user_id)

    return LoginVoiceResponse(
        status="authenticated",
        access_token=str(tokens["access_token"]),
        refresh_token=str(tokens["refresh_token"]),
        expires_in=int(tokens["expires_in"]),
        voice_score=0.0,
        anti_spoof_score=None,
    )


@router.post(
    "/refresh",
    response_model=TokenPairResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid refresh token"},
    },
)
async def refresh_token(
    body: TokenRefreshRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> TokenPairResponse:
    """Refresh an access token.

    Phase 3 stub: accepts any token and issues new stub tokens.
    Phase 4 will validate and rotate refresh tokens.
    """
    # Phase 4: validate refresh token, check expiry, rotate
    return TokenPairResponse(
        access_token="stub-access-new",
        refresh_token="stub-refresh-new",
        token_type="bearer",
        expires_in=900,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Session invalidated"},
    },
)
async def logout(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> None:
    """Invalidate session tokens.

    Phase 3 stub: no-op.  Phase 4 will revoke refresh tokens.
    """
    return None
