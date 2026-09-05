"""VoiceGuard V2 — Auth routes.

POST /api/v1/auth/register        — Register new user (Argon2id hashing)
POST /api/v1/auth/login-password  — Step 1: password verification
POST /api/v1/auth/login-voice     — Step 2: voice (NOT implemented until Phase 6)
POST /api/v1/auth/login-totp      — Step 2 (backup): TOTP verification
POST /api/v1/auth/refresh         — Rotate refresh token
POST /api/v1/auth/logout          — Revoke session(s)
POST /api/v1/auth/totp/setup      — Generate TOTP secret (shown once)
POST /api/v1/auth/totp/confirm    — Activate TOTP after first code

Security notes
**************
- ``login-voice`` NEVER issues tokens: voice biometric verification is not
  implemented until the later ML phases, so it must not simulate a
  successful login (see Phase 4 scope).
- ``login-totp`` issues tokens only after a valid password-authenticated
  flow.  The password step returns a server-side one-time ``login_token``
  bound to the authenticated user; ``login-totp`` consumes that token to
  derive the target account (never trusting a client-supplied UUID) and
  verifies the TOTP code against that account's secret.  An arbitrary user's
  UUID alone cannot be used to obtain tokens.
- ``totp/setup`` and ``totp/confirm`` derive the target account from the
  authenticated JWT (``auth.user_id``).  A client-supplied ``user_id`` is
  never accepted, so an authenticated user cannot configure or enable TOTP
  for another account.
- Password hashes, JWT secrets, and TOTP secrets are never exposed through
  these responses.  TOTP hardware secrets are only returned once at setup
  time.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.dependencies import (
    AuthContext,
    get_auth_context,
    get_db,
    rate_limit_login,
    require_auth,
)
from voiceguard.schemas.auth import (
    LoginPasswordRequest,
    LoginPasswordResponse,
    LoginVoiceRequest,
    LoginVoiceResponse,
    LogoutRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPairResponse,
    TokenRefreshRequest,
    TOTPConfirmRequest,
    TOTPLoginRequest,
    TOTPSetupResponse,
)
from voiceguard.schemas.common import ErrorResponse
from voiceguard.security.rate_limit import RateLimitResult
from voiceguard.services.auth_service import (
    AccountInactiveError,
    AccountLockedError,
    AuthError,
    AuthService,
    InvalidCredentialsError,
    PendingLoginError,
    TokenIssueError,
    TOTPNotEnabledError,
    TOTPVerificationError,
)

logger = logging.getLogger("voiceguard")

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_IP = "ip_address"
_UA = "user_agent"


def _client_meta(request: Request) -> tuple[str | None, str | None]:
    """Extract (ip, user_agent) metadata without logging sensitive data."""
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
    return ip, ua


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def register(
    body: RegisterRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> RegisterResponse:
    """Register a new user account (password hashed with Argon2id)."""
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
        200: {"model": LoginPasswordResponse, "description": "Password verified"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        403: {"model": ErrorResponse, "description": "Account inactive"},
        423: {"model": ErrorResponse, "description": "Account locked"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def login_password(
    body: LoginPasswordRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    _rate: Annotated[RateLimitResult, Depends(rate_limit_login)],
) -> LoginPasswordResponse:
    """Authenticate with username/email + password (step 1 of 2).

    On success returns the user UUID plus a flag that a secondary factor
    is required.  Does **not** yet issue any tokens.
    """
    svc = AuthService(session)
    ip, ua = _client_meta(request)
    try:
        user_id, login_token = await svc.authenticate_password(
            body.username, body.password, user_agent=ua, ip_address=ip
        )
    except AccountLockedError as exc:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail={
                "error": "account_locked",
                "detail": "Account locked due to too many failed attempts.",
                "retry_after": exc.cooldown_seconds,
            },
        ) from None
    except AccountInactiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from None
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    return LoginPasswordResponse(
        status="password_verified",
        user_id=user_id,
        login_token=login_token,
        requires_secondary=True,
    )


@router.post(
    "/login-voice",
    response_model=LoginVoiceResponse,
    responses={
        200: {"model": LoginVoiceResponse, "description": "Voice verification not implemented"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def login_voice(
    body: LoginVoiceRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> LoginVoiceResponse:
    """Voice secondary-factor login (step 2).

    **Phase 4:** Voice biometric verification is NOT implemented until the
    later ML phases.  This endpoint is an API boundary only and does NOT
    issue any tokens and does NOT create a session.  It always returns a
    ``not_implemented`` status.
    """
    return LoginVoiceResponse(
        status="not_implemented",
        reason="voice_verification_not_implemented",
        detail="Voice verification is not available yet. Use password + TOTP.",
    )


@router.post(
    "/login-totp",
    response_model=TokenPairResponse,
    responses={
        200: {"model": TokenPairResponse, "description": "Authenticated"},
        400: {"model": ErrorResponse, "description": "TOTP not enabled / invalid code"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def login_totp(
    body: TOTPLoginRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    _rate: Annotated[RateLimitResult, Depends(rate_limit_login)],
) -> TokenPairResponse:
    """Authenticate with TOTP as the secondary factor (backup MFA).

    Requires the one-time ``login_token`` returned by a prior successful
    password step.  The target user is derived server-side from that token —
    never from a client-supplied ``user_id`` — so a TOTP code cannot be paired
    with an arbitrary account.  Issues access + refresh tokens on success.
    """
    svc = AuthService(session)
    ip, ua = _client_meta(request)
    try:
        pair = await svc.authenticate_totp(
            body.login_token, body.code, user_agent=ua, ip_address=ip
        )
    except PendingLoginError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from None
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from None
    except TOTPNotEnabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except TOTPVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except TokenIssueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from None

    return TokenPairResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        expires_in=pair.expires_in,
    )


@router.post(
    "/refresh",
    response_model=TokenPairResponse,
    responses={
        200: {"model": TokenPairResponse, "description": "Tokens refreshed"},
        401: {"model": ErrorResponse, "description": "Invalid or expired refresh token"},
    },
)
async def refresh_token(
    body: TokenRefreshRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> TokenPairResponse:
    """Rotate the refresh token and return a fresh token pair."""
    svc = AuthService(session)
    ip, ua = _client_meta(request)
    try:
        pair = await svc.refresh_token_pair(
            body.refresh_token, user_agent=ua, ip_address=ip
        )
    except AuthError as exc:
        logger.warning("Token refresh failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired.",
        ) from None

    return TokenPairResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        expires_in=pair.expires_in,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Sessions invalidated"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def logout(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    body: LogoutRequest | None = None,
) -> None:
    """Invalidate one session (by refresh token) or all for the user.

    The request body is optional: a body containing a ``refresh_token``
    revokes exactly that session; an empty/absent body revokes all sessions
    for the authenticated user (and requires a valid access token).
    """
    svc = AuthService(session)
    refresh_token = body.refresh_token if body is not None else None
    if refresh_token:
        await svc.logout(refresh_token, auth.user_id)
        return None
    if not auth.is_authenticated or auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    await svc.logout(None, auth.user_id)
    return None


@router.post(
    "/totp/setup",
    response_model=TOTPSetupResponse,
    responses={
        200: {"model": TOTPSetupResponse, "description": "TOTP secret generated"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def totp_setup(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> TOTPSetupResponse:
    """Generate a TOTP secret + provisioning URI (shown exactly once).

    The target user is the authenticated user derived from the JWT.  A
    client-supplied ``user_id`` is never trusted for authorization.
    """
    if auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    svc = AuthService(session)
    secret, uri = await svc.setup_totp(auth.user_id)
    return TOTPSetupResponse(secret=secret, otpauth_uri=uri)


@router.post(
    "/totp/confirm",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "TOTP enabled"},
        400: {"model": ErrorResponse, "description": "Invalid TOTP code"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def totp_confirm(
    body: TOTPConfirmRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> None:
    """Activate TOTP for the user after verifying their first code.

    The target user is the authenticated user derived from the JWT.  A
    client-supplied ``user_id`` is never trusted for authorization.
    """
    if auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    svc = AuthService(session)
    try:
        await svc.confirm_totp(auth.user_id, body.code)
    except TOTPNotEnabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except TOTPVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    return None
