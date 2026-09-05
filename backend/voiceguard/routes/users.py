"""VoiceGuard V2 — User routes.

GET  /api/v1/users/me           — Get authenticated user's profile
PUT  /api/v1/users/me           — Update profile
PUT  /api/v1/users/me/password  — Change password (revokes sessions)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.dependencies import AuthContext, get_db, require_auth
from voiceguard.schemas.common import ErrorResponse
from voiceguard.schemas.user import (
    PasswordChangeRequest,
    UserResponse,
    UserUpdateRequest,
)
from voiceguard.services.auth_service import AuthService
from voiceguard.services.user_service import UserService

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserResponse,
    responses={
        200: {"model": UserResponse, "description": "User profile"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        404: {"model": ErrorResponse, "description": "User not found"},
    },
)
async def get_profile(
    session: Annotated[AsyncSession, Depends(get_db)],
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> UserResponse:
    """Return the authenticated user's profile."""
    if auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    svc = UserService(session)
    user = await svc.get_by_id(auth.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return UserResponse(
        user_id=user.id,
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        balance=user.balance,
        daily_limit=user.daily_limit,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.put(
    "/me",
    response_model=UserResponse,
    responses={
        200: {"model": UserResponse, "description": "Updated profile"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        404: {"model": ErrorResponse, "description": "User not found"},
        409: {"model": ErrorResponse, "description": "Email already in use"},
    },
)
async def update_profile(
    body: UserUpdateRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> UserResponse:
    """Update the authenticated user's profile (display name / email)."""
    if auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    svc = UserService(session)
    user = await svc.get_by_id(auth.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    try:
        user = await svc.update_profile(
            user,
            display_name=body.display_name,
            email=body.email,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from None
    return UserResponse(
        user_id=user.id,
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        balance=user.balance,
        daily_limit=user.daily_limit,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.put(
    "/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Password updated"},
        400: {"model": ErrorResponse, "description": "Invalid new password"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        403: {"model": ErrorResponse, "description": "Current password incorrect"},
    },
)
async def change_password(
    body: PasswordChangeRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> None:
    """Change the authenticated user's password.

    On success, all existing sessions are revoked so other devices must
    re-authenticate.
    """
    if auth.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    svc = UserService(session)
    auth_svc = AuthService(session)
    user = await svc.get_by_id(auth.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        await svc.change_own_password(
            user,
            current_password=body.current_password,
            new_password=body.new_password,
        )
    except ValueError as exc:
        # Distinguish wrong current password (403) from invalid new password (400).
        if "Current password" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from None
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    # Revoke all existing sessions after a password change.
    await auth_svc.logout(None, auth.user_id)

    # Audit the password change (never log the password itself).
    from voiceguard.models.audit_log import AuditLog

    entry = AuditLog(
        user_id=auth.user_id,
        event_type="auth.password.change",
        event_detail=None,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", ""),
    )
    session.add(entry)
    await session.flush()
    return None
