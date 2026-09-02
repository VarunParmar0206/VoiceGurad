"""VoiceGuard V2 — User routes.

GET  /api/v1/users/me        — Get authenticated user's profile
PUT  /api/v1/users/me        — Update profile
PUT  /api/v1/users/me/password — Change password
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.dependencies import (
    AuthContext,
    get_db,
    require_auth,
)
from voiceguard.schemas.common import ErrorResponse
from voiceguard.schemas.user import (
    PasswordChangeRequest,
    UserResponse,
    UserUpdateRequest,
)

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        404: {"model": ErrorResponse, "description": "User not found"},
    },
)
async def get_profile(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> UserResponse:
    """Return the authenticated user's profile.

    Phase 3 stub: always returns 401 (no JWT verification yet).
    Phase 4 will populate auth.user_id from the JWT.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.put(
    "/me",
    response_model=UserResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        404: {"model": ErrorResponse, "description": "User not found"},
        409: {"model": ErrorResponse, "description": "Email already in use"},
    },
)
async def update_profile(
    body: UserUpdateRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> UserResponse:
    """Update the authenticated user's profile.

    Phase 3 stub: always returns 401.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.put(
    "/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Password updated"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        403: {"model": ErrorResponse, "description": "Current password incorrect"},
    },
)
async def change_password(
    body: PasswordChangeRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> None:
    """Change the authenticated user's password.

    Phase 3 stub: always returns 401.
    Phase 4 will verify current password and hash new one with Argon2id.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )
