"""VoiceGuard V2 — Voice routes.

POST /api/v1/voice/enroll     — Upload enrollment audio samples
GET  /api/v1/voice/status     — Get enrollment status
POST /api/v1/voice/re-enroll  — Re-enroll (replace existing template)

Phase 3: All voice routes are stubs.
Phase 5-6 will wire up the full voice processing pipeline
(preprocessing, feature extraction, embedding, anti-spoof, verification).
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
from voiceguard.schemas.voice import (
    VoiceEnrollResponse,
    VoiceStatusResponse,
)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


@router.post(
    "/enroll",
    response_model=VoiceEnrollResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"model": VoiceEnrollResponse},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        422: {"model": ErrorResponse, "description": "Invalid audio data"},
    },
)
async def enroll_voice(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> VoiceEnrollResponse:
    """Upload voice samples for enrollment.

    Phase 3 stub: always returns 401.
    Phase 6 will implement the full enrollment pipeline.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.get(
    "/status",
    response_model=VoiceStatusResponse,
    responses={
        200: {"model": VoiceStatusResponse},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def voice_status(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> VoiceStatusResponse:
    """Get voice enrollment status.

    Phase 3 stub: always returns 401.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.post(
    "/re-enroll",
    response_model=VoiceEnrollResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"model": VoiceEnrollResponse},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def re_enroll_voice(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> VoiceEnrollResponse:
    """Re-enroll voice (deactivate existing template, create new one).

    Phase 3 stub: always returns 401.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )
