"""VoiceGuard V2 — Transaction routes.

POST /api/v1/transactions          — Create transaction
GET  /api/v1/transactions          — List transactions (paginated)
GET  /api/v1/transactions/{id}     — Get transaction detail
GET  /api/v1/transactions/balance  — Get balance info
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from voiceguard.dependencies import (
    AuthContext,
    get_db,
    rate_limit_transaction,
    require_auth,
)
from voiceguard.schemas.common import ErrorResponse
from voiceguard.schemas.transaction import (
    BalanceResponse,
    TransactionCreateRequest,
    TransactionListResponse,
    TransactionResponse,
)

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])


@router.post(
    "",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"model": TransactionResponse},
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        403: {"model": ErrorResponse, "description": "Insufficient balance or daily limit"},
        409: {"model": ErrorResponse, "description": "Duplicate request (idempotent)"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_transaction(
    body: TransactionCreateRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
    _rate: Annotated[None, Depends(rate_limit_transaction)] = None,
) -> TransactionResponse:
    """Create a new transaction.

    Phase 3 stub: always returns 401 (no auth yet).
    Phase 4-5 will add voice authorization before transaction execution.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.get(
    "",
    response_model=TransactionListResponse,
    responses={
        200: {"model": TransactionListResponse},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def list_transactions(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
) -> TransactionListResponse:
    """List transactions for the authenticated user.

    Phase 3 stub: always returns 401.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.get(
    "/balance",
    response_model=BalanceResponse,
    responses={
        200: {"model": BalanceResponse},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
    },
)
async def get_balance(
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> BalanceResponse:
    """Get balance and daily limit info.

    Phase 3 stub: always returns 401.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    responses={
        200: {"model": TransactionResponse},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        404: {"model": ErrorResponse, "description": "Transaction not found"},
    },
)
async def get_transaction(
    transaction_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[AuthContext, Depends(require_auth)],
) -> TransactionResponse:
    """Get a specific transaction by ID.

    Phase 3 stub: always returns 401.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )
