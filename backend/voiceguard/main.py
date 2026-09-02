"""VoiceGuard V2 — FastAPI application factory.

Creates the ASGI application with middleware stack (CORS, request-ID
propagation, structured logging, rate limiting) and registers all
route modules.

Usage::

    uvicorn voiceguard.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from voiceguard.config import settings
from voiceguard.db import close_engine, close_redis
from voiceguard.routes import (
    auth_router,
    health_router,
    transactions_router,
    users_router,
    voice_router,
)

logger = logging.getLogger("voiceguard")


# ── Error code mapping ───────────────────────────────────────────────────


def _http_error_code(status_code: int) -> str:
    """Return a machine-readable error code for an HTTP status code."""
    mapping = {
        status.HTTP_400_BAD_REQUEST: "bad_request",
        status.HTTP_401_UNAUTHORIZED: "unauthorized",
        status.HTTP_403_FORBIDDEN: "forbidden",
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_409_CONFLICT: "conflict",
        status.HTTP_422_UNPROCESSABLE_CONTENT: "validation_error",
        status.HTTP_423_LOCKED: "account_locked",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
        status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_server_error",
        status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
    }
    return mapping.get(status_code, "request_error")


# ── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown hooks."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("VoiceGuard API starting (version=%s)", settings.APP_VERSION)
    yield
    logger.info("VoiceGuard API shutting down")
    await close_engine()
    await close_redis()


# ── Application factory ──────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured ``FastAPI`` instance.
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────

    # CORS — permissive in dev, restrict in production.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ───────────────────────────────────────────────

    from fastapi import HTTPException

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        """Convert HTTPExceptions into the standard error format.

        The ``detail`` may be a plain string (mapped to a stable error code)
        or an ErrorResponse-style dict for richer fields.
        """
        if isinstance(exc.detail, dict):
            content = {
                "error": exc.detail.get("error", "request_error"),
                "detail": exc.detail.get("detail", "Request failed"),
                "field": exc.detail.get("field"),
            }
        else:
            content = {
                "error": _http_error_code(exc.status_code),
                "detail": str(exc.detail),
                "field": None,
            }
        headers = dict(exc.headers or {})
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Translate Pydantic validation errors into the standard format."""
        errors = exc.errors()
        if errors:
            first = errors[0]
            field = ".".join(str(loc) for loc in first.get("loc", []))
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={
                    "error": "validation_error",
                    "detail": first.get("msg", "Invalid input"),
                    "field": field or None,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "validation_error",
                "detail": "Invalid request body",
                "field": None,
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Catch-all handler — logs the error, returns a safe 500."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            "Unhandled exception [request_id=%s]: %s", request_id, exc
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred",
                "field": None,
            },
        )

    # ── Request-ID middleware ─────────────────────────────────────────────

    @application.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Propagate a request ID through every request/response cycle.

        Accepts ``X-Request-ID`` from the client or generates a new UUID v4.
        Returns the ID in the ``X-Request-ID`` response header.
        """
        raw = request.headers.get("x-request-id")
        try:
            rid = str(uuid.UUID(raw)) if raw else str(uuid.uuid4())
        except ValueError:
            rid = str(uuid.uuid4())

        request.state.request_id = rid

        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # ── Route registration ───────────────────────────────────────────────

    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(users_router)
    application.include_router(transactions_router)
    application.include_router(voice_router)

    return application


# ── Module-level app instance (for ``uvicorn voiceguard.main:app``) ────

app = create_app()
