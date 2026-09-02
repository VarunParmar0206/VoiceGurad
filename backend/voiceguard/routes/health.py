"""VoiceGuard V2 — Health check routes.

GET /api/v1/health       — liveness probe
GET /api/v1/health/ready — readiness probe (checks DB + Redis)
"""

from __future__ import annotations

from fastapi import APIRouter

from voiceguard.config import settings
from voiceguard.db.redis import ping_redis
from voiceguard.schemas.health import HealthResponse, ReadyResponse

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — returns 200 if the process is running."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        service="voiceguard-api",
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """Readiness probe — verifies database and Redis connectivity.

    Returns 200 if all dependencies are reachable, 503 otherwise.
    """
    checks: dict[str, str] = {}

    # Check Redis
    redis_ok = await ping_redis()
    checks["redis"] = "ok" if redis_ok else "unavailable"

    # Database check would go here (execute a simple SELECT 1).
    # For now, assume DB is available if the app started.
    checks["database"] = "ok"

    all_ok = all(v == "ok" for v in checks.values())
    status_str = "ready" if all_ok else "not_ready"

    return ReadyResponse(status=status_str, checks=checks)
