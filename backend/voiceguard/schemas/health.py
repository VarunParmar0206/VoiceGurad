"""VoiceGuard V2 — Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response body."""

    status: str = Field("ok", description="Service health status")
    version: str = Field(..., description="Application version")
    service: str = Field("voiceguard-api", description="Service name")


class ReadyResponse(BaseModel):
    """Readiness probe response body."""

    status: str = Field(..., description="Readiness status: ready | not_ready")
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="Component health checks (e.g. database, redis)",
    )
