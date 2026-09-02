"""VoiceGuard V2 — Common Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response format (Architecture 19.4)."""

    error: str = Field(..., description="Machine-readable error code")
    detail: str = Field(..., description="Human-readable error message")
    field: str | None = Field(
        None, description="Optional field that caused the error"
    )


class PaginationParams(BaseModel):
    """Pagination query parameters."""

    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    limit: int = Field(20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""

    items: list[object] = Field(default_factory=list)
    total_count: int = Field(0, ge=0)
    page: int = Field(1, ge=1)
    limit: int = Field(20, ge=1)
    has_more: bool = Field(False)
