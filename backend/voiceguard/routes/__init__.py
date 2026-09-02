"""VoiceGuard V2 — API route modules."""

from __future__ import annotations

from voiceguard.routes.auth import router as auth_router
from voiceguard.routes.health import router as health_router
from voiceguard.routes.transactions import router as transactions_router
from voiceguard.routes.users import router as users_router
from voiceguard.routes.voice import router as voice_router

__all__ = [
    "auth_router",
    "health_router",
    "transactions_router",
    "users_router",
    "voice_router",
]
