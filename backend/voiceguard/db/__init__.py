"""VoiceGuard V2 — Database layer.

Provides:
- ``db.session``  — async SQLAlchemy engine + session factory + ``init_db()``
- ``db.redis``    — async Redis client + helpers
"""

from __future__ import annotations

from voiceguard.db.redis import (
    close_redis,
    create_redis,
    get_redis,
    ping_redis,
)
from voiceguard.db.session import (
    async_session_factory,
    close_engine,
    engine,
    get_async_session,
    init_db,
)

__all__ = [
    "engine",
    "async_session_factory",
    "get_async_session",
    "init_db",
    "close_engine",
    "create_redis",
    "get_redis",
    "ping_redis",
    "close_redis",
]
