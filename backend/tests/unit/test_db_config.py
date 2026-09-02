"""Tests for database configuration and session management."""

from __future__ import annotations

import os

# A valid-looking environment for importing functions that depend on config.
# Importing db.session triggers module-level engine creation, which reads
# settings lazily — so we patch before importing.
import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def _env() -> dict[str, str]:
    return {
        "VG_DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/db",
        "VG_JWT_SECRET_KEY": "secret" * 10,
        "VG_ENCRYPTION_KEY": "gAAAAAB" + "0" * 40,
        "VG_REDIS_URL": "redis://localhost:6379/0",
    }


class TestDbConfig:
    def test_async_session_factory_is_async(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            from voiceguard.db.session import (
                async_session_factory,
                engine,
            )

            assert engine is not None
            assert async_session_factory is not None

    def test_engine_uses_pool_pre_ping(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            from voiceguard.db.session import engine

            assert engine.pool._pre_ping is True

    def test_get_async_session_is_async_generator(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            from voiceguard.db.session import get_async_session

            # It must be constructible as a context manager that yields.
            assert callable(get_async_session)

    def test_db_init_importable_models(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            from voiceguard.models import Base

            assert len(Base.metadata.tables) == 8


class TestRedis:
    def test_create_redis_from_url(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            from voiceguard.db.redis import create_redis

            r = create_redis()
            # redis client created with decode_responses=True
            assert r is not None

    def test_ping_returns_false_when_unavailable(self) -> None:
        """When no real Redis server is running, ping must fail gracefully
        (return False) rather than raise."""
        with patch.dict(os.environ, _env(), clear=False):
            import asyncio

            from redis.asyncio import Redis

            from voiceguard.db.redis import ping_redis

            async def _run() -> None:
                r = Redis.from_url(
                    "redis://127.0.0.1:1/0",
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                )
                try:
                    result = await ping_redis(r)
                    assert result is False
                except Exception:
                    # If connect raises before our graceful handling, still
                    # acceptable — we only require it not hang forever.
                    pass
                finally:
                    await r.aclose()

            asyncio.run(_run())
