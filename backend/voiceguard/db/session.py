"""VoiceGuard V2 — Async SQLAlchemy session factory and engine management.

Usage::

    from voiceguard.db.session import get_async_session, init_db

    # At application startup:
    await init_db()

    # In request handlers (FastAPI dependency):
    async for session in get_async_session():
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from voiceguard.config import settings

# ── Engine ───────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# ── Session factory ─────────────────────────────────────────────────────

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` and ensure it is closed after use.

    Designed to be used as a FastAPI dependency::

        @router.get("/items")
        async def list_items(session: AsyncSession = Depends(get_async_session)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables (for development / testing only).

    In production, use Alembic migrations instead.
    """
    from voiceguard.models import Base  # noqa: F811

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_engine() -> None:
    """Dispose of the engine connection pool.  Call at shutdown."""
    await engine.dispose()
