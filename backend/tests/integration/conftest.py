"""Shared pytest fixtures for async database tests.

Uses an in-memory SQLite database via aiosqlite.  SQLite does not fully
enforce foreign keys by default, so we enable PRAGMA foreign_keys=ON for
integrity tests.
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from voiceguard.models import Base


@pytest_asyncio.fixture
async def session_factory():
    """Create an isolated in-memory async SQLite database per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Enable foreign key enforcement for integrity tests.
    async with factory() as session:
        await session.execute(
            __import__("sqlalchemy").text("PRAGMA foreign_keys=ON")
        )
        await session.commit()

    yield factory

    await engine.dispose()


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    """Provide an AsyncSession for repository CRUD tests."""
    async with session_factory() as s:
        yield s
        await s.rollback()
