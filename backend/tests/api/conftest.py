"""Shared pytest fixtures for API tests.

Uses FastAPI TestClient with dependency overrides so the app can be
tested without a live PostgreSQL/Redis server.  Route handlers that
would normally require a DB session get an in-memory SQLite session.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from voiceguard.main import app
from voiceguard.models import Base


@pytest_asyncio.fixture
async def session_factory():
    """In-memory async SQLite database for API tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield factory

    await engine.dispose()


@pytest_asyncio.fixture
async def client(session_factory) -> AsyncGenerator[TestClient, None]:
    """FastAPI TestClient with DB session dependency overridden."""

    from voiceguard.dependencies import get_db

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        session = session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
