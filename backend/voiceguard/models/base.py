"""VoiceGuard V2 — SQLAlchemy base and type helpers.

Defines portable column types so the models can be exercised against both
PostgreSQL (production) and SQLite (fast, isolated tests) without changes.
PostgreSQL keeps its native, most-capable types; SQLite falls back to
portable equivalents.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy import DateTime
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all VoiceGuard models."""

    pass


# ── Portable column types ────────────────────────────────────────────────
# On PostgreSQL use the native types; on other dialects (SQLite) fall back
# to portable equivalents so tests can run without a live PostgreSQL server.

def _uuid_type() -> sa.types.TypeEngine[uuid.UUID]:
    # sqlalchemy.Uuid renders as UUID on PostgreSQL and as CHAR(32) on
    # SQLite, so no explicit variant is needed.
    return sa.Uuid()


def _jsonb_type() -> sa.types.TypeEngine[dict[str, object]]:

    return postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def _inet_type() -> sa.types.TypeEngine[str]:

    return postgresql.INET().with_variant(sa.String(45), "sqlite")


def _bigint_pk_type() -> sa.types.TypeEngine[int]:
    """Auto-incrementing big-int primary key.

    Renders as ``BIGSERIAL`` on PostgreSQL (native, most capable) and as
    ``INTEGER PRIMARY KEY`` on SQLite.  SQLite only auto-increments the
    ``INTEGER PRIMARY KEY`` rowid alias — a plain ``BIGINT PRIMARY KEY``
    would NOT auto-generate values, breaking tests.
    """

    return sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def _utcnow() -> datetime:
    """Portable UTC now — used as a Python-side column default.

    A Python-side default (rather than ``func.now()`` server default) is used
    so the timestamp is populated identically on PostgreSQL and SQLite.
    SQLite does not implement ``func.now()``, which would otherwise raise on
    live DML during tests.
    """
    return datetime.now(UTC)


class TimestampMixin:
    """Mixin that adds created_at / updated_at with automatic defaults."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """Mixin that adds a portable UUID primary key column."""

    id: Mapped[uuid.UUID] = mapped_column(
        _uuid_type(),
        primary_key=True,
        default=uuid.uuid4,
    )
