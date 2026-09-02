"""Alembic migration environment — async URL support.

Configures a sync engine for migrations.  The ``DATABASE_URL`` from
settings uses ``postgresql+asyncpg://``; we rewrite it to
``postgresql+psycopg2://`` for the synchronous Alembic engine, so schema
operations run against the same database.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure backend/ is importable so ``voiceguard`` resolves.
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# Load settings to obtain DATABASE_URL without instantiating engines.
from voiceguard.config import settings  # noqa: E402
from voiceguard.models import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Convert async SQLAlchemy URL to a sync-compatible URL for migrations.
_db_url = settings.DATABASE_URL
if _db_url.startswith("postgresql+asyncpg://"):
    _db_url = _db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
elif _db_url.startswith("sqlite+aiosqlite://"):
    _db_url = _db_url.replace("sqlite+aiosqlite://", "sqlite://")

config.set_main_option("sqlalchemy.url", _db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Emits SQL without a live database connection (``alembic upgrade
    head --sql``).
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (against a live database)."""
    # Ensure psycopg2 is available for the sync engine.
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
