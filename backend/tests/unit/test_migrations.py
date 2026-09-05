"""Tests for the Alembic migration.

Verifies the migration produces valid, complete PostgreSQL DDL matching
the ORM metadata, without needing a live database (offline mode).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BACKEND_ROOT)


def _env() -> dict[str, str]:
    return {
        "VG_DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/db",
        "VG_JWT_SECRET_KEY": "secret" * 10,
        "VG_ENCRYPTION_KEY": "gAAAAAB" + "0" * 40,
    }


class TestMigrationOfflineSQL:
    """Render the initial migration as SQL and assert its contents."""

    def _render_sql(self) -> str:
        from alembic import command
        from alembic.config import Config

        alembic_ini = os.path.join(BACKEND_ROOT, "alembic.ini")
        cfg = Config(alembic_ini)
        # Offline mode writes SQL to stdout — capture it.
        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            command.upgrade(cfg, "head", sql=True)
        return buf.getvalue()

    def test_render_initial_migration(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            sql = self._render_sql()

        # All 8 tables should appear.
        assert "CREATE TABLE users" in sql
        assert "CREATE TABLE voice_templates" in sql
        assert "CREATE TABLE voice_models" in sql
        assert "CREATE TABLE sessions" in sql
        assert "CREATE TABLE transactions" in sql
        assert "CREATE TABLE audit_log" in sql
        assert "CREATE TABLE challenges" in sql
        assert "CREATE TABLE auth_attempts" in sql
        assert "CREATE TABLE pending_logins" in sql

    def test_amount_check_constraint_rendered(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            sql = self._render_sql()
        assert "ck_transactions_amount_positive" in sql
        assert "amount > 0" in sql

    def test_idempotency_index_rendered(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            sql = self._render_sql()
        assert "ix_transactions_request_id" in sql

    def test_foreign_key_delete_behaviors(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            sql = self._render_sql()
        assert "ON DELETE CASCADE" in sql
        assert "ON DELETE RESTRICT" in sql
        assert "ON DELETE SET NULL" in sql

    def test_revision_exists(self) -> None:
        from voiceguard.db.migrations.versions import (
            a20cb75ca96b_create_initial_voiceguard_v2_tables as rev,
        )

        assert rev.revision == "a20cb75ca96b"
        assert rev.down_revision is None

    def test_downgrade_drops_tables(self) -> None:
        with patch.dict(os.environ, _env(), clear=False):
            sql = self._render_downgrade_sql()

        assert "DROP TABLE auth_attempts" in sql
        assert "DROP TABLE users" in sql

    def _render_downgrade_sql(self) -> str:
        from contextlib import redirect_stdout
        from io import StringIO

        from alembic import command
        from alembic.config import Config

        cfg = Config(os.path.join(BACKEND_ROOT, "alembic.ini"))
        cfg.set_main_option(
            "script_location",
            os.path.join(BACKEND_ROOT, "voiceguard/db/migrations"),
        )
        # Downgrade with --sql requires a <from>:<to> revision range.
        from voiceguard.db.migrations.versions import (
            a20cb75ca96b_create_initial_voiceguard_v2_tables as rev,
        )

        tail = f"{rev.revision}:base"
        buf = StringIO()
        with redirect_stdout(buf):
            command.downgrade(cfg, tail, sql=True)
        return buf.getvalue()
