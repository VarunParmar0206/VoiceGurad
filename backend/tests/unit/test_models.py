"""Tests for the SQLAlchemy model definitions and schema."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from voiceguard.models import (
    Challenge,
    Session,
)


class TestModelMetadata:
    """Verify that all 8 tables are registered with the metadata."""

    @pytest.mark.parametrize(
        "tablename",
        [
            "users",
            "voice_templates",
            "voice_models",
            "sessions",
            "transactions",
            "audit_log",
            "challenges",
            "auth_attempts",
        ],
    )
    def test_table_registered(self, tablename: str) -> None:
        from voiceguard.models import Base

        assert tablename in Base.metadata.tables

    def test_user_columns(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["users"]
        cols = {c.name for c in table.columns}
        assert {
            "id",
            "username",
            "email",
            "password_hash",
            "display_name",
            "is_active",
            "is_locked",
            "daily_limit",
            "created_at",
            "updated_at",
        } <= cols

    def test_voice_templates_has_encrypted_fields(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["voice_templates"]
        cols = {c.name for c in table.columns}
        assert {"template_data", "salt"} <= cols


class TestConstraints:
    def test_unique_username(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["users"]
        uniques = [c for c in table.columns if c.name == "username"]
        assert len(uniques) == 1
        assert uniques[0].unique is True

    def test_unique_email(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["users"]
        assert table.c.email.unique is True

    def test_amount_positive_check(self) -> None:
        from sqlalchemy.schema import CheckConstraint

        from voiceguard.models import Base

        table = Base.metadata.tables["transactions"]
        checks = [
            x
            for x in table.constraints
            if isinstance(x, CheckConstraint) and x.name == "ck_transactions_amount_positive"
        ]
        assert len(checks) == 1
        assert "amount > 0" in str(checks[0].sqltext)

    def test_session_refresh_token_unique(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["sessions"]
        assert table.c.refresh_token.unique is True


class TestForeignKeys:
    def test_voice_templates_fk_user(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["voice_templates"]
        fks = [x for x in table.foreign_keys if x.parent.name == "user_id"]
        assert fks
        assert fks[0].target_fullname == "users.id"
        assert fks[0].ondelete == "CASCADE"

    def test_transactions_amount_fk_behavior(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["transactions"]
        sender_fk = [x for x in table.foreign_keys if x.parent.name == "sender_id"][0]
        recipient_fk = [x for x in table.foreign_keys if x.parent.name == "recipient_id"][0]
        # Sender cannot be deleted while transactions reference it.
        assert sender_fk.ondelete == "RESTRICT"
        # Recipient deletion nullifies the reference.
        assert recipient_fk.ondelete == "SET NULL"

    def test_audit_log_user_fk_set_null(self) -> None:
        from voiceguard.models import Base

        table = Base.metadata.tables["audit_log"]
        fk = [x for x in table.foreign_keys if x.parent.name == "user_id"][0]
        assert fk.ondelete == "SET NULL"


class TestModelProperties:
    def test_session_is_active(self) -> None:
        future = datetime.now(UTC).replace(tzinfo=None)
        # Create session with a far-future expiry.
        s = Session(
            user_id=uuid.uuid4(),
            refresh_token="tok",
            expires_at=future,
        )
        s.expires_at = datetime.now(UTC) + __import__("datetime").timedelta(hours=1)
        assert s.is_active is True

        s.revoked_at = datetime.now(UTC)
        assert s.is_active is False

    def test_challenge_validity(self) -> None:
        import datetime as dt

        live = Challenge(
            user_id=uuid.uuid4(),
            challenge_text="verify 1234",
            challenge_type="number",
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1),
        )
        assert live.is_valid is True

        expired = Challenge(
            user_id=uuid.uuid4(),
            challenge_text="verify 1234",
            challenge_type="number",
            expires_at=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1),
        )
        assert expired.is_expired is True
        assert expired.is_valid is False
