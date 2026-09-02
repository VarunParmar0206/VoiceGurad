"""create initial VoiceGuard v2 tables

Revision ID: a20cb75ca96b
Revises:
Create Date: 2026-09-02 21:40:55.053265

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op  # noqa: E402
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "a20cb75ca96b"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB
INET = postgresql.INET


def upgrade() -> None:
    # ── users ────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("username", sa.String(32), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(64), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_locked", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "daily_limit",
            sa.Numeric(12, 2),
            server_default=sa.text("50000.00"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── voice_templates ──────────────────────────────────────────────────
    op.create_table(
        "voice_templates",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("template_data", sa.LargeBinary(), nullable=False),
        sa.Column("enrollment_samples", sa.Integer(), nullable=False),
        sa.Column("quality_scores", JSONB, nullable=True),
        sa.Column("salt", sa.LargeBinary(), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_voice_templates_user_id", "voice_templates", ["user_id"]
    )

    # ── voice_models ─────────────────────────────────────────────────────
    op.create_table(
        "voice_models",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "template_id",
            UUID,
            sa.ForeignKey("voice_templates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_type", sa.String(32), nullable=False),
        sa.Column("model_data", sa.LargeBinary(), nullable=False),
        sa.Column("parameters", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_voice_models_user_id", "voice_models", ["user_id"])

    # ── sessions ─────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("refresh_token", sa.String(512), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", INET, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_sessions_user_id", "sessions", ["user_id"]
    )
    op.create_index(
        "ix_sessions_refresh_token", "sessions", ["refresh_token"], unique=True
    )

    # ── transactions ─────────────────────────────────────────────────────
    op.create_table(
        "transactions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "sender_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "recipient_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("recipient_name", sa.String(128), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column(
            "currency",
            sa.String(3),
            server_default=sa.text("'INR'"),
            nullable=False,
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("voice_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("challenge_id", UUID, nullable=True),
        sa.Column("request_id", UUID, nullable=True),
        sa.Column("decline_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "amount > 0", name="ck_transactions_amount_positive"
        ),
    )
    op.create_index(
        "ix_transactions_sender_created", "transactions", ["sender_id", "created_at"]
    )
    op.create_index("ix_transactions_status", "transactions", ["status"])
    op.create_index("ix_transactions_sender_id", "transactions", ["sender_id"])
    op.create_index(
        "ix_transactions_request_id", "transactions", ["request_id"], unique=True
    )

    # ── audit_log ────────────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("event_detail", JSONB, nullable=True),
        sa.Column("ip_address", INET, nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("ix_audit_log_event_type", "audit_log", ["event_type"])
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])

    # ── challenges ───────────────────────────────────────────────────────
    op.create_table(
        "challenges",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("challenge_text", sa.String(128), nullable=False),
        sa.Column("challenge_type", sa.String(32), nullable=False),
        sa.Column(
            "is_used",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_challenges_user_id", "challenges", ["user_id"])

    # ── auth_attempts ────────────────────────────────────────────────────
    op.create_table(
        "auth_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("attempt_type", sa.String(16), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("ip_address", INET, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_auth_attempts_user_id", "auth_attempts", ["user_id"])
    op.create_index("ix_auth_attempts_created_at", "auth_attempts", ["created_at"])


def downgrade() -> None:
    op.drop_table("auth_attempts")
    op.drop_table("challenges")
    op.drop_table("audit_log")
    op.drop_table("transactions")
    op.drop_table("sessions")
    op.drop_table("voice_models")
    op.drop_table("voice_templates")
    op.drop_table("users")
