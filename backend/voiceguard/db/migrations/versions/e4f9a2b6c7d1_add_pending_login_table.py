"""Add pending_logins table for two-step login server-side state.

The pending login binds an opaque one-time login token (stored as a SHA-256
hash) to a user after a successful password verification.  The secondary
factor (TOTP) consumes it to derive the target account server-side rather
than trusting a client-supplied user_id.

Revision ID: e4f9a2b6c7d1
Revises: c7d9e2f1a3b4
Create Date: 2026-09-03
"""

from __future__ import annotations

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e4f9a2b6c7d1"
down_revision: str | None = "c7d9e2f1a3b4"
branch_labels: str | None = None
depends_on: str | None = None


user_id_type = sa.Uuid()
token_hash = sa.String(64)
inet_type = pg.INET().with_variant(sa.String(45), "sqlite")


def upgrade() -> None:
    op.create_table(
        "pending_logins",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            user_id_type,
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", token_hash, nullable=False),
        sa.Column("ip_address", inet_type, nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_pending_logins_user_id", "pending_logins", ["user_id"])
    op.create_index("ix_pending_logins_token_hash", "pending_logins", ["token_hash"])


def downgrade() -> None:
    op.drop_index("ix_pending_logins_token_hash", table_name="pending_logins")
    op.drop_index("ix_pending_logins_user_id", table_name="pending_logins")
    op.drop_table("pending_logins")
