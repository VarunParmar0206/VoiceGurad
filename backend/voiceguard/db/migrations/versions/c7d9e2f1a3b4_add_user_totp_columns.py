"""Add TOTP columns (totp_secret, totp_enabled) to users table.

The TOTP secret is stored AES-256-GCM-encrypted (BYTEA), never plaintext.

Revision ID: c7d9e2f1a3b4
Revises: b30d8f1a2c4e
Create Date: 2026-09-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7d9e2f1a3b4"
down_revision: str | None = "b30d8f1a2c4e"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("totp_secret", sa.LargeBinary(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "totp_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "totp_secret")
    op.drop_column("users", "totp_enabled")
