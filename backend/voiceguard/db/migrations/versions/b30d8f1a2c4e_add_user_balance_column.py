"""Add balance column to users table.

Revision ID: b30d8f1a2c4e
Revises: a20cb75ca96b
Create Date: 2026-09-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b30d8f1a2c4e"
down_revision: str | None = "a20cb75ca96b"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "balance",
            sa.Numeric(12, 2),
            nullable=False,
            server_default="10000.00",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "balance")
