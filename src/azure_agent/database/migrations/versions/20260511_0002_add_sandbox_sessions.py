"""add sandbox_sessions and file_hydrations

Revision ID: 20260511_0002
Revises: 20260507_0001
Create Date: 2026-05-11 00:01:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260511_0002"
down_revision: Union[str, Sequence[str], None] = "20260507_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Sandbox session lifecycle and snapshot baseline.
    op.create_table(
        "sandbox_sessions",
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("session_marker", sa.Text(), nullable=False),
        sa.Column(
            "last_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id", "thread_id"),
    )

    # Per-file hydration cache, scoped to a sandbox session_marker.
    op.create_table(
        "file_hydrations",
        sa.Column("file_id", sa.Text(), nullable=False),
        sa.Column("session_marker", sa.Text(), nullable=False),
        sa.Column(
            "hydrated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["file_id"],
            ["agent_files.file_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("file_id", "session_marker"),
    )
    op.create_index(
        "ix_file_hydrations_marker",
        "file_hydrations",
        ["session_marker"],
    )

    # Prevent duplicate sandbox paths within a thread (idempotency + collision guard).
    op.create_unique_constraint(
        "uq_agent_files_sandbox_path",
        "agent_files",
        ["user_id", "thread_id", "sandbox_path"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_agent_files_sandbox_path",
        "agent_files",
        type_="unique",
    )
    op.drop_index("ix_file_hydrations_marker", table_name="file_hydrations")
    op.drop_table("file_hydrations")
    op.drop_table("sandbox_sessions")
