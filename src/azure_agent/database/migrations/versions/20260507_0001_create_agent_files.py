"""create agent_files

Revision ID: 20260507_0001
Revises:
Create Date: 2026-05-07 00:01:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260507_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_files",
        sa.Column("file_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("blob_path", sa.Text(), nullable=False),
        sa.Column("sandbox_path", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('upload', 'artifact')",
            name="ck_agent_files_role",
        ),
        sa.PrimaryKeyConstraint("file_id"),
    )
    op.create_index(
        "ix_agent_files_thread_created_at",
        "agent_files",
        ["user_id", "thread_id", "created_at"],
    )
    op.create_index(
        "ix_agent_files_job_created_at",
        "agent_files",
        ["user_id", "thread_id", "job_id", "created_at"],
    )
    op.create_index(
        "ix_agent_files_role_created_at",
        "agent_files",
        ["user_id", "thread_id", "role", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_files_role_created_at", table_name="agent_files")
    op.drop_index("ix_agent_files_job_created_at", table_name="agent_files")
    op.drop_index("ix_agent_files_thread_created_at", table_name="agent_files")
    op.drop_table("agent_files")
