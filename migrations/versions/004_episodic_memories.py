"""004 — episodic_memories 情景记忆表"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NOW = sa.text("now()")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "episodic_memories" in inspector.get_table_names():
        return
    op.create_table(
        "episodic_memories",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("turn_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("turn_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "session_id", "turn_end", name="uk_episode_span"),
    )
    op.create_index("idx_episode_user_created", "episodic_memories", ["user_id", "created_at"])
    op.create_index("idx_episode_user_session", "episodic_memories", ["user_id", "session_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "episodic_memories" not in inspector.get_table_names():
        return
    op.drop_index("idx_episode_user_session", table_name="episodic_memories")
    op.drop_index("idx_episode_user_created", table_name="episodic_memories")
    op.drop_table("episodic_memories")
