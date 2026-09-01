"""003 — knowledge_entries.kind 区分用户文档与长期记忆"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "knowledge_entries" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("knowledge_entries")}
    if "kind" not in columns:
        op.add_column(
            "knowledge_entries",
            sa.Column("kind", sa.String(length=16), nullable=False, server_default="user"),
        )
    indexes = {index["name"] for index in inspector.get_indexes("knowledge_entries")}
    if "idx_knowledge_user_kind" not in indexes:
        op.create_index("idx_knowledge_user_kind", "knowledge_entries", ["user_id", "kind"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "knowledge_entries" not in inspector.get_table_names():
        return
    indexes = {index["name"] for index in inspector.get_indexes("knowledge_entries")}
    if "idx_knowledge_user_kind" in indexes:
        op.drop_index("idx_knowledge_user_kind", table_name="knowledge_entries")
    columns = {column["name"] for column in inspector.get_columns("knowledge_entries")}
    if "kind" in columns:
        op.drop_column("knowledge_entries", "kind")
