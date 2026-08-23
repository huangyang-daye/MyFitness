"""Initial schema (PostgreSQL)

Revision ID: 001
Revises:
Create Date: 2026-08-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NOW = sa.text("now()")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False, server_default="default"),
        sa.Column("profile", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "body_metrics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("record_date", sa.Date(), nullable=False),
        sa.Column("metric_type", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("unit", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="xunji_sync"),
        sa.Column("xunji_ref", sa.String(length=128), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "record_date", "metric_type", "source", name="uk_body_metric"),
    )
    op.create_index("idx_body_user_date", "body_metrics", ["user_id", "record_date"])

    op.create_table(
        "foods",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("uniquekey", sa.String(length=128), nullable=True),
        sa.Column("ntr", sa.JSON(), nullable=False),
        sa.Column("units", sa.JSON(), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="xunji"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_food_name", "foods", ["name"])
    op.create_index("idx_food_uniquekey", "foods", ["uniquekey"])

    op.create_table(
        "nutrition_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("record_date", sa.Date(), nullable=False),
        sa.Column("meal_type", sa.String(length=32), nullable=False),
        sa.Column("food_id", sa.BigInteger(), nullable=True),
        sa.Column("food_name", sa.String(length=200), nullable=False),
        sa.Column("amount", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False, server_default="g"),
        sa.Column("nutrients_snapshot", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="xunji_sync"),
        sa.Column("xunji_record_id", sa.String(length=128), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["food_id"], ["foods.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_nutrition_user_date", "nutrition_logs", ["user_id", "record_date"])

    op.create_table(
        "training_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("record_date", sa.Date(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="xunji_sync"),
        sa.Column("xunji_localid", sa.String(length=64), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "xunji_localid", name="uk_training_localid"),
    )
    op.create_index("idx_training_user_date", "training_logs", ["user_id", "record_date"])

    op.create_table(
        "training_exercises",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("training_log_id", sa.BigInteger(), nullable=False),
        sa.Column("movement_name", sa.String(length=100), nullable=False),
        sa.Column("set_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sets_detail", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["training_log_id"], ["training_logs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "user_goals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("goal_type", sa.String(length=32), nullable=False),
        sa.Column("target_value", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("start_value", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_goal_user", "user_goals", ["user_id", "status"])

    op.create_table(
        "daily_reports",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("content_md", sa.Text(), nullable=False),
        sa.Column("agent_outputs", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "report_date", name="uk_report"),
    )

    op.create_table(
        "agent_plans",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("plan_type", sa.String(length=32), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_plan_user", "agent_plans", ["user_id", "plan_type", "status"])

    op.create_table(
        "sync_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("sync_type", sa.String(length=32), nullable=False),
        sa.Column("sync_start_date", sa.Date(), nullable=True),
        sa.Column("sync_end_date", sa.Date(), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("error_log", sa.Text(), nullable=True),
        sa.Column("stats", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_sync_user", "sync_jobs", ["user_id", "sync_type"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", name="uk_session"),
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_msg_session", "chat_messages", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_msg_session", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_index("idx_sync_user", table_name="sync_jobs")
    op.drop_table("sync_jobs")
    op.drop_index("idx_plan_user", table_name="agent_plans")
    op.drop_table("agent_plans")
    op.drop_table("daily_reports")
    op.drop_index("idx_goal_user", table_name="user_goals")
    op.drop_table("user_goals")
    op.drop_table("training_exercises")
    op.drop_index("idx_training_user_date", table_name="training_logs")
    op.drop_table("training_logs")
    op.drop_index("idx_nutrition_user_date", table_name="nutrition_logs")
    op.drop_table("nutrition_logs")
    op.drop_index("idx_food_uniquekey", table_name="foods")
    op.drop_index("idx_food_name", table_name="foods")
    op.drop_table("foods")
    op.drop_index("idx_body_user_date", table_name="body_metrics")
    op.drop_table("body_metrics")
    op.drop_table("users")
