"""SQL 查询日志 — 在控制台打印 ORM / 原生 SQL。"""

from __future__ import annotations

import logging
from typing import Any

from myfitness.config import get_settings

_CONFIGURED = False
logger = logging.getLogger("myfitness.sql")


def is_sql_echo_enabled() -> bool:
    settings = get_settings()
    return bool(settings.sql_echo or settings.debug_mode)


def configure_sql_logging() -> None:
    """启用 SQLAlchemy 引擎 echo 与统一控制台格式。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    if not is_sql_echo_enabled():
        return

    engine_logger = logging.getLogger("sqlalchemy.engine.Engine")
    engine_logger.setLevel(logging.INFO)
    if not engine_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [SQL] %(message)s")
        )
        engine_logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [SQL] %(message)s")
        )
        logger.addHandler(handler)


def log_query_context(tool: str, **context: Any) -> None:
    """记录 Agent 查询工具的参数上下文（便于对照 ORM SQL）。"""
    if not is_sql_echo_enabled():
        return
    parts = [f"{key}={value}" for key, value in context.items() if value is not None]
    logger.info("[%s] %s", tool, ", ".join(parts) if parts else "(no params)")


def log_raw_sql(statement: str, params: dict[str, Any] | None = None) -> None:
    """记录原生 SQL 文本（如 pgvector 检索）。"""
    if not is_sql_echo_enabled():
        return
    safe = _sanitize_params(params or {})
    logger.info("SQL:\n%s\nPARAMS: %s", statement.strip(), safe)


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if key in {"query_vec", "embedding"}:
            safe[key] = "<vector omitted>"
        elif isinstance(value, str) and len(value) > 200:
            safe[key] = value[:200] + "…"
        else:
            safe[key] = value
    return safe
