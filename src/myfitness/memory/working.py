"""工作记忆 — 当前会话热窗口，优先异步 Redis，失败回退进程内。"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from myfitness.config import get_settings
from myfitness.memory.compress import format_messages
from myfitness.memory.redis_client import redis_configured, redis_get, redis_set, run_async
from myfitness.schemas.state import ChatMessage

logger = logging.getLogger(__name__)

_LOCAL_LOCK = threading.Lock()
_LOCAL_STORE: dict[str, str] = {}


@dataclass
class WorkingSnapshot:
    compacted_count: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    backend: str = "memory"


def working_key(user_id: int, session_id: str) -> str:
    return f"mf:wm:{user_id}:{session_id}"


def load_working(user_id: int, session_id: str) -> WorkingSnapshot | None:
    key = working_key(user_id, session_id)
    raw = None
    backend = "memory"
    if redis_configured():
        raw = run_async(redis_get(key))
        if raw:
            backend = "redis"
    if not raw:
        with _LOCAL_LOCK:
            raw = _LOCAL_STORE.get(key)
        backend = "memory"
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("工作记忆 JSON 损坏，忽略 key=%s", key)
        return None
    if not isinstance(payload, dict):
        return None
    messages = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in (payload.get("messages") or [])
        if isinstance(item, dict) and str(item.get("content", "")).strip()
    ]
    try:
        compacted = int(payload.get("compacted_count") or 0)
    except (TypeError, ValueError):
        compacted = 0
    return WorkingSnapshot(compacted_count=max(0, compacted), messages=messages, backend=backend)


def save_working(
    user_id: int,
    session_id: str,
    *,
    compacted_count: int,
    recent: list[ChatMessage],
) -> str:
    """写入工作记忆，返回实际后端 redis | memory。"""
    payload = json.dumps(
        {
            "compacted_count": compacted_count,
            "messages": [{"role": item.role, "content": item.content} for item in recent],
        },
        ensure_ascii=False,
    )
    key = working_key(user_id, session_id)
    ttl = get_settings().memory_working_ttl_seconds
    backend = "memory"
    if redis_configured():
        ok = run_async(redis_set(key, payload, ttl=ttl))
        if ok:
            backend = "redis"
    with _LOCAL_LOCK:
        _LOCAL_STORE[key] = payload
    return backend


def format_working_text(recent: list[ChatMessage]) -> str:
    if not recent:
        return ""
    return "最近对话：\n" + format_messages(recent, max_chars=3000)


def clear_local_working() -> None:
    with _LOCAL_LOCK:
        _LOCAL_STORE.clear()


