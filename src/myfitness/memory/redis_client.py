"""异步 Redis 客户端 — 独立事件循环线程，供同步对话主链路调用。"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from myfitness.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_redis: Any = None
_redis_url: str | None = None


def redis_configured() -> bool:
    return bool(get_settings().resolved_redis_url())


def run_async(coro, *, timeout: float | None = None):
    """在 Redis 专用事件循环上执行协程；失败或超时返回 None。"""
    settings = get_settings()
    wait = timeout if timeout is not None else settings.memory_redis_timeout
    try:
        loop = _ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=wait)
    except Exception as exc:  # noqa: BLE001 - Redis 失败不得打断主对话
        logger.warning("异步 Redis 调用失败: %s", exc)
        return None


async def redis_get(key: str) -> str | None:
    client = await _get_redis()
    if client is None:
        return None
    value = await client.get(key)
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


async def redis_set(key: str, value: str, *, ttl: int) -> bool:
    client = await _get_redis()
    if client is None:
        return False
    await client.set(key, value, ex=max(60, ttl))
    return True


def close_redis() -> None:
    """测试或进程退出时关闭连接与事件循环。"""
    global _loop, _thread, _redis, _redis_url
    with _lock:
        loop = _loop
        redis_client = _redis
        thread = _thread
        _redis = None
        _redis_url = None
        _loop = None
        _thread = None
    if loop is not None and redis_client is not None:
        try:
            asyncio.run_coroutine_threadsafe(redis_client.aclose(), loop).result(timeout=2)
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭 Redis 连接时忽略: %s", exc)
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=2)


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is not None and _thread is not None and _thread.is_alive():
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_run_loop, args=(loop,), name="mf-redis-loop", daemon=True)
        thread.start()
        _loop = loop
        _thread = thread
        return loop


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()
    loop.close()


async def _get_redis():
    global _redis, _redis_url
    url = get_settings().resolved_redis_url()
    if not url:
        return None
    if _redis is not None and _redis_url == url:
        return _redis
    try:
        from redis.asyncio import Redis
    except ImportError:
        logger.warning("未安装 redis 包，工作记忆回退到进程内存储")
        return None
    client = Redis.from_url(url, decode_responses=False)
    try:
        await asyncio.wait_for(client.ping(), timeout=get_settings().memory_redis_timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用，工作记忆回退到进程内存储: %s", exc)
        try:
            await client.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.debug("丢弃失败的 Redis 连接时忽略: %s", exc)
        return None
    _redis = client
    _redis_url = url
    return client
