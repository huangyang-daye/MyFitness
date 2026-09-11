"""Redis 持久化的 LangGraph Checkpointer（普通 Redis，无需 Redis Stack）。

在 MemorySaver 语义上按 thread 切片落盘，支持进程重启后的断点恢复。
"""

from __future__ import annotations

import logging
import pickle
import threading
from collections.abc import Iterator
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_KEY_PREFIX = "mf:lg:ckpt:"


class RedisCheckpointSaver(MemorySaver):
    """MemorySaver + Redis 持久化：get 时 hydrate，put/delete 时 sync。"""

    def __init__(
        self,
        redis_url: str,
        *,
        ttl_seconds: int = 604800,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._redis_url = redis_url
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._client = None
        self._hydrate_lock = threading.Lock()
        self._hydrated: set[str] = set()

    def _redis(self):
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
        return self._client

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"{_KEY_PREFIX}{thread_id}"

    def _hydrate_thread(self, thread_id: str) -> None:
        with self._hydrate_lock:
            if thread_id in self._hydrated:
                return
            try:
                raw = self._redis().get(self._key(thread_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis checkpointer hydrate 失败 thread=%s: %s", thread_id, exc)
                self._hydrated.add(thread_id)
                return
            if raw:
                try:
                    payload = pickle.loads(raw)
                    storage = payload.get("storage") or {}
                    writes = payload.get("writes") or {}
                    blobs = payload.get("blobs") or {}
                    if storage:
                        self.storage[thread_id] = storage
                    for key, value in writes.items():
                        self.writes[key] = value
                    for key, value in blobs.items():
                        self.blobs[key] = value
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Redis checkpointer 反序列化失败 thread=%s: %s", thread_id, exc)
            self._hydrated.add(thread_id)

    def _persist_thread(self, thread_id: str) -> None:
        storage_slice = self.storage.get(thread_id)
        writes_slice = {k: v for k, v in self.writes.items() if k[0] == thread_id}
        blobs_slice = {k: v for k, v in self.blobs.items() if k[0] == thread_id}
        key = self._key(thread_id)
        try:
            client = self._redis()
            if not storage_slice and not writes_slice and not blobs_slice:
                client.delete(key)
                return
            payload = pickle.dumps(
                {
                    "storage": storage_slice,
                    "writes": writes_slice,
                    "blobs": blobs_slice,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            client.set(key, payload, ex=self._ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis checkpointer persist 失败 thread=%s: %s", thread_id, exc)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        self._hydrate_thread(thread_id)
        return super().get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is not None:
            thread_id = config["configurable"]["thread_id"]
            self._hydrate_thread(thread_id)
        yield from super().list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        self._hydrate_thread(thread_id)
        result = super().put(config, checkpoint, metadata, new_versions)
        self._persist_thread(thread_id)
        return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        self._hydrate_thread(thread_id)
        super().put_writes(config, writes, task_id, task_path)
        self._persist_thread(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self._hydrate_thread(thread_id)
        super().delete_thread(thread_id)
        self._hydrated.discard(thread_id)
        try:
            self._redis().delete(self._key(thread_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis checkpointer delete 失败 thread=%s: %s", thread_id, exc)


def build_redis_checkpointer(redis_url: str, *, ttl_seconds: int = 604800) -> RedisCheckpointSaver:
    """探测 Redis 可用后构造 checkpointer；失败则抛出异常由调用方回退。"""
    saver = RedisCheckpointSaver(redis_url, ttl_seconds=ttl_seconds)
    saver._redis()  # 立即 ping，避免静默落到半残状态
    return saver
