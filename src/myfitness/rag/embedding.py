"""Embedding API — OpenAI 兼容 /embeddings。"""

from __future__ import annotations

import logging
from functools import lru_cache
from urllib.parse import urlparse

import httpx

from myfitness.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 官方聊天接口没有 /embeddings，不能把 LLM_BASE_URL 当 embedding 端点。
_NO_EMBEDDING_HOSTS = frozenset({
    "api.deepseek.com",
    "api.deepseek.ai",
    "api.anthropic.com",
})

_UNSUPPORTED_HOST_HINT = (
    "当前接口不提供 Embedding（DeepSeek / Claude 等聊天模型没有 /embeddings）。"
    "请在 .env 设置 EMBEDDING_BASE_URL、EMBEDDING_API_KEY、EMBEDDING_MODEL，"
    "指向 OpenAI 兼容的 embedding 服务（如 OpenAI、硅基流动、智谱）。"
)


class EmbeddingError(RuntimeError):
    pass


@lru_cache
def _cached_settings_key() -> tuple[str, str, str, int]:
    s = get_settings()
    return (
        s.resolved_embedding_base_url(),
        s.resolved_embedding_api_key(),
        s.embedding_model,
        s.embedding_dimensions,
    )


def embedding_host_supported(base_url: str) -> bool:
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    return bool(host) and host not in _NO_EMBEDDING_HOSTS


@lru_cache
def _warn_unsupported_embedding_host(host: str) -> None:
    logger.warning("%s（检测到 %s）", _UNSUPPORTED_HOST_HINT, host)


def is_embedding_configured(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    if not (s.resolved_embedding_api_key() and s.embedding_model):
        return False
    base = s.resolved_embedding_base_url()
    if embedding_host_supported(base):
        return True
    parsed = urlparse(base if "://" in base else f"https://{base}")
    host = (parsed.hostname or base).lower()
    _warn_unsupported_embedding_host(host)
    return False


def embeddings_url(settings: Settings | None = None) -> str:
    s = settings or get_settings()
    base = s.resolved_embedding_base_url().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """批量获取文本 embedding。"""
    if not texts:
        return []
    s = settings or get_settings()
    api_key = s.resolved_embedding_api_key()
    if not api_key:
        raise EmbeddingError("Embedding 未配置：请设置 EMBEDDING_API_KEY 或 LLM_API_KEY")
    if not embedding_host_supported(s.resolved_embedding_base_url()):
        raise EmbeddingError(_UNSUPPORTED_HOST_HINT)

    payload: dict = {
        "model": s.embedding_model,
        "input": texts,
    }
    if s.embedding_dimensions:
        payload["dimensions"] = s.embedding_dimensions

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=float(s.llm_timeout)) as client:
            response = client.post(embeddings_url(s), headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001 - surface as EmbeddingError
        raise EmbeddingError(f"Embedding API 调用失败: {exc}") from exc

    rows = data.get("data") or []
    if len(rows) != len(texts):
        raise EmbeddingError(f"Embedding 返回数量不匹配：期望 {len(texts)}，实际 {len(rows)}")

    rows.sort(key=lambda item: item.get("index", 0))
    expected = s.embedding_dimensions
    vectors: list[list[float]] = []
    for row in rows:
        vector = row.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("Embedding 返回格式无效")
        values = [float(v) for v in vector]
        if len(values) != expected:
            raise EmbeddingError(
                f"Embedding 维度不匹配：模型返回 {len(values)} 维，"
                f"但 EMBEDDING_DIMENSIONS={expected}。"
                "请修正 .env 中的 EMBEDDING_DIMENSIONS 后运行 myfitness rag init。"
            )
        vectors.append(values)
    return vectors


def embed_text(text: str, settings: Settings | None = None) -> list[float]:
    return embed_texts([text], settings=settings)[0]
