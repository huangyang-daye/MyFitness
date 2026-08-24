import json
import logging
import time as _time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from myfitness.config import Settings, get_settings
from myfitness.llm.guard import get_llm_guard

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.5
WARMUP_PROBE_TIMEOUT = 15
WARMUP_PROBE_PROMPT = "OK"


class LlmUnavailableError(RuntimeError):
    """LLM 在重试后仍不可用。"""


@dataclass(frozen=True)
class LlmWarmupResult:
    """chat 启动预热结果。"""

    configured: bool
    loaded: bool
    model: str | None = None
    connected: bool | None = None
    error: str | None = None

    @property
    def ready_for_input(self) -> bool:
        """未配置 LLM 或客户端已加载完成，即可接受用户输入。"""
        return not self.configured or self.loaded


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int | None
    timeout: int

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def masked_api_key(self) -> str:
        if len(self.api_key) <= 4:
            return "****"
        return f"****{self.api_key[-4:]}"


def get_llm_config(settings: Settings | None = None) -> LlmConfig:
    s = settings or get_settings()
    api_key = s.resolved_llm_api_key()
    if not api_key:
        raise ValueError(
            "LLM 未配置：请在 .env 中设置 LLM_API_KEY（或 OPENAI_API_KEY）"
        )
    if not s.llm_model:
        raise ValueError("LLM 未配置：请在 .env 中设置 LLM_MODEL")

    return LlmConfig(
        base_url=s.llm_base_url.rstrip("/"),
        api_key=api_key,
        model=s.llm_model,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout,
    )


def is_llm_configured(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(s.resolved_llm_api_key() and s.llm_model)


@lru_cache
def get_llm():
    """返回 LangChain ChatOpenAI 实例（需安装 agents 可选依赖）。"""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "使用 Agent/LLM 功能需安装：pip install -e \".[agents]\""
        ) from exc

    cfg = get_llm_config()
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "timeout": cfg.timeout,
    }
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens

    logger.info(
        "LLM 已激活: model=%s base_url=%s key=%s",
        cfg.model,
        cfg.base_url,
        cfg.masked_api_key(),
    )
    return ChatOpenAI(**kwargs)


def warmup_llm(
    settings: Settings | None = None,
    *,
    probe: bool = True,
) -> LlmWarmupResult:
    """预加载 LangChain LLM 客户端；可选快速连通性探测。

    chat 启动时调用，避免首条消息才触发 langchain 导入与实例化。
    """
    s = settings or get_settings()
    if not is_llm_configured(s):
        return LlmWarmupResult(configured=False, loaded=True)

    model: str | None = None
    try:
        cfg = get_llm_config(s)
        model = cfg.model
        get_llm()
    except Exception as exc:
        logger.warning("LLM 预热加载失败: %s", exc)
        return LlmWarmupResult(
            configured=True,
            loaded=False,
            model=model,
            error=str(exc),
        )

    connected: bool | None = None
    if probe:
        try:
            _probe_quick(cfg, timeout=min(WARMUP_PROBE_TIMEOUT, cfg.timeout))
            connected = True
        except Exception as exc:
            logger.warning("LLM 预热探测失败: %s", exc)
            connected = False

    return LlmWarmupResult(
        configured=True,
        loaded=True,
        model=model,
        connected=connected,
    )


def _probe_quick(cfg: LlmConfig, timeout: int) -> None:
    """启动预热用的轻量连通性探测（短超时、极少 token）。"""
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": WARMUP_PROBE_PROMPT}],
        "temperature": 0,
        "max_tokens": 5,
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=float(timeout)) as client:
        response = client.post(cfg.chat_completions_url, headers=headers, json=payload)
        response.raise_for_status()


def probe_llm_connection(
    prompt: str = "回复 OK",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """通过 OpenAI 兼容 /chat/completions 探测 LLM 连通性（不依赖 LangChain）。"""
    cfg = get_llm_config(settings)
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
    }
    if cfg.max_tokens is not None:
        payload["max_tokens"] = cfg.max_tokens

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=float(cfg.timeout)) as client:
        response = client.post(cfg.chat_completions_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = ""
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or ""

    usage = data.get("usage") or {}
    return {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "reply": content.strip(),
        "usage": usage,
    }


def chat_completion(
    messages: list[dict[str, str]],
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """非流式 OpenAI 兼容对话补全：限频 → 重试 → 熔断记录。

    供意图识别 Agent 等需要一次性结构化输出的场景使用。
    """
    cfg = get_llm_config(settings)
    guard = get_llm_guard()
    guard.acquire()
    try:
        content = _complete_with_retry(cfg, messages, temperature, max_tokens)
        guard.record_success()
        return content
    except Exception as exc:
        guard.record_failure(str(exc))
        raise


def _complete_with_retry(
    cfg: LlmConfig,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    payload: dict[str, Any] = {"model": cfg.model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    else:
        payload["temperature"] = cfg.temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    elif cfg.max_tokens is not None:
        payload["max_tokens"] = cfg.max_tokens

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=float(cfg.timeout)) as client:
                response = client.post(cfg.chat_completions_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LlmUnavailableError("LLM 返回为空：无 choices")
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            if not content.strip():
                raise LlmUnavailableError("LLM 返回内容为空")
            return content
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retryable = status in RETRYABLE_STATUS
            logger.warning("LLM HTTP %s（attempt %d/%d）", status, attempt + 1, MAX_RETRIES + 1)
            if not retryable or attempt >= MAX_RETRIES:
                raise LlmUnavailableError(f"LLM 服务不可用: HTTP {status}") from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning("LLM 网络异常（attempt %d/%d）: %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt >= MAX_RETRIES:
                raise LlmUnavailableError(f"LLM 连接失败: {exc}") from exc

        _time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise LlmUnavailableError("LLM 调用失败：重试耗尽")


def _parse_stream_line(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if content:
        return content
    message = choice.get("message") or {}
    return message.get("content")


def stream_chat_completion(
    messages: list[dict[str, str]],
    settings: Settings | None = None,
) -> Iterator[str]:
    """OpenAI 兼容 SSE 流式输出：限频 → 重试 → 熔断记录。

    首个 token 已输出后的失败不重试（避免重复输出），直接结束流。
    """
    cfg = get_llm_config(settings)
    guard = get_llm_guard()
    guard.acquire()
    try:
        yield from _stream_with_retry(cfg, messages)
        guard.record_success()
    except Exception as exc:
        guard.record_failure(str(exc))
        raise


def _stream_with_retry(
    cfg: LlmConfig,
    messages: list[dict[str, str]],
) -> Iterator[str]:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "stream": True,
    }
    if cfg.max_tokens is not None:
        payload["max_tokens"] = cfg.max_tokens

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    first_token_emitted = False

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=float(cfg.timeout)) as client:
                with client.stream(
                    "POST",
                    cfg.chat_completions_url,
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        token = _parse_stream_line(line)
                        if token:
                            first_token_emitted = True
                            yield token
                    return
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retryable = status in RETRYABLE_STATUS and not first_token_emitted
            logger.warning("LLM HTTP %s（attempt %d/%d）", status, attempt + 1, MAX_RETRIES + 1)
            if not retryable or attempt >= MAX_RETRIES:
                raise LlmUnavailableError(f"LLM 服务不可用: HTTP {status}") from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if first_token_emitted:
                return  # 已有部分输出，静默结束避免重复
            logger.warning("LLM 网络异常（attempt %d/%d）: %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt >= MAX_RETRIES:
                raise LlmUnavailableError(f"LLM 连接失败: {exc}") from exc

        _time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise LlmUnavailableError("LLM 调用失败：重试耗尽")
