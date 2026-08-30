"""MyFitness debug tracing for Agent, Tool, and routing calls.

Debug tracing is opt-in through ``DEBUG_MODE=true`` or the CLI's global
``--debug`` flag.  Values are bounded and credential-like fields are redacted
before they reach logs.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("myfitness.debug")

P = ParamSpec("P")
R = TypeVar("R")

_debug_override: bool | None = None
_MAX_DEPTH = 4
_MAX_ITEMS = 30
_MAX_STRING = 800
_MAX_PREVIEW = 4_000
_SECRET_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def set_debug_mode(enabled: bool | None) -> None:
    """Override debug mode for the current process; ``None`` restores settings."""
    global _debug_override
    _debug_override = enabled


def debug_enabled() -> bool:
    if _debug_override is not None:
        return _debug_override
    from myfitness.config import get_settings

    return bool(get_settings().debug_mode)


def configure_debug_logging(enabled: bool) -> None:
    """Enable MyFitness DEBUG records without making third-party loggers noisy."""
    set_debug_mode(enabled)
    logging.getLogger("myfitness").setLevel(logging.DEBUG if enabled else logging.NOTSET)


def trace_agent(name: str | None = None):
    """Trace every call and result of an Agent entry point."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        agent_name = name or func.__name__
        if inspect.isgeneratorfunction(func):

            @wraps(func)
            def generator_wrapper(*args: P.args, **kwargs: P.kwargs):
                if not debug_enabled():
                    yield from func(*args, **kwargs)
                    return
                started = time.perf_counter()
                logger.debug(
                    "Agent call | name=%s | input=%s",
                    agent_name,
                    preview_call(args, kwargs),
                )
                chunks: list[Any] = []
                try:
                    for chunk in func(*args, **kwargs):
                        if len(chunks) < _MAX_ITEMS:
                            chunks.append(chunk)
                        yield chunk
                except Exception:
                    logger.exception(
                        "Agent error | name=%s | elapsed_ms=%d",
                        agent_name,
                        _elapsed_ms(started),
                    )
                    raise
                logger.debug(
                    "Agent result | name=%s | elapsed_ms=%d | output=%s",
                    agent_name,
                    _elapsed_ms(started),
                    safe_preview(chunks),
                )

            return generator_wrapper  # type: ignore[return-value]

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            if not debug_enabled():
                return func(*args, **kwargs)
            started = time.perf_counter()
            logger.debug(
                "Agent call | name=%s | input=%s",
                agent_name,
                preview_call(args, kwargs),
            )
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Agent error | name=%s | elapsed_ms=%d",
                    agent_name,
                    _elapsed_ms(started),
                )
                raise
            logger.debug(
                "Agent result | name=%s | elapsed_ms=%d | output=%s",
                agent_name,
                _elapsed_ms(started),
                safe_preview(result),
            )
            return result

        return wrapper

    return decorator


def trace_tool_call(tool: Any, user_id: int, kwargs: dict[str, Any], invoke: Callable[[], R]) -> R:
    """Invoke one Tool with debug start/result/error logging."""
    if not debug_enabled():
        return invoke()
    name = str(getattr(tool, "name", None) or getattr(tool, "__name__", type(tool).__name__))
    started = time.perf_counter()
    logger.debug(
        "Tool call | name=%s | user_id=%s | input=%s",
        name,
        user_id,
        safe_preview(kwargs),
    )
    try:
        result = invoke()
    except Exception:
        logger.exception(
            "Tool error | name=%s | elapsed_ms=%d",
            name,
            _elapsed_ms(started),
        )
        raise
    logger.debug(
        "Tool result | name=%s | elapsed_ms=%d | output=%s",
        name,
        _elapsed_ms(started),
        safe_preview(result),
    )
    return result


def log_intent_result(message: str, result: Any, *, source: str) -> None:
    """Log the final, reconciled intent routing result."""
    if not debug_enabled():
        return
    logger.debug(
        "Intent result | source=%s | message=%s | intents=%s | domain=%s | "
        "start_date=%s | end_date=%s | confirmation=%s",
        source,
        safe_preview(message),
        [getattr(item, "value", str(item)) for item in result.intents],
        result.domain,
        result.start_date,
        result.end_date,
        result.confirmation_action,
    )


def preview_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"args": list(args)} if args else {}
    if kwargs:
        payload["kwargs"] = kwargs
    return safe_preview(payload)


def safe_preview(value: Any) -> str:
    normalized = _normalize(value, depth=0, key="")
    text = repr(normalized)
    if len(text) > _MAX_PREVIEW:
        return text[:_MAX_PREVIEW] + "…<truncated>"
    return text


def _normalize(value: Any, *, depth: int, key: str) -> Any:
    if _is_secret_key(key):
        return "<redacted>"
    if depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "…"
    if isinstance(value, (date, datetime, Path, Enum)):
        return str(getattr(value, "value", value))
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="json"), depth=depth + 1, key=key)
    if isinstance(value, Mapping):
        items = list(value.items())[:_MAX_ITEMS]
        result = {
            str(item_key): _normalize(item_value, depth=depth + 1, key=str(item_key))
            for item_key, item_value in items
        }
        if len(value) > _MAX_ITEMS:
            result["<truncated>"] = len(value) - _MAX_ITEMS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        normalized = [_normalize(item, depth=depth + 1, key=key) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            normalized.append(f"…<{len(items) - _MAX_ITEMS} more>")
        return normalized
    return repr(value)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_PARTS)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1_000)
