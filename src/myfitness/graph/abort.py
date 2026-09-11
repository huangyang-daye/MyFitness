"""协作式取消：客户端断开 / 显式 abort 时停止上游 LLM 订阅。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_AbortCheck = Callable[[], bool]
_abort_check: ContextVar[_AbortCheck | None] = ContextVar("mf_abort_check", default=None)


def is_aborted() -> bool:
    check = _abort_check.get()
    return bool(check and check())


@contextmanager
def abort_scope(check: _AbortCheck | None) -> Iterator[None]:
    """在当前任务上下文绑定取消检查（同线程内 LLM 流会读取）。"""
    token = _abort_check.set(check)
    try:
        yield
    finally:
        _abort_check.reset(token)
