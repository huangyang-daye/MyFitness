"""LLM 调用限制与熔断 — 限频、连续失败熔断、半开恢复。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class LlmCircuitOpenError(RuntimeError):
    """熔断打开期间拒绝调用。"""


@dataclass
class LlmCallStats:
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    throttled_calls: int = 0
    circuit_opens: int = 0
    last_error: str = ""
    last_call_at: float = 0.0


@dataclass
class LlmGuard:
    """单例守卫：最小间隔限频 + 连续失败熔断 + 半开恢复。

    - min_interval_seconds：两次调用最小间隔（限频）
    - failure_threshold：连续失败 N 次后熔断
    - cooldown_seconds：熔断冷却时间，之后进入半开（放行一次探测）
    """

    min_interval_seconds: float = 1.0
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0

    stats: LlmCallStats = field(default_factory=LlmCallStats)
    _last_call: float = field(default=0.0, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)
    _circuit_opened_at: float = field(default=0.0, repr=False)
    _half_open_probe_inflight: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_circuit_open(self) -> bool:
        if self._circuit_opened_at == 0.0:
            return False
        if time.monotonic() - self._circuit_opened_at >= self.cooldown_seconds:
            return False  # 冷却结束 → 半开
        return True

    @property
    def state(self) -> str:
        if not self.is_circuit_open:
            if self._circuit_opened_at != 0.0:
                return "half_open"
            return "closed"
        return "open"

    def acquire(self) -> None:
        """获取调用许可；限频等待 + 熔断检查。"""
        with self._lock:
            if self.is_circuit_open:
                self.stats.throttled_calls += 1
                remaining = self.cooldown_seconds - (time.monotonic() - self._circuit_opened_at)
                raise LlmCircuitOpenError(f"LLM 熔断中，{remaining:.0f}s 后重试")

            wait = self.min_interval_seconds - (time.monotonic() - self._last_call)
            if wait > 0 and wait <= self.min_interval_seconds:
                time.sleep(wait)

            self._last_call = time.monotonic()
            self.stats.total_calls += 1

    def record_success(self) -> None:
        with self._lock:
            self.stats.success_calls += 1
            self._consecutive_failures = 0
            if self._circuit_opened_at != 0.0:
                logger.info("LLM 探测成功，熔断恢复")
                self._circuit_opened_at = 0.0
            self.stats.last_call_at = time.time()

    def record_failure(self, error: str) -> None:
        with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_error = error[:200]
            self.stats.last_call_at = time.time()
            self._consecutive_failures += 1

            if self._consecutive_failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._circuit_opened_at = time.monotonic()
        self.stats.circuit_opens += 1
        logger.warning(
            "LLM 连续失败 %d 次，熔断 %.0fs", self._consecutive_failures, self.cooldown_seconds
        )

    def snapshot(self) -> dict:
        with self._lock:
            s = self.stats
            return {
                "state": self.state,
                "total_calls": s.total_calls,
                "success_calls": s.success_calls,
                "failed_calls": s.failed_calls,
                "throttled_calls": s.throttled_calls,
                "circuit_opens": s.circuit_opens,
                "consecutive_failures": self._consecutive_failures,
                "last_error": s.last_error,
            }


_guard = LlmGuard()


def get_llm_guard() -> LlmGuard:
    return _guard


def reset_llm_guard(
    min_interval_seconds: float | None = None,
    failure_threshold: int | None = None,
    cooldown_seconds: float | None = None,
) -> LlmGuard:
    """测试用：重建守卫。"""
    global _guard
    kwargs: dict = {}
    if min_interval_seconds is not None:
        kwargs["min_interval_seconds"] = min_interval_seconds
    if failure_threshold is not None:
        kwargs["failure_threshold"] = failure_threshold
    if cooldown_seconds is not None:
        kwargs["cooldown_seconds"] = cooldown_seconds
    _guard = LlmGuard(**kwargs)
    return _guard
