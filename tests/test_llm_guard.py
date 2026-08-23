"""LLM 守卫（限频 + 熔断）测试。"""

import httpx
import pytest

from myfitness.config import Settings
from myfitness.llm.factory import LlmUnavailableError, stream_chat_completion
from myfitness.llm.guard import LlmCircuitOpenError, get_llm_guard, reset_llm_guard


@pytest.fixture(autouse=True)
def fast_guard():
    guard = reset_llm_guard(min_interval_seconds=0.0, failure_threshold=2, cooldown_seconds=60)
    yield guard
    reset_llm_guard()


def test_circuit_opens_after_consecutive_failures():
    guard = get_llm_guard()
    guard.record_failure("err1")
    guard.record_failure("err2")
    assert guard.state == "open"

    with pytest.raises(LlmCircuitOpenError):
        guard.acquire()

    snap = guard.snapshot()
    assert snap["circuit_opens"] == 1
    assert snap["throttled_calls"] == 1


def test_circuit_recovers_on_success():
    guard = get_llm_guard()
    guard.record_failure("err1")
    guard.record_failure("err2")
    assert guard.state == "open"

    guard.record_success()
    assert guard.state == "closed"
    assert guard.snapshot()["consecutive_failures"] == 0


def test_stream_records_failure_and_raises(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)
    monkeypatch.setattr("myfitness.llm.factory._time.sleep", lambda *_: None)

    settings = Settings(llm_api_key="sk-test", llm_model="m")

    with pytest.raises(LlmUnavailableError):
        list(
            stream_chat_completion(
                [{"role": "user", "content": "hi"}],
                settings=settings,
            )
        )

    assert get_llm_guard().snapshot()["failed_calls"] >= 1
