"""LLM 启动预热测试。"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from myfitness.config import Settings
from myfitness.llm.factory import LlmWarmupResult, warmup_llm


def test_warmup_not_configured():
    result = warmup_llm(Settings(llm_api_key="", llm_model="gpt-4o"))
    assert result.configured is False
    assert result.loaded is True
    assert result.ready_for_input is True


def test_warmup_loads_llm_without_probe(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr("myfitness.llm.factory.get_llm", lambda: mock_llm)

    result = warmup_llm(
        Settings(llm_api_key="sk-test", llm_model="test-model"),
        probe=False,
    )
    assert result.configured is True
    assert result.loaded is True
    assert result.model == "test-model"
    assert result.connected is None
    assert result.ready_for_input is True


def test_warmup_probe_success(monkeypatch):
    monkeypatch.setattr("myfitness.llm.factory.get_llm", lambda: MagicMock())

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers, json):
            assert json["max_tokens"] == 5
            return FakeResponse()

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)

    result = warmup_llm(
        Settings(llm_api_key="sk-test", llm_model="test-model"),
        probe=True,
    )
    assert result.loaded is True
    assert result.connected is True


def test_warmup_load_failure():
    with patch("myfitness.llm.factory.get_llm", side_effect=ImportError("no langchain")):
        result = warmup_llm(
            Settings(llm_api_key="sk-test", llm_model="test-model"),
            probe=False,
        )
    assert result.loaded is False
    assert result.ready_for_input is False
    assert "langchain" in (result.error or "")


def test_warmup_probe_failure_still_loaded(monkeypatch):
    monkeypatch.setattr("myfitness.llm.factory.get_llm", lambda: MagicMock())

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://x/chat/completions")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("503", request=request, response=response)

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)

    result = warmup_llm(
        Settings(llm_api_key="sk-test", llm_model="test-model"),
        probe=True,
    )
    assert result.loaded is True
    assert result.connected is False
    assert result.ready_for_input is True
