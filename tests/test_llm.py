import pytest

from myfitness.config import Settings
from myfitness.llm.factory import (
    get_llm_config,
    is_llm_configured,
    probe_llm_connection,
)


def test_is_llm_configured():
    assert is_llm_configured(Settings(llm_api_key="sk-test", llm_model="gpt-4o"))
    assert not is_llm_configured(Settings(llm_api_key="", llm_model="gpt-4o"))
    assert is_llm_configured(Settings(openai_api_key="sk-legacy", llm_model="gpt-4o"))


def test_get_llm_config_openai_compat():
    cfg = get_llm_config(
        Settings(
            llm_base_url="https://api.example.com/v1/",
            llm_api_key="sk-abc123456789",
            llm_model="deepseek-chat",
            llm_temperature=0.5,
        )
    )
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "deepseek-chat"
    assert cfg.chat_completions_url == "https://api.example.com/v1/chat/completions"
    assert cfg.masked_api_key() == "****6789"


def test_probe_llm_connection(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 10},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers, json):
            assert url.endswith("/chat/completions")
            assert json["model"] == "test-model"
            return FakeResponse()

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)

    result = probe_llm_connection(
        settings=Settings(llm_api_key="sk-test", llm_model="test-model")
    )
    assert result["reply"] == "OK"
    assert result["usage"]["total_tokens"] == 10
