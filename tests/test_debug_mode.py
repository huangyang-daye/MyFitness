from __future__ import annotations

import logging
from datetime import date

import pytest
from typer.testing import CliRunner

from myfitness.agents.intent_agent import run_intent_agent
from myfitness.agents.tools.base import invoke_tool
from myfitness.api.cli import app
from myfitness.config import get_settings
from myfitness.debug import safe_preview, set_debug_mode, trace_agent
from myfitness.graph.router import classify_intent


@pytest.fixture(autouse=True)
def reset_debug_override():
    set_debug_mode(False)
    yield
    set_debug_mode(None)


def test_debug_mode_logs_agent_call_and_result(caplog, monkeypatch):
    monkeypatch.setattr("myfitness.agents.intent_agent.is_llm_configured", lambda: False)
    set_debug_mode(True)
    with caplog.at_level(logging.DEBUG, logger="myfitness.debug"):
        result = run_intent_agent("你好", today=date(2026, 8, 30))

    assert result is None
    assert "Agent call | name=IntentAgent" in caplog.text
    assert "Agent result | name=IntentAgent" in caplog.text


def test_debug_mode_logs_tool_call_result_and_redacts_secrets(caplog):
    class FakeTool:
        name = "fake_tool"

        def invoke(self, payload, config):
            assert config["configurable"]["user_id"] == 7
            return {"ok": True, "access_token": "result-secret"}

    set_debug_mode(True)
    with caplog.at_level(logging.DEBUG, logger="myfitness.debug"):
        result = invoke_tool(
            FakeTool(),
            object(),
            7,
            api_key="input-secret",
            amount=2,
        )

    assert result["ok"] is True
    assert "Tool call | name=fake_tool" in caplog.text
    assert "Tool result | name=fake_tool" in caplog.text
    assert "input-secret" not in caplog.text
    assert "result-secret" not in caplog.text
    assert "<redacted>" in caplog.text


def test_debug_mode_logs_final_intent_result(caplog):
    set_debug_mode(True)
    with caplog.at_level(logging.DEBUG, logger="myfitness.debug"):
        result = classify_intent(
            "查询今天体重",
            use_llm=False,
            today=date(2026, 8, 30),
        )

    assert result.intent.value == "data_query"
    assert "Intent result | source=keyword" in caplog.text
    assert "data_query" in caplog.text
    assert "domain=body" in caplog.text


def test_debug_logs_are_disabled_by_default(caplog):
    @trace_agent("SilentAgent")
    def run():
        return "ok"

    set_debug_mode(False)
    with caplog.at_level(logging.DEBUG, logger="myfitness.debug"):
        assert run() == "ok"
    assert "SilentAgent" not in caplog.text


def test_safe_preview_redacts_nested_credentials():
    preview = safe_preview(
        {"config": {"api_key": "sk-secret", "Authorization": "Bearer abc"}, "value": 1}
    )
    assert "sk-secret" not in preview
    assert "Bearer abc" not in preview
    assert preview.count("<redacted>") == 2


def test_debug_mode_can_be_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("DEBUG_MODE", "true")
    get_settings.cache_clear()
    set_debug_mode(None)
    try:
        assert get_settings().debug_mode is True
    finally:
        get_settings.cache_clear()


def test_cli_global_debug_flag_enables_tracing(monkeypatch):
    configured = []
    monkeypatch.setattr("myfitness.api.cli.configure_debug_logging", configured.append)

    result = CliRunner().invoke(app, ["--debug", "llm", "providers"])

    assert result.exit_code == 0, result.output
    assert configured == [True]
