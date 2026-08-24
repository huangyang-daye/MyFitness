"""意图识别 Agent 测试 — 全部 mock LLM，不发起真实调用。"""

from datetime import date
from unittest.mock import patch

from myfitness.agents.intent_agent import (
    build_system_prompt,
    parse_agent_response,
    run_intent_agent,
)
from myfitness.schemas.state import Intent

TODAY = date(2026, 8, 24)


def _llm_patch(payload: str):
    return patch(
        "myfitness.agents.intent_agent.chat_completion", return_value=payload
    )


def _configured(flag: bool = True):
    return patch(
        "myfitness.agents.intent_agent.is_llm_configured", return_value=flag
    )


def test_build_system_prompt_contains_current_date():
    prompt = build_system_prompt(TODAY)
    assert "2026-08-24" in prompt
    assert "8月23日" in prompt  # 动态生成的组合意图示例
    # 意图类别定义齐全
    for intent in Intent:
        assert intent.value in prompt
    # 输出格式与多意图规则说明齐全
    assert '"intents"' in prompt
    assert "多意图" in prompt
    assert "date_range" in prompt


def test_run_intent_agent_multi_intent_with_dates():
    payload = (
        '{"intents": ["sync_trigger", "report_trigger"], "domain": null, '
        '"date_range": {"start": "2026-08-24", "end": "2026-08-24"}, '
        '"reasoning": "先同步再生成日报"}'
    )
    with _configured(), _llm_patch(payload) as llm_mock:
        route = run_intent_agent("同步今日数据并生成日报", today=TODAY)

    llm_mock.assert_called_once()
    assert route is not None
    assert route.intents == [Intent.SYNC_TRIGGER, Intent.REPORT_TRIGGER]
    assert route.start_date == TODAY
    assert route.end_date == TODAY


def test_run_intent_agent_passes_system_prompt_and_user_message():
    payload = '{"intents": ["general"], "domain": null, "date_range": null}'
    with _configured(), _llm_patch(payload) as llm_mock:
        run_intent_agent("你好", today=TODAY)

    messages = llm_mock.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert "意图识别 Agent" in messages[0]["content"]
    assert "2026-08-24" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "你好"}
    # 意图识别使用低温度以保证稳定输出
    assert llm_mock.call_args.kwargs.get("temperature") == 0


def test_run_intent_agent_not_configured_returns_none():
    with _configured(False):
        assert run_intent_agent("同步数据", today=TODAY) is None


def test_run_intent_agent_llm_failure_returns_none():
    with _configured(), patch(
        "myfitness.agents.intent_agent.chat_completion",
        side_effect=RuntimeError("LLM 不可用"),
    ):
        assert run_intent_agent("同步数据", today=TODAY) is None


def test_parse_code_fenced_json():
    payload = (
        "```json\n"
        '{"intents": ["sync_trigger"], "domain": null, '
        '"date_range": {"start": "2026-08-24", "end": "2026-08-24"}}\n'
        "```"
    )
    route = parse_agent_response(payload, today=TODAY)
    assert route is not None
    assert route.intents == [Intent.SYNC_TRIGGER]
    assert route.start_date == TODAY


def test_parse_singular_intent_key():
    payload = '{"intent": "data_query", "domain": "nutrition", "date_range": null}'
    route = parse_agent_response(payload, today=TODAY)
    assert route is not None
    assert route.intents == [Intent.DATA_QUERY]
    assert route.domain == "nutrition"


def test_parse_invalid_json_returns_none():
    assert parse_agent_response("抱歉，我无法理解。", today=TODAY) is None
    assert parse_agent_response("", today=TODAY) is None
    assert parse_agent_response("[1, 2, 3]", today=TODAY) is None


def test_parse_unknown_intent_filtered():
    # 未知意图被过滤，仅保留合法意图
    payload = '{"intents": ["sync_trigger", "delete_everything"], "domain": null, "date_range": null}'
    route = parse_agent_response(payload, today=TODAY)
    assert route is not None
    assert route.intents == [Intent.SYNC_TRIGGER]


def test_parse_all_intents_unknown_returns_none():
    payload = '{"intents": ["unknown_a", "unknown_b"], "domain": null, "date_range": null}'
    assert parse_agent_response(payload, today=TODAY) is None


def test_parse_confirmation_response_filtered():
    # 确认/取消由 Router 结合 pending_confirmation 上下文处理，LLM 结果中应被剔除
    payload = '{"intents": ["confirmation_response"], "domain": null, "date_range": null}'
    assert parse_agent_response(payload, today=TODAY) is None


def test_parse_future_date_range_dropped():
    payload = (
        '{"intents": ["sync_trigger"], "domain": null, '
        '"date_range": {"start": "2026-12-01", "end": "2026-12-31"}}'
    )
    route = parse_agent_response(payload, today=TODAY)
    assert route is not None
    assert route.intents == [Intent.SYNC_TRIGGER]
    assert route.start_date is None
    assert route.end_date is None


def test_parse_reversed_range_swapped():
    payload = (
        '{"intents": ["sync_trigger"], "domain": null, '
        '"date_range": {"start": "2026-08-23", "end": "2026-08-20"}}'
    )
    route = parse_agent_response(payload, today=TODAY)
    assert route is not None
    assert route.start_date == date(2026, 8, 20)
    assert route.end_date == date(2026, 8, 23)


def test_parse_domain_aliases_and_invalid():
    route = parse_agent_response(
        '{"intents": ["data_query"], "domain": "training", "date_range": null}',
        today=TODAY,
    )
    assert route is not None
    assert route.domain == "fitness"

    route = parse_agent_response(
        '{"intents": ["data_query"], "domain": "kitchen", "date_range": null}',
        today=TODAY,
    )
    assert route is not None
    assert route.domain is None


def test_parse_duplicate_intents_deduped():
    payload = '{"intents": ["sync_trigger", "sync_trigger"], "domain": null, "date_range": null}'
    route = parse_agent_response(payload, today=TODAY)
    assert route is not None
    assert route.intents == [Intent.SYNC_TRIGGER]
