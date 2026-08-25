import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent, RouteResult

FIXTURES = Path(__file__).parent / "fixtures" / "intent_samples.json"


def test_router_intent_samples_accuracy():
    samples = json.loads(FIXTURES.read_text(encoding="utf-8"))
    correct = 0
    for sample in samples:
        result = classify_intent(sample["text"], use_llm=False)
        if result.intent.value == sample["intent"]:
            correct += 1
    accuracy = correct / len(samples)
    assert accuracy >= 0.9, f"router accuracy {accuracy:.0%} < 90%"


def test_router_manual_entry_domains():
    body = classify_intent("记录体重 71kg", use_llm=False)
    assert body.intent == Intent.MANUAL_ENTRY
    assert body.domain == "body"

    food = classify_intent("午餐吃了鸡胸肉200g", use_llm=False)
    assert food.intent == Intent.MANUAL_ENTRY
    assert food.domain == "nutrition"


def test_router_sync_trigger():
    result = classify_intent("帮我同步训记数据", use_llm=False)
    assert result.intent == Intent.SYNC_TRIGGER


# --- 同步日期范围解析（修复「同步今日数据」却同步最近 7 天的问题） ---


def test_router_sync_today_only():
    today = date(2026, 8, 24)
    result = classify_intent("同步今日数据", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date == today
    assert result.end_date == today


def test_router_sync_today_with_app_name():
    today = date(2026, 8, 24)
    result = classify_intent("同步今天的训记数据", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date == today
    assert result.end_date == today


def test_router_sync_yesterday_only():
    today = date(2026, 8, 24)
    result = classify_intent("同步昨天的数据", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date == date(2026, 8, 23)
    assert result.end_date == date(2026, 8, 23)


def test_router_sync_recent_days_includes_today():
    today = date(2026, 8, 24)
    result = classify_intent("同步最近3天数据", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date == date(2026, 8, 22)
    assert result.end_date == today


def test_router_sync_specific_date():
    today = date(2026, 8, 24)
    result = classify_intent("同步8月24日数据", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date == date(2026, 8, 24)
    assert result.end_date == date(2026, 8, 24)


def test_router_sync_without_date_has_no_range():
    result = classify_intent("帮我同步训记数据", use_llm=False)
    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date is None
    assert result.end_date is None


# --- 多意图：同步 + 日报 ---


def test_router_sync_and_report_multi_intent():
    today = date(2026, 8, 24)
    result = classify_intent("同步8月24日数据并生成日报", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER, Intent.REPORT_TRIGGER]
    assert result.start_date == date(2026, 8, 24)
    assert result.end_date == date(2026, 8, 24)


def test_router_sync_and_report_today():
    today = date(2026, 8, 24)
    result = classify_intent("同步今日数据然后生成日报", use_llm=False, today=today)
    assert result.intents == [Intent.SYNC_TRIGGER, Intent.REPORT_TRIGGER]
    assert result.start_date == today
    assert result.end_date == today


def test_router_report_single_intent_not_multi():
    result = classify_intent("生成昨天日报", use_llm=False, today=date(2026, 8, 24))
    assert result.intents == [Intent.REPORT_TRIGGER]
    assert result.start_date == date(2026, 8, 23)


def test_router_report_dot_date():
    result = classify_intent("生成8.21的报告", use_llm=False, today=date(2026, 8, 24))
    assert result.intents == [Intent.REPORT_TRIGGER]
    assert result.start_date == date(2026, 8, 21)


def test_router_report_without_date_has_no_range():
    result = classify_intent("生成日报", use_llm=False, today=date(2026, 8, 24))
    assert result.intents == [Intent.REPORT_TRIGGER]
    assert result.start_date is None
    assert result.end_date is None


def test_router_schedule_takes_priority_over_daily_report():
    result = classify_intent("每天早上7点生成日报", use_llm=False)
    assert result.intents == [Intent.SCHEDULE_MANAGE]


# --- LLM 优先、关键词兜底的编排 ---


def test_classify_llm_first_when_available():
    """use_llm=True 时优先调用意图 Agent，且采用其结果。"""
    llm_route = RouteResult(
        intents=[Intent.SYNC_TRIGGER, Intent.REPORT_TRIGGER],
        start_date=date(2026, 8, 24),
        end_date=date(2026, 8, 24),
    )
    with patch(
        "myfitness.agents.intent_agent.run_intent_agent", return_value=llm_route
    ) as llm_mock:
        result = classify_intent("同步8月24日数据并生成日报", use_llm=True)

    llm_mock.assert_called_once()
    assert result.intents == [Intent.SYNC_TRIGGER, Intent.REPORT_TRIGGER]


def test_classify_falls_back_to_keyword_when_llm_fails():
    """意图 Agent 失败（返回 None）时回退关键词匹配。"""
    with patch(
        "myfitness.agents.intent_agent.run_intent_agent", return_value=None
    ) as llm_mock:
        result = classify_intent("帮我同步训记数据", use_llm=True)

    llm_mock.assert_called_once()
    assert result.intents == [Intent.SYNC_TRIGGER]


def test_classify_llm_general_falls_back_to_keyword():
    """LLM 误判为 general 而关键词有明确命中时，采用关键词结果。"""
    with patch(
        "myfitness.agents.intent_agent.run_intent_agent",
        return_value=RouteResult(intents=[Intent.GENERAL]),
    ):
        result = classify_intent("帮我同步训记数据", use_llm=True)

    assert result.intents == [Intent.SYNC_TRIGGER]


def test_classify_llm_missing_dates_filled_by_keyword():
    """LLM 识别出同步意图但未提取日期时，用关键词解析的日期补齐。"""
    llm_route = RouteResult(intents=[Intent.SYNC_TRIGGER])  # 无日期
    today = date(2026, 8, 24)
    with patch(
        "myfitness.agents.intent_agent.run_intent_agent", return_value=llm_route
    ):
        result = classify_intent("同步今日数据", use_llm=True, today=today)

    assert result.intents == [Intent.SYNC_TRIGGER]
    assert result.start_date == today
    assert result.end_date == today
