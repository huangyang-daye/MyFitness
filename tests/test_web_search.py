"""联网检索：博查 / 智谱 / HTML 回退，以及对话上下文注入。"""

from datetime import date
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.summary import build_rule_based_summary
from myfitness.agents.tools.web_search import (
    _parse_bing,
    _parse_bocha,
    _parse_duckduckgo,
    _parse_zhipu,
    build_search_query,
    format_web_search_results,
    is_web_search_request,
    needs_web_search,
    search_web,
    web_search,
)
from myfitness.config import Settings
from myfitness.db.models import Base, User
from myfitness.graph.router import classify_intent
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import ContextSnapshot, DateRange, Intent
from myfitness.services.context_with_query import load_context_for_turn


def _settings(**overrides) -> Settings:
    values = {
        "web_search_enabled": True,
        "web_search_provider": "bocha",
        "web_search_api_key": "",
        "bocha_api_key": "sk-bocha",
        "zhipu_api_key": "",
        "web_search_count": 5,
        "web_search_timeout": 10,
        "web_search_freshness": "noLimit",
    }
    values.update(overrides)
    return Settings(**values)


def test_build_search_query_strips_prefix():
    assert build_search_query("搜一下HIIT一周练几次") == "HIIT一周练几次"
    assert build_search_query("帮我搜一下蛋白质推荐摄入量") == "蛋白质推荐摄入量"
    assert build_search_query("蛋白质推荐摄入量") == "蛋白质推荐摄入量"


def test_needs_web_search_explicit_and_knowledge():
    with patch("myfitness.agents.tools.web_search.get_settings", return_value=_settings()):
        assert needs_web_search("搜一下HIIT怎么练", Intent.GENERAL)
        assert needs_web_search("蛋白质推荐摄入量有什么科学依据", Intent.GENERAL)
        assert needs_web_search("HIIT一周练几次", Intent.WEB_SEARCH)
        assert not needs_web_search("你好", Intent.GENERAL)
        assert not needs_web_search("昨天吃了多少蛋白质", Intent.DATA_QUERY)
        assert not needs_web_search("同步今日数据", Intent.SYNC_TRIGGER)
        assert needs_web_search("我昨天蛋白质对照推荐量够不够", Intent.DATA_QUERY)


def test_is_web_search_request_does_not_steal_personal_queries():
    assert is_web_search_request("搜一下今天HIIT怎么练")
    assert is_web_search_request("蛋白质推荐摄入量有什么科学依据")
    assert not is_web_search_request("昨天吃了多少蛋白质")
    assert not is_web_search_request("查询今天热量")


def test_router_classifies_web_search():
    result = classify_intent("搜一下HIIT一周练几次比较好", use_llm=False)
    assert result.intent == Intent.WEB_SEARCH
    assert result.domain == "fitness"

    knowledge = classify_intent("蛋白质推荐摄入量有什么科学依据", use_llm=False)
    assert knowledge.intent == Intent.WEB_SEARCH
    assert knowledge.domain == "nutrition"

    personal = classify_intent("昨天吃了多少蛋白质", use_llm=False)
    assert personal.intent == Intent.DATA_QUERY


def test_web_search_intent_skips_database_query():
    from myfitness.agents.tools.query_planner import needs_database_query

    assert not needs_database_query(Intent.WEB_SEARCH, "蛋白质推荐摄入量有什么科学依据")


def test_parse_bocha_and_zhipu_payloads():
    bocha = _parse_bocha(
        {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "HIIT 训练指南",
                            "url": "https://example.com/hiit",
                            "summary": "每周 2 到 3 次",
                            "siteName": "丁香医生",
                            "datePublished": "2026-01-01",
                        }
                    ]
                }
            }
        }
    )
    assert bocha[0]["title"] == "HIIT 训练指南"
    assert bocha[0]["site"] == "丁香医生"

    zhipu = _parse_zhipu(
        {
            "search_result": [
                {
                    "title": "蛋白质摄入",
                    "link": "https://example.com/protein",
                    "content": "每公斤体重 1.6g",
                    "media": "中国营养学会",
                    "publish_date": "2025-06-01",
                }
            ]
        }
    )
    assert zhipu[0]["url"] == "https://example.com/protein"
    assert "1.6g" in zhipu[0]["snippet"]


def test_parse_html_fallback_results():
    ddg = """
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fcn.example.com%2Fhiit">HIIT 频率</a>
    <a class="result__snippet">每周两到三次高强度间歇</a>
    """
    hits = _parse_duckduckgo(ddg, count=5)
    assert hits[0]["url"] == "https://cn.example.com/hiit"
    assert "高强度" in hits[0]["snippet"]

    bing = """
    <li class="b_algo">
      <h2><a href="https://www.example.cn/protein">蛋白质推荐</a></h2>
      <p>中国居民膳食指南建议</p>
    </li>
    """
    bing_hits = _parse_bing(bing, count=5)
    assert bing_hits[0]["title"] == "蛋白质推荐"
    assert "膳食指南" in bing_hits[0]["snippet"]


def test_search_web_uses_bocha_when_key_present():
    payload = {
        "data": {
            "webPages": {
                "value": [
                    {
                        "name": "标题",
                        "url": "https://example.com/a",
                        "snippet": "摘要",
                        "siteName": "站点",
                    }
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            assert "bocha" in url
            assert headers["Authorization"] == "Bearer sk-bocha"
            assert json["query"] == "HIIT"
            return FakeResponse()

    with (
        patch("myfitness.agents.tools.web_search.get_settings", return_value=_settings()),
        patch("myfitness.agents.tools.web_search.httpx.Client", FakeClient),
    ):
        result = search_web("HIIT")
    assert result["provider"] == "bocha"
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com/a"


def test_search_web_disabled_returns_empty():
    with patch(
        "myfitness.agents.tools.web_search.get_settings",
        return_value=_settings(web_search_enabled=False),
    ):
        result = search_web("HIIT")
    assert result["results"] == []
    assert result["error"] == "disabled"


def test_web_search_tool_is_registered():
    assert web_search.name == "web_search"
    assert "query" in web_search.args
    assert "session" not in web_search.args


def test_format_and_summary_include_citations():
    results = [
        {
            "title": "HIIT 指南",
            "url": "https://example.com/hiit",
            "snippet": "每周 2-3 次",
            "site": "丁香医生",
            "published": "2026-01-01",
        }
    ]
    formatted = format_web_search_results(results)
    assert "[1] HIIT 指南" in formatted
    assert "https://example.com/hiit" in formatted

    context = ContextSnapshot(
        date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 7)),
        web_search_results=results,
    )
    text = build_rule_based_summary(AgentOutputs(), context, Intent.WEB_SEARCH)
    assert "HIIT 指南" in text
    assert "丁香医生" in text


def test_load_context_injects_web_search_results():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()
    try:
        hits = [
            {
                "title": "HIIT 指南",
                "url": "https://example.com/hiit",
                "snippet": "每周 2-3 次",
                "site": "丁香医生",
                "published": "",
            }
        ]
        with (
            patch("myfitness.services.context_with_query.retrieve_for_turn", return_value=[]),
            patch(
                "myfitness.services.context_with_query.needs_web_search",
                return_value=True,
            ),
            patch(
                "myfitness.services.context_with_query.search_web",
                return_value={"tool": "web_search", "results": hits},
            ) as search,
        ):
            context, tools = load_context_for_turn(
                session, 1, "搜一下HIIT一周练几次", Intent.WEB_SEARCH
            )
        search.assert_called_once()
        assert "web_search" in tools
        assert context.web_search_results[0]["title"] == "HIIT 指南"
    finally:
        session.close()


def test_load_context_skips_search_when_not_needed():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()
    try:
        with (
            patch("myfitness.services.context_with_query.retrieve_for_turn", return_value=[]),
            patch(
                "myfitness.services.context_with_query.needs_web_search",
                return_value=False,
            ),
            patch("myfitness.services.context_with_query.search_web") as search,
        ):
            context, tools = load_context_for_turn(session, 1, "你好", Intent.GENERAL)
        search.assert_not_called()
        assert "web_search" not in tools
        assert context.web_search_results == []
    finally:
        session.close()
