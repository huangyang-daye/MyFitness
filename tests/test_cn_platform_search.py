"""中文平台搜索：点名识别、结果归一、与综合网页检索合并。"""

from unittest.mock import patch

from myfitness.agents.tools.cn_platform_search import (
    detect_platforms,
    is_platform_search_request,
    resolve_platforms,
    search_cn_platforms,
    strip_platform_mentions,
)
from myfitness.agents.tools.web_search import (
    _merge_hits,
    build_search_query,
    is_web_search_request,
    needs_web_search,
    search_web,
)
from myfitness.config import Settings
from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent


def _settings(**overrides) -> Settings:
    values = {
        "web_search_enabled": True,
        "web_search_provider": "bocha",
        "bocha_api_key": "sk-bocha",
        "web_search_count": 6,
        "web_search_timeout": 10,
        "cn_scraper_enabled": True,
        "cn_scraper_platforms": "auto",
        "cn_scraper_default_platforms": "bilibili",
    }
    values.update(overrides)
    return Settings(**values)


def test_detect_platforms_from_chinese_aliases():
    assert detect_platforms("小红书搜减脂餐") == ["xiaohongshu"]
    assert detect_platforms("在B站搜HIIT训练") == ["bilibili"]
    assert detect_platforms("知乎和微博都搜一下蛋白粉") == ["zhihu", "weibo"]
    assert detect_platforms("京东搜乳清蛋白") == ["jd"]
    assert detect_platforms("蛋白质推荐摄入量") == []
    assert detect_platforms("今晚吃红薯") == []
    assert detect_platforms("请点评一下我的训练") == []


def test_strip_platform_and_build_query():
    assert strip_platform_mentions("小红书搜一下减脂餐") == "搜一下减脂餐"
    assert build_search_query("小红书搜一下减脂餐") == "减脂餐"
    assert build_search_query("在B站搜HIIT训练") == "HIIT训练"
    assert build_search_query("搜一下HIIT一周练几次") == "HIIT一周练几次"


def test_platform_mention_is_web_search():
    assert is_platform_search_request("小红书搜减脂餐")
    assert is_web_search_request("小红书搜减脂餐")
    assert is_web_search_request("B站搜深蹲教学")
    assert not is_platform_search_request("小红书")
    with patch("myfitness.agents.tools.web_search.get_settings", return_value=_settings()):
        assert needs_web_search("小红书搜减脂餐", Intent.GENERAL)


def test_router_classifies_platform_search():
    result = classify_intent("小红书搜减脂餐推荐", use_llm=False)
    assert result.intent == Intent.WEB_SEARCH
    bili = classify_intent("B站搜HIIT训练", use_llm=False)
    assert bili.intent == Intent.WEB_SEARCH
    assert bili.domain == "fitness"


def test_resolve_platforms_auto_and_explicit():
    settings = _settings()
    names, explicit = resolve_platforms("小红书搜减脂餐", settings)
    assert names == ["xiaohongshu"]
    assert explicit
    names, explicit = resolve_platforms("HIIT一周练几次", settings)
    assert names == ["bilibili"]
    assert not explicit
    names, explicit = resolve_platforms("HIIT", _settings(cn_scraper_platforms="none"))
    assert names == []
    assert not explicit


def test_normalize_and_search_cn_platforms_from_engine(monkeypatch):
    def fake_engine(platform, query, limit):
        assert platform == "bilibili"
        assert query == "HIIT"
        return {
            "keyword": query,
            "items": [
                {
                    "title": "HIIT 教学",
                    "bvid": "BV1xx411c7mD",
                    "description": "20分钟间歇",
                    "author": "教练A",
                    "views": 1000,
                }
            ],
        }

    monkeypatch.setattr(
        "myfitness.agents.tools.cn_platform_search._call_engine_search",
        fake_engine,
    )
    hits = search_cn_platforms("HIIT", ["bilibili"], count=5, settings=_settings())
    assert hits[0]["site"] == "B站"
    assert hits[0]["platform"] == "bilibili"
    assert "BV1xx411c7mD" in hits[0]["url"]
    assert "20分钟" in hits[0]["snippet"]


def test_merge_prefers_named_platform():
    web = [
        {"title": "网页1", "url": "https://example.com/1", "snippet": "", "site": "web"},
        {"title": "网页2", "url": "https://example.com/2", "snippet": "", "site": "web"},
        {"title": "网页3", "url": "https://example.com/3", "snippet": "", "site": "web"},
    ]
    platform = [
        {"title": "笔记", "url": "https://www.xiaohongshu.com/explore/1", "snippet": "", "site": "小红书", "platform": "xiaohongshu"},
        {"title": "笔记2", "url": "https://www.xiaohongshu.com/explore/2", "snippet": "", "site": "小红书", "platform": "xiaohongshu"},
    ]
    merged = _merge_hits(web, platform, limit=3, prefer_platform=True)
    assert [item["site"] for item in merged] == ["小红书", "小红书", "web"]


def test_search_web_merges_platform_hits():
    web_hit = {
        "title": "指南",
        "url": "https://example.com/hiit",
        "snippet": "每周两次",
        "site": "丁香医生",
        "published": "",
    }
    platform_hit = {
        "title": "HIIT 教学",
        "url": "https://www.bilibili.com/video/BV1xx411c7mD",
        "snippet": "20分钟",
        "site": "B站",
        "platform": "bilibili",
        "published": "",
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": web_hit["title"],
                                "url": web_hit["url"],
                                "snippet": web_hit["snippet"],
                                "siteName": web_hit["site"],
                            }
                        ]
                    }
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            return FakeResponse()

    with (
        patch(
            "myfitness.agents.tools.web_search.get_settings",
            return_value=_settings(),
        ),
        patch("myfitness.agents.tools.web_search.httpx.Client", FakeClient),
        patch(
            "myfitness.agents.tools.web_search.cn_scraper_available",
            return_value=True,
        ),
        patch(
            "myfitness.agents.tools.web_search.search_cn_platforms",
            return_value=[platform_hit],
        ) as platform_search,
    ):
        result = search_web("B站搜HIIT训练")
    platform_search.assert_called_once()
    assert platform_search.call_args.args[0] == "HIIT训练"
    assert platform_search.call_args.args[1] == ["bilibili"]
    assert result["provider"] == "bocha"
    assert result["results"][0]["site"] == "B站"
    assert result["results"][1]["url"] == "https://example.com/hiit"
    assert "bilibili" in result["sources"]
