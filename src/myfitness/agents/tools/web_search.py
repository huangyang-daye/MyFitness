"""联网检索 Tool — 对话时检索中国互联网公开资料。

优先国内搜索 API（博查 Bocha、智谱 GLM Web Search），未配置密钥时回退到
DuckDuckGo HTML / 必应中国版页面解析，无需额外依赖。
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import httpx
from langchain_core.tools import tool

from myfitness.config import get_settings
from myfitness.schemas.state import Intent

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_BOCHA_URLS = (
    "https://api.bochaai.com/v1/web-search",
    "https://api.bocha.cn/v1/web-search",
)
_ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"

_EXPLICIT_SEARCH = (
    "搜一下",
    "搜一搜",
    "搜索一下",
    "帮我搜",
    "网上搜",
    "网上查",
    "上网查",
    "联网",
    "查资料",
    "查一下资料",
    "百度一下",
    "谷歌一下",
    "搜索网上",
    "互联网",
    "网页搜索",
)
_KNOWLEDGE_HINTS = (
    "什么是",
    "怎么练",
    "如何练",
    "如何做",
    "为什么",
    "有没有用",
    "科学依据",
    "最新研究",
    "推荐剂量",
    "推荐摄入",
    "推荐量",
    "摄入推荐",
    "指南",
    "ACSM",
    "GI值",
    "升糖",
    "副作用",
    "禁忌",
    "原理",
    "机制",
    "有没有研究",
    "文献",
)
_SKIP_INTENTS = {
    Intent.SYNC_TRIGGER,
    Intent.MANUAL_ENTRY,
    Intent.CONFIRMATION_RESPONSE,
    Intent.SCHEDULE_MANAGE,
    Intent.REPORT_TRIGGER,
    Intent.CHART_TRIGGER,
}
_GREETINGS = {"你好", "您好", "谢谢", "感谢", "在吗", "你能做什么"}
_SEARCH_PREFIXES = (
    "帮我搜一下",
    "帮我搜索一下",
    "帮我搜一搜",
    "请搜索一下",
    "请搜一下",
    "搜索一下",
    "搜一搜",
    "搜一下",
    "网上搜一下",
    "网上查一下",
    "上网查一下",
    "联网搜",
    "百度一下",
    "谷歌一下",
)
_TAG_RE = re.compile(r"<[^>]+>")
_DDG_LINK_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def is_explicit_search(message: str) -> bool:
    return any(token in message for token in _EXPLICIT_SEARCH)


def is_knowledge_question(message: str) -> bool:
    text = message.strip()
    if not text or text in _GREETINGS:
        return False
    if len(text) < 12 and any(g in text for g in _GREETINGS):
        return False
    return any(hint in text for hint in _KNOWLEDGE_HINTS)


def is_personal_data_query(message: str) -> bool:
    has_time_or_self = bool(
        re.search(r"(昨天|今日|今天|昨日|前天|我的|我吃|我练|查询)", message)
    )
    has_metric = bool(re.search(r"(蛋白|热量|体重|体脂|训练|吃|卡路里|围度)", message))
    return has_time_or_self and has_metric


def is_web_search_request(message: str) -> bool:
    """关键词层：显式搜网，或公开知识问（且不是查自己某天的数据）。"""
    if is_explicit_search(message):
        return True
    if is_personal_data_query(message):
        return False
    return is_knowledge_question(message)


def needs_web_search(message: str, intent: Intent | None) -> bool:
    settings = get_settings()
    if not settings.web_search_enabled:
        return False
    if intent in _SKIP_INTENTS:
        return False
    if is_explicit_search(message) or intent == Intent.WEB_SEARCH:
        return True
    if is_knowledge_question(message):
        return True
    return False


def build_search_query(message: str) -> str:
    text = message.strip()
    for prefix in _SEARCH_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].lstrip("：:，, ")
            break
    return text or message.strip()


def format_web_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = [
        "【联网检索结果 — 综合以下公开网页回答知识性问题，关键结论后标注 [n]；"
        "与用户本地数据冲突时以数据库为准。文末列出参考资料（标题 + 链接）】"
    ]
    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "未命名").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        site = str(item.get("site") or "").strip()
        published = str(item.get("published") or "").strip()
        meta = " · ".join(part for part in (site, published) if part)
        lines.append(f"[{index}] {title}")
        if meta:
            lines.append(f"来源：{meta}")
        if url:
            lines.append(f"链接：{url}")
        if snippet:
            lines.append(snippet[:500])
        lines.append("")
    return "\n".join(lines).strip()


def search_web(
    query: str,
    *,
    count: int | None = None,
    freshness: str | None = None,
) -> dict[str, Any]:
    """执行网页搜索并返回规范化结果。失败时 results 为空并带 error。"""
    settings = get_settings()
    query = query.strip()
    if not query:
        return _payload(query, [], provider=None, error="empty_query")
    if not settings.web_search_enabled:
        return _payload(query, [], provider=None, error="disabled")

    limit = count or settings.web_search_count
    freshness = freshness or settings.web_search_freshness
    last_error: str | None = None

    for provider in _provider_chain(settings):
        try:
            hits = _dispatch(provider, query, limit, freshness, settings)
        except Exception as exc:  # noqa: BLE001 - 任一提供方失败都尝试下一个
            last_error = str(exc)
            logger.warning("web_search provider %s failed: %s", provider, exc)
            continue
        if hits:
            return _payload(query, hits[:limit], provider=provider, error=None)
        last_error = last_error or "no_results"

    return _payload(query, [], provider=_resolve_provider(settings), error=last_error)


@tool
def web_search(
    query: str,
    count: int | None = None,
    freshness: str | None = None,
) -> dict:
    """在中国互联网上搜索公开资料（健身、营养、训练研究等）。

    Args:
        query: 搜索关键词或完整问题。
        count: 返回条数，默认使用配置 WEB_SEARCH_COUNT。
        freshness: 时效 noLimit / oneDay / oneWeek / oneMonth / oneYear。
    """
    return search_web(query, count=count, freshness=freshness)


def _payload(
    query: str,
    results: list[dict[str, Any]],
    *,
    provider: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "tool": "web_search",
        "provider": provider,
        "query": query,
        "count": len(results),
        "results": results,
        "error": error,
    }


def _resolve_provider(settings) -> str:
    provider = (settings.web_search_provider or "auto").strip().lower()
    if provider != "auto":
        return provider
    if settings.resolved_bocha_api_key():
        return "bocha"
    if settings.resolved_zhipu_search_key():
        return "zhipu"
    return "duckduckgo"


def _provider_chain(settings) -> list[str]:
    primary = _resolve_provider(settings)
    fallbacks = ["bocha", "zhipu", "duckduckgo", "bing"]
    chain = [primary]
    for name in fallbacks:
        if name not in chain:
            if name == "bocha" and not settings.resolved_bocha_api_key():
                continue
            if name == "zhipu" and not settings.resolved_zhipu_search_key():
                continue
            chain.append(name)
    return chain


def _dispatch(
    provider: str,
    query: str,
    count: int,
    freshness: str,
    settings,
) -> list[dict[str, Any]]:
    if provider == "bocha":
        return _search_bocha(query, count, freshness, settings)
    if provider == "zhipu":
        return _search_zhipu(query, count, freshness, settings)
    if provider == "bing":
        return _search_bing(query, count, settings)
    return _search_duckduckgo(query, count, settings)


def _normalize_hit(
    *,
    title: str,
    url: str,
    snippet: str = "",
    site: str = "",
    published: str = "",
) -> dict[str, Any] | None:
    title = _clean_text(title)
    url = (url or "").strip()
    if not title or not url or url.startswith("javascript:"):
        return None
    if not site:
        host = urlparse(url).hostname or ""
        site = host.removeprefix("www.")
    return {
        "title": title[:200],
        "url": url,
        "snippet": _clean_text(snippet)[:500],
        "site": site[:80],
        "published": _clean_text(published)[:40],
    }


def _clean_text(value: str) -> str:
    text = html.unescape(_TAG_RE.sub("", value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _timeout(settings) -> float:
    return float(settings.web_search_timeout)


def _search_bocha(query: str, count: int, freshness: str, settings) -> list[dict[str, Any]]:
    api_key = settings.resolved_bocha_api_key()
    if not api_key:
        return []
    body = {
        "query": query,
        "freshness": freshness or "noLimit",
        "summary": True,
        "count": count,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    with httpx.Client(timeout=_timeout(settings), headers={"User-Agent": _USER_AGENT}) as client:
        for url in _BOCHA_URLS:
            try:
                response = client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            hits = _parse_bocha(data)
            if hits:
                return hits
    if last_error:
        raise last_error
    return []


def _parse_bocha(data: dict[str, Any]) -> list[dict[str, Any]]:
    pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    results: list[dict[str, Any]] = []
    for item in pages:
        if not isinstance(item, dict):
            continue
        hit = _normalize_hit(
            title=str(item.get("name") or item.get("title") or ""),
            url=str(item.get("url") or ""),
            snippet=str(item.get("summary") or item.get("snippet") or ""),
            site=str(item.get("siteName") or ""),
            published=str(item.get("datePublished") or item.get("dateLastCrawled") or ""),
        )
        if hit:
            results.append(hit)
    return results


def _search_zhipu(query: str, count: int, freshness: str, settings) -> list[dict[str, Any]]:
    api_key = settings.resolved_zhipu_search_key()
    if not api_key:
        return []
    recency = {
        "noLimit": "noLimit",
        "oneDay": "oneDay",
        "oneWeek": "oneWeek",
        "oneMonth": "oneMonth",
        "oneYear": "oneYear",
    }.get(freshness, "noLimit")
    body = {
        "search_query": query,
        "search_engine": "search_pro",
        "search_intent": True,
        "count": count,
        "search_recency_filter": recency,
        "content_size": "medium",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_timeout(settings), headers={"User-Agent": _USER_AGENT}) as client:
        response = client.post(_ZHIPU_URL, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
    return _parse_zhipu(data)


def _parse_zhipu(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("search_result") or data.get("search_results") or []
    if isinstance(raw, dict):
        raw = raw.get("results") or raw.get("value") or []
    results: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        hit = _normalize_hit(
            title=str(item.get("title") or item.get("name") or ""),
            url=str(item.get("link") or item.get("url") or ""),
            snippet=str(item.get("content") or item.get("snippet") or item.get("media") or ""),
            site=str(item.get("media") or item.get("site") or ""),
            published=str(item.get("publish_date") or item.get("refer") or ""),
        )
        if hit:
            results.append(hit)
    return results


def _search_duckduckgo(query: str, count: int, settings) -> list[dict[str, Any]]:
    with httpx.Client(
        timeout=_timeout(settings),
        headers={"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
        follow_redirects=True,
    ) as client:
        response = client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "kl": "cn-zh"},
        )
        response.raise_for_status()
        return _parse_duckduckgo(response.text, count)


def _parse_duckduckgo(page: str, count: int) -> list[dict[str, Any]]:
    snippets = [_clean_text(item) for item in _DDG_SNIPPET_RE.findall(page)]
    results: list[dict[str, Any]] = []
    for index, (href, title_html) in enumerate(_DDG_LINK_RE.findall(page)):
        url = _unwrap_ddg_url(html.unescape(href))
        snippet = snippets[index] if index < len(snippets) else ""
        hit = _normalize_hit(title=title_html, url=url, snippet=snippet)
        if hit:
            results.append(hit)
        if len(results) >= count:
            break
    return results


def _unwrap_ddg_url(href: str) -> str:
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        return unquote(qs["uddg"][0])
    return href


def _search_bing(query: str, count: int, settings) -> list[dict[str, Any]]:
    url = f"https://cn.bing.com/search?q={quote(query)}&setlang=zh-hans&ensearch=0"
    with httpx.Client(
        timeout=_timeout(settings),
        headers={"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
        follow_redirects=True,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return _parse_bing(response.text, count)


def _parse_bing(page: str, count: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block in re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', page, re.IGNORECASE | re.DOTALL):
        link = re.search(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not link:
            continue
        url = html.unescape(link.group(1))
        if url.startswith("/"):
            url = urljoin("https://cn.bing.com", url)
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.IGNORECASE | re.DOTALL)
        snippet = snippet_match.group(1) if snippet_match else ""
        hit = _normalize_hit(title=link.group(2), url=url, snippet=snippet)
        if hit:
            results.append(hit)
        if len(results) >= count:
            break
    return results
