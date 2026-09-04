"""中文平台搜索 — 封装 cn-scraper-mcp 的搜索能力。

优先调用本机已安装的 `cn_scraper_mcp` 引擎；若配置了 `CN_SCRAPER_MCP_URL`，
则改为调用远程 MCP（Cookie 留在 MCP 所在机器）。未安装依赖时静默跳过。
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from myfitness.config import get_settings

logger = logging.getLogger(__name__)

# 站点展示名
PLATFORM_LABELS: dict[str, str] = {
    "xiaohongshu": "小红书",
    "zhihu": "知乎",
    "weibo": "微博",
    "bilibili": "B站",
    "taobao": "淘宝",
    "jd": "京东",
    "douban": "豆瓣",
    "dianping": "大众点评",
    "douyin": "抖音",
    "pdd": "拼多多",
}

# 别名按长度降序匹配；知识星球需要 group_id，不纳入关键词搜索。
PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {
    "xiaohongshu": ("xiaohongshu", "小红书", "xhs"),
    "bilibili": ("bilibili", "哔哩哔哩", "哔哩", "b 站", "b站"),
    "dianping": ("大众点评", "dianping"),
    "taobao": ("taobao", "tmall", "淘宝", "天猫"),
    "douyin": ("douyin", "抖音"),
    "zhihu": ("zhihu", "知乎"),
    "weibo": ("weibo", "微博"),
    "douban": ("douban", "豆瓣"),
    "pdd": ("pinduoduo", "拼多多"),
    "jd": ("jingdong", "京东"),
}
# 纯拉丁短别名按词边界匹配，避免误伤普通英文。
_LATIN_WORD_ALIASES: dict[str, str] = {
    "xhs": "xiaohongshu",
    "jd": "jd",
}

# 需要本地 Chrome / 登录态的平台；未点名时不自动搜。
CHROME_PLATFORMS = frozenset({"xiaohongshu", "jd", "pdd", "douyin"})
# 默认只补充无需登录的 B 站，避免无 Cookie 时拖慢对话。
DEFAULT_PLATFORMS = ("bilibili",)
SKIP_DEFAULT = frozenset({"pdd"})  # 官方不推荐，仅用户点名才搜

MCP_TOOL_NAMES: dict[str, str] = {
    "xiaohongshu": "xiaohongshu_search",
    "zhihu": "zhihu_search",
    "weibo": "weibo_search",
    "bilibili": "bilibili_search",
    "taobao": "taobao_search",
    "jd": "jd_search",
    "douban": "douban_search",
    "dianping": "dianping_search",
    "douyin": "douyin_search",
    "pdd": "pdd_search",
}

_ENGINE_SPECS: dict[str, tuple[str, str]] = {
    "xiaohongshu": ("cn_scraper_mcp.engines.xiaohongshu", "XiaohongshuEngine"),
    "zhihu": ("cn_scraper_mcp.engines.zhihu", "ZhihuEngine"),
    "weibo": ("cn_scraper_mcp.engines.weibo", "WeiboEngine"),
    "bilibili": ("cn_scraper_mcp.engines.bilibili", "BilibiliEngine"),
    "taobao": ("cn_scraper_mcp.engines.taobao", "TaobaoEngine"),
    "jd": ("cn_scraper_mcp.engines.jd", "JDEngine"),
    "douban": ("cn_scraper_mcp.engines.douban", "DoubanEngine"),
    "dianping": ("cn_scraper_mcp.engines.dianping", "DianpingEngine"),
    "douyin": ("cn_scraper_mcp.engines.douyin", "DouyinEngine"),
    "pdd": ("cn_scraper_mcp.engines.pdd", "PDDEngine"),
}

_ALL_ALIASES: list[tuple[str, str]] = sorted(
    ((alias, platform) for platform, aliases in PLATFORM_ALIASES.items() for alias in aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)
_ALIAS_PATTERN = re.compile(
    "|".join(re.escape(alias) for alias, _ in _ALL_ALIASES),
    re.IGNORECASE,
)


def detect_platforms(message: str) -> list[str]:
    """从用户话里识别点名的平台，保持别名出现顺序且去重。"""
    text = (message or "").strip()
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    lower = text.casefold()
    spans: list[tuple[int, str]] = []
    for alias, platform in _ALL_ALIASES:
        start = 0
        needle = alias.casefold()
        while True:
            index = lower.find(needle, start)
            if index < 0:
                break
            spans.append((index, platform))
            start = index + max(len(needle), 1)
    for alias, platform in _LATIN_WORD_ALIASES.items():
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lower):
            spans.append((match.start(), platform))
    spans.sort(key=lambda item: item[0])
    for _, platform in spans:
        if platform not in seen:
            seen.add(platform)
            found.append(platform)
    return found


def strip_platform_mentions(text: str) -> str:
    """去掉平台名，便于把「小红书搜减脂餐」收成「减脂餐」。"""
    cleaned = _ALIAS_PATTERN.sub(" ", text or "")
    cleaned = re.sub(r"[（）()【】\[\]「」]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[在上里中]+", "", cleaned).strip()
    return cleaned


def is_platform_search_request(message: str) -> bool:
    """话里点了中文平台，且去掉平台名后仍有检索内容。"""
    platforms = detect_platforms(message)
    if not platforms:
        return False
    leftover = strip_platform_mentions(message)
    leftover = re.sub(
        r"^(?:帮我)?(?:搜一下|搜一搜|搜索一下|搜索|搜|查一下|查|找一下|找)\s*",
        "",
        leftover,
    ).strip("：:，, ")
    return bool(leftover)


def resolve_platforms(message: str, settings=None) -> tuple[list[str], bool]:
    """返回 (要搜的平台, 是否用户点名)。

    `CN_SCRAPER_PLATFORMS=auto` 时：点名用点名的平台，否则用默认平台（B 站）。
    设为逗号列表时始终搜这些平台，并与点名合并。
    """
    settings = settings or get_settings()
    if not getattr(settings, "cn_scraper_enabled", True):
        return [], False
    explicit = detect_platforms(message)
    configured = _parse_platform_list(getattr(settings, "cn_scraper_platforms", "auto"))
    if configured == ["none"] or configured == []:
        return [], False
    if configured == ["auto"]:
        if explicit:
            return _filter_skip(explicit, honor_skip=False), True
        defaults = _parse_platform_list(
            getattr(settings, "cn_scraper_default_platforms", "bilibili")
        )
        return _filter_skip(defaults, honor_skip=True), False
    merged: list[str] = []
    seen: set[str] = set()
    for name in [*explicit, *configured]:
        if name in seen or name in {"auto", "none"}:
            continue
        if name not in PLATFORM_LABELS:
            continue
        seen.add(name)
        merged.append(name)
    return merged, bool(explicit)


def search_cn_platforms(
    query: str,
    platforms: list[str],
    *,
    count: int = 5,
    settings=None,
) -> list[dict[str, Any]]:
    """按平台搜索并归一化为 web_search 的 hit 结构。任一平台失败不影响其余。"""
    query = (query or "").strip()
    if not query or not platforms:
        return []
    settings = settings or get_settings()
    limit = max(1, min(count, 20))
    mcp_url = (getattr(settings, "cn_scraper_mcp_url", "") or "").strip()
    timeout = float(getattr(settings, "cn_scraper_timeout", 25))

    http_platforms = [name for name in platforms if name not in CHROME_PLATFORMS]
    chrome_platforms = [name for name in platforms if name in CHROME_PLATFORMS]

    hits: list[dict[str, Any]] = []
    if http_platforms:
        workers = min(4, len(http_platforms))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _search_one_platform, name, query, limit, mcp_url, timeout
                ): name
                for name in http_platforms
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    hits.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("cn platform %s failed: %s", name, exc)
    for name in chrome_platforms:
        try:
            hits.extend(_search_one_platform(name, query, limit, mcp_url, timeout))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cn platform %s failed: %s", name, exc)
    return hits


def cn_scraper_available(settings=None) -> bool:
    settings = settings or get_settings()
    if not getattr(settings, "cn_scraper_enabled", True):
        return False
    if (getattr(settings, "cn_scraper_mcp_url", "") or "").strip():
        return _fastmcp_importable()
    return _engines_importable()


def _parse_platform_list(raw: str) -> list[str]:
    text = (raw or "auto").strip().lower()
    if not text:
        return ["auto"]
    parts = [part.strip().lower() for part in re.split(r"[,，\s]+", text) if part.strip()]
    alias_map = {
        alias.casefold(): platform
        for platform, aliases in PLATFORM_ALIASES.items()
        for alias in aliases
    }
    alias_map.update({name: name for name in PLATFORM_LABELS})
    resolved: list[str] = []
    for part in parts:
        name = alias_map.get(part, part)
        if name not in resolved:
            resolved.append(name)
    return resolved


def _filter_skip(platforms: list[str], *, honor_skip: bool) -> list[str]:
    out: list[str] = []
    for name in platforms:
        if name not in PLATFORM_LABELS:
            continue
        if honor_skip and name in SKIP_DEFAULT:
            continue
        if name not in out:
            out.append(name)
    return out


def _search_one_platform(
    platform: str,
    query: str,
    limit: int,
    mcp_url: str,
    timeout: float,
) -> list[dict[str, Any]]:
    if mcp_url:
        payload = _call_mcp_search(mcp_url, platform, query, limit, timeout)
    else:
        payload = _call_engine_search(platform, query, limit)
    return _normalize_payload(platform, payload)


def _call_engine_search(platform: str, query: str, limit: int) -> dict[str, Any]:
    spec = _ENGINE_SPECS.get(platform)
    if spec is None:
        return {}
    module_name, class_name = spec
    try:
        module = __import__(module_name, fromlist=[class_name])
        engine_cls = getattr(module, class_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cn-scraper-mcp not available for %s: %s", platform, exc)
        return {}
    engine = engine_cls()
    result = engine.search(query, limit=limit)
    return result if isinstance(result, dict) else {}


def _call_mcp_search(
    url: str,
    platform: str,
    query: str,
    limit: int,
    timeout: float,
) -> dict[str, Any]:
    tool_name = MCP_TOOL_NAMES.get(platform)
    if not tool_name:
        return {}
    arguments: dict[str, Any] = {"keyword": query, "limit": limit}

    async def _invoke() -> Any:
        from fastmcp import Client

        async with Client(url) as client:
            return await client.call_tool(tool_name, arguments)

    raw = _run_async(_invoke, timeout=timeout)
    return _unwrap_mcp_result(raw)


def _unwrap_mcp_result(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    for attr in ("data", "structured_content", "output"):
        value = getattr(raw, attr, None)
        if isinstance(value, dict):
            return value
    content = getattr(raw, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str) and text.strip().startswith("{"):
            import json

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _run_async(coro_factory, *, timeout: float) -> Any:
    async def _with_timeout():
        return await asyncio.wait_for(coro_factory(), timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_with_timeout())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _with_timeout()).result(timeout=timeout + 2)


def _normalize_payload(platform: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not payload:
        return []
    if payload.get("error") or payload.get("status") in {"error", "ACTION_REQUIRED"}:
        logger.info("cn platform %s returned error: %s", platform, payload.get("message") or payload)
        return []
    items = payload.get("items") or payload.get("results") or payload.get("data") or []
    if isinstance(items, dict):
        items = items.get("items") or items.get("list") or []
    if not isinstance(items, list):
        return []
    hits: list[dict[str, Any]] = []
    site = PLATFORM_LABELS.get(platform, platform)
    for item in items:
        if not isinstance(item, dict):
            continue
        hit = _normalize_item(platform, item, site)
        if hit:
            hits.append(hit)
    return hits


def _normalize_item(platform: str, item: dict[str, Any], site: str) -> dict[str, Any] | None:
    title = str(
        item.get("title")
        or item.get("name")
        or item.get("text")
        or item.get("question_title")
        or ""
    ).strip()
    url = str(item.get("url") or item.get("href") or item.get("link") or "").strip()
    if not url:
        url = _infer_url(platform, item)
    if not title or not url or url.startswith("javascript:"):
        return None
    snippet = _snippet_from_item(item)
    published = str(
        item.get("published")
        or item.get("published_at")
        or item.get("time")
        or item.get("created_at")
        or ""
    ).strip()
    return {
        "title": title[:200],
        "url": url,
        "snippet": snippet[:500],
        "site": site[:80],
        "published": str(published)[:40],
        "platform": platform,
    }


def _infer_url(platform: str, item: dict[str, Any]) -> str:
    if platform == "xiaohongshu":
        note_id = str(item.get("noteId") or item.get("id") or "").strip()
        if note_id:
            return f"https://www.xiaohongshu.com/explore/{note_id}"
    if platform == "bilibili":
        bvid = str(item.get("bvid") or "").strip()
        if bvid:
            return f"https://www.bilibili.com/video/{bvid}"
    if platform == "weibo":
        mid = str(item.get("id") or item.get("mid") or "").strip()
        if mid:
            return f"https://m.weibo.cn/detail/{mid}"
    if platform == "douban":
        subject_id = str(item.get("id") or "").strip()
        if subject_id:
            return f"https://www.douban.com/subject/{subject_id}/"
    if platform in {"taobao", "jd", "pdd"}:
        item_id = str(item.get("id") or item.get("sku") or "").strip()
        if platform == "jd" and item_id:
            return f"https://item.jd.com/{item_id}.html"
        if platform == "taobao" and item_id:
            return f"https://item.taobao.com/item.htm?id={item_id}"
    return ""


def _snippet_from_item(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "excerpt",
        "description",
        "desc",
        "summary",
        "snippet",
        "text",
        "content",
        "author",
        "shop",
        "user",
    ):
        value = item.get(key)
        if value:
            parts.append(str(value).strip())
    extras = []
    if item.get("price") not in (None, ""):
        extras.append(f"价格 {item.get('price')}")
    if item.get("sales") not in (None, ""):
        extras.append(f"销量 {item.get('sales')}")
    if item.get("likes") not in (None, ""):
        extras.append(f"点赞 {item.get('likes')}")
    if item.get("views") not in (None, ""):
        extras.append(f"播放 {item.get('views')}")
    if item.get("votes") not in (None, ""):
        extras.append(f"赞同 {item.get('votes')}")
    if item.get("rating") not in (None, ""):
        extras.append(f"评分 {item.get('rating')}")
    parts.extend(extras)
    text = " · ".join(part for part in parts if part)
    return re.sub(r"\s+", " ", text).strip()


def _engines_importable() -> bool:
    try:
        import cn_scraper_mcp  # noqa: F401
    except ImportError:
        return False
    return True


def _fastmcp_importable() -> bool:
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        return False
    return True
