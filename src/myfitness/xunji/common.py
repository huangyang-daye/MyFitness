import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

AuthHeaderStyle = Literal["bearer", "x-api-key", "x-agent-key"]


class XunjiRateLimitError(Exception):
    def __init__(self, retry_after_ms: int, message: str = "too frequent"):
        self.retry_after_ms = retry_after_ms
        super().__init__(message)


class XunjiApiError(Exception):
    def __init__(self, message: str, response: dict | None = None):
        self.response = response
        super().__init__(message)


class XunjiConfirmationRequiredError(XunjiApiError):
    """Skill: user confirmation required — 需先 dry_run 展示摘要并等用户确认。"""


def mask_api_key(key: str) -> str:
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"


def new_client_request_id() -> str:
    return str(uuid.uuid4())


@dataclass
class _CacheEntry:
    data: Any
    expires_at: float


@dataclass
class XunjiHttpClient:
    """训记 Skill 共用 HTTP 层。

    遵循 skills/xunji-*/SKILL.md：
    - Authorization Bearer / x-api-key
    - 同一 key + 同一 endpoint 限频
    - 相同查询条件缓存
    - too frequent → 等待 retry_after_ms
    """

    cache_ttl_seconds: int = 300
    max_retries: int = 3

    _cache: dict[str, _CacheEntry] = field(default_factory=dict, repr=False)
    _last_call: dict[str, float] = field(default_factory=dict, repr=False)

    def _cache_key(self, endpoint: str, payload: dict) -> str:
        raw = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _rate_limit_key(self, url: str, api_key: str) -> str:
        return f"{url}:{mask_api_key(api_key)}"

    def _get_cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and entry.expires_at > time.time():
            return entry.data
        if entry:
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        self._cache[key] = _CacheEntry(data=data, expires_at=time.time() + self.cache_ttl_seconds)

    def invalidate_cache(self, endpoint: str | None = None) -> None:
        if endpoint is None:
            self._cache.clear()
            return
        for key in [k for k in self._cache if endpoint in k]:
            del self._cache[key]

    @staticmethod
    def _build_headers(api_key: str, auth_style: AuthHeaderStyle) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        }
        if auth_style == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_style == "x-api-key":
            headers["x-api-key"] = api_key
        elif auth_style == "x-agent-key":
            headers["x-agent-key"] = api_key
        return headers

    def post(
        self,
        url: str,
        api_key: str,
        payload: dict,
        *,
        min_interval_seconds: float = 15.0,
        use_cache: bool = True,
        require_success: bool = True,
        auth_style: AuthHeaderStyle = "bearer",
        return_full: bool = False,
    ) -> dict:
        if not api_key:
            raise XunjiApiError("apikey missing")

        cache_key = self._cache_key(url, payload) if use_cache else ""
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug("训记缓存命中: %s", url)
                return cached

        rl_key = self._rate_limit_key(url, api_key)
        elapsed = time.time() - self._last_call.get(rl_key, 0.0)
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)

        headers = self._build_headers(api_key, auth_style)

        for attempt in range(self.max_retries):
            try:
                self._last_call[rl_key] = time.time()
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                self._handle_skill_errors(data)

                if require_success and data.get("success") is False and "success" in data:
                    raise XunjiApiError(data.get("message", "API request failed"), data)

                result = data if return_full else data.get("res", data)
                if use_cache:
                    self._set_cache(cache_key, result)
                return result

            except XunjiRateLimitError as exc:
                if attempt + 1 >= self.max_retries:
                    raise
                logger.warning(
                    "训记限频 endpoint=%s key=%s retry=%sms",
                    url,
                    mask_api_key(api_key),
                    exc.retry_after_ms,
                )
                time.sleep(exc.retry_after_ms / 1000.0)
            except httpx.HTTPStatusError as exc:
                if attempt + 1 >= self.max_retries:
                    raise XunjiApiError(f"HTTP {exc.response.status_code}: {exc}") from exc
                time.sleep(2**attempt)

        raise XunjiApiError("Max retries exceeded")

    def _handle_skill_errors(self, data: dict) -> None:
        msg = str(data.get("message", ""))
        lower = msg.lower()

        if "too frequent" in lower:
            retry_ms = int(data.get("retry_after_ms", 15000))
            raise XunjiRateLimitError(retry_ms, msg)

        if "apikey missing" in lower or "apikey invalid" in lower:
            raise XunjiApiError(msg, data)

        if "user confirmation required" in lower:
            raise XunjiConfirmationRequiredError(msg, data)

        if "仅vip可用" in lower or ("vip" in lower and "可用" in msg):
            raise XunjiApiError("仅VIP可用", data)
