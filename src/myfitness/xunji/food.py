"""训记饮食数据 — skills/xunji-food-open-api/SKILL.md"""

from datetime import date, timedelta
from typing import Any, Iterator

from myfitness.xunji.common import XunjiHttpClient, new_client_request_id
from myfitness.xunji.skills import (
    FOOD_BASE_URL,
    FOOD_CUSTOM_UPSERT_PATH,
    FOOD_QUERY_MAX_FUTURE_DAYS,
    FOOD_QUERY_MAX_PAST_DAYS,
    FOOD_QUERY_PATH,
    FOOD_RATE_LIMIT_SECONDS,
    FOOD_SEARCH_BASE_URL,
    FOOD_SEARCH_PATH,
    FOOD_TEMPLATES_APPLY_PATH,
    FOOD_TEMPLATES_LIST_PATH,
    FOOD_UPSERT_PATH,
)


def clamp_food_query_range(start: date, end: date) -> tuple[date, date, list[str]]:
    """Skill: 查询范围限制在过去一年到未来 3 个月。"""
    today = date.today()
    min_date = today - timedelta(days=FOOD_QUERY_MAX_PAST_DAYS)
    max_date = today + timedelta(days=FOOD_QUERY_MAX_FUTURE_DAYS)
    notes: list[str] = []
    clamped_start = max(start, min_date)
    clamped_end = min(end, max_date)
    if clamped_start != start or clamped_end != end:
        notes.append(
            f"饮食查询范围已按 Skill 限制调整为 {clamped_start} ~ {clamped_end}"
            f"（允许：{min_date} ~ {max_date}）"
        )
    if clamped_start > clamped_end:
        clamped_start = clamped_end
    return clamped_start, clamped_end, notes


def iter_food_query_chunks(start: date, end: date, chunk_days: int = 31) -> Iterator[tuple[date, date]]:
    """Skill: 大范围查询拆分为多个允许范围内的小段。"""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


class FoodOpenApi:
    SKILL = "xunji-food-open-api"

    def __init__(
        self,
        food_api_key: str,
        search_api_key: str = "",
        http: XunjiHttpClient | None = None,
    ):
        self.food_api_key = food_api_key
        self.search_api_key = search_api_key
        self.http = http or XunjiHttpClient()

    def query(
        self,
        start_date: str,
        end_date: str,
        *,
        include_detail: bool = True,
    ) -> dict:
        start, end, _ = clamp_food_query_range(
            date.fromisoformat(start_date),
            date.fromisoformat(end_date),
        )
        payload = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "include_detail": include_detail,
        }
        return self.http.post(
            f"{FOOD_BASE_URL}{FOOD_QUERY_PATH}",
            self.food_api_key,
            payload,
            min_interval_seconds=FOOD_RATE_LIMIT_SECONDS,
            require_success=True,
            auth_style="bearer",
        )

    def query_range_merged(
        self,
        start_date: str,
        end_date: str,
        *,
        include_detail: bool = True,
    ) -> tuple[dict[str, Any], list[str]]:
        """按 Skill 限制 clamp 并分段 query，合并 days/foods。"""
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        clamped_start, clamped_end, notes = clamp_food_query_range(start, end)

        merged_days: list[dict] = []
        merged_foods: list[dict] = []

        for chunk_start, chunk_end in iter_food_query_chunks(clamped_start, clamped_end):
            result = self.query(
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                include_detail=include_detail,
            )
            days = result.get("days")
            if isinstance(days, list):
                merged_days.extend(days)
            foods = result.get("foods")
            if isinstance(foods, list):
                merged_foods.extend(foods)

        payload: dict[str, Any] = {}
        if merged_days:
            payload["days"] = merged_days
        if merged_foods:
            payload["foods"] = merged_foods
        return payload, notes

    def search(self, keyword: str, limit: int = 8) -> dict:
        """Skill: 搜索走 api.xunjiapp.cn，兼容 x-agent-key / x-api-key。"""
        if not self.search_api_key:
            raise ValueError("food search api key missing")

        payload = {"keyword": keyword, "limit": limit}
        return self.http.post(
            f"{FOOD_SEARCH_BASE_URL}{FOOD_SEARCH_PATH}",
            self.search_api_key,
            payload,
            min_interval_seconds=FOOD_RATE_LIMIT_SECONDS,
            require_success=True,
            auth_style="x-agent-key",
        )

    def upsert_foods(
        self,
        foods: list[dict],
        *,
        dry_run: bool = False,
        client_request_id: str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "client_request_id": client_request_id or new_client_request_id(),
            "dry_run": dry_run,
            "foods": foods,
        }
        result = self.http.post(
            f"{FOOD_BASE_URL}{FOOD_UPSERT_PATH}",
            self.food_api_key,
            payload,
            min_interval_seconds=FOOD_RATE_LIMIT_SECONDS,
            use_cache=False,
            require_success=True,
        )
        if not dry_run:
            self.http.invalidate_cache(FOOD_QUERY_PATH)
        return result

    def upsert_custom_food(
        self,
        food: dict,
        *,
        dry_run: bool = False,
        client_request_id: str | None = None,
    ) -> dict:
        payload = {
            "client_request_id": client_request_id or new_client_request_id(),
            "dry_run": dry_run,
            "food": food,
        }
        return self.http.post(
            f"{FOOD_BASE_URL}{FOOD_CUSTOM_UPSERT_PATH}",
            self.food_api_key,
            payload,
            min_interval_seconds=FOOD_RATE_LIMIT_SECONDS,
            use_cache=False,
            require_success=True,
        )

    def list_templates(self) -> dict:
        return self.http.post(
            f"{FOOD_BASE_URL}{FOOD_TEMPLATES_LIST_PATH}",
            self.food_api_key,
            {},
            min_interval_seconds=FOOD_RATE_LIMIT_SECONDS,
            require_success=True,
        )

    def apply_template(self, payload: dict) -> dict:
        result = self.http.post(
            f"{FOOD_BASE_URL}{FOOD_TEMPLATES_APPLY_PATH}",
            self.food_api_key,
            payload,
            min_interval_seconds=FOOD_RATE_LIMIT_SECONDS,
            use_cache=False,
            require_success=True,
        )
        self.http.invalidate_cache(FOOD_QUERY_PATH)
        return result
