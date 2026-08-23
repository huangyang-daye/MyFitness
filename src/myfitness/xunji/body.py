"""训记身体数据 — skills/xunji-body-open-api/SKILL.md"""

from typing import Any

from myfitness.xunji.common import XunjiHttpClient, new_client_request_id
from myfitness.xunji.parsers.body import validate_body_write_record
from myfitness.xunji.skills import (
    BODY_BASE_URL,
    BODY_QUERY_PAGE_SIZE,
    BODY_QUERY_PATH,
    BODY_RATE_LIMIT_SECONDS,
    BODY_SCHEMA_VERSION,
    BODY_UPSERT_PATH,
)


class BodyOpenApi:
    SKILL = "xunji-body-open-api"

    def __init__(self, api_key: str, http: XunjiHttpClient | None = None):
        self.api_key = api_key
        self.http = http or XunjiHttpClient()

    def query(
        self,
        start_date: str,
        end_date: str,
        types: list[str] | None = None,
        *,
        include_latest: bool = True,
        include_records: bool = True,
        limit: int = BODY_QUERY_PAGE_SIZE,
        offset: int = 0,
    ) -> dict:
        payload: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "include_latest": include_latest,
            "include_records": include_records,
            "limit": limit,
            "offset": offset,
        }
        if types:
            payload["types"] = types

        return self.http.post(
            f"{BODY_BASE_URL}{BODY_QUERY_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=BODY_RATE_LIMIT_SECONDS,
            require_success=True,
        )

    def query_all_records(
        self,
        start_date: str,
        end_date: str,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """分页拉取全部 records（Skill: limit/offset）。"""
        offset = 0
        all_records: list[dict] = []
        latest: dict[str, Any] = {}
        by_type: dict[str, Any] = {}

        while True:
            page = self.query(
                start_date,
                end_date,
                types=types,
                offset=offset,
                limit=BODY_QUERY_PAGE_SIZE,
            )
            records = page.get("records") or []
            all_records.extend(records)
            latest.update(page.get("latest") or {})
            by_type.update(page.get("by_type") or {})

            if len(records) < BODY_QUERY_PAGE_SIZE:
                break
            offset += BODY_QUERY_PAGE_SIZE

        return {"records": all_records, "latest": latest, "by_type": by_type}

    def upsert_dry_run(
        self,
        records: list[dict],
        client_request_id: str | None = None,
    ) -> dict:
        for record in records:
            validate_body_write_record(record)

        payload = {
            "schema_version": BODY_SCHEMA_VERSION,
            "client_request_id": client_request_id or new_client_request_id(),
            "dry_run": True,
            "records": records,
        }
        return self.http.post(
            f"{BODY_BASE_URL}{BODY_UPSERT_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=BODY_RATE_LIMIT_SECONDS,
            use_cache=False,
            require_success=True,
        )

    def upsert_confirmed(
        self,
        records: list[dict],
        client_request_id: str | None = None,
    ) -> dict:
        for record in records:
            validate_body_write_record(record)

        payload = {
            "schema_version": BODY_SCHEMA_VERSION,
            "client_request_id": client_request_id or new_client_request_id(),
            "dry_run": False,
            "confirmed": True,
            "records": records,
        }
        result = self.http.post(
            f"{BODY_BASE_URL}{BODY_UPSERT_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=BODY_RATE_LIMIT_SECONDS,
            use_cache=False,
            require_success=True,
        )
        self.http.invalidate_cache(BODY_QUERY_PATH)
        return result
