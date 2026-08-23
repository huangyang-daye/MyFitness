"""训记训练数据 — skills/xunji-training-open-api/SKILL.md"""

from typing import Any

from myfitness.xunji.common import XunjiHttpClient, new_client_request_id
from myfitness.xunji.parsers.training import validate_train_upsert_batch
from myfitness.xunji.skills import (
    PLAN_BASE_URL,
    PLAN_QUERY_PATH,
    PLAN_RATE_LIMIT_SECONDS,
    PLAN_SCHEMA_VERSION,
    TRAINING_BASE_URL,
    TRAINING_RATE_LIMIT_FULL_SECONDS,
    TRAINING_RATE_LIMIT_LIGHT_SECONDS,
    TRAINING_RATE_LIMIT_WRITE_SECONDS,
    TRAINING_READ_PATH,
    TRAINING_SCHEMA_VERSION,
    TRAINING_UPSERT_PATH,
)


class TrainingOpenApi:
    SKILL = "xunji-training-open-api"

    def __init__(self, api_key: str, http: XunjiHttpClient | None = None):
        self.api_key = api_key
        self.http = http or XunjiHttpClient()

    def read(
        self,
        datestr: str,
        *,
        include_full_data: bool = False,
    ) -> dict:
        payload = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "datestr": datestr,
            "include_full_data": include_full_data,
        }
        interval = (
            TRAINING_RATE_LIMIT_FULL_SECONDS if include_full_data else TRAINING_RATE_LIMIT_LIGHT_SECONDS
        )
        return self.http.post(
            f"{TRAINING_BASE_URL}{TRAINING_READ_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=interval,
            require_success=False,
        )

    def upsert(
        self,
        trains: list[dict],
        *,
        dry_run: bool = False,
        include_full_data: bool = False,
        client_request_id: str | None = None,
    ) -> dict:
        validate_train_upsert_batch(trains)

        payload: dict[str, Any] = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "client_request_id": client_request_id or new_client_request_id(),
            "dry_run": dry_run,
            "include_full_data": include_full_data,
            "res": trains,
        }
        result = self.http.post(
            f"{TRAINING_BASE_URL}{TRAINING_UPSERT_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=TRAINING_RATE_LIMIT_WRITE_SECONDS,
            use_cache=False,
            require_success=False,
        )
        if not dry_run:
            self.http.invalidate_cache(TRAINING_READ_PATH)
        return result

    def list_plans(self) -> dict:
        payload = {"schema_version": PLAN_SCHEMA_VERSION, "action": "list"}
        return self.http.post(
            f"{PLAN_BASE_URL}{PLAN_QUERY_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=PLAN_RATE_LIMIT_SECONDS,
            require_success=False,
        )

    def get_plan(
        self,
        plan_ref: str,
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        include_movements: bool = True,
    ) -> dict:
        payload: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "action": "get",
            "plan_ref": plan_ref,
            "include_movements": include_movements,
        }
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        return self.http.post(
            f"{PLAN_BASE_URL}{PLAN_QUERY_PATH}",
            self.api_key,
            payload,
            min_interval_seconds=PLAN_RATE_LIMIT_SECONDS,
            require_success=False,
        )
