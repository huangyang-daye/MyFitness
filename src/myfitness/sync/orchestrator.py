import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.db.repositories.metrics import SyncJobRepository
from myfitness.db.session import get_or_create_default_user
from myfitness.sync.body_sync import sync_body_metrics
from myfitness.sync.food_sync import sync_nutrition_logs
from myfitness.sync.training_sync import sync_training_logs
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.common import XunjiApiError
from myfitness.xunji.keys import ensure_sync_keys
from myfitness.xunji.skill_keys import resolve_xunji_keys

logger = logging.getLogger(__name__)


def resolve_sync_range(
    session: Session,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
    days: int | None,
) -> tuple[date, date]:
    settings = get_settings()
    end = end_date or (date.today() - timedelta(days=1))
    if start_date:
        start = start_date
    elif days is not None:
        start = end - timedelta(days=days - 1)
    else:
        job_repo = SyncJobRepository(session, user_id)
        last_end = job_repo.last_success_end_date("all")
        if last_end:
            start = last_end + timedelta(days=1)
        else:
            start = end - timedelta(days=settings.sync_default_days - 1)
    if start > end:
        start = end
    return start, end


def build_xunji_client() -> XunjiClient:
    settings = get_settings()
    keys = resolve_xunji_keys(settings)
    return XunjiClient(
        body_api_key=keys["body"],
        food_api_key=keys["food"],
        food_search_key=keys["food_search"],
        training_api_key=keys["training"],
        cache_ttl_seconds=settings.xunji_cache_ttl_seconds,
    )


def run_sync(
    session: Session,
    user_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    days: int | None = 7,
    types: list[str] | None = None,
) -> dict:
    get_or_create_default_user(session, user_id)
    start, end = resolve_sync_range(session, user_id, start_date, end_date, days)
    job_repo = SyncJobRepository(session, user_id)

    sync_types = types or ["body", "food", "training"]
    ensure_sync_keys(sync_types)

    client = build_xunji_client()
    job = job_repo.create("all" if len(sync_types) == 3 else sync_types[0], start, end)

    results: dict[str, dict] = {}
    errors: list[str] = []

    if "body" in sync_types:
        try:
            results["body"] = sync_body_metrics(session, client, user_id, start, end)
        except XunjiApiError as exc:
            errors.append(f"body: {exc}")
            results["body"] = {"error": str(exc)}

    if "food" in sync_types:
        try:
            results["food"] = sync_nutrition_logs(session, client, user_id, start, end)
        except XunjiApiError as exc:
            errors.append(f"food: {exc}")
            results["food"] = {"error": str(exc)}

    if "training" in sync_types:
        try:
            results["training"] = sync_training_logs(session, client, user_id, start, end)
        except XunjiApiError as exc:
            errors.append(f"training: {exc}")
            results["training"] = {"error": str(exc)}

    if errors and len(errors) < len(sync_types):
        status = "partial"
    elif errors:
        status = "failed"
    else:
        status = "success"

    job_repo.finish(
        job,
        status=status,
        stats={"start": start.isoformat(), "end": end.isoformat(), "results": results},
        error_log="\n".join(errors) if errors else None,
    )

    logger.info("Sync %s for %s ~ %s: %s", status, start, end, results)
    result = {
        "status": status,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "results": results,
        "errors": errors,
    }
    if status in {"success", "partial"}:
        try:
            from myfitness.rag.pipeline import maybe_index_after_sync

            maybe_index_after_sync(session, user_id, start, end)
        except Exception as exc:  # noqa: BLE001 - indexing must not break sync
            logger.warning("RAG 增量索引失败: %s", exc)
    return result
