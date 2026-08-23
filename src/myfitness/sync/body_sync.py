from datetime import UTC, datetime

from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import BodyMetricRepository
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.parsers.body import iter_body_records


def sync_body_metrics(
    session: Session,
    client: XunjiClient,
    user_id: int,
    start_date,
    end_date,
) -> dict[str, int]:
    repo = BodyMetricRepository(session, user_id)
    stats = {"fetched": 0, "upserted": 0, "skipped_manual": 0}

    result = client.body.query_all_records(
        start_date.isoformat(),
        end_date.isoformat(),
    )

    now = datetime.now(UTC)
    for record in iter_body_records(result):
        stats["fetched"] += 1
        upserted = repo.upsert_from_sync(
            record_date=record["record_date"],
            metric_type=record["metric_type"],
            value=record["value"],
            unit=record["unit"],
            xunji_ref=record["xunji_ref"],
            synced_at=now,
        )
        if upserted:
            stats["upserted"] += 1
        else:
            stats["skipped_manual"] += 1

    return stats
