from datetime import timedelta

from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import TrainingLogRepository
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.common import XunjiApiError
from myfitness.xunji.parsers.training import extract_trains, normalize_train_for_sync


def sync_training_logs(
    session: Session,
    client: XunjiClient,
    user_id: int,
    start_date,
    end_date,
    include_full_data: bool = False,
) -> dict[str, int]:
    repo = TrainingLogRepository(session, user_id)
    stats = {"days": 0, "fetched": 0, "upserted": 0, "errors": 0}

    current = start_date
    while current <= end_date:
        stats["days"] += 1
        datestr = current.isoformat()
        try:
            read_result = client.training.read(datestr, include_full_data=include_full_data)
            for train in extract_trains(read_result):
                normalized = normalize_train_for_sync(train, current)
                if not normalized:
                    continue

                stats["fetched"] += 1
                repo.upsert_from_sync(
                    record_date=normalized["record_date"],
                    title=normalized["title"],
                    raw_payload=normalized["raw_payload"],
                    xunji_localid=normalized["xunji_localid"],
                    exercises=normalized["exercises"],
                )
                stats["upserted"] += 1
        except XunjiApiError:
            raise
        except Exception:
            stats["errors"] += 1

        current += timedelta(days=1)

    return stats
