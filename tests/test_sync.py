from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, BodyMetric, NutritionLog
from myfitness.db.repositories.metrics import (
    BodyMetricRepository,
    NutritionLogRepository,
    SOURCE_MANUAL,
    SOURCE_XUNJI,
)
from myfitness.sync.body_sync import sync_body_metrics
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.common import mask_api_key
from myfitness.xunji.food import clamp_food_query_range
from myfitness.xunji.parsers.food import calc_nutrients_from_ntr, iter_food_entries
from myfitness.xunji.parsers.training import parse_exercise_summaries


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_body_manual_priority(db_session):
    from myfitness.db.models import User

    user = User(id=1, name="test")
    db_session.add(user)
    db_session.flush()

    repo = BodyMetricRepository(db_session, user_id=1)
    repo.session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 20),
            metric_type="weight",
            value=72.0,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    upserted = repo.upsert_from_sync(
        record_date=date(2026, 8, 20),
        metric_type="weight",
        value=73.0,
        unit="kg",
        xunji_ref="2026-08-20:weight",
    )
    assert upserted is False

    metric = db_session.query(BodyMetric).one()
    assert float(metric.value) == 72.0
    assert metric.source == SOURCE_MANUAL


def test_body_sync_from_api(db_session):
    from myfitness.db.models import User

    user = User(id=1, name="test")
    db_session.add(user)
    db_session.flush()

    client = XunjiClient(body_api_key="test-key")
    mock_response = {
        "records": [
            {"datestr": "2026-08-20", "type": "weight", "value": 71.5, "unit": "kg"},
            {"datestr": "2026-08-20", "type": "bodyfat", "value": 18.2, "unit": "%"},
        ]
    }

    with patch.object(client.body, "query_all_records", return_value=mock_response):
        stats = sync_body_metrics(db_session, client, 1, date(2026, 8, 20), date(2026, 8, 20))

    assert stats["fetched"] == 2
    assert stats["upserted"] == 2
    assert db_session.query(BodyMetric).count() == 2


def test_calc_nutrients():
    ntr = {"cal": 165, "protein": 31, "fat": 3.6, "carb": 0}
    result = calc_nutrients_from_ntr(ntr, amount=200)
    assert result["cal"] == 330.0
    assert result["protein"] == 62.0


def test_iter_food_entries_days_shape():
    payload = {
        "days": [
            {
                "date": "2026-08-20",
                "foods": [
                    {"name": "鸡胸肉", "amount": 150, "unit": "g", "meal_type": "lunch"},
                ],
            }
        ]
    }
    entries = list(iter_food_entries(payload))
    assert len(entries) == 1
    assert entries[0]["food_name"] == "鸡胸肉"


def test_iter_food_entries_xunji_foods_records_shape():
    payload = {
        "days": [
            {
                "date": "2026-08-25",
                "foods": {
                    "records": [
                        {
                            "record_id": "14",
                            "meal_type": "noon",
                            "name": "米饭",
                            "amount": 50,
                            "unit": "g",
                            "uniquekey": "/shiwu/mifan_zheng",
                            "ntr": {"cal": 116, "protein": 2.6, "fat": 0.3, "carb": 25.9},
                        }
                    ]
                },
            }
        ]
    }

    entries = list(iter_food_entries(payload))

    assert len(entries) == 1
    assert entries[0]["record_date"] == date(2026, 8, 25)
    assert entries[0]["meal_type"] == "lunch"
    assert entries[0]["food_name"] == "米饭"
    assert entries[0]["xunji_record_id"].startswith("food:2026-08-25:lunch:14:")


def test_nutrition_sync_key_does_not_collapse_same_food_across_dates(db_session):
    from myfitness.db.models import User

    user = User(id=1, name="test")
    db_session.add(user)
    db_session.flush()

    repo = NutritionLogRepository(db_session, user_id=1)
    repo.upsert_from_sync(
        record_date=date(2026, 8, 24),
        meal_type="lunch",
        food_name="米饭",
        amount=50,
        unit="g",
        nutrients_snapshot={"cal": 58},
        food_id=None,
        xunji_record_id="food:2026-08-24:lunch:14:/shiwu/mifan_zheng:g",
    )
    repo.upsert_from_sync(
        record_date=date(2026, 8, 25),
        meal_type="dinner",
        food_name="米饭",
        amount=100,
        unit="g",
        nutrients_snapshot={"cal": 116},
        food_id=None,
        xunji_record_id="food:2026-08-25:dinner:14:/shiwu/mifan_zheng:g",
    )
    db_session.flush()

    rows = db_session.query(NutritionLog).order_by(NutritionLog.record_date).all()
    assert len(rows) == 2
    assert [row.record_date for row in rows] == [date(2026, 8, 24), date(2026, 8, 25)]


def test_parse_exercises():
    train = {
        "movements": [
            {"name": "杠铃卧推", "sets": [{"weight": "60", "reps": "10"}, {"weight": "65", "reps": "8"}]},
        ]
    }
    exercises = parse_exercise_summaries(train)
    assert len(exercises) == 1
    assert exercises[0]["movement_name"] == "杠铃卧推"
    assert exercises[0]["set_count"] == 2


def test_mask_api_key():
    assert mask_api_key("abcdefghijklmnop") == "****mnop"
