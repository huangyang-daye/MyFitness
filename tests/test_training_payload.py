"""训练 raw_payload 解析测试。"""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.tools.query_format import format_query_results
from myfitness.agents.tools.query_tools import query_training_logs
from myfitness.db.models import Base, TrainingLog, User
from myfitness.xunji.parsers.training import (
    format_movement_sets,
    format_training_session,
    parse_training_payload,
)

SAMPLE_PAYLOAD = {
    "localid": 1787305387305,
    "datestr": "2026-08-21",
    "title": "腿臀",
    "note": "calorie:269 personalworkout_id:1786999087642 personalplanid:286198",
    "start": 1787305391061,
    "end": 1787310285984,
    "started_at": 1787305391061,
    "ended_at": 1787310285984,
    "movements": [
        {
            "index": 1,
            "name": "单腿哑铃硬拉",
            "type": "臀部",
            "exetype": "",
            "sets": [
                {"index": 1, "done": True, "weight": "12.5", "unit": "kg", "reps": "12", "time": 60},
                {"index": 2, "done": True, "weight": "12.5", "unit": "kg", "reps": "12", "time": 60},
            ],
            "truncated": False,
        },
        {
            "index": 2,
            "name": "山羊挺身",
            "type": "背",
            "exetype": "plus_weight",
            "sets": [
                {"index": 1, "done": True, "weight": "", "unit": "kg", "reps": "10", "time": 60},
            ],
            "truncated": False,
        },
    ],
    "truncated": False,
}


def test_parse_training_payload():
    parsed = parse_training_payload(SAMPLE_PAYLOAD)
    assert parsed["title"] == "腿臀"
    assert parsed["calories"] == 269
    assert parsed["movement_count"] == 2
    assert parsed["total_sets"] == 3
    assert parsed["movements"][0]["name"] == "单腿哑铃硬拉"
    assert parsed["movements"][0]["sets"][0]["weight"] == 12.5
    assert parsed["movements"][0]["sets"][0]["reps"] == 12
    assert parsed["movements"][0]["volume_kg"] == 300.0


def test_format_movement_sets():
    parsed = parse_training_payload(SAMPLE_PAYLOAD)
    text = format_movement_sets(parsed["movements"][0])
    assert "12.5kg×12" in text
    bodyweight = format_movement_sets(parsed["movements"][1])
    assert "10次" in bodyweight


def test_format_training_session():
    parsed = parse_training_payload(SAMPLE_PAYLOAD)
    text = format_training_session(parsed)
    assert "腿臀" in text
    assert "单腿哑铃硬拉" in text
    assert "269 kcal" in text


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, name="test")
    session.add(user)
    session.flush()
    yield session
    session.close()


def test_query_training_logs_parses_raw_payload(db_session):
    db_session.add(
        TrainingLog(
            user_id=1,
            record_date=date(2026, 8, 21),
            title="腿臀",
            raw_payload=SAMPLE_PAYLOAD,
            source="xunji_sync",
            xunji_localid="1787305387305",
        )
    )
    db_session.flush()

    result = query_training_logs(db_session, 1, date(2026, 8, 21), date(2026, 8, 21))
    assert result["count"] == 1
    session = result["sessions"][0]
    assert session["title"] == "腿臀"
    assert session["calories"] == 269
    assert session["movements"][0]["name"] == "单腿哑铃硬拉"
    assert session["movements"][0]["sets"][0]["reps"] == 12

    formatted = format_query_results({"training": result})
    assert "12.5kg×12" in formatted
    assert "单腿哑铃硬拉" in formatted
