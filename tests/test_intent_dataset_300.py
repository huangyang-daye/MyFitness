"""300 条意图测评集回归测试。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent, PendingConfirmation

FIXTURE = Path(__file__).parent / "fixtures" / "intent_dataset_300.json"
TODAY = __import__("datetime").date(2026, 8, 23)


@pytest.fixture
def dataset():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_dataset_has_300_samples_with_three_difficulties(dataset):
    assert len(dataset) == 300
    by_diff = {}
    for item in dataset:
        by_diff[item["difficulty"]] = by_diff.get(item["difficulty"], 0) + 1
    assert by_diff == {"simple": 100, "medium": 100, "complex": 100}


def _pending() -> PendingConfirmation:
    return PendingConfirmation(
        action_type="db_write",
        summary="test",
        payload={},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="nutrition",
    )


def _evaluate(dataset: list[dict]) -> dict[str, dict[str, float]]:
    pending = _pending()
    stats: dict[str, dict[str, list[bool]]] = {
        "all": {"primary": [], "full": []},
        "simple": {"primary": [], "full": []},
        "medium": {"primary": [], "full": []},
        "complex": {"primary": [], "full": []},
    }
    for item in dataset:
        use_pending = "confirmation_response" in item["intents"]
        route = classify_intent(
            item["text"],
            pending if use_pending else None,
            use_llm=False,
            today=TODAY,
        )
        predicted = [intent.value for intent in route.intents]
        expected = item["intents"]
        primary_ok = predicted[0] == expected[0] if predicted else False
        full_ok = predicted == expected
        for key in ("all", item["difficulty"]):
            stats[key]["primary"].append(primary_ok)
            stats[key]["full"].append(full_ok)

    def rate(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 1.0

    return {
        tier: {"primary": rate(buckets["primary"]), "full": rate(buckets["full"])}
        for tier, buckets in stats.items()
    }


def test_intent_dataset_300_accuracy_baseline(dataset):
    """关键词 Router 基线：简单集要求更高，复杂集允许较低完整序列命中。"""
    scores = _evaluate(dataset)
    assert scores["simple"]["primary"] >= 0.90
    assert scores["simple"]["full"] >= 0.90
    assert scores["medium"]["primary"] >= 0.65
    assert scores["all"]["primary"] >= 0.65
