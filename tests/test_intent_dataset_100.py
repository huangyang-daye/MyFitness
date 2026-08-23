"""100 条意图数据集回归测试。"""

import json
from pathlib import Path

import pytest

from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent, PendingConfirmation
from datetime import UTC, datetime, timedelta

FIXTURE = Path(__file__).parent / "fixtures" / "intent_dataset_100.json"


@pytest.fixture
def dataset():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_dataset_has_100_samples(dataset):
    assert len(dataset) == 100


def test_intent_dataset_accuracy(dataset):
    pending = PendingConfirmation(
        action_type="db_write",
        summary="test",
        payload={},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="nutrition",
    )
    correct = 0
    for item in dataset:
        use_pending = item["intent"] == "confirmation_response"
        route = classify_intent(item["text"], pending if use_pending else None)
        if route.intent.value == item["intent"]:
            correct += 1
    accuracy = correct / len(dataset)
    assert accuracy >= 0.85, f"intent accuracy {accuracy:.0%} < 85%"
