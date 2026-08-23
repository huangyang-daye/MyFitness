import json
from pathlib import Path

from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent

FIXTURES = Path(__file__).parent / "fixtures" / "intent_samples.json"


def test_router_intent_samples_accuracy():
    samples = json.loads(FIXTURES.read_text(encoding="utf-8"))
    correct = 0
    for sample in samples:
        result = classify_intent(sample["text"])
        if result.intent.value == sample["intent"]:
            correct += 1
    accuracy = correct / len(samples)
    assert accuracy >= 0.9, f"router accuracy {accuracy:.0%} < 90%"


def test_router_manual_entry_domains():
    body = classify_intent("记录体重 71kg")
    assert body.intent == Intent.MANUAL_ENTRY
    assert body.domain == "body"

    food = classify_intent("午餐吃了鸡胸肉200g")
    assert food.intent == Intent.MANUAL_ENTRY
    assert food.domain == "nutrition"


def test_router_sync_trigger():
    result = classify_intent("帮我同步训记数据")
    assert result.intent == Intent.SYNC_TRIGGER
