import pytest

from myfitness.config import Settings
from myfitness.xunji.keys import ensure_sync_keys, get_key_statuses, missing_keys_for_sync
from myfitness.xunji.skill_keys import load_keys_from_skills, resolve_xunji_keys


def _settings(**overrides) -> Settings:
    defaults = dict(
        xunji_body_api_key="",
        xunji_food_api_key="",
        xunji_food_search_key="",
        xunji_training_api_key="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_load_keys_from_skill_docs():
    keys = load_keys_from_skills()
    assert keys["body"].startswith("xjbody_")
    assert keys["food"].startswith("xjfood_")
    assert keys["training"].startswith("xjllm_")
    assert keys["food_search"]


def test_resolve_prefers_env_over_skill():
    resolved = resolve_xunji_keys(_settings(xunji_body_api_key="env-body-key"))
    assert resolved["body"] == "env-body-key"


def test_missing_keys_for_sync_uses_skill_docs():
    missing = missing_keys_for_sync(["body", "food", "training"], _settings())
    assert missing == []


def test_missing_keys_when_skill_empty(monkeypatch):
    monkeypatch.setattr(
        "myfitness.xunji.keys.resolve_xunji_keys",
        lambda _s=None: {"body": "", "food": "", "food_search": "", "training": ""},
    )
    missing = missing_keys_for_sync(["body", "training"], _settings())
    assert missing == ["body", "training"]


def test_ensure_sync_keys_raises_when_missing(monkeypatch):
    monkeypatch.setattr(
        "myfitness.xunji.keys.resolve_xunji_keys",
        lambda _s=None: {"body": "", "food": "", "food_search": "", "training": ""},
    )
    with pytest.raises(ValueError, match="xunji-body-open-api"):
        ensure_sync_keys(["body"], _settings())


def test_get_key_statuses_from_skill():
    statuses = get_key_statuses(_settings())
    assert statuses["body"].configured is True
    assert statuses["body"].source == "skill"
    assert statuses["body"].masked.startswith("****")
