"""训记 Skill 与解析器测试。"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from myfitness.xunji.body import BodyOpenApi
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.food import FoodOpenApi, clamp_food_query_range
from myfitness.xunji.parsers.body import iter_body_records, normalize_body_record
from myfitness.xunji.parsers.food import (
    calc_nutrients_from_ntr,
    iter_food_entries,
    parse_search_foods,
)
from myfitness.xunji.parsers.training import extract_trains, validate_train_upsert_batch
from myfitness.xunji.registry import assert_skill_docs_exist, skill_doc_path
from myfitness.xunji.training import TrainingOpenApi


def test_skill_docs_exist():
    assert_skill_docs_exist()
    assert skill_doc_path("xunji-body-open-api").endswith("SKILL.md")


def test_body_query_payload():
    api = BodyOpenApi("test-key")
    with patch.object(api.http, "post", return_value={"records": []}) as mock_post:
        api.query("2026-01-01", "2026-01-31", types=["weight"])
    assert mock_post.call_args[0][2]["types"] == ["weight"]


def test_body_query_all_records_pagination():
    api = BodyOpenApi("test-key")

    def fake_query(*args, **kwargs):
        offset = kwargs.get("offset", 0)
        if offset == 0:
            return {"records": [{"datestr": "2026-01-01", "type": "weight", "value": 70, "unit": "kg"}] * 500}
        return {"records": [{"datestr": "2026-01-02", "type": "weight", "value": 71, "unit": "kg"}]}

    with patch.object(api, "query", side_effect=fake_query):
        result = api.query_all_records("2026-01-01", "2026-01-31")
    assert len(result["records"]) == 501


def test_normalize_body_record():
    record = normalize_body_record({"datestr": "2026-08-20", "type": "weight", "value": 72.0, "unit": "kg"})
    assert record["metric_type"] == "weight"
    assert record["xunji_ref"] == "2026-08-20:weight"


def test_food_search_uses_agent_key_header():
    api = FoodOpenApi("food-key", search_api_key="search-key")
    with patch.object(api.http, "post", return_value={"foods": []}) as mock_post:
        api.search("鸡蛋")
    assert mock_post.call_args.kwargs["auth_style"] == "x-agent-key"


def test_parse_search_foods_compressed():
    result = parse_search_foods({"d": [[1, "鸡蛋", 144, 2, 10, 13, "", "uk1", []]]})
    assert result[0]["name"] == "鸡蛋"
    assert result[0]["uniquekey"] == "uk1"


def test_iter_food_entries_meals_shape():
    payload = {
        "days": [
            {
                "date": "2026-08-20",
                "meals": [
                    {
                        "meal_type": "lunch",
                        "foods": [{"name": "鸡胸肉", "amount": 150, "unit": "g", "ntr": {"cal": 165, "protein": 31, "fat": 3.6, "carb": 0}}],
                    }
                ],
            }
        ]
    }
    entries = list(iter_food_entries(payload))
    assert len(entries) == 1
    assert entries[0]["meal_type"] == "lunch"


def test_extract_trains():
    assert len(extract_trains({"trains": [{"localid": 1}]})) == 1
    assert len(extract_trains([{"localid": 2}])) == 1


def test_validate_train_upsert_batch_rejects_multi_day():
    with pytest.raises(ValueError, match="same datestr"):
        validate_train_upsert_batch(
            [{"datestr": "2026-01-01", "movements": []}, {"datestr": "2026-01-02", "movements": []}]
        )


def test_training_read_no_require_success():
    api = TrainingOpenApi("train-key")
    with patch.object(api.http, "post", return_value={"trains": []}) as mock_post:
        api.read("2026-04-02")
    assert mock_post.call_args.kwargs["require_success"] is False


def test_xunji_client_delegates_to_skill_modules():
    client = XunjiClient(body_api_key="b", food_api_key="f", training_api_key="t")
    assert isinstance(client.body, BodyOpenApi)
    with patch.object(client.body, "query_all_records", return_value={"records": []}) as m:
        client.query_body("2026-01-01", "2026-01-31")
        m.assert_called_once()


def test_clamp_food_query_range():
    today = date.today()
    start, end, notes = clamp_food_query_range(today - timedelta(days=500), today + timedelta(days=200))
    assert start >= today - timedelta(days=365)
    assert end <= today + timedelta(days=90)
    assert notes
