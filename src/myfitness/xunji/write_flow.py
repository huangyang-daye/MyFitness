"""训记 Skill 写操作确认流程。

所有写回须：dry_run/摘要 → 用户确认 → 正式写入。
详见 skills/xunji-*/SKILL.md 原则章节。
"""

from dataclasses import dataclass
from typing import Callable

from myfitness.xunji.body import BodyOpenApi
from myfitness.xunji.food import FoodOpenApi
from myfitness.xunji.parsers.body import format_body_write_summary
from myfitness.xunji.parsers.food import format_food_write_summary
from myfitness.xunji.parsers.training import format_train_write_summary
from myfitness.xunji.training import TrainingOpenApi


@dataclass
class WritePreview:
    domain: str
    summary: str
    payload: dict | list


ConfirmCallback = Callable[[WritePreview], bool]


def preview_body_write(api: BodyOpenApi, records: list[dict]) -> WritePreview:
    dry_run_result = api.upsert_dry_run(records)
    return WritePreview(
        domain="body",
        summary=format_body_write_summary(dry_run_result),
        payload=records,
    )


def commit_body_write(api: BodyOpenApi, records: list[dict]) -> dict:
    return api.upsert_confirmed(records)


def preview_food_write(api: FoodOpenApi, foods: list[dict], *, dry_run: bool = True) -> WritePreview:
    if dry_run:
        api.upsert_foods(foods, dry_run=True)
    return WritePreview(
        domain="food",
        summary=format_food_write_summary(foods),
        payload=foods,
    )


def commit_food_write(api: FoodOpenApi, foods: list[dict]) -> dict:
    return api.upsert_foods(foods, dry_run=False)


def preview_training_write(api: TrainingOpenApi, trains: list[dict]) -> WritePreview:
    api.upsert(trains, dry_run=True)
    return WritePreview(
        domain="training",
        summary=format_train_write_summary(trains),
        payload=trains,
    )


def commit_training_write(api: TrainingOpenApi, trains: list[dict]) -> dict:
    return api.upsert(trains, dry_run=False)


def write_with_confirmation(
    preview: WritePreview,
    commit_fn: Callable[[], dict],
    confirm: ConfirmCallback,
) -> dict | None:
    if not confirm(preview):
        return None
    return commit_fn()
