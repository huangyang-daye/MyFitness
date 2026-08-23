"""训练数据响应解析 — skills/xunji-training-open-api/SKILL.md"""

import re
from datetime import date, datetime
import re
from typing import Any

from myfitness.xunji.skills import (
    DIFFICULTY_VALUES,
    RPE_VALUES,
    TRAINING_UPSERT_MAX_MOVEMENTS,
    TRAINING_UPSERT_MAX_SETS,
    TRAINING_UPSERT_MAX_TRAINS,
)


def parse_record_date(datestr: str) -> date:
    return datetime.strptime(str(datestr)[:10], "%Y-%m-%d").date()


def extract_trains(read_result: dict[str, Any] | list) -> list[dict[str, Any]]:
    """Skill: 核心数据在 res.trains；res 也可能是训练数组。"""
    if isinstance(read_result, list):
        return read_result
    trains = read_result.get("trains")
    if isinstance(trains, list):
        return trains
    return []


def parse_exercise_summaries(train: dict[str, Any]) -> list[dict[str, Any]]:
    exercises = []
    for movement in train.get("movements") or []:
        name = movement.get("name") or "unknown"
        sets = movement.get("sets") or []
        exercises.append(
            {
                "movement_name": name,
                "set_count": len(sets),
                "sets_detail": sets[:TRAINING_UPSERT_MAX_SETS],
                "difficulty": movement.get("difficulty"),
            }
        )
    return exercises


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_set_detail(raw_set: dict[str, Any]) -> dict[str, Any]:
    """解析单组训练数据。"""
    weight = _parse_float(raw_set.get("weight"))
    reps = _parse_int(raw_set.get("reps"))
    volume_kg = weight * reps if weight is not None and reps is not None else None
    return {
        "set_index": raw_set.get("index"),
        "done": bool(raw_set.get("done", True)),
        "weight": weight,
        "weight_display": str(raw_set.get("weight") or "").strip(),
        "unit": raw_set.get("unit") or "kg",
        "reps": reps,
        "reps_display": str(raw_set.get("reps") or "").strip(),
        "rest_seconds": _parse_int(raw_set.get("time")),
        "volume_kg": volume_kg,
        "self_weight": bool(raw_set.get("selfWeight")),
    }


def parse_movement_detail(raw_movement: dict[str, Any]) -> dict[str, Any]:
    """解析单个动作及其各组明细。"""
    sets = [parse_set_detail(s) for s in raw_movement.get("sets") or []]
    done_sets = [s for s in sets if s.get("done", True)]
    volume_kg = sum(s["volume_kg"] for s in done_sets if s.get("volume_kg") is not None)
    reps_values = [s["reps"] for s in done_sets if s.get("reps") is not None]

    return {
        "index": raw_movement.get("index"),
        "name": raw_movement.get("name") or "unknown",
        "muscle_type": raw_movement.get("type") or "",
        "exercise_type": raw_movement.get("exetype") or "",
        "set_count": len(sets),
        "done_set_count": len(done_sets),
        "sets": sets,
        "volume_kg": round(volume_kg, 1) if volume_kg else 0.0,
        "rep_range": f"{min(reps_values)}-{max(reps_values)}" if reps_values else "",
    }


def _parse_calories_from_note(note: str | None) -> int | None:
    if not note:
        return None
    match = re.search(r"calorie:(\d+)", note)
    return int(match.group(1)) if match else None


def _calc_duration_minutes(raw: dict[str, Any]) -> int | None:
    start = raw.get("started_at") or raw.get("start")
    end = raw.get("ended_at") or raw.get("end")
    if start is None or end is None:
        return None
    try:
        seconds = (int(end) - int(start)) / 1000
        return max(0, round(seconds / 60))
    except (TypeError, ValueError):
        return None


def parse_training_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    """解析 training_logs.raw_payload 完整训练 JSON。"""
    movements = [parse_movement_detail(m) for m in raw_payload.get("movements") or []]
    total_sets = sum(m["set_count"] for m in movements)
    total_volume = round(sum(m["volume_kg"] for m in movements), 1)
    duration = _calc_duration_minutes(raw_payload)

    return {
        "localid": raw_payload.get("localid"),
        "date": str(raw_payload.get("datestr") or "")[:10],
        "title": raw_payload.get("title") or "训练",
        "note": raw_payload.get("note") or "",
        "calories": _parse_calories_from_note(raw_payload.get("note")),
        "duration_minutes": duration,
        "movement_count": len(movements),
        "total_sets": total_sets,
        "total_volume_kg": total_volume,
        "movements": movements,
        "truncated": bool(raw_payload.get("truncated")),
    }


def format_movement_sets(movement: dict[str, Any]) -> str:
    """格式化单个动作的组次明细，如：12.5kg×12, 12.5kg×12, ..."""
    parts: list[str] = []
    for s in movement.get("sets") or []:
        if not s.get("done", True):
            continue
        weight = s.get("weight_display") or (str(s["weight"]) if s.get("weight") is not None else "")
        reps = s.get("reps_display") or (str(s["reps"]) if s.get("reps") is not None else "")
        unit = s.get("unit") or "kg"
        if weight and reps:
            parts.append(f"{weight}{unit}×{reps}")
        elif reps:
            parts.append(f"{reps}次")
    return ", ".join(parts) if parts else "（无完成组次）"


def format_training_session(parsed: dict[str, Any]) -> str:
    """将解析后的训练会话格式化为可读文本。"""
    lines = [f"{parsed.get('date')} {parsed.get('title')}（{parsed.get('movement_count')} 个动作）"]
    if parsed.get("duration_minutes"):
        lines[0] += f"，时长约 {parsed['duration_minutes']} 分钟"
    if parsed.get("calories"):
        lines[0] += f"，消耗约 {parsed['calories']} kcal"
    if parsed.get("total_volume_kg"):
        lines[0] += f"，总容量约 {parsed['total_volume_kg']} kg"

    for m in parsed.get("movements") or []:
        muscle = f"[{m['muscle_type']}]" if m.get("muscle_type") else ""
        sets_text = format_movement_sets(m)
        lines.append(
            f"  · {m['name']}{muscle}：{m['done_set_count']} 组 — {sets_text}"
        )
    return "\n".join(lines)


def normalize_train_for_sync(train: dict[str, Any], fallback_date: date) -> dict[str, Any] | None:
    localid = train.get("localid")
    if localid is None:
        return None

    datestr = train.get("datestr") or fallback_date.isoformat()
    return {
        "record_date": parse_record_date(str(datestr)),
        "title": train.get("title"),
        "raw_payload": train,
        "xunji_localid": str(localid),
        "exercises": parse_exercise_summaries(train),
    }


def validate_train_upsert_batch(trains: list[dict[str, Any]]) -> None:
    """Skill: 单次最多 4 条且必须同一天；每条最多 15 动作、20 组。"""
    if not trains:
        raise ValueError("empty train batch")
    if len(trains) > TRAINING_UPSERT_MAX_TRAINS:
        raise ValueError(f"at most {TRAINING_UPSERT_MAX_TRAINS} trains per upsert")

    dates = {t.get("datestr") for t in trains if t.get("datestr")}
    if len(dates) > 1:
        raise ValueError("all trains in one upsert must share the same datestr")

    for train in trains:
        movements = train.get("movements") or []
        if len(movements) > TRAINING_UPSERT_MAX_MOVEMENTS:
            raise ValueError(f"at most {TRAINING_UPSERT_MAX_MOVEMENTS} movements per train")
        for movement in movements:
            sets = movement.get("sets") or []
            if len(sets) > TRAINING_UPSERT_MAX_SETS:
                raise ValueError(f"at most {TRAINING_UPSERT_MAX_SETS} sets per movement")

            difficulty = movement.get("difficulty")
            if difficulty is not None and difficulty not in DIFFICULTY_VALUES:
                raise ValueError(f"invalid difficulty: {difficulty}")

            for s in sets:
                rpe = s.get("rpe")
                if rpe is not None and str(rpe) not in RPE_VALUES:
                    raise ValueError(f"invalid rpe: {rpe}")


def format_train_write_summary(trains: list[dict[str, Any]]) -> str:
    lines = []
    for train in trains:
        title = train.get("title") or "训练"
        datestr = train.get("datestr", "")
        movement_names = [m.get("name") for m in train.get("movements") or [] if m.get("name")]
        lines.append(f"- {datestr} {title}: {', '.join(movement_names) or '(无动作)'}")
    return "\n".join(lines)
