"""身体数据响应解析 — skills/xunji-body-open-api/SKILL.md"""

from datetime import date, datetime
from typing import Any, Iterator

from myfitness.xunji.skills import BODY_METRIC_TYPES, BODY_UNITS


def parse_record_date(datestr: str) -> date:
    return datetime.strptime(str(datestr)[:10], "%Y-%m-%d").date()


def normalize_body_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Skill: records[] 含 datestr、type、value、unit。"""
    datestr = raw.get("datestr") or raw.get("date")
    metric_type = raw.get("type")
    value = raw.get("value")

    if not datestr or metric_type is None or value is None:
        return None

    metric_type = str(metric_type)
    unit = raw.get("unit") or BODY_UNITS.get(metric_type, "cm")

    return {
        "datestr": str(datestr)[:10],
        "record_date": parse_record_date(str(datestr)),
        "metric_type": metric_type,
        "value": float(value),
        "unit": str(unit),
        "label": raw.get("label"),
        "label_en": raw.get("label_en"),
        "xunji_ref": f"{str(datestr)[:10]}:{metric_type}",
    }


def iter_body_records(query_result: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """从 query 返回的 res 迭代 records。"""
    records = query_result.get("records") or []
    for raw in records:
        normalized = normalize_body_record(raw)
        if normalized:
            yield normalized


def validate_body_write_record(record: dict[str, Any]) -> None:
    metric_type = record.get("type")
    if metric_type not in BODY_METRIC_TYPES:
        raise ValueError(f"unsupported body metric type: {metric_type}")
    if "datestr" not in record or "value" not in record:
        raise ValueError("body write record requires datestr and value")


def format_body_write_summary(dry_run_result: dict[str, Any]) -> str:
    """Skill: 展示 res.summary。"""
    summary = dry_run_result.get("summary")
    if isinstance(summary, str):
        return summary
    if isinstance(summary, list):
        lines = []
        for item in summary:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('datestr')} {item.get('type')}: "
                    f"{item.get('value')} {item.get('unit', '')}".strip()
                )
            else:
                lines.append(str(item))
        return "\n".join(lines) if lines else str(dry_run_result)
    return str(dry_run_result)
