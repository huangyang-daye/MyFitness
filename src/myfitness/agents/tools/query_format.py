"""将 DB 查询结果格式化为 Agent / LLM 可读文本。"""

from __future__ import annotations

from typing import Any

from myfitness.xunji.parsers.training import format_movement_sets


def format_query_results(query_results: dict[str, dict[str, Any]]) -> str:
    if not query_results:
        return "（无查询结果）"

    sections: list[str] = []
    body = query_results.get("body")
    if body:
        sections.append(_format_body(body))
    nutrition = query_results.get("nutrition")
    if nutrition:
        sections.append(_format_nutrition(nutrition))
    training = query_results.get("training")
    if training:
        sections.append(_format_training(training))

    return "\n\n".join(sections) if sections else "（查询范围内无记录）"


def _format_body(data: dict) -> str:
    lines = [
        f"【数据库·身体数据】{data['start_date']} ~ {data['end_date']}，共 {data['count']} 条"
    ]
    for r in data.get("records", [])[:50]:
        lines.append(
            f"- {r['date']} {r['metric_type']}: {r['value']}{r['unit']} ({r['source']})"
        )
    if data["count"] > 50:
        lines.append(f"... 其余 {data['count'] - 50} 条省略")
    if data["count"] == 0:
        lines.append("- 无记录")
    return "\n".join(lines)


def _format_nutrition(data: dict) -> str:
    lines = [
        f"【数据库·饮食数据】{data['start_date']} ~ {data['end_date']}，共 {data['count']} 条"
    ]
    for d, totals in sorted(data.get("daily_totals", {}).items()):
        lines.append(
            f"- {d} 合计: {totals['calories']:.0f} kcal，"
            f"蛋白 {totals['protein_g']:.0f}g，"
            f"碳水 {totals['carbs_g']:.0f}g，"
            f"脂肪 {totals['fat_g']:.0f}g"
        )
    for entry in data.get("entries", [])[:30]:
        ntr = entry["nutrients"]
        lines.append(
            f"  · {entry['date']} {entry['meal_type']} {entry['food_name']} "
            f"{entry['amount']}{entry['unit']} → {ntr['calories']:.0f}kcal / "
            f"蛋白{ntr['protein_g']:.0f}g"
        )
    if data["count"] > 30:
        lines.append(f"... 其余 {data['count'] - 30} 条明细省略")
    if data["count"] == 0:
        lines.append("- 无记录")
    return "\n".join(lines)


def _format_training(data: dict) -> str:
    lines = [
        f"【数据库·训练数据】{data['start_date']} ~ {data['end_date']}，共 {data['count']} 次"
    ]
    for s in data.get("sessions", [])[:10]:
        title = s.get("title") or "训练"
        header = f"- {s['date']} {title}"
        extras: list[str] = []
        if s.get("duration_minutes"):
            extras.append(f"时长{s['duration_minutes']}min")
        if s.get("calories"):
            extras.append(f"消耗{s['calories']}kcal")
        if s.get("total_volume_kg"):
            extras.append(f"总容量{s['total_volume_kg']}kg")
        if extras:
            header += f"（{'，'.join(extras)}）"
        lines.append(header)

        for m in s.get("movements", [])[:15]:
            name = m.get("name") or m.get("movement_name") or "未知动作"
            muscle = f"[{m['muscle_type']}]" if m.get("muscle_type") else ""
            if m.get("sets") and isinstance(m["sets"][0], dict) and "reps" in m["sets"][0]:
                sets_text = format_movement_sets(m)
                set_count = m.get("done_set_count") or m.get("set_count") or len(m.get("sets") or [])
            else:
                set_count = m.get("set_count") or len(m.get("sets") or [])
                sets_text = "（明细未解析）"
            lines.append(f"    · {name}{muscle}：{set_count} 组 — {sets_text}")

    if data["count"] > 10:
        lines.append(f"... 其余 {data['count'] - 10} 次省略")
    if data["count"] == 0:
        lines.append("- 无记录")
    return "\n".join(lines)
