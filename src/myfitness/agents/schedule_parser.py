"""从对话解析定时任务（日报 / 同步）。"""

from __future__ import annotations

import re

TASK_LABELS = {
    "daily_report": "每日健康日报",
    "sync": "训记数据同步",
}

TASK_KEYWORDS = {
    "daily_report": ("日报", "健康报告", "每日报告", "晨报"),
    "sync": ("同步", "拉取", "更新训记"),
}


def parse_schedule_request(message: str) -> dict | None:
    """解析创建/修改定时任务。返回 None 表示无法解析。"""
    text = message.strip()
    lower = text.lower()

    if _is_schedule_list_request(text):
        return {"action": "list"}

    cancel_type = _parse_cancel_task_type(text)
    if cancel_type:
        return {"action": "cancel", "task_type": cancel_type}

    if not _looks_like_schedule_create(text):
        return None

    task_type = _infer_task_type(text)
    time_of_day = _parse_time_of_day(text)
    if not time_of_day:
        return None

    return {
        "action": "upsert",
        "task_type": task_type,
        "label": TASK_LABELS[task_type],
        "time_of_day": time_of_day,
        "enabled": True,
    }


def _looks_like_schedule_create(text: str) -> bool:
    if any(k in text for k in ("定时", "定时任务", "每天", "每日", "自动")):
        return True
    return bool(re.search(r"\d{1,2}[:：点时]", text) and any(
        kw in text for labels in TASK_KEYWORDS.values() for kw in labels
    ))


def _is_schedule_list_request(text: str) -> bool:
    return any(k in text for k in ("查看定时", "定时任务列表", "有哪些定时", "我的定时"))


def _parse_cancel_task_type(text: str) -> str | None:
    if not any(k in text for k in ("取消", "关闭", "停用", "删除")):
        return None
    if not any(k in text for k in ("定时", "任务", "日报", "同步")):
        return None
    return _infer_task_type(text)


def _infer_task_type(text: str) -> str:
    for task_type, keywords in TASK_KEYWORDS.items():
        if any(k in text for k in keywords):
            return task_type
    return "daily_report"


def _parse_time_of_day(text: str) -> str | None:
    m = re.search(r"(\d{1,2})[:：](\d{2})", text)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return f"{h:02d}:{mm:02d}"

    m = re.search(r"(\d{1,2})\s*点\s*(\d{1,2})?\s*分?", text)
    if not m:
        m = re.search(r"(\d{1,2})\s*点", text)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2)) if m.lastindex and m.group(2) else 0
        if any(k in text for k in ("下午", "晚上", "傍晚")) and h < 12:
            h += 12
        if "中午" in text and h <= 11:
            h = 12
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return f"{h:02d}:{mm:02d}"
    return None


def format_schedule_confirmation(payload: dict) -> str:
    task_type = payload["task_type"]
    label = payload.get("label") or TASK_LABELS.get(task_type, task_type)
    time_of_day = payload["time_of_day"]
    lines = [
        f"请确认定时任务：",
        f"- 任务：{label}（{task_type}）",
        f"- 执行时间：每天 {time_of_day}",
        "",
        "回复「确认」保存，或「取消」放弃。",
        "（需运行 `myfitness scheduler run` 启动调度器后才会自动执行）",
    ]
    return "\n".join(lines)


def format_schedule_list(tasks: list) -> str:
    if not tasks:
        return "当前没有已保存的定时任务。你可以说：「每天早上 7 点生成日报」。"
    lines = ["当前定时任务："]
    for t in tasks:
        status = "启用" if t.enabled else "停用"
        last = t.last_run_at.isoformat() if t.last_run_at else "从未执行"
        lines.append(f"- [{status}] {t.label} — 每天 {t.time_of_day}（上次：{last}）")
    lines.append("\n修改可说「每天早上7点生成日报」；取消可说「取消日报定时任务」。")
    return "\n".join(lines)
