"""CLI 对话进度 — 与 Web SSE 共用 progress 事件，展示任务 Todo 与当前步骤。"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

TASK_STATUS_LABELS: dict[str, str] = {
    "pending": "待执行",
    "running": "进行中",
    "success": "已完成",
    "failed": "失败",
    "skipped": "已跳过",
    "pending_confirmation": "待确认",
}


class CliTurnProgress:
    """累积一轮对话的 Planner 任务与步骤，供 Rich Live 实时渲染。"""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.task_order: list[str] = []
        self.requirements: str = ""
        self.current_step: str = "正在理解你的问题…"
        self.steps: list[Any] = []

    def handle(self, payload: str | dict[str, Any]) -> None:
        self.steps.append(payload)
        if isinstance(payload, str):
            text = payload.strip()
            if text:
                self.current_step = text.rstrip("…").rstrip(".")
            return

        event_type = str(payload.get("type") or "")
        if event_type == "task_plan":
            self.requirements = str(payload.get("user_requirements") or "")
            self.tasks.clear()
            self.task_order.clear()
            for task in payload.get("tasks") or []:
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id") or "")
                if not task_id:
                    continue
                self.tasks[task_id] = dict(task)
                self.task_order.append(task_id)
            if self.tasks:
                self.current_step = "任务计划已生成，开始执行…"
            return

        if event_type == "task_status":
            task_id = str(payload.get("task_id") or "")
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = str(payload.get("status") or "pending")
                description = str(payload.get("description") or "").strip()
                if description:
                    self.tasks[task_id]["description"] = description
            status = TASK_STATUS_LABELS.get(
                str(payload.get("status") or ""),
                str(payload.get("status") or ""),
            )
            if task_id:
                self.current_step = f"任务 {task_id}：{status}"
            return

        text = str(payload.get("text") or payload.get("type") or "")
        if text:
            self.current_step = text

    def renderable(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self.tasks:
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            table.add_column("#", style="dim", width=3)
            table.add_column("任务", min_width=24)
            table.add_column("域", style="dim", width=10)
            table.add_column("状态", width=8)
            for index, task_id in enumerate(self.task_order, start=1):
                task = self.tasks[task_id]
                status = str(task.get("status") or "pending")
                table.add_row(
                    str(index),
                    str(task.get("description") or task_id),
                    str(task.get("domain") or task.get("intent") or ""),
                    TASK_STATUS_LABELS.get(status, status),
                )
            if self.requirements:
                parts.append(Text(f"任务计划：{self.requirements}", style="bold"))
            parts.append(table)
        parts.append(Text(f"› {self.current_step}", style="cyan"))
        return Group(*parts) if len(parts) > 1 else parts[0]

    def summary_lines(self) -> list[str]:
        """回合结束后的单行摘要（写入 progress_log）。"""
        lines: list[str] = []
        for step in self.steps:
            if isinstance(step, dict):
                if step.get("type") == "task_plan":
                    tasks = step.get("tasks") or []
                    lines.append(
                        "任务计划: "
                        + " | ".join(str(t.get("description", "")) for t in tasks)
                    )
                elif step.get("type") == "task_status":
                    lines.append(f"{step.get('task_id')}:{step.get('status')}")
                else:
                    lines.append(str(step.get("text") or step.get("type") or step))
            else:
                lines.append(str(step).rstrip("…").rstrip("."))
        return lines
