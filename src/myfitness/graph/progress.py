"""对话进度回调 — CLI / Web SSE 展示 Agent / Tool / 任务计划执行过程。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from myfitness.graph.task_plan import TaskPlan

ProgressPayload: TypeAlias = str | dict[str, Any]
ProgressCallback: TypeAlias = Callable[[ProgressPayload], None]


def emit(on_progress: ProgressCallback | None, message: ProgressPayload) -> None:
    if on_progress is not None:
        on_progress(message)


def emit_plan(on_progress: ProgressCallback | None, plan: TaskPlan) -> None:
    if on_progress is None:
        return
    emit(
        on_progress,
        {
            "type": "task_plan",
            "user_requirements": plan.user_requirements,
            "tasks": [
                {
                    "id": task.id,
                    "description": task.description or task.intent.value,
                    "intent": task.intent.value,
                    "domain": task.domain,
                    "status": "pending",
                }
                for task in plan.tasks
            ],
        },
    )


def emit_task_status(
    on_progress: ProgressCallback | None,
    task_id: str,
    status: str,
    *,
    description: str = "",
) -> None:
    if on_progress is None:
        return
    emit(
        on_progress,
        {
            "type": "task_status",
            "task_id": task_id,
            "status": status,
            "description": description,
        },
    )


def emit_step(on_progress: ProgressCallback | None, message: str) -> None:
    """用户可见的当前步骤（兼容旧版纯文本 progress）。"""
    emit(on_progress, message)


# 用户可见的短标签
TOOL_LABELS: dict[str, str] = {
    "query_body_metrics": "查询身体数据",
    "query_nutrition_logs": "查询饮食记录",
    "query_training_logs": "查询训练记录",
    "load_context": "加载上下文",
    "classify_intent": "识别意图",
    "planner": "Planner 拆分任务",
    "context_reflection": "上下文反思",
    "judge": "Judge 质量评估",
    "body_monitor": "BodyMonitor 分析中",
    "nutritionist": "Nutritionist 分析中",
    "fitness_planner": "FitnessPlanner 分析中",
    "summary": "Summary 生成回复中",
    "manual_entry": "解析手动录入",
    "confirmation": "处理确认操作",
    "sync": "同步训记数据",
    "schedule": "管理定时任务",
    "daily_report": "生成日报",
    "period_report": "生成周期报表",
    "chart": "生成统计图",
    "memory": "整理记忆",
    "web_search": "联网检索",
    "read_document": "读取文档",
    "write_document": "生成文档",
    "list_documents": "列出文档",
}


def label_for(key: str) -> str:
    return TOOL_LABELS.get(key, key)
