"""对话进度回调 — CLI status 展示 Agent / Tool 调用过程。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

ProgressCallback: TypeAlias = Callable[[str], None]


def emit(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


# 用户可见的短标签
TOOL_LABELS: dict[str, str] = {
    "query_body_metrics": "查询身体数据",
    "query_nutrition_logs": "查询饮食记录",
    "query_training_logs": "查询训练记录",
    "load_context": "加载上下文",
    "classify_intent": "识别意图",
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
}


def label_for(key: str) -> str:
    return TOOL_LABELS.get(key, key)
