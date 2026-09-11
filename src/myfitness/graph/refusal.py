"""无效意图拦截 — 超出健身助手能力范围时硬拒绝，不走检索/编排/生成。"""

from __future__ import annotations

from myfitness.schemas.state import Intent, RouteResult

UNSUPPORTED_REPLY = (
    "这个问题超出了我的能力范围，我无法回答。"
    "我是 MyFitness 健康助手，只能帮你查询和记录身体/饮食/训练数据、"
    "分析趋势、生成日报与统计图、同步训记数据，以及检索公开的健身与营养资料。"
    "请换一个与健身、饮食或身体数据相关的问题。"
)


def should_refuse(route: RouteResult) -> bool:
    """仅当本轮意图全部为 unsupported 时拦截（混有有效意图则继续执行）。"""
    return bool(route.intents) and all(item == Intent.UNSUPPORTED for item in route.intents)
