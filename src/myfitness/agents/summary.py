"""SummaryAgent — 汇聚各 Agent 输出为最终回复。"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from myfitness.agents.tools.query_format import format_query_results
from myfitness.debug import trace_agent
from myfitness.llm.factory import is_llm_configured, stream_chat_completion
from myfitness.schemas.agent_outputs import AgentOutputs, SummaryAgentOutput
from myfitness.schemas.constants import DISCLAIMER
from myfitness.schemas.state import ContextSnapshot, Intent

logger = logging.getLogger(__name__)


@trace_agent("SummaryAgent")
def run_summary_agent(
    agent_outputs: AgentOutputs,
    context: ContextSnapshot | None,
    intent: Intent,
    user_message: str = "",
    output_type: str = "chat_reply",
    *,
    include_query_results: bool = True,
) -> SummaryAgentOutput:
    content_md = build_rule_based_summary(
        agent_outputs, context, intent, include_query_results=include_query_results
    )
    return SummaryAgentOutput(
        output_type=output_type,  # type: ignore[arg-type]
        content_md=content_md,
        data_quality_notes=list(context.data_gaps) if context else [],
        disclaimer=DISCLAIMER,
    )


def build_rule_based_summary(
    agent_outputs: AgentOutputs,
    context: ContextSnapshot | None,
    intent: Intent,
    *,
    include_query_results: bool = True,
) -> str:
    """规则模板摘要。

    include_query_results=False 时不附「数据库查询结果」段——周期报表已有
    每日明细表 / 趋势图，避免重复罗列。
    """
    sections: list[str] = []
    data_notes = list(context.data_gaps) if context else []

    if agent_outputs.body and agent_outputs.body.narrative:
        sections.append(f"**身体**\n{agent_outputs.body.narrative}")
    if agent_outputs.nutrition and agent_outputs.nutrition.narrative:
        sections.append(f"**饮食**\n{agent_outputs.nutrition.narrative}")
    if agent_outputs.fitness and agent_outputs.fitness.narrative:
        sections.append(f"**训练**\n{agent_outputs.fitness.narrative}")

    if include_query_results and context and context.query_results:
        sections.append(f"**数据库查询结果**\n{format_query_results(context.query_results)}")

    if not sections:
        if intent == Intent.GENERAL:
            sections.append(
                "你好！我是 MyFitness 健康助手，可以帮你查询数据、记录饮食/体重、分析趋势。"
            )
        else:
            sections.append("暂无足够数据生成分析，请先同步训记数据或手动录入。")

    if data_notes:
        sections.append("**数据提示**\n" + "\n".join(f"- {n}" for n in data_notes))

    return "\n\n".join(sections)


def build_summary_messages(
    agent_outputs: AgentOutputs,
    context: ContextSnapshot | None,
    intent: Intent,
    user_message: str,
) -> list[dict[str, str]]:
    parts: list[str] = [f"用户问题：{user_message}", f"意图：{intent.value}"]

    if agent_outputs.body:
        parts.append(f"【身体分析】\n{agent_outputs.body.narrative}")
        if agent_outputs.body.recommendations:
            parts.append("建议：" + "；".join(agent_outputs.body.recommendations))
    if agent_outputs.nutrition:
        parts.append(f"【饮食分析】\n{agent_outputs.nutrition.narrative}")
    if agent_outputs.fitness:
        parts.append(f"【训练分析】\n{agent_outputs.fitness.narrative}")
    if context and context.query_results:
        parts.append(
            "【数据库查询明细 — 必须基于以下真实数据回答，禁止编造】\n"
            + format_query_results(context.query_results)
        )
    if context and context.data_gaps:
        parts.append("【数据缺口】\n" + "\n".join(f"- {g}" for g in context.data_gaps))

    user_content = "\n\n".join(parts)
    return [
        {
            "role": "system",
            "content": (
                "你是 MyFitness 多 Agent 健康助手的 Summary Agent。"
                "根据各 Specialist Agent 的结构化分析及数据库查询明细，用简洁清晰的中文回复用户。"
                "要求：必须优先引用数据库查询结果中的具体数字；不做医疗诊断；不要编造未提供的数据；"
                "不要输出免责声明（会由系统追加）。"
            ),
        },
        {"role": "user", "content": user_content},
    ]


def should_stream_summary(intent: Intent) -> bool:
    """手动录入确认、同步等固定文案不走 LLM 流式。"""
    return intent not in {
        Intent.MANUAL_ENTRY,
        Intent.CONFIRMATION_RESPONSE,
        Intent.SYNC_TRIGGER,
    }


@trace_agent("SummaryAgent.stream")
def iter_summary_reply(
    agent_outputs: AgentOutputs,
    context: ContextSnapshot | None,
    intent: Intent,
    user_message: str = "",
) -> Iterator[str]:
    """LLM 流式生成 Summary；失败/熔断/未配置时回退规则模板。

    兜底链：LLM 流式 → （部分输出则原样结束）→ 规则模板。
    """
    if is_llm_configured() and should_stream_summary(intent):
        emitted: list[str] = []
        try:
            messages = build_summary_messages(agent_outputs, context, intent, user_message)
            for chunk in stream_chat_completion(messages):
                emitted.append(chunk)
                yield chunk
            if emitted:
                return
            # 空响应 → 规则兜底
            logger.warning("LLM 返回空回复，使用规则模板兜底")
        except Exception as exc:  # noqa: BLE001 - streaming failures use rule fallback
            logger.warning("LLM 流式失败，使用规则模板兜底: %s", exc)
            if not emitted:
                yield build_rule_based_summary(agent_outputs, context, intent)
                return
            # 已有部分输出，追加提示后结束
            yield "\n\n（回复因 LLM 服务中断而不完整）"
            return

    yield build_rule_based_summary(agent_outputs, context, intent)
