"""SummaryAgent — 汇聚各 Agent 输出为最终回复。"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from myfitness.agents.tools.query_format import format_query_results
from myfitness.agents.tools.web_search import format_web_search_results
from myfitness.debug import trace_agent
from myfitness.llm.factory import is_llm_configured, stream_chat_completion
from myfitness.rag.format import format_retrieved_chunks
from myfitness.rag.schemas import RetrievedChunk
from myfitness.schemas.agent_outputs import AgentOutputs, SummaryAgentOutput
from myfitness.schemas.state import ContextSnapshot, Intent

logger = logging.getLogger(__name__)

_AGENT_NAME_RE = re.compile(
    r"(?:好的[,，]?\s*)?(?:我作为|我是)\s*(?:MyFitness\s*)?"
    r"(?:Summary|Body|Fitness|Nutrition|Document)?\s*Agent[,，]?\s*",
    re.IGNORECASE,
)
_INTERNAL_AGENT_RE = re.compile(
    r"(?:Summary|Body|Fitness|Nutrition|Document|Specialist)\s*Agent",
    re.IGNORECASE,
)


def sanitize_user_facing_reply(text: str) -> str:
    """去掉回复中暴露的内部 Agent 名称。"""
    cleaned = _AGENT_NAME_RE.sub("", text)
    cleaned = _INTERNAL_AGENT_RE.sub("健康助手", cleaned)
    cleaned = re.sub(r"多\s*Agent\s*健康助手", "健康助手", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
        disclaimer="",
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

    rag_section = _format_context_retrieval(context)
    if rag_section:
        sections.append(rag_section)

    web_section = _format_web_search(context)
    if web_section:
        sections.append(web_section)

    memory_section = _format_memory(context)
    if memory_section:
        sections.append(memory_section)

    reflection_section = _format_reflection(context)
    if reflection_section:
        sections.append(reflection_section)

    if intent == Intent.GENERAL:
        greeting = "你好！我是 MyFitness 健康助手，可以帮你查询数据、记录饮食/体重、分析趋势。"
        if not any("MyFitness" in section for section in sections):
            sections.insert(0, greeting)

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
    rag_section = _format_context_retrieval(context)
    if rag_section:
        parts.append(rag_section)
    web_section = _format_web_search(context)
    if web_section:
        parts.append(web_section)
    memory_section = _format_memory(context)
    if memory_section:
        parts.append(memory_section)
    reflection_section = _format_reflection(context)
    if reflection_section:
        parts.append(reflection_section)
    if context and context.data_gaps:
        parts.append("【数据缺口】\n" + "\n".join(f"- {g}" for g in context.data_gaps))

    user_content = "\n\n".join(parts)
    system = (
        "你是 MyFitness 健康助手。根据提供的分析要点、数据库查询明细、联网检索结果"
        "以及用户画像/会话记忆，用简洁清晰的中文回复用户。"
        "要求：必须优先引用数据库查询结果中的具体数字；结合长期画像保持建议连贯；"
        "若有联网检索结果，综合网页资料回答知识性问题，关键结论后标注 [n]，"
        "文末用「参考资料」列出标题和链接；本地用户数据优先于网页；"
        "不做医疗诊断；不要编造未提供的数据；"
        "禁止在回复中出现 Agent、Summary Agent、Specialist 等内部系统名称，"
        "以第一人称「我」直接回答即可。"
    )
    if user_message.strip():
        from myfitness.agents.tools.document_tools import needs_document_write

        if needs_document_write(user_message):
            system += (
                "用户还要求将内容保存为文档文件（如 PDF/Word/Markdown）。"
                "系统会在回复后自动写入文件，你只需在对话中给出分析/建议正文；"
                "禁止声称无法生成、导出或保存文件，也不要指导用户去用其他工具手动保存。"
            )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def should_stream_summary(intent: Intent, user_message: str = "") -> bool:
    """手动录入确认、同步等固定文案不走 LLM 流式。"""
    if user_message:
        from myfitness.agents.tools.document_tools import wants_minimal_chat_for_document

        if wants_minimal_chat_for_document(user_message):
            return False
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
    if is_llm_configured() and should_stream_summary(intent, user_message):
        emitted: list[str] = []
        try:
            messages = build_summary_messages(agent_outputs, context, intent, user_message)
            for chunk in stream_chat_completion(messages):
                emitted.append(chunk)
                yield chunk
            if emitted:
                combined = sanitize_user_facing_reply("".join(emitted))
                if combined != "".join(emitted):
                    # 流式已发出原文字，仅在最终 state 落盘时由 finalize 再清洗
                    pass
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


def _format_context_retrieval(context: ContextSnapshot | None) -> str:
    if not context or not context.retrieved_chunks:
        return ""
    from datetime import date as date_cls

    chunks: list[RetrievedChunk] = []
    for item in context.retrieved_chunks:
        record_date = item.get("record_date")
        parsed_date = date_cls.fromisoformat(record_date) if isinstance(record_date, str) else None
        chunks.append(
            RetrievedChunk(
                id=int(item.get("id", 0)),
                source_type=str(item.get("source_type", "")),
                source_id=str(item.get("source_id", "")),
                domain=str(item.get("domain", "")),
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                record_date=parsed_date,
                similarity=float(item.get("similarity", 0.0)),
                metadata=item.get("metadata"),
            )
        )
    return (
        "【语义检索结果 — 优先引用以下历史片段，注明来源日期/类型；"
        "与数据库明细冲突时以数据库为准】\n"
        + format_retrieved_chunks(chunks)
    )


def _format_web_search(context: ContextSnapshot | None) -> str:
    if not context or not context.web_search_results:
        return ""
    return format_web_search_results(context.web_search_results)


def _format_reflection(context: ContextSnapshot | None) -> str:
    if not context or not context.reflection_notes.strip():
        return ""
    return "【上下文反思 — 已从数据库确认的个体事实】\n" + context.reflection_notes.strip()


def _format_memory(context: ContextSnapshot | None) -> str:
    if not context:
        return ""
    parts: list[str] = []
    if context.memory_long_term.strip():
        parts.append(
            "【用户画像 / 长期记忆 — 回答时保持一致，不要编造画像中没有的事实】\n"
            + context.memory_long_term.strip()
        )
    if context.memory_short_term.strip():
        parts.append("【本轮会话记忆】\n" + context.memory_short_term.strip())
    return "\n\n".join(parts)
