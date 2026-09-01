"""LLM 报告生成 — 根据用户诉求与数据动态撰写报告，替代固定模板。"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from myfitness.agents.summary import build_rule_based_summary
from myfitness.agents.tools.query_format import format_query_results
from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.rag.format import format_retrieved_chunks
from myfitness.rag.schemas import RetrievedChunk
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import ContextSnapshot, Intent

logger = logging.getLogger(__name__)

DEFAULT_DAILY_PROMPT = (
    "请生成一份完整的健康日报，涵盖身体、饮食、训练各方面，"
    "结合数据分析并给出可执行建议。"
)
DEFAULT_PERIOD_PROMPT = (
    "请生成一份完整的健康周期报告，涵盖身体、饮食、训练各方面，"
    "分析区间内的变化趋势并给出建议。"
)


def build_report_messages(
    agent_outputs: AgentOutputs,
    context: ContextSnapshot | None,
    user_message: str,
    *,
    start_date: date,
    end_date: date,
    report_kind: str,
) -> list[dict[str, str]]:
    span_days = (end_date - start_date).days + 1
    request = user_message.strip() or (
        DEFAULT_DAILY_PROMPT if report_kind == "daily" else DEFAULT_PERIOD_PROMPT
    )
    period_label = (
        end_date.isoformat()
        if report_kind == "daily"
        else f"{start_date.isoformat()} ~ {end_date.isoformat()}（{span_days} 天）"
    )

    parts = [
        f"用户诉求：{request}",
        f"报告类型：{'单日日报' if report_kind == 'daily' else '多日周期报告'}",
        f"报告区间：{period_label}",
    ]

    if agent_outputs.body and agent_outputs.body.narrative:
        parts.append(f"【身体 Specialist 分析】\n{agent_outputs.body.narrative}")
        if agent_outputs.body.recommendations:
            parts.append("身体建议：" + "；".join(agent_outputs.body.recommendations))
    if agent_outputs.nutrition and agent_outputs.nutrition.narrative:
        parts.append(f"【饮食 Specialist 分析】\n{agent_outputs.nutrition.narrative}")
    if agent_outputs.fitness and agent_outputs.fitness.narrative:
        parts.append(f"【训练 Specialist 分析】\n{agent_outputs.fitness.narrative}")
    if context and context.query_results:
        parts.append(
            "【数据库查询明细 — 必须基于以下真实数据撰写，禁止编造】\n"
            + format_query_results(context.query_results)
        )
    if context and context.retrieved_chunks:
        parts.append(_format_report_retrieval(context))
    if context and context.data_gaps:
        parts.append("【数据缺口】\n" + "\n".join(f"- {g}" for g in context.data_gaps))

    system = (
        "你是 MyFitness 健康报告撰写 Agent。根据用户诉求、Specialist 分析及数据库真实数据，"
        "用 Markdown 撰写报告正文。\n\n"
        "要求：\n"
        "1. **紧扣用户诉求**：若用户只要体重变化，就聚焦体重，不要写无关的完整日报结构；"
        "若用户要完整日报/综合报告，则全面覆盖身体、饮食、训练。\n"
        "2. **必须引用数据库中的具体数字**（日期、体重、热量、训练次数等），禁止编造。\n"
        "3. 结构灵活：用合适的 Markdown 标题组织，可含表格、列表、对比分析；"
        "不要套用固定「分析摘要 / 原始指标速览」等模板章节名。\n"
        "4. 给出简洁、可执行的观察与建议；不做医疗诊断。\n"
        "5. 不要输出报告标题行（# 开头的主标题）或元数据块（生成时间等）——这些由系统追加。\n"
        "6. 只输出报告正文 Markdown。"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def generate_report_body(
    agent_outputs: AgentOutputs,
    context: ContextSnapshot | None,
    user_message: str,
    *,
    start_date: date,
    end_date: date,
    report_kind: str,
) -> str:
    """LLM 生成报告正文；未配置或失败时回退规则摘要。"""
    if is_llm_configured():
        try:
            messages = build_report_messages(
                agent_outputs,
                context,
                user_message,
                start_date=start_date,
                end_date=end_date,
                report_kind=report_kind,
            )
            content = chat_completion(messages, temperature=0.3)
            if content.strip():
                return content.strip()
            logger.warning("LLM 报告生成为空，使用规则兜底")
        except Exception as exc:  # noqa: BLE001 - LLM failure uses rule fallback
            logger.warning("LLM 报告生成失败，使用规则兜底: %s", exc)

    return build_rule_based_summary(
        agent_outputs, context, Intent.TREND_ANALYSIS, include_query_results=True
    )


def build_report_title(
    *,
    report_kind: str,
    start_date: date,
    end_date: date,
) -> str:
    span_days = (end_date - start_date).days + 1
    if span_days <= 1:
        return f"MyFitness 日报 — {end_date.isoformat()}"
    return (
        f"MyFitness 周期报表 — {start_date.isoformat()} ~ {end_date.isoformat()}"
        f"（{span_days} 天）"
    )


def wrap_report_document(
    *,
    title: str,
    body_md: str,
    start_date: date,
    end_date: date,
    context: ContextSnapshot | None,
    sync_result: dict | None = None,
    charts_md: str = "",
) -> str:
    """为 LLM 正文追加元数据头与可选趋势图。"""
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    span_days = (end_date - start_date).days + 1
    dr = context.date_range if context else None

    sync_line = ""
    if sync_result:
        sync_line = f"\n> 同步状态：{sync_result.get('status', 'unknown')}"

    if span_days <= 1:
        period_line = f"> 报告日：{end_date.isoformat()}{sync_line}\n"
    else:
        period_line = (
            f"> 报告区间：{start_date.isoformat()} ~ {end_date.isoformat()}"
            f"（{span_days} 天）{sync_line}\n"
        )

    data_line = ""
    if dr:
        data_line = f"> 数据覆盖：{dr.start.isoformat()} ~ {dr.end.isoformat()}\n"

    header = f"""# {title}

> 生成时间：{generated_at}
{period_line}{data_line}
---

"""

    charts_block = ""
    if charts_md.strip():
        charts_block = f"\n\n---\n\n## 数据趋势图\n\n{charts_md.strip()}\n"

    return f"{header}{body_md.strip()}{charts_block}\n"


def _format_report_retrieval(context: ContextSnapshot) -> str:
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
        "【语义检索结果 — 可参考以下历史片段，与数据库明细冲突时以数据库为准】\n"
        + format_retrieved_chunks(chunks)
    )
