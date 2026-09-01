"""RAG 检索结果格式化。"""

from __future__ import annotations

from myfitness.rag.schemas import RetrievedChunk


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "（无语义检索结果）"

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        date_label = chunk.record_date.isoformat() if chunk.record_date else "未知日期"
        lines.append(
            f"[{index}] {chunk.title}（{chunk.source_type} · {date_label} · "
            f"相似度 {chunk.similarity:.2f}）\n{chunk.content.strip()}"
        )
    return "\n\n".join(lines)
