"""会话产物读取 — 只允许读取 data_dir 之内的文件。

产物（报告 / 统计图文档）由对话过程生成，前端需要按路径取回内容展示。
路径由前端传入，因此必须校验落在数据目录之内，避免退化成任意文件读取。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from myfitness.config import Settings, get_settings

MAX_CHARS = 400_000  # 约 400 KB，超出截断以免前端渲染卡死


class ArtifactError(ValueError):
    """产物不可读：路径越界 / 不存在 / 不是文件。"""


def artifact_root(settings: Settings | None = None) -> Path:
    return Path((settings or get_settings()).data_dir).resolve()


def resolve_artifact(path: str, settings: Settings | None = None) -> Path:
    """校验产物路径落在 data_dir 之内，返回解析后的绝对路径。"""
    raw = str(path or "").strip()
    if not raw:
        raise ArtifactError("缺少产物路径")
    root = artifact_root(settings)
    try:
        target = Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:  # Windows 非法盘符 / 过长路径
        raise ArtifactError(f"产物路径无效：{raw}") from exc
    if not target.is_relative_to(root):
        raise ArtifactError("产物必须位于数据目录之内")
    if not target.is_file():
        raise ArtifactError("产物不存在或不是文件")
    return target


def read_artifact(path: str, settings: Settings | None = None) -> dict[str, Any]:
    """读取产物内容（UTF-8，超长截断），返回可直接下发给前端的结构。"""
    target = resolve_artifact(path, settings)
    raw = target.read_bytes()
    content = raw.decode("utf-8", errors="replace")
    truncated = len(content) > MAX_CHARS
    if truncated:
        content = content[:MAX_CHARS] + "\n\n…（内容过长，已截断显示）"

    stat = target.stat()
    posix = target.as_posix()
    return {
        "path": posix,
        "name": target.name,
        "title": target.stem,
        "kind": "chart" if "/charts/" in f"/{posix}" else "report",
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "content": content,
        "truncated": truncated,
    }
