"""File-backed chat session persistence.

Each conversation is stored as one JSON document.  The filename and the state's
``session_id`` are the same canonical UUID, which makes restoring a conversation
deterministic and keeps the store portable.

对话记录属于「使用记录」，与项目本体分离：默认落在 settings.chat_history_dir
（即 <DATA_DIR>/chat-history），不再写入项目目录下的 .chatHistory。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from myfitness.config import get_settings
from myfitness.schemas.state import MyFitnessGraphState

SCHEMA_VERSION = 1
_EXPORT_TAIL_RE = re.compile(
    r"[，,]?\s*(?:产出|保存|导出|输出|写成|整理成|生成).{0,48}"
    r"(?:pdf|docx|md|word|markdown|文档).*$",
    re.IGNORECASE,
)


class ChatHistoryError(ValueError):
    """Raised when a conversation id or history document is invalid."""


class ChatSessionNotFound(ChatHistoryError):
    """Raised when the requested conversation does not exist."""


@dataclass(frozen=True)
class ChatSessionSummary:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "preview": self.preview,
        }


class ChatHistoryStore:
    """Persist and restore :class:`MyFitnessGraphState` by UUID."""

    def __init__(self, history_dir: str | Path | None = None) -> None:
        target = history_dir if history_dir is not None else get_settings().chat_history_dir
        self.history_dir = Path(target).expanduser().resolve()

    @staticmethod
    def normalize_session_id(session_id: str) -> str:
        try:
            parsed = uuid.UUID(str(session_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ChatHistoryError("session id 必须是有效的 UUID") from exc
        if parsed.version != 4:
            raise ChatHistoryError("session id 必须是 UUID v4")
        return str(parsed)

    def path_for(self, session_id: str) -> Path:
        canonical = self.normalize_session_id(session_id)
        return self.history_dir / f"{canonical}.json"

    def save(self, state: MyFitnessGraphState) -> Path:
        session_id = self.normalize_session_id(state.session_id)
        if state.session_id != session_id:
            state.session_id = session_id

        existing = self._read_document(self.path_for(session_id), missing_ok=True)
        now = datetime.now(UTC).isoformat()
        created_at = (existing or {}).get("created_at") or self._created_at_for(state, now)
        document = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "title": self._title_for(state),
            "created_at": created_at,
            "updated_at": now,
            "state": state.model_dump(mode="json"),
        }

        self.history_dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(session_id)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.history_dir,
                prefix=f".{session_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temp_name = handle.name
            os.replace(temp_name, target)
        finally:
            if temp_name and Path(temp_name).exists():
                Path(temp_name).unlink()
        return target

    def load(self, session_id: str) -> MyFitnessGraphState:
        canonical = self.normalize_session_id(session_id)
        document = self._read_document(self.path_for(canonical))
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ChatHistoryError("不支持的会话历史版本")
        if document.get("session_id") != canonical:
            raise ChatHistoryError("会话文件名与文档 session_id 不一致")
        try:
            state = MyFitnessGraphState.model_validate(document["state"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ChatHistoryError("会话历史内容损坏") from exc
        if state.session_id != canonical:
            raise ChatHistoryError("会话状态中的 session_id 不一致")
        return state

    def get(self, session_id: str) -> dict[str, Any]:
        canonical = self.normalize_session_id(session_id)
        document = self._read_document(self.path_for(canonical))
        # Validate before returning data to callers such as the Web API.
        self.load(canonical)
        return document

    def list_sessions(self) -> list[ChatSessionSummary]:
        if not self.history_dir.exists():
            return []
        summaries: list[ChatSessionSummary] = []
        for path in self.history_dir.glob("*.json"):
            try:
                document = self._read_document(path)
                state = MyFitnessGraphState.model_validate(document["state"])
                canonical = self.normalize_session_id(document["session_id"])
                if path.stem != canonical or state.session_id != canonical:
                    continue
                messages = state.messages
                first_user = next((m.content.strip() for m in messages if m.role == "user"), "")
                preview = summarize_session_title(first_user) if first_user else ""
                summaries.append(
                    ChatSessionSummary(
                        session_id=canonical,
                        title=summarize_session_title(first_user) if first_user else str(document.get("title") or "新对话"),
                        created_at=str(document.get("created_at") or ""),
                        updated_at=str(document.get("updated_at") or ""),
                        message_count=len(messages),
                        preview=preview,
                    )
                )
            except (ChatHistoryError, KeyError, OSError, ValueError, TypeError):
                # One damaged file must not make the whole sidebar unavailable.
                continue
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def _read_document(self, path: Path, *, missing_ok: bool = False) -> dict[str, Any] | None:
        if not path.is_file():
            if missing_ok:
                return None
            raise ChatSessionNotFound(f"未找到会话：{path.stem}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChatHistoryError(f"无法读取会话历史：{path.name}") from exc
        if not isinstance(value, dict):
            raise ChatHistoryError("会话历史根节点必须是 JSON object")
        return value

    @staticmethod
    def _created_at_for(state: MyFitnessGraphState, fallback: str) -> str:
        started = state.metadata.started_at
        return started.isoformat() if started else fallback

    @staticmethod
    def _title_for(state: MyFitnessGraphState) -> str:
        first_user = next((m.content.strip() for m in state.messages if m.role == "user"), "")
        if not first_user:
            return "新对话"
        return summarize_session_title(first_user)


def summarize_session_title(message: str) -> str:
    """把用户首条消息压缩为侧栏可读的会话摘要。"""
    text = _EXPORT_TAIL_RE.sub("", message.strip()).strip(" ，,。.")
    if not text:
        text = message.strip()

    if re.search(r"训练建议|训练计划|怎么练|训练记录", text):
        if re.search(r"减重|减脂|减肥", text):
            return "减重期训练建议"
        return "训练建议"
    if re.search(r"饮食规划|饮食计划|吃什么|营养", text):
        return "饮食规划"
    if re.search(r"减肥|减脂|减重", text):
        return "减重管理"
    if re.search(r"体重|体脂|围度", text):
        return "身体数据"

    for sep in ("，", ",", "。", "？", "?", "；", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    text = re.sub(r"^(?:根据最近的?|根据|请|帮我|给我|能不能|可以)", "", text).strip()
    one_line = " ".join(text.split())
    if not one_line:
        return "新对话"
    if len(one_line) <= 24:
        return one_line
    return f"{one_line[:24]}…"
