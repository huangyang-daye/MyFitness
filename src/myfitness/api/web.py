"""Dependency-free local Web UI server for MyFitness Agent."""

from __future__ import annotations

import json
import logging
import re
import threading
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from myfitness.chat_history import (
    ChatHistoryError,
    ChatHistoryStore,
    ChatSessionNotFound,
)
from myfitness.config import get_settings
from myfitness.db.repositories.reports import ScheduledTaskRepository
from myfitness.db.session import get_or_create_default_user, session_scope
from myfitness.graph.chat import (
    finalize_streamed_reply,
    iter_chat_turn,
    new_chat_state,
    run_chat_turn,
)
from myfitness.llm.factory import LlmConfig, probe_llm_config
from myfitness.llm.registry import ModelRegistryError, get_registry
from myfitness.paths import PROJECT_ROOT
from myfitness.rag.document_parser import MAX_FILE_BYTES, DocumentParseError, parse_document
from myfitness.rag.knowledge_service import KnowledgeError, KnowledgeNotFound
from myfitness.services.artifacts import ArtifactError, read_artifact

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parents[1] / "web_static"
PROBE_TIMEOUT = 30
MAX_JSON_BYTES = 100_000
MAX_UPLOAD_BYTES = MAX_FILE_BYTES + 64_000
TASK_TYPES = {
    ScheduledTaskRepository.TASK_DAILY_REPORT: "生成健康日报",
    ScheduledTaskRepository.TASK_SYNC: "同步训记数据",
}


def parse_multipart_file(body: bytes, content_type: str) -> tuple[str, bytes]:
    """从 multipart/form-data 取出名为 file 的上传内容。"""
    match = re.search(r"boundary=([^;]+)", content_type or "", re.IGNORECASE)
    if not match:
        raise ValueError("请使用 multipart/form-data 上传文件")
    boundary = match.group(1).strip().strip('"')
    delim = b"--" + boundary.encode("ascii", "replace")
    parts = body.split(delim)
    for raw_part in parts:
        part = raw_part.lstrip(b"\r\n")
        if not part or part.startswith(b"--"):
            continue
        header_blob, separator, payload = part.partition(b"\r\n\r\n")
        if not separator:
            header_blob, separator, payload = part.partition(b"\n\n")
        if not separator:
            continue
        headers = header_blob.decode("utf-8", "replace")
        disposition = ""
        for line in headers.splitlines():
            if line.lower().startswith("content-disposition:"):
                disposition = line
                break
        if not disposition:
            continue
        filename = _multipart_filename(disposition)
        name_match = re.search(r'\bname="?([^";]+)"?', disposition, re.IGNORECASE)
        field_name = name_match.group(1).strip() if name_match else ""
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        elif payload.endswith(b"\n"):
            payload = payload[:-1]
        if field_name == "file" or filename:
            return filename or "untitled", payload
    raise ValueError("未找到上传文件，请使用字段名 file")


def _multipart_filename(disposition: str) -> str:
    star = re.search(r"filename\*=(?:UTF-8''|utf-8'')([^;]+)", disposition, re.IGNORECASE)
    if star:
        return Path(unquote(star.group(1).strip().strip('"'))).name
    normal = re.search(r'filename="([^"]*)"|filename=([^;]+)', disposition, re.IGNORECASE)
    if not normal:
        return ""
    raw = (normal.group(1) or normal.group(2) or "").strip().strip('"')
    return Path(unquote(raw)).name


class ScheduledTaskNotFound(ValueError):
    pass


class AgentWebApplication:
    """Small application layer shared by the HTTP handler and tests."""

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        *,
        history_dir: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        # 对话记录是运行时数据，落在 <DATA_DIR>/chat-history，不占用项目目录
        self.history = ChatHistoryStore(history_dir)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def list_sessions(self) -> dict[str, Any]:
        return {"sessions": [item.as_dict() for item in self.history.list_sessions()]}

    def create_session(self) -> dict[str, Any]:
        settings = get_settings()
        state = new_chat_state(user_id=settings.default_user_id)
        self.history.save(state)
        return self.session_payload(state.session_id)

    def session_payload(self, session_id: str) -> dict[str, Any]:
        document = self.history.get(session_id)
        state = document["state"]
        return {
            "session_id": document["session_id"],
            "title": document["title"],
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
            "messages": state.get("messages", []),
            "pending_confirmation": state.get("pending_confirmation"),
            "metadata": state.get("metadata", {}),
        }

    def list_scheduled_tasks(self) -> dict[str, Any]:
        settings = get_settings()
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            rows = ScheduledTaskRepository(session, settings.default_user_id).list_all()
            tasks = [self._scheduled_task_payload(task) for task in rows]
        return {
            "tasks": tasks,
            "task_types": TASK_TYPES,
            "timezone": "Asia/Shanghai",
            "scheduler_running": self._scheduler_running(),
        }

    def update_scheduled_task(self, task_id: int, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"task_type", "label", "time_of_day", "enabled"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"不支持的定时任务字段：{', '.join(sorted(unknown))}")
        if not changes:
            raise ValueError("没有可提交的修改")

        values: dict[str, Any] = {}
        if "task_type" in changes:
            task_type = str(changes["task_type"]).strip()
            if task_type not in TASK_TYPES:
                raise ValueError("任务内容只支持生成健康日报或同步训记数据")
            values["task_type"] = task_type
        if "label" in changes:
            label = str(changes["label"]).strip()
            if not label or len(label) > 128:
                raise ValueError("任务名称长度必须为 1～128 个字符")
            values["label"] = label
        if "time_of_day" in changes:
            time_of_day = str(changes["time_of_day"]).strip()
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", time_of_day):
                raise ValueError("执行时间必须是 HH:MM 格式")
            values["time_of_day"] = time_of_day
        if "enabled" in changes:
            if not isinstance(changes["enabled"], bool):
                raise ValueError("enabled 必须是布尔值")
            values["enabled"] = changes["enabled"]

        settings = get_settings()
        try:
            with session_scope() as session:
                repo = ScheduledTaskRepository(session, settings.default_user_id)
                task = repo.update(task_id, **values)
                if task is None:
                    raise ScheduledTaskNotFound(f"未找到定时任务：{task_id}")
                result = self._scheduled_task_payload(task)
        except ScheduledTaskNotFound:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "uk_scheduled_task_user_type" in message or "unique" in message:
                raise ValueError("同类型的定时任务已经存在，不能修改为重复内容") from exc
            raise

        scheduler_error = None
        try:
            from myfitness.scheduler.manager import reload_scheduler_jobs

            reload_scheduler_jobs(settings.default_user_id)
        except Exception as exc:
            logger.exception("定时任务已保存，但调度器重载失败")
            scheduler_error = str(exc)
        return {
            "task": result,
            "scheduler_running": self._scheduler_running(),
            "scheduler_error": scheduler_error,
        }

    def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        text = self._validated_message(message)
        lock = self._lock_for(session_id)
        with lock:
            state = self.history.load(session_id)
            progress: list[str] = []
            with session_scope() as session:
                get_or_create_default_user(session, state.user_id)
                state = run_chat_turn(session, state, text, on_progress=progress.append)
            self.history.save(state)
            payload = self.session_payload(state.session_id)
            payload.update({"reply": state.reply, "progress": progress})
            return payload

    def stream_message(
        self,
        session_id: str | None,
        message: str,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """处理一轮对话并通过回调推送 SSE 事件。

        未提供 session_id 时先在内存中创建会话，等本轮用户消息写入后再注册到
        历史仓库，避免打开空白页就落下一个空会话文件。
        """
        text = self._validated_message(message)
        created = not session_id
        if created:
            settings = get_settings()
            state = new_chat_state(user_id=settings.default_user_id)
            canonical = state.session_id
        else:
            canonical = self.history.normalize_session_id(str(session_id))
            state = None

        def emit_event(event: str, payload: dict[str, Any]) -> None:
            if emit is not None:
                emit(event, payload)

        lock = self._lock_for(canonical)
        with lock:
            if not created:
                state = self.history.load(canonical)
            assert state is not None
            progress: list[str] = []

            def on_progress(msg: str) -> None:
                progress.append(msg)
                emit_event("progress", {"text": msg})

            with session_scope() as session:
                get_or_create_default_user(session, state.user_id)
                state, chunks = iter_chat_turn(
                    session, state, text, on_progress=on_progress
                )
                # 首轮用户消息已经写入 state，此时才落盘并出现在侧栏。
                self.history.save(state)
                emit_event("session", self.session_payload(state.session_id))
                reply_parts: list[str] = []
                for chunk in chunks:
                    if not chunk:
                        continue
                    reply_parts.append(chunk)
                    emit_event("delta", {"text": chunk})
                finalize_streamed_reply(state, "".join(reply_parts))
            self.history.save(state)
            payload = self.session_payload(state.session_id)
            payload.update({"reply": state.reply, "progress": progress})
            emit_event("done", payload)
            return payload

    @staticmethod
    def _validated_message(message: str) -> str:
        text = str(message).strip()
        if not text:
            raise ValueError("消息不能为空")
        if len(text) > 20_000:
            raise ValueError("消息不能超过 20000 个字符")
        return text

    # ------------------------------------------------------------------- 产物
    def read_artifact_file(self, path: str) -> dict[str, Any]:
        """读取会话产物内容；路径必须落在 data_dir 之内。"""
        return read_artifact(path)

    # ------------------------------------------------------------------- 模型
    def list_models(self) -> dict[str, Any]:
        return get_registry().public_payload()

    def save_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = get_registry().upsert(payload)
        return {"model": preset.public_dict(), **get_registry().public_payload()}

    def delete_model(self, model_id: str) -> dict[str, Any]:
        get_registry().delete(model_id)
        return get_registry().public_payload()

    def activate_model(self, model_id: str) -> dict[str, Any]:
        get_registry().set_active(model_id)
        return get_registry().public_payload()

    def test_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        """连通性测试：可测已保存模型（给 id），也可测尚未保存的表单值。"""
        body = payload or {}
        model_id = str(body.get("id") or "").strip()
        preset = get_registry().get(model_id) if model_id else None
        if model_id and preset is None:
            raise ModelRegistryError(f"未找到模型：{model_id}")

        api_key = str(body.get("api_key") or "").strip() or (preset.api_key if preset else "")
        base_url = str(body.get("base_url") or "").strip() or (preset.base_url if preset else "")
        model_name = str(body.get("model") or "").strip() or (preset.model if preset else "")
        try:
            timeout = int(body.get("timeout") or (preset.timeout if preset else PROBE_TIMEOUT))
        except (TypeError, ValueError):
            timeout = PROBE_TIMEOUT

        if not api_key:
            raise ValueError("请先填写 API Key")
        if not base_url:
            raise ValueError("请先填写 Base URL")
        if not model_name:
            raise ValueError("请先填写模型标识")

        config = LlmConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model_name,
            temperature=0.0,
            max_tokens=None,
            timeout=max(5, min(timeout, PROBE_TIMEOUT)),
        )
        try:
            result = probe_llm_config(config, prompt="回复 OK")
        except Exception as exc:  # noqa: BLE001 - 探测失败是预期分支，回传给前端展示
            logger.info("模型连通性测试失败: %s", exc)
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **result}

    # ------------------------------------------------------------------- 知识库
    def list_knowledge(self) -> dict[str, Any]:
        from myfitness.rag.knowledge_service import list_knowledge

        settings = get_settings()
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            return list_knowledge(session, settings.default_user_id)

    def create_knowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        from myfitness.rag.knowledge_service import create_knowledge

        settings = get_settings()
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            return create_knowledge(
                session,
                settings.default_user_id,
                title=str(payload.get("title") or ""),
                content=str(payload.get("content") or ""),
            )

    def parse_knowledge_file(self, filename: str, data: bytes) -> dict[str, Any]:
        parsed = parse_document(filename, data)
        return {
            "title": parsed.title,
            "content": parsed.content,
            "filename": parsed.filename,
            "format": parsed.format,
            "truncated": parsed.truncated,
            "char_count": parsed.char_count,
        }

    def update_knowledge(self, entry_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        from myfitness.rag.knowledge_service import update_knowledge

        settings = get_settings()
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            changes: dict[str, str] = {}
            if "title" in payload:
                changes["title"] = str(payload.get("title") or "")
            if "content" in payload:
                changes["content"] = str(payload.get("content") or "")
            if not changes:
                raise ValueError("没有可提交的修改")
            return update_knowledge(session, settings.default_user_id, entry_id, **changes)

    def delete_knowledge(self, entry_id: int) -> dict[str, Any]:
        from myfitness.rag.knowledge_service import delete_knowledge

        settings = get_settings()
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            return delete_knowledge(session, settings.default_user_id, entry_id)

    def reindex_knowledge(self, *, full: bool = False) -> dict[str, Any]:
        from myfitness.rag.knowledge_service import reindex_all_data, reindex_all_knowledge

        settings = get_settings()
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            if full:
                return reindex_all_data(session, settings.default_user_id)
            return reindex_all_knowledge(session, settings.default_user_id)

    def _lock_for(self, session_id: str) -> threading.Lock:
        canonical = self.history.normalize_session_id(session_id)
        with self._locks_guard:
            return self._locks.setdefault(canonical, threading.Lock())

    @staticmethod
    def _scheduled_task_payload(task: Any) -> dict[str, Any]:
        return {
            "id": task.id,
            "task_type": task.task_type,
            "content_label": TASK_TYPES.get(task.task_type, task.task_type),
            "label": task.label,
            "time_of_day": task.time_of_day,
            "enabled": bool(task.enabled),
            "last_run_at": task.last_run_at.isoformat() if task.last_run_at else None,
            "last_status": task.last_status,
            "last_error": task.last_error,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }

    @staticmethod
    def _scheduler_running() -> bool:
        try:
            from myfitness.scheduler.manager import scheduler_running

            return scheduler_running()
        except Exception:  # noqa: BLE001 - optional scheduler dependency/status
            return False


class AgentUiRequestHandler(BaseHTTPRequestHandler):
    server: AgentUiHttpServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._json({"status": "ok"})
                return
            if parsed.path == "/api/sessions":
                self._json(self.server.app.list_sessions())
                return
            if parsed.path == "/api/scheduled-tasks":
                self._json(self.server.app.list_scheduled_tasks())
                return
            if parsed.path.startswith("/api/sessions/"):
                session_id = unquote(parsed.path.removeprefix("/api/sessions/"))
                self._json(self.server.app.session_payload(session_id))
                return
            if parsed.path == "/api/models":
                self._json(self.server.app.list_models())
                return
            if parsed.path == "/api/knowledge":
                self._json(self.server.app.list_knowledge())
                return
            if parsed.path == "/api/artifact":
                query = parse_qs(parsed.query)
                self._json(
                    self.server.app.read_artifact_file(query.get("path", [""])[0])
                )
                return
            self._serve_static(parsed.path)
        except Exception as exc:  # noqa: BLE001 - HTTP exception boundary
            self._handle_error(exc)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/knowledge/"):
                raw_id = parsed.path.removeprefix("/api/knowledge/").strip("/")
                if not raw_id.isdigit():
                    raise ValueError("知识条目 id 无效")
                self._json(
                    self.server.app.update_knowledge(int(raw_id), self._read_json())
                )
                return
            if parsed.path.startswith("/api/scheduled-tasks/"):
                raw_id = parsed.path.removeprefix("/api/scheduled-tasks/").strip("/")
                if not raw_id.isdigit():
                    raise ValueError("定时任务 id 无效")
                self._json(
                    self.server.app.update_scheduled_task(int(raw_id), self._read_json())
                )
                return
            self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - HTTP exception boundary
            self._handle_error(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/sessions/stream":
                body = self._read_json()
                session_id = str(body.get("session_id") or "").strip() or None
                self._stream_message(session_id, body.get("message", ""))
                return
            if parsed.path == "/api/sessions":
                self._json(self.server.app.create_session(), status=HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/messages"):
                session_id = unquote(
                    parsed.path.removeprefix("/api/sessions/").removesuffix("/messages")
                ).strip("/")
                body = self._read_json()
                self._json(self.server.app.send_message(session_id, body.get("message", "")))
                return
            if parsed.path == "/api/knowledge/reindex":
                body = self._read_json()
                self._json(
                    self.server.app.reindex_knowledge(full=bool(body.get("full")))
                )
                return
            if parsed.path == "/api/knowledge/parse":
                filename, file_bytes = self._read_multipart_file()
                self._json(self.server.app.parse_knowledge_file(filename, file_bytes))
                return
            if parsed.path == "/api/knowledge":
                self._json(
                    self.server.app.create_knowledge(self._read_json()),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/models":
                self._json(self.server.app.save_model(self._read_json()))
                return
            if parsed.path == "/api/models/test":
                self._json(self.server.app.test_model(self._read_json()))
                return
            if parsed.path.startswith("/api/models/") and parsed.path.endswith("/activate"):
                model_id = unquote(
                    parsed.path.removeprefix("/api/models/").removesuffix("/activate")
                ).strip("/")
                self._json(self.server.app.activate_model(model_id))
                return
            self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - HTTP exception boundary
            self._handle_error(exc)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/knowledge/"):
                raw_id = parsed.path.removeprefix("/api/knowledge/").strip("/")
                if not raw_id.isdigit():
                    raise ValueError("知识条目 id 无效")
                self._json(self.server.app.delete_knowledge(int(raw_id)))
                return
            if parsed.path.startswith("/api/models/"):
                model_id = unquote(parsed.path.removeprefix("/api/models/")).strip("/")
                if not model_id:
                    raise ValueError("模型 id 无效")
                self._json(self.server.app.delete_model(model_id))
                return
            self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - HTTP exception boundary
            self._handle_error(exc)

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body(MAX_JSON_BYTES)
        try:
            value = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError("请求必须是合法 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON 根节点必须是 object")  # noqa: TRY004
        return value

    def _read_body(self, max_bytes: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length 无效") from exc
        if length < 0:
            raise ValueError("Content-Length 无效")
        if length > max_bytes:
            raise ValueError("请求内容过大")
        return self.rfile.read(length) if length else b""

    def _read_multipart_file(self) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        filename, payload = parse_multipart_file(
            self._read_body(MAX_UPLOAD_BYTES),
            content_type,
        )
        if not payload:
            raise ValueError("文件为空")
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError(f"文件不能超过 {MAX_FILE_BYTES // (1024 * 1024)}MB")
        return filename, payload

    def _serve_static(self, request_path: str) -> None:
        name = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC_DIR / name).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self._json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(target.suffix.lower(), "application/octet-stream")
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _stream_message(self, session_id: str | None, message: str) -> None:
        try:
            self.server.app._validated_message(message)
        except Exception as exc:  # noqa: BLE001 - HTTP exception boundary
            self._handle_error(exc)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True
        try:
            self.server.app.stream_message(session_id, message, emit=self._write_sse)
        except Exception as exc:
            logger.exception("Agent UI stream failed")
            error = str(exc) if isinstance(
                exc, (ChatHistoryError, ArtifactError, ValueError)
            ) else "请求处理失败，请查看服务端日志"
            try:
                self._write_sse("error", {"error": error})
            except Exception:
                logger.debug("无法写入 SSE 错误事件", exc_info=True)

    def _write_sse(self, event: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        frame = f"event: {event}\ndata: {data}\n\n".encode()
        self.wfile.write(frame)
        self.wfile.flush()

    def _json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, (ChatSessionNotFound, ScheduledTaskNotFound, KnowledgeNotFound)):
            self._json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        elif isinstance(
            exc,
            (ChatHistoryError, ArtifactError, ValueError, KnowledgeError, DocumentParseError),
        ):
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        else:
            logger.exception("Agent UI request failed")
            self._json({"error": "请求处理失败，请查看服务端日志"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("Agent UI: " + format, *args)


class AgentUiHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: AgentWebApplication) -> None:
        self.app = app
        super().__init__(address, AgentUiRequestHandler)


def run_web_ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    """Run the local UI until interrupted."""
    # Bind first. If the port is occupied, fail before APScheduler creates a
    # background thread that could keep an otherwise failed process alive.
    app = AgentWebApplication(project_root)
    server = AgentUiHttpServer((host, port), app)
    scheduler_started = False
    try:
        from myfitness.scheduler.manager import start_scheduler

        count = start_scheduler()
        scheduler_started = True
        logger.info("Agent UI 调度器已启动，共加载 %d 个任务", count)
    except Exception:
        logger.exception("Agent UI 调度器启动失败")

    url = f"http://{host}:{server.server_port}"
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    print(f"MyFitness Agent UI: {url}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if scheduler_started:
            try:
                from myfitness.scheduler.manager import stop_scheduler

                stop_scheduler()
            except Exception:
                logger.exception("Agent UI 调度器停止失败")
