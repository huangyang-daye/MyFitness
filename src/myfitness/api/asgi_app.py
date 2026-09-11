"""FastAPI ASGI 应用 — MyFitness Agent UI 唯一 HTTP 入口。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from myfitness.api.web import (
    STATIC_DIR,
    AgentWebApplication,
    ScheduledTaskNotFound,
    inline_content_disposition,
)
from myfitness.chat_history import ChatHistoryError, ChatSessionNotFound
from myfitness.llm.registry import ModelRegistryError
from myfitness.paths import PROJECT_ROOT
from myfitness.rag.document_parser import DocumentParseError
from myfitness.rag.knowledge_service import KnowledgeError, KnowledgeNotFound
from myfitness.services.artifacts import ArtifactError

logger = logging.getLogger(__name__)


class StreamBody(BaseModel):
    message: str = ""
    session_id: str | None = None


class EditBody(BaseModel):
    message: str = ""
    message_index: int = 0


class MessageBody(BaseModel):
    message: str = ""


def create_app(
    web_app: AgentWebApplication | None = None,
    *,
    start_scheduler: bool = False,
) -> FastAPI:
    """构造 FastAPI 应用；业务逻辑委托 AgentWebApplication。

    start_scheduler=True 时在 lifespan 中启停 APScheduler（生产 UI 进程使用）。
    """
    app_layer = web_app or AgentWebApplication()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        scheduler_started = False
        if start_scheduler:
            try:
                from myfitness.scheduler.manager import start_scheduler as _start

                count = _start()
                scheduler_started = True
                logger.info("Agent UI 调度器已启动，共加载 %d 个任务", count)
            except Exception:
                logger.exception("Agent UI 调度器启动失败")
        yield
        if scheduler_started:
            try:
                from myfitness.scheduler.manager import stop_scheduler

                stop_scheduler()
            except Exception:
                logger.exception("Agent UI 调度器停止失败")

    api = FastAPI(
        title="MyFitness Agent UI",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    api.state.web_app = app_layer

    @api.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        return _error_response(exc)

    @api.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        return app_layer.list_sessions()

    @api.post("/api/sessions", status_code=201)
    async def create_session() -> dict[str, Any]:
        return app_layer.create_session()

    @api.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        return app_layer.session_payload(session_id)

    @api.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: MessageBody) -> dict[str, Any]:
        return app_layer.send_message(session_id, body.message)

    @api.post("/api/sessions/{session_id}/abort")
    async def abort_session(session_id: str) -> dict[str, Any]:
        return app_layer.abort_stream(session_id)

    @api.post("/api/sessions/stream")
    async def stream_session(request: Request, body: StreamBody) -> StreamingResponse:
        session_id = (body.session_id or "").strip() or None
        try:
            app_layer._validated_message(body.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def runner(emit, client_gone):
            return app_layer.stream_message(
                session_id, body.message, emit=emit, client_gone=client_gone
            )

        return _sse_response(request, runner=runner)

    @api.post("/api/sessions/{session_id}/edit")
    async def edit_session(
        request: Request, session_id: str, body: EditBody
    ) -> StreamingResponse:
        try:
            app_layer._validated_message(body.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def runner(emit, client_gone):
            return app_layer.edit_message(
                session_id,
                body.message_index,
                body.message,
                emit=emit,
                client_gone=client_gone,
            )

        return _sse_response(request, runner=runner)

    @api.post("/api/sessions/{session_id}/resume")
    async def resume_session(request: Request, session_id: str) -> StreamingResponse:
        """从 LangGraph / Redis 断点恢复未完成的分析。"""

        def runner(emit, client_gone):
            return app_layer.stream_message(
                session_id, "继续", emit=emit, client_gone=client_gone
            )

        return _sse_response(request, runner=runner)

    @api.get("/api/scheduled-tasks")
    async def list_tasks() -> dict[str, Any]:
        return app_layer.list_scheduled_tasks()

    @api.patch("/api/scheduled-tasks/{task_id}")
    async def patch_task(task_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return app_layer.update_scheduled_task(task_id, body)

    @api.get("/api/models")
    async def list_models() -> dict[str, Any]:
        return app_layer.list_models()

    @api.post("/api/models")
    async def save_model(body: dict[str, Any]) -> dict[str, Any]:
        return app_layer.save_model(body)

    @api.post("/api/models/test")
    async def test_model(body: dict[str, Any]) -> dict[str, Any]:
        return app_layer.test_model(body)

    @api.post("/api/models/{model_id}/activate")
    async def activate_model(model_id: str) -> dict[str, Any]:
        return app_layer.activate_model(model_id)

    @api.delete("/api/models/{model_id}")
    async def delete_model(model_id: str) -> dict[str, Any]:
        return app_layer.delete_model(model_id)

    @api.get("/api/knowledge")
    async def list_knowledge() -> dict[str, Any]:
        return app_layer.list_knowledge()

    @api.post("/api/knowledge", status_code=201)
    async def create_knowledge(body: dict[str, Any]) -> dict[str, Any]:
        return app_layer.create_knowledge(body)

    @api.patch("/api/knowledge/{item_id}")
    async def update_knowledge(item_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return app_layer.update_knowledge(item_id, body)

    @api.delete("/api/knowledge/{item_id}")
    async def delete_knowledge(item_id: int) -> dict[str, Any]:
        return app_layer.delete_knowledge(item_id)

    @api.post("/api/knowledge/reindex")
    async def reindex_knowledge(body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = body or {}
        return app_layer.reindex_knowledge(full=bool(payload.get("full")))

    @api.post("/api/knowledge/parse")
    async def parse_knowledge(file: UploadFile = File(...)) -> dict[str, Any]:
        data = await file.read()
        return app_layer.parse_knowledge_file(file.filename or "untitled", data)

    @api.get("/api/artifact")
    async def read_artifact(path: str = "") -> dict[str, Any]:
        return app_layer.read_artifact_file(path)

    @api.get("/api/artifact/file")
    async def artifact_file(path: str = "") -> Response:
        payload, content_type, filename = app_layer.artifact_file_payload(path)
        return Response(
            content=payload,
            media_type=content_type,
            headers={
                "Content-Disposition": inline_content_disposition(filename),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    if STATIC_DIR.is_dir():
        api.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return api


def app_factory() -> FastAPI:
    """uvicorn --factory 入口（含调度器）。"""
    return create_app(AgentWebApplication(), start_scheduler=True)


def run_web_ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    """启动 FastAPI + uvicorn 本地 Agent UI。"""
    from myfitness.db.sql_logging import configure_sql_logging, is_sql_echo_enabled

    configure_sql_logging()
    if is_sql_echo_enabled():
        print("SQL 查询日志已开启（SQL_ECHO 或 DEBUG_MODE）")

    web_app = AgentWebApplication(project_root)
    asgi = create_app(web_app, start_scheduler=True)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    print(f"MyFitness Agent UI (FastAPI): {url}")
    print("按 Ctrl+C 停止服务")
    uvicorn.run(asgi, host=host, port=port, log_level="info")


def _sse_response(
    request: Request,
    *,
    runner: Callable[[Callable[[str, dict[str, Any]], None], threading.Event], Any],
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    stop = threading.Event()
    client_gone = threading.Event()

    def emit(event: str, payload: dict[str, Any]) -> None:
        if stop.is_set() or client_gone.is_set():
            raise BrokenPipeError("client disconnected")
        future = asyncio.run_coroutine_threadsafe(queue.put((event, payload)), loop)
        future.result(timeout=30)

    def worker() -> None:
        try:
            runner(emit, client_gone)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            logger.info("SSE 客户端断开，已取消上游")
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE 流失败")
            error = str(exc) if isinstance(
                exc,
                (ChatHistoryError, ArtifactError, ValueError, KnowledgeError, DocumentParseError),
            ) else "请求处理失败，请查看服务端日志"
            try:
                emit("error", {"error": error})
            except Exception:
                logger.debug("无法推送 error 事件", exc_info=True)
        finally:
            stop.set()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)

    async def watch_disconnect() -> None:
        while not stop.is_set():
            try:
                if await request.is_disconnected():
                    logger.info("request.is_disconnected=True，取消上游 LLM / 图执行")
                    client_gone.set()
                    stop.set()
                    return
            except Exception:
                logger.debug("is_disconnected 探测失败", exc_info=True)
                return
            await asyncio.sleep(0.25)

    async def event_stream() -> AsyncIterator[bytes]:
        watcher = asyncio.create_task(watch_disconnect())
        thread = threading.Thread(target=worker, name="mf-sse-worker", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, payload = item
                data = json.dumps(payload, ensure_ascii=False)
                yield f"event: {event}\ndata: {data}\n\n".encode()
        finally:
            stop.set()
            client_gone.set()
            watcher.cancel()
            thread.join(timeout=1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "close",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        return JSONResponse(
            {"error": detail if isinstance(detail, str) else str(detail)},
            status_code=int(exc.status_code),
        )
    if isinstance(exc, (ChatSessionNotFound, ScheduledTaskNotFound, KnowledgeNotFound)):
        return JSONResponse({"error": str(exc)}, status_code=404)
    if isinstance(
        exc,
        (
            ChatHistoryError,
            ArtifactError,
            ValueError,
            KnowledgeError,
            DocumentParseError,
            ModelRegistryError,
        ),
    ):
        return JSONResponse({"error": str(exc)}, status_code=400)
    logger.exception("Agent UI request failed")
    return JSONResponse({"error": "请求处理失败，请查看服务端日志"}, status_code=500)
