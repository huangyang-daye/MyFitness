import logging
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import typer
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.chart_tools import generate_chart, resolve_metric
from myfitness.chat_history import ChatHistoryError, ChatHistoryStore
from myfitness.config import get_settings
from myfitness.db.repositories.reports import DailyReportRepository, ScheduledTaskRepository
from myfitness.db.session import get_or_create_default_user, session_scope
from myfitness.debug import configure_debug_logging
from myfitness.graph.chat import (
    finalize_streamed_reply,
    iter_chat_turn,
    new_chat_state,
    run_chat_turn,
)
from myfitness.llm.factory import (
    LlmConfig,
    LlmWarmupResult,
    get_llm_config,
    is_llm_configured,
    probe_llm_config,
    probe_llm_connection,
    warmup_llm,
)
from myfitness.llm.guard import get_llm_guard
from myfitness.llm.registry import (
    ENV_PRESET_ID,
    PROVIDER_PRESETS,
    ModelPreset,
    ModelRegistryError,
    get_registry,
)
from myfitness.paths import PROJECT_ROOT
from myfitness.services.artifacts import ArtifactError, read_artifact
from myfitness.services.daily_report import run_daily_report
from myfitness.services.period_report import run_period_report
from myfitness.sync.orchestrator import run_sync
from myfitness.xunji.keys import get_key_statuses, missing_keys_for_sync

app = typer.Typer(help="MyFitness — 多 Agent 健康监控 CLI")
db_app = typer.Typer(help="数据库管理")
llm_app = typer.Typer(help="LLM 配置与测试")
xunji_app = typer.Typer(help="训记 Open API 配置")
report_app = typer.Typer(help="健康日报 / 周期报表")
chart_app = typer.Typer(help="Mermaid 统计图")
scheduler_app = typer.Typer(help="定时任务调度")
session_app = typer.Typer(help="历史对话管理")
artifact_app = typer.Typer(help="报表与图表产物查看")
rag_app = typer.Typer(help="RAG 语义检索（pgvector）")
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
app.add_typer(xunji_app, name="xunji")
app.add_typer(report_app, name="report")
app.add_typer(chart_app, name="chart")
app.add_typer(scheduler_app, name="scheduler")
app.add_typer(session_app, name="session")
app.add_typer(artifact_app, name="artifact")
app.add_typer(rag_app, name="rag")

console = Console()
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def _setup_logging(debug: bool | None = None) -> None:
    settings = get_settings()
    debug_enabled = bool(getattr(settings, "debug_mode", False)) if debug is None else debug
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=debug_enabled,
    )
    configure_debug_logging(debug_enabled)
    if debug_enabled:
        logging.getLogger(__name__).debug("Debug mode enabled")


@app.callback()
def main(
    debug: bool | None = typer.Option(
        None,
        "--debug/--no-debug",
        help="打印 Agent、Tool 调用及意图识别结果（也可设置 DEBUG_MODE=true）",
    ),
) -> None:
    _setup_logging(debug)


@db_app.command("migrate")
def db_migrate() -> None:
    """执行 Alembic 数据库迁移到最新版本。"""
    root = PROJECT_ROOT
    alembic_ini = root / "alembic.ini"
    if not alembic_ini.exists():
        console.print("[red]未找到 alembic.ini[/red]")
        raise typer.Exit(1)

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(alembic_ini), "upgrade", "head"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
    console.print("[green]数据库迁移完成[/green]")


def _parse_date_option(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


@rag_app.command("init")
def rag_init() -> None:
    """初始化 pgvector 扩展与 RAG 表/索引。"""
    from myfitness.db.session import get_engine
    from myfitness.rag.pgvector_setup import ensure_rag_schema, is_postgresql

    engine = get_engine()
    if not is_postgresql(engine):
        console.print("[red]RAG 需要 PostgreSQL 数据库（当前非 PostgreSQL）[/red]")
        raise typer.Exit(1)
    try:
        ensure_rag_schema(engine)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print("[green]RAG 表与 pgvector 扩展已就绪[/green]")


@rag_app.command("index")
def rag_index(
    full: bool = typer.Option(False, "--full", help="全量重建索引（忽略日期范围）"),
    user_id: int = typer.Option(1, "--user-id", help="用户 ID"),
    start: str | None = typer.Option(None, "--start", help="起始日期 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束日期 YYYY-MM-DD"),
) -> None:
    """将身体/饮食/训练/报告数据索引到 pgvector。"""
    from myfitness.rag.indexer import index_user_data

    start_date = _parse_date_option(start)
    end_date = _parse_date_option(end)
    with session_scope() as session:
        get_or_create_default_user(session, user_id)
        result = index_user_data(
            session,
            user_id,
            start_date=start_date,
            end_date=end_date,
            full=full,
        )
    table = Table(title="RAG 索引结果")
    table.add_column("字段")
    table.add_column("值")
    for key, value in result.items():
        table.add_row(str(key), str(value))
    console.print(table)
    if result.get("status") == "skipped":
        raise typer.Exit(1)


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="检索语句"),
    user_id: int = typer.Option(1, "--user-id", help="用户 ID"),
    top_k: int = typer.Option(5, "--top-k", help="返回条数"),
) -> None:
    """语义检索 RAG 向量库。"""
    from myfitness.rag.format import format_retrieved_chunks
    from myfitness.rag.store import search_chunks

    with session_scope() as session:
        get_or_create_default_user(session, user_id)
        chunks = search_chunks(session, user_id, query, top_k=top_k)
    if not chunks:
        console.print("[yellow]未找到匹配片段（请先 myfitness rag index）[/yellow]")
        raise typer.Exit(1)
    console.print(Markdown(format_retrieved_chunks(chunks)))


@rag_app.command("stats")
def rag_stats(user_id: int = typer.Option(1, "--user-id", help="用户 ID")) -> None:
    """查看 RAG 索引统计。"""
    from myfitness.rag.pgvector_setup import rag_is_available
    from myfitness.rag.store import chunk_stats, count_chunks

    with session_scope() as session:
        get_or_create_default_user(session, user_id)
        available = rag_is_available(session)
        total = count_chunks(session, user_id)
        stats = chunk_stats(session, user_id)
    table = Table(title="RAG 统计")
    table.add_column("字段")
    table.add_column("值")
    table.add_row("可用", "是" if available else "否")
    table.add_row("总块数", str(total))
    for source_type, count in sorted(stats.items()):
        table.add_row(source_type, str(count))
    console.print(table)


@app.command("sync")
def sync(
    days: int | None = typer.Option(7, help="同步最近 N 天（默认 7）"),
    start: str | None = typer.Option(None, "--start", help="起始日期 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束日期 YYYY-MM-DD"),
    types: str | None = typer.Option(None, help="同步类型，逗号分隔：body,food,training"),
) -> None:
    """从训记 Open API 同步数据到 PostgreSQL。"""
    settings = get_settings()
    sync_types = [t.strip() for t in types.split(",")] if types else None
    start_date = _parse_date_option(start)
    end_date = _parse_date_option(end)
    sync_types = sync_types or ["body", "food", "training"]
    missing = missing_keys_for_sync(sync_types)
    if missing:
        statuses = get_key_statuses()
        console.print("[red]训记鉴权未就绪，无法同步。[/red]")
        for name in missing:
            console.print(f"  - {statuses[name].hint}")
        console.print(
            "\n训记 App 通过 Skill 文档下发 Token，不再单独提供 API Key。"
            "\n运行 [bold]myfitness xunji keys[/bold] 查看来源与状态。"
        )
        raise typer.Exit(1)

    with session_scope() as session:
        result = run_sync(
            session,
            user_id=settings.default_user_id,
            start_date=start_date,
            end_date=end_date,
            days=days if start_date is None else None,
            types=sync_types,
        )

    table = Table(title="同步结果")
    table.add_column("字段")
    table.add_column("值")
    table.add_row("状态", result["status"])
    table.add_row("日期范围", f"{result['start_date']} ~ {result['end_date']}")

    for domain, stats in result["results"].items():
        if isinstance(stats, dict):
            detail = ", ".join(f"{k}={v}" for k, v in stats.items())
            table.add_row(domain, detail)

    if result["errors"]:
        table.add_row("错误", "\n".join(result["errors"]))

    console.print(table)

    if result["status"] == "failed":
        raise typer.Exit(1)


@xunji_app.command("keys")
def xunji_keys() -> None:
    """显示训记鉴权状态（从 Skill 文档或 .env 读取，脱敏）。"""
    statuses = get_key_statuses()
    table = Table(title="训记鉴权状态")
    table.add_column("类型")
    table.add_column("来源")
    table.add_column("状态")
    table.add_column("值")

    labels = {
        "body": "身体数据",
        "food": "饮食数据",
        "food_search": "食物搜索（可选）",
        "training": "训练数据",
    }
    for name, status in statuses.items():
        state = "[green]已配置[/green]" if status.configured else "[yellow]未配置[/yellow]"
        table.add_row(labels.get(name, name), status.source_label, state, status.masked)

    console.print(table)

    sync_missing = missing_keys_for_sync(["body", "food", "training"])
    if sync_missing:
        console.print("\n[yellow]同步还需配置：[/yellow]")
        for name in sync_missing:
            console.print(f"  - {statuses[name].hint}")
    else:
        console.print("\n[green]同步所需鉴权已就绪（Skill 文档或 .env）。[/green]")


@report_app.command("generate")
def report_generate(
    report_date: str | None = typer.Option(None, "--date", help="报告日期 YYYY-MM-DD，默认昨天"),
    start: str | None = typer.Option(None, "--start", help="周期报表起始日期 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="周期报表结束日期 YYYY-MM-DD"),
    no_sync: bool = typer.Option(False, "--no-sync", help="跳过训记同步"),
) -> None:
    """立即生成健康日报；给出 --start/--end 时生成区间周期报表（含趋势图）。"""
    settings = get_settings()
    d = date.fromisoformat(report_date) if report_date else None
    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    is_period = bool(start_date or end_date)
    with session_scope() as session:
        get_or_create_default_user(session, settings.default_user_id)
        with console.status("[bold cyan]正在生成报表…[/bold cyan]", spinner="dots"):
            if is_period:
                result = run_period_report(
                    session,
                    settings.default_user_id,
                    start_date=start_date,
                    end_date=end_date,
                    sync_first=not no_sync,
                )
            else:
                result = run_daily_report(
                    session,
                    settings.default_user_id,
                    report_date=d,
                    sync_first=not no_sync,
                )

    if result.get("report_kind") == "period":
        label = f"{result['period_start']} ~ {result['period_end']}（{result['period_days']} 天）"
        console.print(f"[green]周期报表已生成[/green]：{label}")
        if result.get("charts"):
            console.print("趋势图：" + "、".join(c["title"] for c in result["charts"]))
    else:
        console.print(f"[green]日报已生成[/green]：{result['report_date']}")
    if result.get("file_path"):
        console.print(f"文件：{result['file_path']}")
    console.print(result["content_md"][:1500])
    if len(result["content_md"]) > 1500:
        console.print("\n…（已截断）")


@chart_app.command("show")
def chart_show(
    metric: str = typer.Option(
        "weight", "--metric", help="指标：weight/bodyfat/calories/protein_g/volume_kg …"
    ),
    days: int = typer.Option(7, "--days", help="最近 N 天（未给 --start/--end 时生效）"),
    start: str | None = typer.Option(None, "--start", help="起始日期 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束日期 YYYY-MM-DD"),
    chart_type: str = typer.Option("line", "--type", help="line（折线）| bar（柱状）"),
    save: bool = typer.Option(False, "--save", help="同时保存为独立 Markdown 文档"),
) -> None:
    """生成 Mermaid 统计图（横轴日期、纵轴数值）。"""
    settings = get_settings()
    domain, label, _unit = resolve_metric(metric)
    end_date = date.fromisoformat(end) if end else datetime.now(LOCAL_TZ).date()
    start_date = date.fromisoformat(start) if start else end_date - timedelta(days=days - 1)

    with session_scope() as session:
        result = invoke_tool(
            generate_chart,
            session,
            settings.default_user_id,
            domain=domain,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            chart_type=chart_type,
            output_mode="document" if save else "inline",
        )

    if result.get("is_empty"):
        console.print(f"[yellow]{start_date} ~ {end_date} 内没有{label}记录。[/yellow]")
        raise typer.Exit(1)

    console.print(result["markdown"])
    if save and result.get("path"):
        console.print(f"[green]已保存[/green]：{result['path']}")


@chart_app.command("insert")
def chart_insert(
    file: str = typer.Argument(..., help="目标 Markdown 文档路径"),
    metric: str = typer.Option("weight", "--metric", help="指标名"),
    days: int = typer.Option(7, "--days", help="最近 N 天"),
    start: str | None = typer.Option(None, "--start", help="起始日期 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束日期 YYYY-MM-DD"),
    chart_type: str = typer.Option("line", "--type", help="line | bar"),
    anchor: str | None = typer.Option(
        None, "--anchor", help="插入到该小节下，例如「## 趋势」；默认追加到末尾"
    ),
) -> None:
    """生成统计图并插入已有文档（如某天的日报 / 周期报表）。"""
    settings = get_settings()
    domain, label, _unit = resolve_metric(metric)
    end_date = date.fromisoformat(end) if end else datetime.now(LOCAL_TZ).date()
    start_date = date.fromisoformat(start) if start else end_date - timedelta(days=days - 1)

    with session_scope() as session:
        result = invoke_tool(
            generate_chart,
            session,
            settings.default_user_id,
            domain=domain,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            chart_type=chart_type,
            output_mode="insert",
            target_path=file,
            anchor=anchor,
        )

    if result.get("is_empty"):
        console.print(f"[yellow]{start_date} ~ {end_date} 内没有{label}记录。[/yellow]")
        raise typer.Exit(1)
    if result.get("duplicate"):
        console.print(f"[yellow]{result['message']}[/yellow]")
        return
    console.print(f"[green]已插入[/green]：{result.get('path')}")


@report_app.command("list")
def report_list(
    limit: int = typer.Option(7, help="最近 N 份"),
) -> None:
    """列出已保存的日报 / 周期报表。"""
    settings = get_settings()
    with session_scope() as session:
        reports = DailyReportRepository(session, settings.default_user_id).list_recent(limit)
        # session 关闭后 ORM 对象会 detach，先在会话内取出需要的字段
        rows = [
            {
                "date": _report_span(r),
                "kind": ((r.agent_outputs or {}).get("period") or {}).get("report_kind") or "daily",
                "length": len(r.content_md),
                "created": r.created_at.isoformat() if r.created_at else "-",
            }
            for r in reports
        ]

    table = Table(title="报表列表")
    table.add_column("区间 / 日期")
    table.add_column("类型")
    table.add_column("长度")
    table.add_column("生成时间")
    for row in rows:
        table.add_row(
            row["date"],
            "周期报表" if row["kind"] == "period" else "日报",
            str(row["length"]),
            row["created"],
        )
    console.print(table)


def _report_span(report) -> str:
    period = (report.agent_outputs or {}).get("period") or {}
    if (
        period.get("start_date")
        and period.get("end_date")
        and period["start_date"] != period["end_date"]
    ):
        return f"{period['start_date']} ~ {period['end_date']}"
    return report.report_date.isoformat()


@scheduler_app.command("run")
def scheduler_run(
    foreground: bool = typer.Option(True, "--foreground/--background", help="前台阻塞运行"),
) -> None:
    """启动定时任务调度器（从 DB 加载任务；默认种子日报任务）。"""
    try:
        from myfitness.scheduler.manager import start_scheduler, stop_scheduler
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    settings = get_settings()
    count = start_scheduler(settings.default_user_id)
    console.print(f"[green]调度器已启动[/green]，已加载 {count} 个任务。Ctrl+C 退出。")
    if not foreground:
        console.print("[yellow]后台模式暂未支持，请前台运行。[/yellow]")
        return
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_scheduler()
        console.print("\n调度器已停止。")


@scheduler_app.command("list")
def scheduler_list() -> None:
    """列出已保存的定时任务及最近执行状态。"""
    settings = get_settings()
    with session_scope() as session:
        tasks = ScheduledTaskRepository(session, settings.default_user_id).list_all()
        rows = [
            {
                "id": task.id,
                "task_type": task.task_type,
                "label": task.label,
                "time_of_day": task.time_of_day,
                "enabled": bool(task.enabled),
                "last_run_at": task.last_run_at.isoformat() if task.last_run_at else "-",
                "last_status": task.last_status or "-",
                "last_error": task.last_error or "",
            }
            for task in tasks
        ]
    table = Table(title="定时任务")
    table.add_column("ID", justify="right")
    table.add_column("类型")
    table.add_column("名称")
    table.add_column("时间")
    table.add_column("状态")
    table.add_column("上次执行")
    table.add_column("结果")
    for task in rows:
        result = task["last_status"]
        if task["last_error"]:
            result = f"{result}: {task['last_error']}"
        table.add_row(
            str(task["id"]),
            task["task_type"],
            task["label"],
            task["time_of_day"],
            "[green]启用[/green]" if task["enabled"] else "[dim]停用[/dim]",
            task["last_run_at"],
            result,
        )
    console.print(table)
    try:
        from myfitness.scheduler.manager import scheduler_running

        running = scheduler_running()
    except Exception:  # noqa: BLE001 - APScheduler 是可选依赖
        running = False
    console.print("调度器：[green]运行中[/green]" if running else "调度器：[dim]未运行[/dim]")


def _validate_schedule_values(task_type: str, label: str, time_of_day: str) -> None:
    allowed = {
        ScheduledTaskRepository.TASK_DAILY_REPORT,
        ScheduledTaskRepository.TASK_SYNC,
    }
    if task_type not in allowed:
        raise ValueError(f"未知类型：{task_type}，可选：{', '.join(sorted(allowed))}")
    if not label.strip() or len(label.strip()) > 128:
        raise ValueError("任务名称长度必须为 1～128 个字符")
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", time_of_day.strip()):
        raise ValueError("执行时间必须是 HH:MM 格式")


def _reload_scheduler_after_change(user_id: int) -> None:
    """数据库提交后刷新内存任务；调度器依赖缺失时保留已保存的修改。"""
    try:
        from myfitness.scheduler.manager import reload_scheduler_jobs

        reload_scheduler_jobs(user_id)
    except Exception as exc:  # noqa: BLE001 - 持久化成功不能被可选调度依赖回滚
        console.print(f"[yellow]任务已保存，但调度器重载失败：{exc}[/yellow]")


@scheduler_app.command("add")
def scheduler_add(
    task_type: str = typer.Argument(..., help="daily_report | sync"),
    time_of_day: str = typer.Option("07:00", "--time", help="HH:MM"),
    label: str | None = typer.Option(None, "--label", help="任务名称，默认使用内置名称"),
    enabled: bool = typer.Option(True, "--enabled/--disabled", help="创建后是否启用"),
) -> None:
    """添加或更新定时任务。"""
    from myfitness.agents.schedule_parser import TASK_LABELS

    task_label = label or TASK_LABELS.get(task_type, task_type)
    try:
        _validate_schedule_values(task_type, task_label, time_of_day)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    settings = get_settings()
    with session_scope() as session:
        task = ScheduledTaskRepository(session, settings.default_user_id).upsert(
            task_type=task_type,
            label=task_label.strip(),
            time_of_day=time_of_day.strip(),
            enabled=enabled,
        )
        task_id = task.id
    _reload_scheduler_after_change(settings.default_user_id)
    state = "启用" if enabled else "停用"
    console.print(
        f"[green]已保存定时任务[/green] #{task_id}：{task_label}，每天 {time_of_day}（{state}）"
    )


@scheduler_app.command("edit")
def scheduler_edit(
    task_id: int = typer.Argument(..., help="任务 ID（见 scheduler list）"),
    task_type: str | None = typer.Option(None, "--type", help="daily_report | sync"),
    label: str | None = typer.Option(None, "--label", help="任务名称"),
    time_of_day: str | None = typer.Option(None, "--time", help="HH:MM"),
    enabled: bool | None = typer.Option(None, "--enabled/--disabled", help="启用或停用"),
) -> None:
    """按 ID 编辑任务内容、名称、时间或启停状态。"""
    if all(value is None for value in (task_type, label, time_of_day, enabled)):
        console.print("[red]请至少提供一个修改项[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    with session_scope() as session:
        repo = ScheduledTaskRepository(session, settings.default_user_id)
        current = repo.get_by_id(task_id)
        if current is None:
            console.print(f"[red]未找到定时任务：{task_id}[/red]")
            raise typer.Exit(1)
        next_type = task_type or current.task_type
        next_label = label if label is not None else current.label
        next_time = time_of_day or current.time_of_day
        try:
            _validate_schedule_values(next_type, next_label, next_time)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        try:
            updated = repo.update(
                task_id,
                task_type=task_type,
                label=label.strip() if label is not None else None,
                time_of_day=time_of_day.strip() if time_of_day is not None else None,
                enabled=enabled,
            )
        except Exception as exc:
            if "unique" in str(exc).lower():
                console.print("[red]同类型的定时任务已经存在，不能修改为重复内容[/red]")
                raise typer.Exit(1) from exc
            raise
        assert updated is not None
        summary = (updated.label, updated.time_of_day, bool(updated.enabled))
    _reload_scheduler_after_change(settings.default_user_id)
    console.print(
        f"[green]已更新定时任务[/green] #{task_id}：{summary[0]}，每天 {summary[1]}，"
        f"{'启用' if summary[2] else '停用'}"
    )


def _set_scheduler_enabled(task_id: int, enabled: bool) -> None:
    settings = get_settings()
    with session_scope() as session:
        repo = ScheduledTaskRepository(session, settings.default_user_id)
        task = repo.update(task_id, enabled=enabled)
        if task is None:
            console.print(f"[red]未找到定时任务：{task_id}[/red]")
            raise typer.Exit(1)
        label = task.label
    _reload_scheduler_after_change(settings.default_user_id)
    console.print(f"[green]已{'启用' if enabled else '停用'}定时任务[/green] #{task_id}：{label}")


@scheduler_app.command("enable")
def scheduler_enable(task_id: int = typer.Argument(..., help="任务 ID")) -> None:
    """启用一个定时任务。"""
    _set_scheduler_enabled(task_id, True)


@scheduler_app.command("disable")
def scheduler_disable(task_id: int = typer.Argument(..., help="任务 ID")) -> None:
    """停用一个定时任务。"""
    _set_scheduler_enabled(task_id, False)


def _model_by_id(model_id: str) -> ModelPreset:
    preset = next((item for item in get_registry().all_models() if item.id == model_id), None)
    if preset is None:
        raise ModelRegistryError(f"未找到模型：{model_id}")
    return preset


def _print_model_table(title: str = "LLM 模型", *, numbered: bool = False) -> None:
    registry = get_registry()
    active_id = registry.active_id()
    models = registry.all_models()
    table = Table(title=title)
    if numbered:
        table.add_column("序号", justify="right")
    table.add_column("生效", justify="center")
    table.add_column("ID")
    table.add_column("名称")
    table.add_column("模型标识")
    table.add_column("Base URL")
    table.add_column("Key")
    table.add_column("温度", justify="right")
    table.add_column("超时", justify="right")
    for index, preset in enumerate(models, start=1):
        row = [
            "[green]●[/green]" if preset.id == active_id else "",
            preset.id,
            preset.name,
            preset.model,
            preset.base_url,
            preset.masked_key() or "[yellow]未配置[/yellow]",
            str(preset.temperature),
            f"{preset.timeout}s",
        ]
        if numbered:
            row.insert(0, str(index))
        table.add_row(*row)
    console.print(table)
    if not models:
        console.print("[yellow]尚未配置任何模型。可运行 myfitness llm add。[/yellow]")


@llm_app.command("list")
def llm_list() -> None:
    """列出 Web 与 CLI 共用的模型预设。"""
    _print_model_table()


@llm_app.command("providers")
def llm_providers() -> None:
    """列出内置的 OpenAI 兼容服务商模板。"""
    table = Table(title="模型服务商模板")
    table.add_column("名称")
    table.add_column("Base URL")
    table.add_column("默认模型")
    for provider in PROVIDER_PRESETS:
        table.add_row(provider["name"], provider["base_url"] or "-", provider["model"] or "-")
    console.print(table)


@llm_app.command("add")
def llm_add(
    name: str = typer.Option(..., "--name", help="显示名称"),
    base_url: str = typer.Option(..., "--base-url", help="OpenAI 兼容 API Base URL"),
    model: str = typer.Option(..., "--model", help="模型标识"),
    api_key: str = typer.Option(
        ...,
        "--api-key",
        prompt="API Key",
        hide_input=True,
        help="API Key；省略选项时安全提示输入",
    ),
    temperature: float = typer.Option(0.7, "--temperature", min=0.0, max=2.0),
    timeout: int = typer.Option(120, "--timeout", min=10, max=600, help="请求超时秒数"),
    activate: bool = typer.Option(False, "--activate", help="保存后立即设为生效模型"),
) -> None:
    """新增一个模型预设，与 Web 设置页共用同一份配置。"""
    registry = get_registry()
    try:
        preset = registry.upsert(
            {
                "name": name,
                "base_url": base_url,
                "model": model,
                "api_key": api_key,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        if activate:
            registry.set_active(preset.id)
    except ModelRegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]模型已保存[/green]：{preset.name}（ID: {preset.id}）")
    if activate:
        console.print("[green]已设为当前生效模型[/green]")


@llm_app.command("edit")
def llm_edit(
    model_id: str = typer.Argument(..., help="模型 ID（见 llm list）"),
    name: str | None = typer.Option(None, "--name", help="显示名称"),
    base_url: str | None = typer.Option(None, "--base-url", help="API Base URL"),
    model: str | None = typer.Option(None, "--model", help="模型标识"),
    api_key: str | None = typer.Option(None, "--api-key", help="新 API Key；省略则保留"),
    clear_api_key: bool = typer.Option(False, "--clear-api-key", help="清空已保存的 API Key"),
    temperature: float | None = typer.Option(None, "--temperature", min=0.0, max=2.0),
    timeout: int | None = typer.Option(None, "--timeout", min=10, max=600),
) -> None:
    """编辑一个用户模型预设；环境配置 env 不可编辑。"""
    if api_key is not None and clear_api_key:
        console.print("[red]--api-key 与 --clear-api-key 不能同时使用[/red]")
        raise typer.Exit(1)
    if (
        all(value is None for value in (name, base_url, model, api_key, temperature, timeout))
        and not clear_api_key
    ):
        console.print("[red]请至少提供一个修改项[/red]")
        raise typer.Exit(1)
    try:
        current = _model_by_id(model_id)
        payload: dict[str, Any] = {
            "id": model_id,
            "name": name if name is not None else current.name,
            "base_url": base_url if base_url is not None else current.base_url,
            "model": model if model is not None else current.model,
            "temperature": temperature if temperature is not None else current.temperature,
            "timeout": timeout if timeout is not None else current.timeout,
        }
        if api_key is not None or clear_api_key:
            payload["api_key"] = "" if clear_api_key else api_key
        preset = get_registry().upsert(payload)
    except ModelRegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]模型已更新[/green]：{preset.name}（ID: {preset.id}）")


@llm_app.command("activate")
def llm_activate(model_id: str = typer.Argument(..., help="模型 ID；环境配置为 env")) -> None:
    """切换当前生效模型。"""
    try:
        get_registry().set_active(model_id)
        preset = _model_by_id(model_id)
    except ModelRegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]已切换模型[/green]：{preset.name} / {preset.model}")


@llm_app.command("delete")
def llm_delete(
    model_id: str = typer.Argument(..., help="用户模型 ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
) -> None:
    """删除一个用户模型预设；删除当前模型后自动回落。"""
    try:
        preset = _model_by_id(model_id)
        if model_id == ENV_PRESET_ID:
            raise ModelRegistryError("环境配置不可删除")
        if not yes and not typer.confirm(f"确认删除模型 {preset.name}（{model_id}）？"):
            raise typer.Abort()
        get_registry().delete(model_id)
    except ModelRegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]模型已删除[/green]：{preset.name}")


@llm_app.command("config")
def llm_config() -> None:
    """显示当前生效的 LLM 配置（API Key 脱敏）。"""
    if not is_llm_configured():
        console.print(
            "[yellow]LLM 未激活：请运行 llm add，或配置 LLM_API_KEY 与 LLM_MODEL[/yellow]"
        )
        raise typer.Exit(1)

    cfg = get_llm_config()
    active = get_registry().active()
    table = Table(title="LLM 配置")
    table.add_column("参数")
    table.add_column("值")
    table.add_row("PRESET", f"{active.name} ({active.id})" if active else "-")
    table.add_row("BASE_URL", cfg.base_url)
    table.add_row("MODEL", cfg.model)
    table.add_row("API_KEY", cfg.masked_api_key())
    table.add_row("TEMPERATURE", str(cfg.temperature))
    table.add_row("MAX_TOKENS", str(cfg.max_tokens or "(未限制)"))
    table.add_row("TIMEOUT", f"{cfg.timeout}s")
    table.add_row("ENDPOINT", cfg.chat_completions_url)
    console.print(table)


@llm_app.command("status")
def llm_status() -> None:
    """显示 LLM 守卫状态（限频 / 熔断 / 调用统计）。"""
    snap = get_llm_guard().snapshot()
    state_color = {"closed": "green", "open": "red", "half_open": "yellow"}.get(
        snap["state"], "white"
    )
    table = Table(title="LLM 守卫状态")
    table.add_column("指标")
    table.add_column("值")
    table.add_row(
        "熔断状态",
        f"[{state_color}]{snap['state']}[/{state_color}]",
    )
    table.add_row("总调用次数", str(snap["total_calls"]))
    table.add_row("成功", str(snap["success_calls"]))
    table.add_row("失败", str(snap["failed_calls"]))
    table.add_row("限频拒绝（熔断期）", str(snap["throttled_calls"]))
    table.add_row("熔断次数", str(snap["circuit_opens"]))
    table.add_row("连续失败", str(snap["consecutive_failures"]))
    table.add_row("最近错误", snap["last_error"] or "-")
    console.print(table)


@llm_app.command("test")
def llm_test(
    prompt: str = typer.Option("用一句话回复：连接成功", help="测试提示词"),
    model_id: str | None = typer.Option(None, "--id", help="测试指定预设；默认测试当前生效模型"),
) -> None:
    """测试当前或指定模型预设的 API 连通性。"""
    try:
        if model_id is None:
            result = probe_llm_connection(prompt)
        else:
            preset = _model_by_id(model_id)
            if not preset.api_key:
                raise ModelRegistryError(f"模型 {model_id} 尚未配置 API Key")
            result = probe_llm_config(
                LlmConfig(
                    base_url=preset.base_url,
                    api_key=preset.api_key,
                    model=preset.model,
                    temperature=preset.temperature,
                    max_tokens=None,
                    timeout=preset.timeout,
                ),
                prompt=prompt,
            )
    except (ValueError, ModelRegistryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]LLM 请求失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print("[green]LLM 连接成功[/green]")
    console.print(f"模型: {result['model']}")
    console.print(f"回复: {result['reply']}")
    if result.get("usage"):
        console.print(f"Token: {result['usage']}")


def _run_llm_warmup() -> LlmWarmupResult:
    """启动 chat 前预加载 LLM，加载完成后再允许用户输入。"""
    if not is_llm_configured():
        return warmup_llm()

    with console.status("[bold cyan]正在加载 LLM…[/bold cyan]", spinner="dots"):
        return warmup_llm()


def _warmup_database(user_id: int) -> None:
    """预热数据库连接，避免首条消息才建立 PostgreSQL 连接。"""
    try:
        with (
            console.status("[bold cyan]正在连接数据库…[/bold cyan]", spinner="dots"),
            session_scope() as session,
        ):
            get_or_create_default_user(session, user_id)
    except Exception as exc:
        settings = get_settings()
        host = (
            settings.database_url.split("@")[-1]
            if "@" in settings.database_url
            else settings.database_url
        )
        console.print(f"[red]数据库连接失败：{exc}[/red]")
        console.print(f"[yellow]当前目标：{host}[/yellow]")
        console.print(
            "[yellow]Windows 请将 DATABASE_URL 中的 localhost 改为 127.0.0.1，"
            "并确认 PostgreSQL 已启动。[/yellow]"
        )
        raise typer.Exit(1) from exc


def _print_warmup_status(result: LlmWarmupResult) -> None:
    if not result.configured:
        console.print("[yellow]LLM 未配置，将使用规则模板回复[/yellow]")
        return
    if not result.loaded:
        console.print(f"[red]LLM 加载失败：{result.error}[/red]")
        console.print("[yellow]将使用规则模板兜底[/yellow]")
        return
    model_label = result.model or "unknown"
    if result.connected is False:
        console.print(
            f"[yellow]LLM 已加载（{model_label}），但连通性探测失败，"
            "首条回复可能降级为规则模板[/yellow]"
        )
        return
    console.print(f"[green]LLM 已就绪[/green]（{model_label}）")


_HOME_SILHOUETTE = """
▄█▄            ██            ▄█▄
██ ▄▄▄▄       ████       ▄▄▄▄ ██
▀████████▀████████████▀████████▀
   ▀▀▀▀    ▀████████▀    ▀▀▀▀
             ▀████▀
           ██████████
            ████████
            ████████
           ███    ███
        ███          ███
        ███          ███
        ███          ███
      █████          █████
""".strip("\n")


def _active_model_label() -> str:
    preset = get_registry().active()
    if preset is None:
        return "未配置"
    return f"{preset.name} · {preset.model}"


def _render_chat_home() -> None:
    """Render the initial CLI home without creating a chat session."""
    console.clear()
    left = Panel(
        Align.center(Text(_HOME_SILHOUETTE, style="bold green"), vertical="middle"),
        border_style="green",
        box=box.SQUARE,
        padding=(1, 1),
    )
    info = Text()
    info.append("MYFITNESS\n", style="bold bright_green")
    info.append("Multi-Agent Fitness Console\n\n", style="dim")
    info.append("HELP\n", style="bold green")
    info.append("  /model   切换当前 LLM\n")
    info.append("  /resume  恢复历史会话\n")
    info.append("  /help    查看交互命令\n")
    info.append("  exit     退出\n\n")
    info.append("TIPS\n", style="bold green")
    info.append("  直接描述目标、饮食、训练或身体数据。\n")
    info.append("  首条消息提交后才会创建会话。\n\n")
    info.append("MODEL\n", style="bold green")
    info.append(f"  {_active_model_label()}", style="bright_white")
    right = Panel(info, border_style="green", box=box.SQUARE, padding=(1, 2))

    layout = Table.grid(expand=True, padding=0)
    layout.add_column(ratio=1)
    layout.add_column(width=1)
    layout.add_column(ratio=2)
    separator = Text("\n".join("│" for _ in range(19)), style="green")
    layout.add_row(left, separator, right)
    console.print(layout)
    console.print(
        Panel(
            "[dim]输入消息并回车开始对话；空输入不会创建会话[/dim]",
            title="[bold green]对话输入[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _render_conversation_page(state: Any, *, replay_history: bool = False) -> None:
    """Switch to the dedicated conversation page and optionally replay history."""
    console.clear()
    title = "MyFitness 对话"
    model = _active_model_label()
    console.print(
        Panel(
            f"[bold bright_green]{title}[/bold bright_green]\n"
            f"[dim]会话 {state.session_id} · 模型 {model}[/dim]",
            border_style="green",
            box=box.ROUNDED,
        )
    )
    console.print("[dim]/model 切换模型  ·  /resume 恢复会话  ·  /help 帮助  ·  exit 退出[/dim]\n")
    if replay_history:
        _print_conversation_history(state)


def _print_conversation_history(state: Any, *, raw: bool = False) -> None:
    """Print every stored user/assistant message in chronological order."""
    for message in state.messages:
        role = "你" if message.role == "user" else "MyFitness"
        color = "bold" if message.role == "user" else "bold cyan"
        console.print(f"[{color}]{role}[/{color}]")
        if raw:
            console.print(message.content, markup=False)
        else:
            console.print(Markdown(message.content))
        console.print()


def _supports_arrow_menu() -> bool:
    return bool(console.is_terminal and sys.stdin.isatty())


def _read_menu_key() -> str:
    """Read one navigation key in raw mode on Windows or POSIX terminals."""
    if sys.platform == "win32":
        import msvcrt

        char = msvcrt.getwch()
        if char in {"\x00", "\xe0"}:
            code = msvcrt.getwch()
            return {"H": "up", "P": "down"}.get(code, "other")
        if char in {"\r", "\n"}:
            return "enter"
        if char == "\x1b":
            return "escape"
        return {"k": "up", "j": "down"}.get(char.lower(), "other")

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char in {"\r", "\n"}:
            return "enter"
        if char == "\x1b":
            sequence = ""
            while select.select([sys.stdin], [], [], 0.02)[0] and len(sequence) < 2:
                sequence += sys.stdin.read(1)
            return {"[A": "up", "[B": "down"}.get(sequence, "escape")
        return {"k": "up", "j": "down"}.get(char.lower(), "other")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _move_menu_focus(index: int, key: str, count: int) -> int:
    if count <= 0:
        return 0
    if key == "up":
        return (index - 1) % count
    if key == "down":
        return (index + 1) % count
    return index


def _menu_renderable(title: str, options: Sequence[tuple[str, str]], focus: int) -> Panel:
    rows: list[Text] = []
    for index, (_value, label) in enumerate(options):
        selected = index == focus
        row = Text("> " if selected else "  ")
        row.append("● " if selected else "○ ", style="bold green" if selected else "dim")
        row.append(label, style="bold bright_green" if selected else "white")
        rows.append(row)
    rows.append(Text("\n↑/↓ 移动  Enter 确认  Esc 取消", style="dim"))
    return Panel(
        Group(*rows),
        title=f"[bold green]{title}[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _select_option(
    title: str,
    options: Sequence[tuple[str, str]],
    *,
    current_value: str | None = None,
) -> str | None:
    """Select an option with arrows; fall back to a numbered prompt when non-interactive."""
    if not options:
        return None
    focus = next(
        (index for index, (value, _label) in enumerate(options) if value == current_value),
        0,
    )
    if not _supports_arrow_menu():
        table = Table(title=title)
        table.add_column("序号", justify="right")
        table.add_column("选项")
        for index, (value, label) in enumerate(options, start=1):
            marker = " ●" if value == current_value else ""
            table.add_row(str(index), f"{label}{marker}")
        console.print(table)
        choice = console.input("选择序号或 ID（直接回车取消）：").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][0]
        known = {value for value, _label in options}
        return choice if choice in known else None

    with Live(
        _menu_renderable(title, options, focus),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        while True:
            key = _read_menu_key()
            if key == "enter":
                return options[focus][0]
            if key == "escape":
                return None
            next_focus = _move_menu_focus(focus, key, len(options))
            if next_focus != focus:
                focus = next_focus
                live.update(_menu_renderable(title, options, focus), refresh=True)


def _chat_select_model() -> bool:
    """Interactive ``/model`` handler. Returns whether a model was switched."""
    registry = get_registry()
    models = registry.all_models()
    if not models:
        console.print("[yellow]尚未配置任何模型。可运行 myfitness llm add。[/yellow]")
        return False
    model_id = _select_option(
        "选择 LLM 模型",
        [
            (
                preset.id,
                " · ".join(
                    (
                        preset.name,
                        preset.model,
                        preset.base_url,
                        preset.masked_key() or "未配置 Key",
                    )
                ),
            )
            for preset in models
        ],
        current_value=registry.active_id(),
    )
    if model_id is None:
        console.print("[dim]已取消模型切换。[/dim]")
        return False
    try:
        registry.set_active(model_id)
        preset = _model_by_id(model_id)
    except ModelRegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        return False
    console.print(f"[green]已切换模型[/green]：{preset.name} / {preset.model}")
    return True


def _chat_resume_session(history: ChatHistoryStore, current_state: Any | None) -> Any | None:
    """Interactive ``/resume`` handler. Invalid/cancelled choices keep current state."""
    sessions = history.list_sessions()
    if not sessions:
        console.print("[yellow]暂无可恢复的历史会话。[/yellow]")
        return current_state
    current_id = current_state.session_id if current_state is not None else None
    session_id = _select_option(
        "选择历史会话",
        [
            (
                item.session_id,
                " · ".join(
                    (
                        item.title,
                        f"{item.message_count} 条消息",
                        item.updated_at,
                        item.preview[:48],
                    )
                ),
            )
            for item in sessions
        ],
        current_value=current_id,
    )
    if session_id is None:
        console.print("[dim]已取消会话恢复。[/dim]")
        return current_state
    try:
        restored = history.load(session_id)
    except ChatHistoryError as exc:
        console.print(f"[red]恢复对话失败：{exc}[/red]")
        return current_state
    _render_conversation_page(restored, replay_history=True)
    return restored


def _handle_chat_command(
    text: str, history: ChatHistoryStore, current_state: Any | None
) -> tuple[bool, Any | None]:
    """Handle chat slash commands and return ``(handled, possibly_new_state)``."""
    command = text.strip().lower()
    if command == "/model":
        _chat_select_model()
        return True, current_state
    if command == "/resume":
        return True, _chat_resume_session(history, current_state)
    if command == "/help":
        console.print(
            "[bold]/model[/bold] 切换 LLM  ·  "
            "[bold]/resume[/bold] 恢复历史会话  ·  "
            "[bold]exit[/bold] 退出"
        )
        return True, current_state
    return False, current_state


@session_app.command("list")
def session_list(
    limit: int = typer.Option(20, "--limit", min=1, help="最多显示最近 N 个会话"),
) -> None:
    """列出 Web 与 CLI 共用的历史对话。"""
    sessions = ChatHistoryStore().list_sessions()[:limit]
    table = Table(title="历史对话")
    table.add_column("UUID", no_wrap=True, overflow="ignore")
    table.add_column("标题")
    table.add_column("消息", justify="right")
    table.add_column("更新时间")
    table.add_column("预览")
    for item in sessions:
        table.add_row(
            item.session_id,
            item.title,
            str(item.message_count),
            item.updated_at,
            item.preview,
        )
    console.print(table)
    if not sessions:
        console.print("[yellow]暂无历史对话。[/yellow]")


@session_app.command("show")
def session_show(
    session_id: str = typer.Argument(..., help="会话 UUID"),
    raw: bool = typer.Option(False, "--raw", help="原样输出 Markdown，不做终端渲染"),
) -> None:
    """查看一段历史对话的完整消息。"""
    store = ChatHistoryStore()
    try:
        document = store.get(session_id)
        state = store.load(session_id)
    except ChatHistoryError as exc:
        console.print(f"[red]读取对话失败：{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[bold]{document['title']}[/bold]")
    console.print(
        f"[dim]{document['session_id']} · {document['created_at']} → {document['updated_at']}[/dim]\n"
    )
    _print_conversation_history(state, raw=raw)


@artifact_app.command("show")
def artifact_show(
    path: str = typer.Argument(..., help="DATA_DIR 内的报表或图表文件路径"),
    raw: bool = typer.Option(False, "--raw", help="原样输出 Markdown，不做终端渲染"),
) -> None:
    """安全读取并显示 Web 产物查看器可打开的 Markdown 文件。"""
    try:
        artifact = read_artifact(path)
    except ArtifactError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[bold]{artifact['name']}[/bold]  "
        f"[dim]{artifact['size']} bytes · {artifact['modified_at']}[/dim]\n"
    )
    if raw:
        console.print(artifact["content"], markup=False)
    else:
        console.print(Markdown(artifact["content"]))
    if artifact["truncated"]:
        console.print("[yellow]内容过长，已截断显示。[/yellow]")


@app.command("chat")
def chat(
    once: bool = typer.Option(False, "--once", help="单轮模式：处理一条消息后退出"),
    message: str | None = typer.Option(None, "-m", "--message", help="单轮消息内容"),
    no_stream: bool = typer.Option(False, "--no-stream", help="禁用 LLM 流式输出"),
    session_id: str | None = typer.Option(
        None, "--session", help="按 UUID 恢复已有对话；不指定则创建新对话"
    ),
) -> None:
    """启动多 Agent 对话（对话自动保存到 <DATA_DIR>/chat-history/<UUID>.json）。"""
    settings = get_settings()
    history = ChatHistoryStore()
    state: Any | None = None
    runtime_ready = False
    if session_id:
        try:
            state = history.load(session_id)
        except ChatHistoryError as exc:
            console.print(f"[red]恢复对话失败：{exc}[/red]")
            raise typer.Exit(1) from exc
        _render_conversation_page(state, replay_history=True)

    def _ensure_runtime_ready() -> None:
        nonlocal runtime_ready
        if runtime_ready:
            return
        warmup = _run_llm_warmup()
        if not warmup.ready_for_input:
            console.print("[red]LLM 未能加载，无法启动对话。[/red]")
            raise typer.Exit(1)
        _print_warmup_status(warmup)
        _warmup_database(settings.default_user_id)
        runtime_ready = True

    def _process(text: str) -> None:
        nonlocal state
        # 首页先显示且允许切模型/恢复会话；仅首条普通消息触发运行时预热。
        _ensure_runtime_ready()
        if state is None:
            # 首页阶段不注册空会话；第一条普通消息提交后才创建。
            state = new_chat_state(user_id=settings.default_user_id)
            history.save(state)
            _render_conversation_page(state)
        progress_log: list[str] = []

        def on_progress(msg: str) -> None:
            progress_log.append(msg)
            status.update(f"[bold cyan]{msg}[/bold cyan]")

        with console.status("[bold cyan]处理中…[/bold cyan]", spinner="dots") as status:
            if no_stream:
                with session_scope() as session:
                    get_or_create_default_user(session, settings.default_user_id)
                    state = run_chat_turn(session, state, text, on_progress=on_progress)
                # 数据库事务提交成功后再持久化对话，避免 JSON 与数据库状态不一致。
                history.save(state)
                status.stop()
                _print_progress_log(progress_log)
                console.print(f"\n[bold cyan]MyFitness[/bold cyan]\n{state.reply}\n")
                return

            with session_scope() as session:
                get_or_create_default_user(session, settings.default_user_id)
                state, chunks = iter_chat_turn(session, state, text, on_progress=on_progress)
                status.stop()
                _print_progress_log(progress_log)
                console.print("\n[bold cyan]MyFitness[/bold cyan]")
                reply_parts: list[str] = []
                for chunk in chunks:
                    console.print(chunk, end="")
                    reply_parts.append(chunk)
                console.print("\n")
                finalize_streamed_reply(state, "".join(reply_parts))
            history.save(state)
            snap = get_llm_guard().snapshot()
            if snap["state"] == "open":
                console.print(
                    "[yellow]提示：LLM 已熔断，后续回复将使用规则模板，"
                    f"约 {60}s 后自动恢复探测。[/yellow]"
                )

    if message:
        _process(message)
        return

    if once:
        if state is None:
            _render_chat_home()
        text = typer.prompt("你")
        _process(text)
        return

    if state is None:
        _render_chat_home()
    while True:
        try:
            text = console.input("[bold green]>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见。")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q", "退出"}:
            console.print("再见。")
            break
        handled, state = _handle_chat_command(text, history, state)
        if handled:
            if text.lower() == "/model":
                if state is None:
                    _render_chat_home()
                else:
                    _render_conversation_page(state, replay_history=True)
            continue
        _process(text)


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", help="监听地址（默认仅本机可访问）"),
    port: int = typer.Option(8765, help="监听端口"),
    no_open: bool = typer.Option(False, "--no-open", help="启动后不自动打开浏览器"),
) -> None:
    """启动三栏式 MyFitness Agent 本地可视化界面。"""
    from myfitness.api.web import run_web_ui

    run_web_ui(host=host, port=port, open_browser=not no_open)


def _print_progress_log(steps: list[str]) -> None:
    """将本轮 Agent/Tool 调用步骤打印为一行摘要。"""
    if not steps:
        return
    cleaned = [s.rstrip("…").rstrip(".") for s in steps]
    console.print(f"[dim]› {' → '.join(cleaned)}[/dim]")


if __name__ == "__main__":
    app()
