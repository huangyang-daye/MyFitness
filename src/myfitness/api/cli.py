import logging
import subprocess
import sys
from datetime import date, timedelta

import typer
from rich.console import Console
from rich.table import Table

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.chart_tools import generate_chart, resolve_metric
from myfitness.agents.tools.schedule_tools import list_scheduled_tasks
from myfitness.chat_history import ChatHistoryError, ChatHistoryStore
from myfitness.config import get_settings
from myfitness.db.repositories.reports import DailyReportRepository
from myfitness.db.session import get_or_create_default_user, session_scope
from myfitness.graph.chat import (
    finalize_streamed_reply,
    iter_chat_turn,
    new_chat_state,
    run_chat_turn,
)
from myfitness.llm.factory import (
    LlmWarmupResult,
    get_llm_config,
    is_llm_configured,
    probe_llm_connection,
    warmup_llm,
)
from myfitness.llm.guard import get_llm_guard
from myfitness.paths import PROJECT_ROOT
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
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
app.add_typer(xunji_app, name="xunji")
app.add_typer(report_app, name="report")
app.add_typer(chart_app, name="chart")
app.add_typer(scheduler_app, name="scheduler")

console = Console()


def _setup_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@app.callback()
def main() -> None:
    _setup_logging()


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


@app.command("sync")
def sync(
    days: int | None = typer.Option(7, help="同步最近 N 天（默认 7）"),
    start: str | None = typer.Option(None, "--start", help="起始日期 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束日期 YYYY-MM-DD"),
    types: str | None = typer.Option(
        None, help="同步类型，逗号分隔：body,food,training"
    ),
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
    domain, label, unit = resolve_metric(metric)
    end_date = date.fromisoformat(end) if end else date.today()
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
    domain, label, unit = resolve_metric(metric)
    end_date = date.fromisoformat(end) if end else date.today()
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
                "kind": ((r.agent_outputs or {}).get("period") or {}).get("report_kind")
                or "daily",
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
    period = ((report.agent_outputs or {}).get("period") or {})
    if period.get("start_date") and period.get("end_date"):
        if period["start_date"] != period["end_date"]:
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
    """列出已保存的定时任务。"""
    settings = get_settings()
    with session_scope() as session:
        tasks = invoke_tool(list_scheduled_tasks, session, settings.default_user_id)
    table = Table(title="定时任务")
    table.add_column("类型")
    table.add_column("名称")
    table.add_column("时间")
    table.add_column("状态")
    table.add_column("上次执行")
    for t in tasks:
        table.add_row(
            t["task_type"],
            t["label"],
            t["time_of_day"],
            "启用" if t["enabled"] else "停用",
            t.get("last_run_at") or "-",
        )
    console.print(table)


@scheduler_app.command("add")
def scheduler_add(
    task_type: str = typer.Argument(..., help="daily_report | sync"),
    time_of_day: str = typer.Option("07:00", "--time", help="HH:MM"),
) -> None:
    """添加或更新定时任务。"""
    from myfitness.agents.schedule_parser import TASK_LABELS
    from myfitness.agents.tools.schedule_tools import apply_schedule_upsert

    if task_type not in TASK_LABELS:
        console.print(f"[red]未知类型：{task_type}，可选：{', '.join(TASK_LABELS)}[/red]")
        raise typer.Exit(1)
    settings = get_settings()
    with session_scope() as session:
        msg = invoke_tool(
            apply_schedule_upsert,
            session,
            settings.default_user_id,
            payload={
                "task_type": task_type,
                "label": TASK_LABELS[task_type],
                "time_of_day": time_of_day,
                "enabled": True,
            },
        )
    console.print(f"[green]{msg}[/green]")


@llm_app.command("config")
def llm_config() -> None:
    """显示当前 LLM 配置（API Key 脱敏）。"""
    if not is_llm_configured():
        console.print("[yellow]LLM 未激活：请配置 LLM_API_KEY 与 LLM_MODEL[/yellow]")
        raise typer.Exit(1)

    cfg = get_llm_config()
    table = Table(title="LLM 配置")
    table.add_column("参数")
    table.add_column("值")
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
) -> None:
    """测试 LLM 通用 API 连通性。"""
    try:
        result = probe_llm_connection(prompt)
    except ValueError as exc:
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
        with console.status("[bold cyan]正在连接数据库…[/bold cyan]", spinner="dots"):
            with session_scope() as session:
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
    warmup = _run_llm_warmup()
    if not warmup.ready_for_input:
        console.print("[red]LLM 未能加载，无法启动对话。[/red]")
        raise typer.Exit(1)

    _print_warmup_status(warmup)
    _warmup_database(settings.default_user_id)
    history = ChatHistoryStore()
    if session_id:
        try:
            state = history.load(session_id)
        except ChatHistoryError as exc:
            console.print(f"[red]恢复对话失败：{exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(
            f"[green]已恢复对话[/green] {state.session_id}（{len(state.messages)} 条消息）"
        )
    else:
        state = new_chat_state(user_id=settings.default_user_id)
        history.save(state)
        console.print(f"[dim]会话 UUID：{state.session_id}[/dim]")

    def _process(text: str) -> None:
        nonlocal state
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
        text = typer.prompt("你")
        _process(text)
        return

    console.print("[green]MyFitness 对话已启动[/green]（输入 exit 退出）")
    while True:
        try:
            text = console.input("[bold]你[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见。")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q", "退出"}:
            console.print("再见。")
            break
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
