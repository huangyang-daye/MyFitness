import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from myfitness.config import get_settings
from myfitness.db.session import get_or_create_default_user, session_scope
from myfitness.graph.chat import finalize_streamed_reply, iter_chat_turn, new_chat_state, run_chat_turn
from myfitness.llm.factory import get_llm_config, is_llm_configured, probe_llm_connection
from myfitness.llm.guard import get_llm_guard
from myfitness.sync.orchestrator import run_sync
from myfitness.xunji.keys import get_key_statuses, missing_keys_for_sync

app = typer.Typer(help="MyFitness — 多 Agent 健康监控 CLI")
db_app = typer.Typer(help="数据库管理")
llm_app = typer.Typer(help="LLM 配置与测试")
xunji_app = typer.Typer(help="训记 Open API 配置")
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
app.add_typer(xunji_app, name="xunji")

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
    root = Path(__file__).resolve().parents[3]
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
    days: Optional[int] = typer.Option(7, help="同步最近 N 天（默认 7）"),
    start: Optional[str] = typer.Option(None, "--start", help="起始日期 YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, "--end", help="结束日期 YYYY-MM-DD"),
    types: Optional[str] = typer.Option(
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


@llm_app.command("config")
def llm_config() -> None:
    """显示当前 LLM 配置（API Key 脱敏）。"""
    settings = get_settings()
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


@app.command("chat")
def chat(
    once: bool = typer.Option(False, "--once", help="单轮模式：处理一条消息后退出"),
    message: Optional[str] = typer.Option(None, "-m", "--message", help="单轮消息内容"),
    no_stream: bool = typer.Option(False, "--no-stream", help="禁用 LLM 流式输出"),
) -> None:
    """启动多 Agent 对话（LangGraph 编排，LLM 流式输出）。"""
    settings = get_settings()
    state = new_chat_state(user_id=settings.default_user_id)

    def _process(text: str) -> None:
        nonlocal state
        with session_scope() as session:
            get_or_create_default_user(session, settings.default_user_id)
            if no_stream:
                state = run_chat_turn(session, state, text)
                console.print(f"\n[bold cyan]MyFitness[/bold cyan]\n{state.reply}\n")
                return

            state, chunks = iter_chat_turn(session, state, text)
            console.print("\n[bold cyan]MyFitness[/bold cyan]")
            reply_parts: list[str] = []
            for chunk in chunks:
                console.print(chunk, end="")
                reply_parts.append(chunk)
            console.print("\n")
            finalize_streamed_reply(state, "".join(reply_parts))
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


if __name__ == "__main__":
    app()
