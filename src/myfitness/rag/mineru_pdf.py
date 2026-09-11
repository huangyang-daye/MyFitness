"""MinerU PDF 解析适配 — 优先远程 API，其次本机 do_parse / CLI。"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

from myfitness.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TASKS_ENDPOINT = "/tasks"
_HEALTH_ENDPOINT = "/health"


class MinerUParseError(RuntimeError):
    """MinerU 解析失败。"""


def parse_pdf_with_mineru(
    data: bytes,
    *,
    filename: str = "document.pdf",
    settings: Settings | None = None,
) -> str:
    """用 MinerU 将 PDF 转为 Markdown / 纯文本。"""
    if not data:
        raise MinerUParseError("PDF 内容为空")
    cfg = settings or get_settings()
    safe_name = Path(filename).name or "document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{Path(safe_name).stem or 'document'}.pdf"

    api_url = (cfg.mineru_api_url or "").strip()
    if api_url:
        return _parse_via_api(data, safe_name, cfg)

    if _mineru_package_available():
        return _parse_via_do_parse(data, safe_name, cfg)

    if shutil.which("mineru"):
        return _parse_via_cli(data, safe_name, cfg)

    raise MinerUParseError(
        "未找到 MinerU：请安装 `pip install -e \".[mineru]\"`，"
        "或配置 MINERU_API_URL 指向已启动的 mineru-api，"
        "或确保 mineru CLI 在 PATH 中"
    )


def _mineru_package_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mineru.cli.common") is not None
    except Exception:  # noqa: BLE001
        return False


def _parse_via_do_parse(data: bytes, filename: str, cfg: Settings) -> str:
    try:
        from mineru.cli.common import do_parse
    except ImportError as exc:
        raise MinerUParseError(
            '本机 MinerU 未安装完整，请执行：pip install -e ".[mineru]"'
        ) from exc

    stem = Path(filename).stem or "document"
    with tempfile.TemporaryDirectory(prefix="myfitness-mineru-") as tmp:
        output_dir = Path(tmp)
        try:
            do_parse(
                str(output_dir),
                [stem],
                [data],
                [cfg.mineru_lang],
                backend=cfg.mineru_backend,
                parse_method=cfg.mineru_parse_method,
                formula_enable=cfg.mineru_formula_enable,
                table_enable=cfg.mineru_table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise MinerUParseError(f"MinerU do_parse 失败：{exc}") from exc
        markdown = _collect_markdown(output_dir)
        if not markdown.strip():
            raise MinerUParseError("MinerU 未产出有效 Markdown")
        return markdown


def _parse_via_cli(data: bytes, filename: str, cfg: Settings) -> str:
    with tempfile.TemporaryDirectory(prefix="myfitness-mineru-cli-") as tmp:
        root = Path(tmp)
        pdf_path = root / filename
        pdf_path.write_bytes(data)
        output_dir = root / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mineru",
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-b",
            cfg.mineru_backend,
            "-l",
            cfg.mineru_lang,
            "-m",
            cfg.mineru_parse_method,
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(cfg.mineru_timeout, 30),
            )
        except subprocess.TimeoutExpired as exc:
            raise MinerUParseError(f"MinerU CLI 超时（>{cfg.mineru_timeout}s）") from exc
        except OSError as exc:
            raise MinerUParseError(f"无法启动 MinerU CLI：{exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MinerUParseError(
                f"MinerU CLI 失败（exit={completed.returncode}）"
                + (f"：{detail[:500]}" if detail else "")
            )
        markdown = _collect_markdown(output_dir)
        if not markdown.strip():
            raise MinerUParseError("MinerU CLI 未产出有效 Markdown")
        return markdown


def _parse_via_api(data: bytes, filename: str, cfg: Settings) -> str:
    base_url = cfg.mineru_api_url.strip().rstrip("/")
    timeout = httpx.Timeout(cfg.mineru_timeout, connect=min(30.0, float(cfg.mineru_timeout)))
    form_data = {
        "lang_list": cfg.mineru_lang,
        "backend": cfg.mineru_backend,
        "parse_method": cfg.mineru_parse_method,
        "formula_enable": str(cfg.mineru_formula_enable).lower(),
        "table_enable": str(cfg.mineru_table_enable).lower(),
        "image_analysis": "false",
        "return_md": "true",
        "return_middle_json": "false",
        "return_model_output": "false",
        "return_content_list": "false",
        "return_images": "false",
        "response_format_zip": "true",
        "return_original_file": "false",
        "start_page_id": "0",
        "end_page_id": "99999",
    }
    headers: dict[str, str] = {}
    token = (cfg.mineru_api_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with tempfile.TemporaryDirectory(prefix="myfitness-mineru-api-") as tmp:
        root = Path(tmp)
        pdf_path = root / filename
        pdf_path.write_bytes(data)
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            _ensure_api_healthy(client, base_url)
            with pdf_path.open("rb") as handle:
                response = client.post(
                    f"{base_url}{_TASKS_ENDPOINT}",
                    data=form_data,
                    files={"files": (filename, handle, "application/pdf")},
                )
            if response.status_code not in {200, 202}:
                raise MinerUParseError(
                    f"MinerU API 提交失败：HTTP {response.status_code} {_response_detail(response)}"
                )
            payload = response.json()
            status_url = _absolute_url(base_url, str(payload.get("status_url") or ""))
            result_url = _absolute_url(base_url, str(payload.get("result_url") or ""))
            task_id = payload.get("task_id")
            if not status_url or not result_url:
                raise MinerUParseError(
                    f"MinerU API 返回无效任务信息（task_id={task_id!r}）"
                )
            _wait_api_task(client, status_url, cfg.mineru_timeout)
            zip_response = client.get(result_url)
            if zip_response.status_code != 200:
                raise MinerUParseError(
                    f"MinerU API 下载结果失败：HTTP {zip_response.status_code}"
                )
            zip_path = root / "result.zip"
            zip_path.write_bytes(zip_response.content)
            extract_dir = root / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            markdown = _collect_markdown(extract_dir)
            if not markdown.strip():
                raise MinerUParseError("MinerU API 结果中未找到 Markdown")
            return markdown


def _ensure_api_healthy(client: httpx.Client, base_url: str) -> None:
    try:
        response = client.get(f"{base_url}{_HEALTH_ENDPOINT}")
    except httpx.HTTPError as exc:
        raise MinerUParseError(f"无法连接 MinerU API（{base_url}）：{exc}") from exc
    if response.status_code != 200:
        # 部分部署可能无 /health，不强制失败
        logger.warning(
            "MinerU API health check returned HTTP %s，继续提交任务",
            response.status_code,
        )


def _wait_api_task(client: httpx.Client, status_url: str, timeout: int) -> None:
    deadline = time.monotonic() + max(timeout, 30)
    last_status = ""
    while time.monotonic() < deadline:
        try:
            response = client.get(status_url)
        except httpx.HTTPError as exc:
            raise MinerUParseError(f"轮询 MinerU 任务状态失败：{exc}") from exc
        if response.status_code != 200:
            raise MinerUParseError(
                f"MinerU 任务状态查询失败：HTTP {response.status_code} {_response_detail(response)}"
            )
        payload = response.json() if response.content else {}
        status = str(payload.get("status") or "").lower()
        if status != last_status:
            logger.info("MinerU task status: %s", status or "(empty)")
            last_status = status
        if status in {"completed", "success", "done"}:
            return
        if status in {"failed", "error", "cancelled", "canceled"}:
            detail = payload.get("error") or payload.get("message") or payload
            raise MinerUParseError(f"MinerU 任务失败：{detail}")
        time.sleep(1.0)
    raise MinerUParseError(f"等待 MinerU 解析超时（>{timeout}s）")


def _absolute_url(base_url: str, value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if not text.startswith("/"):
        text = "/" + text
    return base_url.rstrip("/") + text


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return (response.text or "").strip()[:300]
    if isinstance(payload, dict):
        for key in ("detail", "error", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]
    return str(payload)[:300]


def _collect_markdown(root: Path) -> str:
    """收集 MinerU 输出目录中的 Markdown，跳过明显的附属文件。"""
    candidates: list[Path] = []
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered in {"readme.md", "license.md"}:
            continue
        if "images" in {part.lower() for part in path.parts}:
            continue
        candidates.append(path)
    if not candidates:
        return ""
    primary = [
        path
        for path in candidates
        if not path.stem.lower().endswith("_content_list")
    ]
    chosen = primary or candidates
    chosen.sort(key=lambda item: (-item.stat().st_size, str(item)))
    texts: list[str] = []
    seen: set[str] = set()
    for path in chosen:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return "\n\n".join(texts)
