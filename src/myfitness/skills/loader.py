"""解析 SKILL.md（YAML frontmatter + Markdown 正文）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from myfitness.skills.models import SkillSpec

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 YAML frontmatter；无合法 frontmatter 时返回 ({}, 全文)。"""
    if not text.startswith("---"):
        return {}, text
    rest = text[3:]
    if rest.startswith("\r\n"):
        rest = rest[2:]
    elif rest.startswith("\n"):
        rest = rest[1:]
    match = re.search(r"\n---\s*(?:\n|$)", rest)
    if match is None:
        return {}, text
    raw = rest[: match.start()]
    body = rest[match.end() :]
    return _parse_simple_yaml(raw), body


def load_skill_spec(skill_md: Path, *, source: str) -> SkillSpec | None:
    """从 SKILL.md 构建 SkillSpec；文件缺失或名称为空时返回 None。"""
    if not skill_md.is_file():
        return None
    text = skill_md.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    folder_name = skill_md.parent.name.strip().lower()
    name = str(meta.get("name") or folder_name).strip().lower()
    if not name or not _NAME_RE.match(name):
        return None

    handler = meta.get("handler")
    handler_str = str(handler).strip() if handler else None
    if not handler_str:
        local_handler = skill_md.parent / "handler.py"
        if local_handler.is_file():
            handler_str = "handler.py:run"

    when = str(meta.get("when") or "context").strip().lower()
    if when not in {"context", "task", "both"}:
        when = "context"

    enabled = meta.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"false", "0", "no"}

    triggers = meta.get("triggers") or {}
    if not isinstance(triggers, dict):
        triggers = {}
    normalized: dict[str, list[str]] = {}
    for key, value in triggers.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(item).strip() for item in value if str(item).strip()]
        elif value:
            normalized[str(key)] = [str(value).strip()]

    return SkillSpec(
        name=name,
        description=str(meta.get("description") or "").strip(),
        path=skill_md.parent,
        source=source,
        handler=handler_str,
        when=when,
        enabled=bool(enabled),
        triggers=normalized,
        body=body.strip(),
    )


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    root, _ = _parse_mapping(lines, 0, -1)
    return root


def _parse_mapping(lines: list[str], index: int, parent_indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        indent = _indent_of(raw)
        if indent <= parent_indent:
            break
        stripped = raw.strip()
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            index += 1
            continue
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        index += 1
        if rest in {">", ">-", "|", "|-"}:
            value, index = _parse_folded(lines, index, indent, folded=rest.startswith(">"))
            result[key] = value
            continue
        if rest == "":
            value, index = _parse_nested(lines, index, indent)
            result[key] = value
            continue
        result[key] = _parse_scalar(rest)
    return result, index


def _parse_nested(lines: list[str], index: int, parent_indent: int) -> tuple[Any, int]:
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
        index += 1
    if index >= len(lines):
        return {}, index
    nxt = lines[index]
    indent = _indent_of(nxt)
    if indent <= parent_indent:
        return {}, index
    if nxt.strip().startswith("- "):
        return _parse_list(lines, index, parent_indent)
    return _parse_mapping(lines, index, parent_indent)


def _parse_list(lines: list[str], index: int, parent_indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        indent = _indent_of(raw)
        if indent <= parent_indent:
            break
        stripped = raw.strip()
        if not stripped.startswith("- "):
            break
        item = stripped[2:].strip()
        index += 1
        if item and ":" in item and not item.startswith("{") and not item.startswith("["):
            key, _, rest = item.partition(":")
            nested = {key.strip(): _parse_scalar(rest.strip()) if rest.strip() else {}}
            if rest.strip() == "":
                child, index = _parse_nested(lines, index, indent)
                nested[key.strip()] = child
            items.append(nested)
        else:
            items.append(_parse_scalar(item) if item else {})
    return items, index


def _parse_folded(lines: list[str], index: int, parent_indent: int, *, folded: bool) -> tuple[str, int]:
    collected: list[str] = []
    while index < len(lines):
        raw = lines[index]
        if not raw.strip():
            collected.append("")
            index += 1
            continue
        indent = _indent_of(raw)
        if indent <= parent_indent:
            break
        collected.append(raw[indent:])
        index += 1
    if folded:
        parts = [line.strip() for line in collected if line.strip()]
        return " ".join(parts), index
    return "\n".join(collected).strip("\n"), index


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
    except ValueError:
        pass
    return value


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))
