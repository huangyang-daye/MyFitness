"""发现并缓存 builtin + 项目目录下的 Skill。"""

from __future__ import annotations

from pathlib import Path

from myfitness.paths import BUILTIN_SKILLS_DIR, SKILLS_DIR
from myfitness.skills.loader import load_skill_spec
from myfitness.skills.models import SkillSpec

_CACHE: dict[str, SkillSpec] | None = None


def skill_search_dirs() -> list[tuple[Path, str]]:
    """(目录, 来源)。后出现的同名 Skill 覆盖前者，便于项目目录覆盖内置实现。"""
    return [
        (BUILTIN_SKILLS_DIR, "builtin"),
        (SKILLS_DIR, "project"),
    ]


def list_skills(*, enabled_only: bool = True) -> list[SkillSpec]:
    specs = list(load_registry().values())
    if enabled_only:
        specs = [item for item in specs if item.enabled]
    return sorted(specs, key=lambda item: (item.source != "builtin", item.name))


def get_skill(name: str) -> SkillSpec | None:
    return load_registry().get(name.strip().lower())


def load_registry() -> dict[str, SkillSpec]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    found: dict[str, SkillSpec] = {}
    for directory, source in skill_search_dirs():
        if not directory.is_dir():
            continue
        for spec in _scan_dir(directory, source):
            found[spec.name] = spec
    _CACHE = found
    return found


def reset_skill_registry() -> None:
    """测试或热加载后清空进程内缓存。"""
    global _CACHE
    _CACHE = None


def format_skills_for_prompt(skills: list[SkillSpec] | None = None) -> str:
    """供 Planner 注入的短目录；只用 frontmatter，不读正文（避免泄露训记 Token）。"""
    items = skills if skills is not None else list_skills()
    lines: list[str] = []
    for spec in items:
        if not spec.description and not spec.has_handler:
            continue
        desc = spec.description.replace("\n", " ").strip()
        if len(desc) > 180:
            desc = desc[:177] + "…"
        kind = "可执行" if spec.has_handler else "文档"
        lines.append(f"- {spec.name}（{kind}）: {desc or '（无描述）'}")
    return "\n".join(lines) if lines else "- （当前未发现带描述的 Skill）"


def _scan_dir(directory: Path, source: str) -> list[SkillSpec]:
    specs: list[SkillSpec] = []
    for child in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        spec = load_skill_spec(child / "SKILL.md", source=source)
        if spec is not None:
            specs.append(spec)
    return specs
