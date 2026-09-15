"""即插即用 Skill 运行时。

扫描顺序：
1. 内置目录 ``src/myfitness/skills/catalog/``
2. 项目目录 ``skills/``（可覆盖同名内置 Skill）

每个 Skill 是含 ``SKILL.md`` 的文件夹；可选 ``handler.py``（``run(ctx)``）
或 frontmatter 里的 ``handler: module:function``。
"""

from myfitness.skills.models import SkillContext, SkillResult, SkillSpec
from myfitness.skills.registry import (
    format_skills_for_prompt,
    get_skill,
    list_skills,
    reset_skill_registry,
)
from myfitness.skills.runtime import invoke_skill, reset_skill_modules, run_context_skills

__all__ = [
    "SkillContext",
    "SkillResult",
    "SkillSpec",
    "format_skills_for_prompt",
    "get_skill",
    "invoke_skill",
    "list_skills",
    "reset_skill_modules",
    "reset_skill_registry",
    "run_context_skills",
]
