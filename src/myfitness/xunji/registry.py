"""训记 Skill 注册与文档引用。"""

from myfitness.xunji.skills import SKILL_DOC_PATHS, SKILL_BODY, SKILL_FOOD, SKILL_TRAINING


def skill_doc_path(name: str) -> str:
    path = SKILL_DOC_PATHS.get(name)
    if path is None:
        raise KeyError(f"unknown skill: {name}")
    return str(path)


def assert_skill_docs_exist() -> None:
    missing = [name for name, path in SKILL_DOC_PATHS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 Skill 文档: {', '.join(missing)}")


__all__ = [
    "SKILL_BODY",
    "SKILL_FOOD",
    "SKILL_TRAINING",
    "skill_doc_path",
    "assert_skill_docs_exist",
]
