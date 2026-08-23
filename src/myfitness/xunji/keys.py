"""训记鉴权配置检查 — 支持 .env 覆盖或从 Skill 文档读取。"""

from dataclasses import dataclass

from myfitness.config import Settings, get_settings
from myfitness.xunji.common import mask_api_key
from myfitness.xunji.skill_keys import key_source, resolve_xunji_keys
from myfitness.xunji.skills import PROJECT_ROOT, SKILL_DOC_PATHS, SKILL_BODY, SKILL_FOOD, SKILL_TRAINING

SKILL_NAMES = {
    "body": SKILL_BODY,
    "food": SKILL_FOOD,
    "food_search": SKILL_FOOD,
    "training": SKILL_TRAINING,
}


@dataclass(frozen=True)
class XunjiKeyStatus:
    name: str
    env_var: str
    configured: bool
    masked: str
    source: str  # env | skill | missing
    skill_path: str

    @property
    def source_label(self) -> str:
        if self.source == "env":
            return "环境变量"
        if self.source == "skill":
            return "Skill 文档"
        return "未配置"

    @property
    def hint(self) -> str:
        if self.configured:
            return f"{self.source_label}: {self.masked}"
        skill = SKILL_NAMES.get(self.name, "")
        path = SKILL_DOC_PATHS.get(skill)
        rel = str(path.relative_to(PROJECT_ROOT)) if path else "skills/.../SKILL.md"
        return f"请从训记 App 复制最新 {skill} Skill，覆盖 {rel}"


ENV_VAR_NAMES = {
    "body": "XUNJI_BODY_API_KEY",
    "food": "XUNJI_FOOD_API_KEY",
    "food_search": "XUNJI_FOOD_SEARCH_KEY",
    "training": "XUNJI_TRAINING_API_KEY",
}

SYNC_TYPE_KEYS = {
    "body": ["body"],
    "food": ["food"],
    "training": ["training"],
}


def get_key_statuses(settings: Settings | None = None) -> dict[str, XunjiKeyStatus]:
    s = settings or get_settings()
    resolved = resolve_xunji_keys(s)
    statuses: dict[str, XunjiKeyStatus] = {}
    for name, env_var in ENV_VAR_NAMES.items():
        value = resolved.get(name, "")
        configured = bool(value)
        skill = SKILL_NAMES.get(name, "")
        path = SKILL_DOC_PATHS.get(skill, "")
        statuses[name] = XunjiKeyStatus(
            name=name,
            env_var=env_var,
            configured=configured,
            masked=mask_api_key(value) if configured else "(empty)",
            source=key_source(name, s),
            skill_path=str(path) if path else "",
        )
    return statuses


def missing_keys_for_sync(types: list[str], settings: Settings | None = None) -> list[str]:
    """返回缺失的同步域名称（body/food/training），便于 CLI 提示。"""
    statuses = get_key_statuses(settings)
    missing: list[str] = []
    for sync_type in types:
        for key_name in SYNC_TYPE_KEYS.get(sync_type, []):
            if not statuses[key_name].configured:
                missing.append(key_name)
    return sorted(set(missing))


def ensure_sync_keys(types: list[str], settings: Settings | None = None) -> None:
    missing = missing_keys_for_sync(types, settings)
    if missing:
        statuses = get_key_statuses(settings)
        lines = []
        for name in missing:
            st = statuses[name]
            skill = SKILL_NAMES.get(name, "")
            lines.append(f"  - {skill}: {st.skill_path}")
        raise ValueError(
            "训记鉴权未就绪，无法同步。请从训记 App 复制最新 Skill 并覆盖：\n"
            + "\n".join(lines)
            + "\n（也可在 .env 中设置 XUNJI_* 覆盖 Skill 内嵌 Token。）"
        )
