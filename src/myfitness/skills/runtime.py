"""按约定加载并调用 Skill handler。"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from myfitness.skills.models import SkillContext, SkillResult, SkillSpec
from myfitness.skills.registry import get_skill, list_skills

logger = logging.getLogger(__name__)

_LOADED_MODULES: dict[str, ModuleType] = {}


def run_context_skills(ctx: SkillContext) -> list[SkillResult]:
    """执行本轮应自动运行的 context-phase Skill。"""
    results: list[SkillResult] = []
    for spec in list_skills():
        if not spec.has_handler or not spec.runs_in_context:
            continue
        if not _should_run(spec, ctx):
            continue
        results.append(_invoke_spec(spec, ctx))
    return results


def invoke_skill(name: str, ctx: SkillContext) -> SkillResult:
    spec = get_skill(name)
    if spec is None:
        return SkillResult(name=name, error=f"未知 Skill: {name}")
    if not spec.has_handler:
        return SkillResult(name=spec.name, error=f"Skill {spec.name} 没有 handler")
    return _invoke_spec(spec, ctx)


def reset_skill_modules() -> None:
    """测试用：卸载动态加载的 handler 模块。"""
    for key in list(_LOADED_MODULES):
        module = _LOADED_MODULES.pop(key)
        sys.modules.pop(module.__name__, None)


def _should_run(spec: SkillSpec, ctx: SkillContext) -> bool:
    module = _load_handler_module(spec)
    should_run = getattr(module, "should_run", None)
    if callable(should_run):
        return bool(should_run(ctx))
    return _trigger_match(spec, ctx)


def _trigger_match(spec: SkillSpec, ctx: SkillContext) -> bool:
    intents = spec.triggers.get("intents") or []
    keywords = spec.triggers.get("keywords") or []
    if not intents and not keywords:
        return False
    intent_hit = ctx.intent.value in intents if intents else False
    keyword_hit = any(keyword in ctx.message for keyword in keywords) if keywords else False
    if intents and keywords:
        return intent_hit or keyword_hit
    return intent_hit or keyword_hit


def _invoke_spec(spec: SkillSpec, ctx: SkillContext) -> SkillResult:
    try:
        module = _load_handler_module(spec)
        run = _resolve_run(spec, module)
        result = run(ctx)
        if isinstance(result, SkillResult):
            return result
        if isinstance(result, dict):
            return SkillResult(name=spec.name, data=result)
        return SkillResult(name=spec.name, extra={"raw": result})
    except Exception as exc:
        if spec.source == "builtin":
            raise
        logger.exception("项目 Skill %s 执行失败", spec.name)
        return SkillResult(name=spec.name, error=str(exc))


def _resolve_run(spec: SkillSpec, module: ModuleType) -> Callable:
    attr = "run"
    if spec.handler and ":" in spec.handler:
        attr = spec.handler.rsplit(":", 1)[-1]
    run = getattr(module, attr, None)
    if not callable(run):
        raise TypeError(f"Skill {spec.name} 缺少可调用的 {attr}()")
    return run


def _load_handler_module(spec: SkillSpec) -> ModuleType:
    cache_key = f"{spec.source}:{spec.name}:{spec.handler}"
    cached = _LOADED_MODULES.get(cache_key)
    if cached is not None:
        return cached

    handler = spec.handler or "handler.py:run"
    module_ref, _, _attr = handler.partition(":")
    module_ref = module_ref.strip()
    if _looks_like_file(module_ref):
        path = Path(module_ref)
        if not path.is_absolute():
            path = spec.path / module_ref
        module = _load_from_path(spec.name, path)
    else:
        module = importlib.import_module(module_ref)
    _LOADED_MODULES[cache_key] = module
    return module


def _looks_like_file(ref: str) -> bool:
    return ref.endswith(".py") or "/" in ref or "\\" in ref


def _load_from_path(skill_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Skill handler 不存在: {path}")
    mod_name = f"myfitness_skill_{skill_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Skill handler: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
