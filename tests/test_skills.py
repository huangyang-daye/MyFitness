"""即插即用 Skill 运行时与 query-database 抽取。"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from myfitness.api.cli import app
from myfitness.db.models import Base, BodyMetric, User
from myfitness.db.repositories.metrics import SOURCE_MANUAL
from myfitness.graph.planner import _build_planner_prompt
from myfitness.paths import BUILTIN_SKILLS_DIR
from myfitness.schemas.state import Intent
from myfitness.skills.loader import parse_frontmatter
from myfitness.skills.models import SkillContext
from myfitness.skills.registry import get_skill, list_skills, reset_skill_registry
from myfitness.skills.runtime import invoke_skill, reset_skill_modules, run_context_skills


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, name="test")
    session.add(user)
    session.flush()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _reset_skills():
    reset_skill_registry()
    reset_skill_modules()
    yield
    reset_skill_registry()
    reset_skill_modules()


def test_parse_frontmatter_folded_description_and_triggers():
    text = """---
name: query-database
description: >
  查询本地身体与训练数据。
handler: myfitness.skills.handlers.query_database:run
when: context
triggers:
  intents:
    - data_query
    - trend_analysis
  keywords:
    - 体重
---

# 正文
hello
"""
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "query-database"
    assert "身体" in meta["description"]
    assert meta["handler"].endswith(":run")
    assert meta["triggers"]["intents"] == ["data_query", "trend_analysis"]
    assert meta["triggers"]["keywords"] == ["体重"]
    assert body.startswith("# 正文")


def test_builtin_query_database_skill_is_discovered():
    spec = get_skill("query-database")
    assert spec is not None
    assert spec.source == "builtin"
    assert spec.has_handler
    assert spec.runs_in_context
    assert (BUILTIN_SKILLS_DIR / "query-database" / "SKILL.md").is_file()
    names = {item.name for item in list_skills()}
    assert "query-database" in names


def test_query_database_skill_executes_body_query(db_session):
    today = date(2026, 8, 23)
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=today,
            metric_type="weight",
            value=71.5,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    ctx = SkillContext(
        session=db_session,
        user_id=1,
        message="最近7天的体重",
        intent=Intent.TREND_ANALYSIS,
        domain="body",
        today=today,
    )
    result = invoke_skill("query-database", ctx)
    assert result.error is None
    assert "query_body_metrics" in result.tools_invoked
    body = result.data["body"]
    assert body["count"] == 1
    assert body["records"][0]["value"] == 71.5
    assert result.extra["plan"].lookback_days == 7


def test_project_skill_is_discovered_and_runs(tmp_path, monkeypatch, db_session):
    skill_dir = tmp_path / "echo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: echo-skill
description: 测试用回声 Skill
when: context
triggers:
  keywords:
    - 回声测试
---
""",
        encoding="utf-8",
    )
    (skill_dir / "handler.py").write_text(
        """
from myfitness.skills.models import SkillResult

def should_run(ctx):
    return "回声测试" in ctx.message

def run(ctx):
    return SkillResult(
        name="echo-skill",
        data={"echo": {"text": ctx.message}},
        tools_invoked=["echo-skill"],
    )
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "myfitness.skills.registry.skill_search_dirs",
        lambda: [(BUILTIN_SKILLS_DIR, "builtin"), (tmp_path, "project")],
    )
    reset_skill_registry()

    spec = get_skill("echo-skill")
    assert spec is not None
    assert spec.source == "project"
    assert spec.has_handler

    results = run_context_skills(
        SkillContext(
            session=db_session,
            user_id=1,
            message="回声测试一下",
            intent=Intent.GENERAL,
        )
    )
    by_name = {item.name: item for item in results}
    assert "echo-skill" in by_name
    assert by_name["echo-skill"].data["echo"]["text"] == "回声测试一下"
    assert "query-database" not in by_name


def test_project_skill_overrides_builtin(tmp_path, monkeypatch, db_session):
    skill_dir = tmp_path / "query-database"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: query-database
description: 覆盖用查询 Skill
handler: handler.py:run
when: context
---
""",
        encoding="utf-8",
    )
    (skill_dir / "handler.py").write_text(
        """
from myfitness.skills.models import SkillResult

def should_run(ctx):
    return True

def run(ctx):
    return SkillResult(
        name="query-database",
        data={"body": {"tool": "query_body_metrics", "count": 0, "records": []}},
        tools_invoked=["query_body_metrics"],
        extra={"overridden": True},
    )
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "myfitness.skills.registry.skill_search_dirs",
        lambda: [(BUILTIN_SKILLS_DIR, "builtin"), (tmp_path, "project")],
    )
    reset_skill_registry()

    spec = get_skill("query-database")
    assert spec.source == "project"
    result = invoke_skill(
        "query-database",
        SkillContext(session=db_session, user_id=1, message="体重", intent=Intent.DATA_QUERY),
    )
    assert result.extra.get("overridden") is True


def test_cli_skills_list_includes_builtin():
    runner = CliRunner()
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0, result.output
    assert "query-database" in result.output
    assert "内置" in result.output


def test_planner_prompt_lists_query_database_skill():
    prompt = _build_planner_prompt(date(2026, 9, 13))
    assert "query-database" in prompt
    assert "可用 Skill" in prompt
