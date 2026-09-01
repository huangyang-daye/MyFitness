"""记忆系统：短期压缩、长期画像、知识库写入。"""

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.summary import build_rule_based_summary
from myfitness.db.models import Base, User
from myfitness.db.repositories.knowledge import KnowledgeRepository
from myfitness.graph.chat import new_chat_state, prepare_chat_turn
from myfitness.memory.long_term import update_long_term_from_message
from myfitness.memory.manager import apply_memory_for_turn, attach_memory
from myfitness.memory.profile import extract_profile_facts, merge_profile, profile_to_markdown
from myfitness.memory.short_term import build_short_term
from myfitness.memory.types import PROFILE_TITLE
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import ChatMessage, ContextSnapshot, DateRange, Intent


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test", profile={}))
    session.flush()
    yield session
    session.close()


def test_rule_extract_goals_and_diet():
    facts = extract_profile_facts(
        "我的目标体重是70kg，而且乳糖不耐",
        intent=Intent.GOAL_SETTING,
        use_llm=False,
    )
    assert any("70" in item for item in facts["goals"])
    assert any("乳糖" in item for item in facts["diet"])


def test_skip_greeting_and_confirmation():
    assert extract_profile_facts("你好", intent=Intent.GENERAL, use_llm=False) == {}
    assert extract_profile_facts("确认", intent=Intent.CONFIRMATION_RESPONSE, use_llm=False) == {}


def test_merge_profile_dedup_and_cap():
    merged = merge_profile(
        {"goals": ["减脂", "目标体重 72kg"]},
        {"goals": ["减脂", "目标体重70kg"], "diet": ["少油"]},
        max_items=8,
    )
    assert merged["goals"].count("减脂") == 1
    assert "少油" in merged["diet"]
    markdown = profile_to_markdown(merged)
    assert "用户画像" in markdown
    assert "少油" in markdown


def test_short_term_compresses_overflow():
    state = new_chat_state(user_id=1)
    for index in range(12):
        role = "user" if index % 2 == 0 else "assistant"
        state.messages.append(ChatMessage(role=role, content=f"回合{index} 内容"))
    with patch("myfitness.memory.short_term.get_settings") as settings:
        settings.return_value.memory_short_term_turns = 4
        settings.return_value.memory_compress_chars = 400
        text, compressed = build_short_term(state, use_llm=False)
    assert compressed
    assert state.memory_compacted_count == 8
    assert "较早对话摘要" in text
    assert "最近对话" in text
    assert "回合11" in text
    assert len(state.messages) == 12


def test_long_term_writes_knowledge(db_session):
    with patch(
        "myfitness.memory.long_term.index_knowledge_entry",
        return_value={"indexed": 1, "skipped": 0, "failed": 0},
    ):
        profile, markdown, changed = update_long_term_from_message(
            db_session,
            1,
            "我的目标体重是70kg，乳糖不耐",
            intent=Intent.GOAL_SETTING,
            use_llm=False,
        )
    assert changed
    assert any("70" in item for item in profile["goals"])
    assert "用户画像" in markdown
    entries = KnowledgeRepository(db_session, 1).list_all()
    assert len(entries) == 1
    assert entries[0].kind == "memory"
    assert entries[0].title == PROFILE_TITLE
    user = db_session.get(User, 1)
    assert user is not None
    assert "goals" in (user.profile or {})


def test_attach_memory_to_context():
    context = ContextSnapshot(date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 7)))
    from myfitness.memory.types import MemoryBundle

    updated = attach_memory(
        context,
        MemoryBundle(short_term="最近对话：用户：近7天体重", long_term="# 用户画像\n- 减脂"),
    )
    assert "减脂" in updated.memory_long_term
    assert "近7天体重" in updated.memory_short_term


def test_rule_summary_includes_memory():
    context = ContextSnapshot(
        date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 7)),
        memory_long_term="# 用户画像\n- 目标体重 70kg",
        memory_short_term="用户：近7天体重怎么变了",
    )
    text = build_rule_based_summary(AgentOutputs(), context, Intent.GENERAL)
    assert "70kg" in text
    assert "近7天体重" in text


def test_prepare_chat_turn_updates_profile(db_session):
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.memory.profile.is_llm_configured", return_value=False),
        patch("myfitness.memory.compress.is_llm_configured", return_value=False),
        patch(
            "myfitness.memory.long_term.index_knowledge_entry",
            return_value={"indexed": 0, "skipped": 1, "failed": 0},
        ),
    ):
        result = prepare_chat_turn(db_session, state, "我的目标体重是70kg")
    assert result.state.context is not None
    assert "70" in result.state.context.memory_long_term
    repo_rows = KnowledgeRepository(db_session, 1).list_all()
    assert repo_rows
    assert repo_rows[0].kind == "memory"


def test_apply_memory_disabled(db_session):
    state = new_chat_state(user_id=1)
    state.user_message = "目标体重70kg"
    state.messages.append(ChatMessage(role="user", content=state.user_message))
    with patch("myfitness.memory.manager.get_settings") as settings:
        settings.return_value.memory_enabled = False
        bundle = apply_memory_for_turn(db_session, state, intent=Intent.GOAL_SETTING)
    assert bundle.short_term == ""
    assert bundle.long_term == ""
