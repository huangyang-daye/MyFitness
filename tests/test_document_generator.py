"""文档生成器测试。"""

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.document_generator import generate_document_body, infer_document_title
from myfitness.agents.tools.document_tools import apply_document_export, wants_minimal_chat_for_document
from myfitness.config import get_settings
from myfitness.db.models import Base
from myfitness.schemas.agent_outputs import AgentOutputs, NutritionAgentOutput
from myfitness.schemas.state import ContextSnapshot, DateRange


@pytest.fixture
def document_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data))
    get_settings.cache_clear()
    yield data / "documents"
    get_settings.cache_clear()


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_infer_document_title():
    assert infer_document_title("生成饮食规划文档") == "饮食规划"
    assert infer_document_title("写训练计划文档") == "训练计划"


def test_wants_minimal_chat_for_document():
    message = "规划饮食构成，不要输出其他内容，生成规划然后输出为文档"
    assert wants_minimal_chat_for_document(message)


def test_generate_document_body_rule_based_excludes_query_dump():
    outputs = AgentOutputs(
        nutrition=NutritionAgentOutput(
            analysis_date=date(2026, 9, 1),
            narrative="建议每日蛋白 160g，碳水 300g。",
        ),
    )
    context = ContextSnapshot(
        date_range=DateRange(start=date(2026, 9, 1), end=date(2026, 9, 1)),
        query_results={
            "body": {
                "start_date": "2026-09-01",
                "end_date": "2026-09-01",
                "count": 1,
                "records": [],
            }
        },
    )
    fallback = "**数据库查询结果**\n【数据库·身体数据】...\n\n**饮食**\n不应出现在文档里"

    with patch("myfitness.agents.document_generator.is_llm_configured", return_value=False):
        body = generate_document_body(
            "根据3倍碳水原则生成饮食规划文档",
            outputs,
            context,
            fallback=fallback,
        )

    assert "数据库查询结果" not in body
    assert "160g" in body
    assert "# 饮食规划" in body


def test_apply_document_export_uses_generated_body_not_chat_reply(db_session, document_dir):
    outputs = AgentOutputs(
        nutrition=NutritionAgentOutput(
            analysis_date=date(2026, 9, 1),
            narrative="每日蛋白目标 150g。",
        ),
    )
    chat_reply = "**数据库查询结果**\n原始聊天内容不应写入文件"

    with patch(
        "myfitness.agents.document_generator.chat_completion",
        return_value="# 饮食规划\n\n## 每日目标\n\n蛋白 150g。",
    ):
        result = apply_document_export(
            db_session,
            1,
            "生成饮食规划然后输出为文档，不要输出其他内容",
            chat_reply,
            agent_outputs=outputs,
            context=None,
        )

    assert result is not None
    assert result.get("document_only") is True
    saved = open(result["path"], encoding="utf-8").read()
    assert "数据库查询结果" not in saved
    assert "饮食规划" in saved
