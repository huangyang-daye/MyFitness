from datetime import date

from myfitness.schemas.agent_outputs import (
    BodyAgentOutput,
    FitnessAgentOutput,
    NutritionAgentOutput,
    SummaryAgentOutput,
)


def test_body_agent_output_schema():
    out = BodyAgentOutput(analysis_date=date.today(), narrative="test")
    assert out.agent == "body_monitor"
    assert out.model_dump()["agent"] == "body_monitor"


def test_nutrition_agent_output_schema():
    out = NutritionAgentOutput(analysis_date=date.today())
    assert out.agent == "nutritionist"


def test_fitness_agent_output_schema():
    out = FitnessAgentOutput(analysis_date=date.today())
    assert out.agent == "fitness_planner"


def test_summary_disclaimer_empty_by_default():
    out = SummaryAgentOutput(output_type="chat_reply", content_md="hello")
    assert out.disclaimer == ""
