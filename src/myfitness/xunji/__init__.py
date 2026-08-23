"""训记 Open API — 按 skills/xunji-*/SKILL.md 实现。"""

from myfitness.xunji.body import BodyOpenApi
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.common import XunjiApiError, XunjiRateLimitError, mask_api_key
from myfitness.xunji.food import FoodOpenApi, clamp_food_query_range
from myfitness.xunji.registry import assert_skill_docs_exist, skill_doc_path
from myfitness.xunji.training import TrainingOpenApi
from myfitness.xunji.write_flow import WritePreview, preview_body_write

__all__ = [
    "BodyOpenApi",
    "FoodOpenApi",
    "TrainingOpenApi",
    "XunjiClient",
    "XunjiApiError",
    "XunjiRateLimitError",
    "WritePreview",
    "assert_skill_docs_exist",
    "clamp_food_query_range",
    "mask_api_key",
    "preview_body_write",
    "skill_doc_path",
]
