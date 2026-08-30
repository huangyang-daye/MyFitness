"""训记 Skill 常量 — 与 skills/xunji-*/SKILL.md 保持同步。"""

from myfitness.paths import SKILLS_DIR

SKILL_BODY = "xunji-body-open-api"
SKILL_FOOD = "xunji-food-open-api"
SKILL_TRAINING = "xunji-training-open-api"

SKILL_DOC_PATHS = {
    SKILL_BODY: SKILLS_DIR / SKILL_BODY / "SKILL.md",
    SKILL_FOOD: SKILLS_DIR / SKILL_FOOD / "SKILL.md",
    SKILL_TRAINING: SKILLS_DIR / SKILL_TRAINING / "SKILL.md",
}

# --- Body Skill ---
BODY_BASE_URL = "https://api.xunjiapp.cn"
BODY_QUERY_PATH = "/open/body/query_gzip"
BODY_UPSERT_PATH = "/open/body/upsert_gzip"
BODY_SCHEMA_VERSION = "body_open_api_v1"
BODY_RATE_LIMIT_SECONDS = 15.0
BODY_QUERY_PAGE_SIZE = 500

BODY_METRIC_TYPES = frozenset(
    {
        "weight",
        "bodyfat",
        "neck",
        "chest",
        "weist",
        "shoulder",
        "bot",
        "arm_left",
        "arm_right",
        "forearm_left",
        "forearm_right",
        "leg_left",
        "leg_right",
        "cav_left",
        "cav_right",
    }
)

BODY_UNITS: dict[str, str] = {
    "weight": "kg",
    "bodyfat": "%",
}

# --- Food Skill ---
FOOD_BASE_URL = "https://eatings.xunjiapp.cn"
FOOD_QUERY_PATH = "/open/food/query_gzip"
FOOD_UPSERT_PATH = "/open/food/upsert_gzip"
FOOD_CUSTOM_UPSERT_PATH = "/open/food/custom/upsert_gzip"
FOOD_TEMPLATES_LIST_PATH = "/open/food/templates/list_gzip"
FOOD_TEMPLATES_APPLY_PATH = "/open/food/templates/apply_gzip"
FOOD_SEARCH_BASE_URL = "https://api.xunjiapp.cn"
FOOD_SEARCH_PATH = "/open_agent/food/search_gzip"
FOOD_RATE_LIMIT_SECONDS = 15.0
FOOD_QUERY_MAX_PAST_DAYS = 365
FOOD_QUERY_MAX_FUTURE_DAYS = 90

MEAL_TYPES = frozenset({"breakfast", "lunch", "dinner", "snack", "other"})

# --- Training Skill ---
TRAINING_BASE_URL = "https://trains.xunjiapp.cn"
TRAINING_READ_PATH = "/api_trains_for_llm_v2"
TRAINING_UPSERT_PATH = "/api_upsert_trains_for_llm_v2"
TRAINING_SCHEMA_VERSION = "train_open_api_v2"
TRAINING_RATE_LIMIT_LIGHT_SECONDS = 15.0
TRAINING_RATE_LIMIT_FULL_SECONDS = 30.0
TRAINING_RATE_LIMIT_WRITE_SECONDS = 45.0
TRAINING_UPSERT_MAX_TRAINS = 4
TRAINING_UPSERT_MAX_MOVEMENTS = 15
TRAINING_UPSERT_MAX_SETS = 20

PLAN_BASE_URL = "https://api.xunjiapp.cn"
PLAN_QUERY_PATH = "/open/plan/query_gzip"
PLAN_SCHEMA_VERSION = "plan_open_api_v1"
PLAN_RATE_LIMIT_SECONDS = 15.0
PLAN_MAX_RANGE_DAYS = 92

MOVEMENTS_URL = "https://github.com/Foveluy/Xunji-movements"

RPE_VALUES = frozenset({"6", "6.5", "7", "7.5", "8", "8.5", "9", "9.5", "10", ""})
DIFFICULTY_VALUES = frozenset({"easy", "normal", "hard"})

DISCLAIMER = "以上内容仅供参考，不构成医疗建议。如有健康问题请咨询专业医生。"
