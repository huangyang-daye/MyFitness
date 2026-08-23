"""已迁移至 myfitness.xunji，保留此模块以兼容旧导入。"""

from myfitness.xunji import (  # noqa: F401
    BodyOpenApi,
    FoodOpenApi,
    TrainingOpenApi,
    XunjiApiError,
    XunjiClient,
    XunjiRateLimitError,
    mask_api_key,
)
