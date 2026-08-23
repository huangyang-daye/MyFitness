"""训记 Open API 聚合客户端。"""

from dataclasses import dataclass, field

from myfitness.xunji.body import BodyOpenApi
from myfitness.xunji.common import XunjiHttpClient, mask_api_key
from myfitness.xunji.food import FoodOpenApi
from myfitness.xunji.training import TrainingOpenApi


@dataclass
class XunjiClient:
    """组合 skills/xunji-*/SKILL.md 三个域客户端，共享 HTTP 限频与缓存。"""

    body_api_key: str = ""
    food_api_key: str = ""
    food_search_key: str = ""
    training_api_key: str = ""
    cache_ttl_seconds: int = 300

    _http: XunjiHttpClient = field(init=False, repr=False)
    body: BodyOpenApi = field(init=False, repr=False)
    food: FoodOpenApi = field(init=False, repr=False)
    training: TrainingOpenApi = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = XunjiHttpClient(cache_ttl_seconds=self.cache_ttl_seconds)
        self.body = BodyOpenApi(self.body_api_key, self._http)
        self.food = FoodOpenApi(self.food_api_key, self.food_search_key, self._http)
        self.training = TrainingOpenApi(self.training_api_key, self._http)

    def query_body(self, start_date: str, end_date: str, types: list[str] | None = None) -> dict:
        return self.body.query_all_records(start_date, end_date, types=types)

    def query_food(self, start_date: str, end_date: str, include_detail: bool = True) -> dict:
        result, notes = self.food.query_range_merged(
            start_date, end_date, include_detail=include_detail
        )
        if notes:
            result["_skill_notes"] = notes
        return result

    def query_training(self, datestr: str, include_full_data: bool = False) -> dict:
        return self.training.read(datestr, include_full_data=include_full_data)

    @staticmethod
    def mask_key(key: str) -> str:
        return mask_api_key(key)

    _mask_key = mask_key
