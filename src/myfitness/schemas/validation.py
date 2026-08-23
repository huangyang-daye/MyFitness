"""Schema 校验工具。"""

from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def validate_model(model_cls: type[T], data: dict) -> T:
    return model_cls.model_validate(data)


def safe_validate(model_cls: type[T], data: dict) -> tuple[T | None, str | None]:
    try:
        return model_cls.model_validate(data), None
    except ValidationError as exc:
        return None, str(exc)
