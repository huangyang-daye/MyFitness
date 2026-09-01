"""数据库自定义类型 — pgvector 与 SQLite 测试兼容。"""

from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator, UserDefinedType


class EmbeddingVector(UserDefinedType):
    """PostgreSQL 使用 pgvector；SQLite 测试环境回落 JSON 数组。"""

    cache_ok = True

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **kw: object) -> str:  # noqa: ARG002
        return f"vector({self.dimensions})"

    def bind_processor(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            return None

        def process(value):  # type: ignore[no-untyped-def]
            return value

        return process

    def result_processor(self, dialect, coltype):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            return None

        def process(value):  # type: ignore[no-untyped-def]
            return value

        return process

    class Comparator:
        def cosine_distance(self, other):  # type: ignore[no-untyped-def]
            return self.op("<=>")(other)

    comparator_factory = Comparator


class SqliteEmbeddingFallback(TypeDecorator):
    """非 PostgreSQL 方言下以 JSON 存储 embedding。"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            return dialect.type_descriptor(EmbeddingVector())
        return dialect.type_descriptor(JSON())
