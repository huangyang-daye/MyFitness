"""MyFitness RAG — pgvector 语义检索。"""

from myfitness.rag.indexer import index_user_data
from myfitness.rag.pipeline import (
    maybe_index_after_report,
    maybe_index_after_sync,
    retrieve_for_turn,
)
from myfitness.rag.store import chunk_stats, count_chunks, search_chunks

__all__ = [
    "chunk_stats",
    "count_chunks",
    "index_user_data",
    "maybe_index_after_report",
    "maybe_index_after_sync",
    "retrieve_for_turn",
    "search_chunks",
]
