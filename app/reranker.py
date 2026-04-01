from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings
from app.utils import tokenize

if TYPE_CHECKING:
    from app.retriever import RetrievalResult


class HeuristicReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        from app.retriever import RetrievalResult

        if not results:
            return []

        query_tokens = set(tokenize(query))
        reranked: list[RetrievalResult] = []
        for result in results[: self.settings.rerank_candidate_count]:
            symbol_tokens = set(tokenize(result.chunk["symbol"]))
            path_tokens = set(tokenize(result.chunk["file_path"]))
            exact_path_bonus = 0.0
            for token in query_tokens:
                if token and token in result.chunk["file_path"].lower():
                    exact_path_bonus = 1.0
                    break

            symbol_overlap = len(query_tokens & symbol_tokens) / max(1, len(query_tokens))
            path_overlap = len(query_tokens & path_tokens) / max(1, len(query_tokens))
            chunk_type_bonus = 0.15 if result.chunk["chunk_type"] in query.lower() else 0.0
            rerank_score = min(1.0, 0.45 * symbol_overlap + 0.30 * path_overlap + 0.15 * exact_path_bonus + chunk_type_bonus)
            final_score = ((1 - self.settings.rerank_weight) * result.final_score) + (self.settings.rerank_weight * rerank_score)

            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    semantic_score=result.semantic_score,
                    keyword_score=result.keyword_score,
                    bm25_score=result.bm25_score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                )
            )

        remainder = results[self.settings.rerank_candidate_count :]
        reranked.extend(remainder)
        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked[:top_k]
