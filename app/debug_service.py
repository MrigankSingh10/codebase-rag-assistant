from __future__ import annotations

import re
import logging

from app.config import Settings
from app.prompts import DEBUG_SYSTEM_PROMPT, build_debug_prompt
from app.qa_service import BaseLLMClient, LocalGroundedClient, build_llm_client
from app.retriever import HybridRetriever, RetrievalResult
from app.utils import tokenize, unique_by_key


ERROR_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]+")
logger = logging.getLogger(__name__)


class DebugService:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm_client: BaseLLMClient | None = None) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm_client = llm_client or build_llm_client(settings)

    def debug(self, error_context: str, question: str | None = None, repo_id: str | None = None) -> dict:
        retrieval_query = self._build_debug_query(question=question, error_context=error_context)
        results = self.retriever.retrieve(retrieval_query, repo_id=repo_id, top_k=self.settings.top_k)
        if not results:
            return {"repo_id": repo_id, "answer": "not found", "citations": [], "confidence": "low"}

        citations = self._build_citations(results)
        confidence = self._confidence_from_results(results)
        selected_repo_id = results[0].chunk.get("repo_id")

        if isinstance(self.llm_client, LocalGroundedClient):
            answer = self._build_local_debug_answer(error_context=error_context, question=question, results=results)
        else:
            prompt = build_debug_prompt(question=question, error_context=error_context, chunks=[result.chunk for result in results])
            try:
                answer = self.llm_client.generate(DEBUG_SYSTEM_PROMPT, prompt)
            except Exception as exc:
                logger.warning("Falling back to local debug answer after LLM failure: %s", exc)
                answer = self._build_local_debug_answer(error_context=error_context, question=question, results=results)

        return {"repo_id": selected_repo_id, "answer": answer, "citations": citations, "confidence": confidence}

    def _build_debug_query(self, question: str | None, error_context: str) -> str:
        keywords = ERROR_TOKEN_PATTERN.findall(error_context)
        deduped_keywords: list[str] = []
        seen: set[str] = set()
        for token in keywords:
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped_keywords.append(token)
        joined = " ".join(deduped_keywords[:25])
        return f"{question or ''}\n{error_context}\n{joined}".strip()

    def _build_local_debug_answer(
        self,
        error_context: str,
        question: str | None,
        results: list[RetrievalResult],
    ) -> str:
        exception_names = [token for token in ERROR_TOKEN_PATTERN.findall(error_context) if token.endswith(("Error", "Exception"))]
        top_files = ", ".join(f"{result.chunk['file_path']}:{result.chunk['start_line']}-{result.chunk['end_line']}" for result in results[:3])
        relevant_lines = self._extract_debug_lines(error_context=error_context, results=results)
        likely_cause = relevant_lines[0] if relevant_lines else "The indexed context does not expose a clear failure point."
        checks = relevant_lines[1:3] if len(relevant_lines) > 1 else []

        sections = [
            f"Likely cause: {likely_cause}",
            f"Affected files: {top_files}",
        ]
        if exception_names:
            sections.append(f"Error signals: {', '.join(exception_names[:3])}")
        if checks:
            sections.append(f"Checks: {' | '.join(checks)}")
        if question:
            sections.append(f"Focus: {question}")
        return "\n".join(sections)

    def _extract_debug_lines(self, error_context: str, results: list[RetrievalResult]) -> list[str]:
        query_tokens = set(tokenize(error_context))
        scored: list[tuple[int, str]] = []
        for result in results:
            for line in result.chunk["text"].splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                overlap = len(query_tokens & set(tokenize(stripped)))
                if overlap > 0:
                    scored.append((overlap, stripped))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in scored[:4]]

    def _build_citations(self, results: list[RetrievalResult]) -> list[dict]:
        citations = [
            {
                "ref": f"{result.chunk['repo_id']}:{result.chunk['file_path']}:{result.chunk['symbol']}:{result.chunk['start_line']}-{result.chunk['end_line']}",
                "repo_id": result.chunk["repo_id"],
                "file_path": result.chunk["file_path"],
                "symbol": result.chunk["symbol"],
                "lines": f"{result.chunk['start_line']}-{result.chunk['end_line']}",
                "score": round(result.final_score, 4),
            }
            for result in results
        ]
        unique = unique_by_key(citations, "ref")
        for citation in unique:
            citation.pop("ref", None)
        return unique[:5]

    def _confidence_from_results(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "low"
        if results[0].final_score >= 0.62:
            return "high"
        if results[0].final_score >= 0.40:
            return "medium"
        return "low"
