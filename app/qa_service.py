from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Literal

from openai import OpenAI

from app.config import Settings
from app.prompts import QA_SYSTEM_PROMPT, build_qa_prompt
from app.retriever import HybridRetriever, RetrievalResult
from app.utils import tokenize, unique_by_key

if TYPE_CHECKING:
    from app.summary_service import SummaryService

logger = logging.getLogger(__name__)
REPOSITORY_OVERVIEW_PATTERNS = (
    r"\bexplain (this|the) (repository|repo|project|codebase)\b",
    r"\bwhat does (this|the) (repository|repo|project|codebase) do\b",
    r"\boverview of (this|the) (repository|repo|project|codebase)\b",
    r"\bsummariz(?:e|ing) (this|the) (repository|repo|project|codebase)\b",
    r"\bdescribe (this|the) (repository|repo|project|codebase)\b",
)
FRONTEND_SCOPE_PATTERN = re.compile(r"\b(frontend|front-end|ui|angular|client-side|client side|browser)\b")
BACKEND_SCOPE_PATTERN = re.compile(r"\b(backend|back-end|server-side|server side|api|django)\b")
DOCUMENT_QUERY_PATTERN = re.compile(
    r"^\s*(what are|list|summarize|summary of|explain|types of)\b|\b(pattern|patterns|document|notes|theory)\b"
)
PATTERN_HEADING_PATTERN = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*{1,2})?(?:\d+\s*[.)]\s*)?(?P<name>[A-Z][A-Za-z /&-]*?Pattern)\b",
)
TOKEN_CREATION_PATTERN = re.compile(
    r"\b(jwt|token|refresh token|access token)\b.*\b(create|created|generate|generated|issue|issued|mint|minted)\b"
    r"|\b(create|created|generate|generated|issue|issued|mint|minted)\b.*\b(jwt|token|refresh token|access token)\b"
)
TOKEN_VALIDATION_PATTERN = re.compile(
    r"\b(jwt|token|refresh token|access token)\b.*\b(validate|validated|verify|verified|blacklist|blacklisted|revoke|revoked)\b"
    r"|\b(validate|validated|verify|verified|blacklist|blacklisted|revoke|revoked)\b.*\b(jwt|token|refresh token|access token)\b"
)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()


class GeminiClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("google-genai must be installed when LLM_PROVIDER=gemini") from exc

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return (response.text or "").strip()


class LocalGroundedClient(BaseLLMClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Local grounded answers are assembled directly in the service layer.")


def build_llm_client(settings: Settings) -> BaseLLMClient:
    if settings.llm_provider == "openai":
        return OpenAIClient(settings)
    if settings.llm_provider == "gemini":
        return GeminiClient(settings)
    return LocalGroundedClient()


class QAService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm_client: BaseLLMClient | None = None,
        summary_service: "SummaryService | None" = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm_client = llm_client or build_llm_client(settings)
        self.summary_service = summary_service

    def answer_question(self, question: str, repo_id: str | None = None) -> dict:
        if self._is_repository_overview_question(question):
            return self._answer_repository_overview(question=question, repo_id=repo_id)

        scope = self._question_scope(question)
        retrieval_query = self._build_retrieval_query(question)
        include_document_neighbors = self._is_document_question(question)
        include_topic_note_aggregation = self._is_topic_note_question(question)
        prefer_full_note_file = self._is_structured_note_list_question(question)
        if prefer_full_note_file:
            results = self.retriever.retrieve_note_documents(question, repo_id=repo_id, max_files=3)
            if not results:
                return {"repo_id": repo_id, "answer": "not found", "citations": [], "confidence": "low"}

            citations = self._build_citations(results)
            confidence = self._confidence_from_results(results)
            selected_repo_id = results[0].chunk.get("repo_id")
            answer = self._build_note_list_answer(question, results)
            return {"repo_id": selected_repo_id, "answer": answer, "citations": citations, "confidence": confidence}

        retrieval_top_k = max(self.settings.top_k, 12) if prefer_full_note_file else self.settings.top_k
        logger.info(
            "QA retrieval question=%r scope=%s retrieval_query=%r document_neighbors=%s topic_note_aggregation=%s full_note_file=%s top_k=%s",
            question,
            scope or "none",
            retrieval_query,
            include_document_neighbors,
            include_topic_note_aggregation,
            prefer_full_note_file,
            retrieval_top_k,
        )
        results = self.retriever.retrieve(
            retrieval_query,
            repo_id=repo_id,
            top_k=retrieval_top_k,
            scope=scope,
            include_document_neighbors=include_document_neighbors,
            include_topic_note_aggregation=include_topic_note_aggregation,
            prefer_full_note_file=prefer_full_note_file,
        )
        if not results:
            return {"repo_id": repo_id, "answer": "not found", "citations": [], "confidence": "low"}

        logger.info(
            "QA top retrievals=%s",
            [
                {
                    "file_path": result.chunk["file_path"],
                    "symbol": result.chunk["symbol"],
                    "score": round(result.final_score, 4),
                }
                for result in results[:3]
            ],
        )

        citations = self._build_citations(results)
        confidence = self._confidence_from_results(results)
        selected_repo_id = results[0].chunk.get("repo_id")

        if prefer_full_note_file:
            answer = self._build_note_list_answer(question, results)
        elif isinstance(self.llm_client, LocalGroundedClient):
            answer = self._build_local_answer(question, results)
        else:
            prompt = build_qa_prompt(question, [result.chunk for result in results])
            try:
                answer = self.llm_client.generate(QA_SYSTEM_PROMPT, prompt)
            except Exception as exc:
                logger.warning("Falling back to local QA answer after LLM failure: %s", exc)
                answer = self._build_local_answer(question, results)

        return {"repo_id": selected_repo_id, "answer": answer, "citations": citations, "confidence": confidence}

    def _answer_repository_overview(self, question: str, repo_id: str | None = None) -> dict:
        if self.summary_service is None:
            return {"repo_id": repo_id, "answer": "not found", "citations": [], "confidence": "low"}

        payload = self.summary_service.summarize_repository(repo_id=repo_id, refresh=False)
        citations = self._build_summary_citations(payload)
        confidence = self.summary_service.overview_confidence(payload)
        return {
            "repo_id": payload["repo_id"],
            "answer": self._build_repository_overview_answer(question, payload),
            "citations": citations,
            "confidence": confidence,
        }

    def _build_repository_overview_answer(self, question: str, payload: dict) -> str:
        summary = (payload.get("summary") or "not found").strip()
        stats = payload.get("stats") or {}
        directories = stats.get("directories") or []
        extensions = stats.get("extensions") or []
        highlights = payload.get("highlights") or []

        notable_areas = ", ".join(f"{name} ({count})" for name, count in directories[:3])
        notable_types = ", ".join(f"{ext} ({count})" for ext, count in extensions[:3])
        highlight_text = ", ".join(f"{item['file_path']}::{item['symbol']}" for item in highlights[:3])
        backend_highlights = [item for item in highlights if "pricing_optimizations/" in item["file_path"]]
        frontend_highlights = [item for item in highlights if "price-optimization-ui/" in item["file_path"]]
        directory_names = [name for name, _ in directories]

        lead = self._build_repository_overview_lead(payload, directory_names, backend_highlights, frontend_highlights)
        sections = [lead]
        if summary and summary.lower() != "not found":
            sections.append(f"Summary: {summary}")
        if backend_highlights:
            backend_items = ", ".join(f"{item['file_path']}::{item['symbol']}" for item in backend_highlights[:2])
            sections.append(f"Backend implementation points: {backend_items}.")
        if frontend_highlights:
            frontend_items = ", ".join(f"{item['file_path']}::{item['symbol']}" for item in frontend_highlights[:2])
            sections.append(f"Frontend implementation points: {frontend_items}.")
        if notable_areas:
            sections.append(f"Main areas: {notable_areas}.")
        if notable_types:
            sections.append(f"Primary file types: {notable_types}.")
        if highlight_text:
            sections.append(f"Representative implementation points: {highlight_text}.")
        if "child" in question.lower():
            sections[0] = f"In simple terms: {sections[0]}"
        return "\n\n".join(sections)

    def _build_repository_overview_lead(
        self,
        payload: dict,
        directory_names: list[str],
        backend_highlights: list[dict],
        frontend_highlights: list[dict],
    ) -> str:
        repo_name = payload.get("repo_name") or "This repository"
        has_backend = "pricing_optimizations" in directory_names or bool(backend_highlights)
        has_frontend = "price-optimization-ui" in directory_names or bool(frontend_highlights)

        if has_backend and has_frontend:
            return (
                f"{repo_name} is a full-stack price optimization application with a Python/Django-style backend "
                f"and a TypeScript/Angular-style frontend."
            )
        if has_backend:
            return f"{repo_name} is primarily a backend application focused on product and pricing workflows."
        if has_frontend:
            return f"{repo_name} is primarily a frontend application focused on product and pricing workflows."
        return f"{repo_name} is a codebase with multiple application areas."

    def _build_summary_citations(self, payload: dict) -> list[dict]:
        highlights = payload.get("highlights") or []
        return [
            {
                "repo_id": item["repo_id"],
                "file_path": item["file_path"],
                "symbol": item["symbol"],
                "lines": item["lines"],
                "score": None,
            }
            for item in highlights[:5]
        ]

    def _is_repository_overview_question(self, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        return any(re.search(pattern, normalized) for pattern in REPOSITORY_OVERVIEW_PATTERNS)

    def _question_scope(self, question: str) -> Literal["frontend", "backend"] | None:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        has_frontend = bool(FRONTEND_SCOPE_PATTERN.search(normalized))
        has_backend = bool(BACKEND_SCOPE_PATTERN.search(normalized))
        if has_frontend and not has_backend:
            return "frontend"
        if has_backend and not has_frontend:
            return "backend"
        return None

    def _is_document_question(self, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        return bool(DOCUMENT_QUERY_PATTERN.search(normalized))

    def _is_topic_note_question(self, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        return self._is_document_question(question) and any(
            phrase in normalized
            for phrase in (
                "list all",
                "mentioned in this document",
                "from my notes",
                "what are",
                "summarize",
            )
        )

    def _is_structured_note_list_question(self, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        return self._is_document_question(question) and any(
            phrase in normalized
            for phrase in (
                "list all",
                "what are",
                "which are",
                "all the",
            )
        )

    def _build_retrieval_query(self, question: str) -> str:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        expansions: list[str] = []

        if TOKEN_CREATION_PATTERN.search(normalized):
            expansions.extend(
                [
                    "RefreshToken.for_user",
                    "refresh",
                    "access",
                    "login",
                    "LoginView",
                    "token generation",
                ]
            )

        if TOKEN_VALIDATION_PATTERN.search(normalized):
            expansions.extend(
                [
                    "Authorization Bearer",
                    "blacklist",
                    "BlacklistJWTMiddleware",
                    "blacklisted token",
                    "refresh.blacklist",
                ]
            )

        if not expansions:
            return question

        return f"{question}\n{' '.join(expansions)}"

    def _build_local_answer(self, question: str, results: list[RetrievalResult]) -> str:
        ranked_sentences = self._extract_relevant_sentences(question, (result.chunk["text"] for result in results))
        if not ranked_sentences:
            return "not found"

        citations = ", ".join(
            f"{result.chunk['file_path']}:{result.chunk['start_line']}-{result.chunk['end_line']}"
            for result in results[:3]
        )
        summary = " ".join(ranked_sentences[:3]).strip()
        return f"{summary}\n\nCitations: {citations}"

    def _build_note_list_answer(self, question: str, results: list[RetrievalResult]) -> str:
        pattern_items = self._extract_pattern_items(question, results)
        if pattern_items:
            lines = [f"*   {name}" for name in pattern_items[:12]]
            if "creational" in question.lower():
                intro = "The creational design patterns mentioned are:"
            else:
                intro = "The design patterns mentioned are:"
            return f"{intro}\n\n" + "\n".join(lines)
        return self._build_local_answer(question, results)

    def _extract_pattern_items(self, question: str, results: list[RetrievalResult]) -> list[str]:
        file_focus_tokens = self._question_note_focus_tokens(question) or self._note_focus_tokens(results)
        found: list[str] = []
        seen: set[str] = set()
        for result in results:
            chunk = result.chunk
            if not chunk["file_path"].lower().endswith(".md"):
                continue
            if file_focus_tokens and not self._chunk_matches_note_focus(chunk, file_focus_tokens):
                continue
            for raw_line in chunk["text"].splitlines():
                line = raw_line.strip()
                if not line or line.startswith("```"):
                    continue
                match = PATTERN_HEADING_PATTERN.match(line)
                if not match:
                    continue
                name = re.sub(r"\s+", " ", match.group("name")).strip(" -:\u2013")
                canonical = name.lower()
                if not canonical.endswith("pattern"):
                    continue
                if canonical in seen:
                    continue
                seen.add(canonical)
                found.append(name)
        return found

    def _question_note_focus_tokens(self, question: str) -> set[str]:
        note_focus_terms = {
            "creational",
            "structural",
            "behavioral",
            "behavorial",
            "singleton",
            "factory",
            "abstract",
            "builder",
            "prototype",
            "adapter",
            "bridge",
            "composite",
            "decorator",
            "facade",
            "strategy",
            "observer",
            "command",
            "mediator",
            "chain",
        }
        return set(token for token in tokenize(question) if token in note_focus_terms)

    def _note_focus_tokens(self, results: list[RetrievalResult]) -> set[str]:
        focus_tokens = NOTE_TOPIC_TERMS = {
            "creational",
            "structural",
            "behavioral",
            "behavorial",
            "singleton",
            "factory",
            "builder",
            "prototype",
            "adapter",
            "bridge",
            "composite",
            "decorator",
            "facade",
            "strategy",
            "observer",
            "command",
            "mediator",
            "chain",
        }
        ranked_tokens: list[str] = []
        for result in results[:4]:
            text = f"{result.chunk['file_path']} {result.chunk['text']}".lower()
            for token in NOTE_TOPIC_TERMS:
                if token in text and token not in ranked_tokens:
                    ranked_tokens.append(token)
        return set(ranked_tokens[:3])

    def _chunk_matches_note_focus(self, chunk: dict, focus_tokens: set[str]) -> bool:
        searchable = f"{chunk['file_path']} {chunk['text']}".lower()
        return any(token in searchable for token in focus_tokens)

    def _extract_relevant_sentences(self, question: str, texts: Iterable[str]) -> list[str]:
        query_terms = tokenize(question)
        scored: list[tuple[int, str]] = []
        for text in texts:
            for raw_sentence in text.splitlines():
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                overlap = sum(1 for token in query_terms if token in sentence.lower())
                if overlap > 0:
                    scored.append((overlap, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [sentence for _, sentence in scored[:5]]

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

        top_score = results[0].final_score
        score_spread = sum(result.final_score for result in results[:3]) / min(3, len(results))
        if top_score >= 0.64 and score_spread >= 0.54:
            return "high"
        if top_score >= 0.42:
            return "medium"
        return "low"
