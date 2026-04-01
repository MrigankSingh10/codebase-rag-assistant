from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.indexer import EmbeddingIndexer
from app.reranker import HeuristicReranker
from app.repository_store import RepositoryStore
from app.utils import keyword_overlap_score, normalize_semantic_score, tokenize

AUTH_QUERY_TERMS = {
    "auth",
    "authenticate",
    "authentication",
    "authorization",
    "login",
    "logout",
    "jwt",
    "token",
    "permission",
    "permissions",
    "guard",
    "session",
    "middleware",
    "user",
}

FRONTEND_QUERY_TERMS = {
    "frontend",
    "front-end",
    "ui",
    "client",
    "angular",
    "browser",
}

FRONTEND_PATH_TERMS = (
    "price-optimization-ui/",
    "/src/app/",
    ".component.",
    ".service.",
    ".guard.",
)

BACKEND_PATH_TERMS = (
    "pricing_optimizations/",
    "middleware.py",
    "serializers.py",
    "permissions.py",
    "views.py",
    "models.py",
)

FRONTEND_AUTH_PATH_TERMS = (
    "/src/app/auth/",
    "auth.service.ts",
    "auth.guard.ts",
    "auth-page.component.ts",
)

NOTE_TOPIC_TERMS = {
    "design",
    "pattern",
    "patterns",
    "creational",
    "structural",
    "behavioral",
    "behavorial",
    "singleton",
    "factory",
    "abstract",
    "builder",
    "prototype",
    "oop",
    "oops",
}

MAX_NOTE_EXPANDED_CHUNKS_PER_FILE = 4
MAX_NOTE_TOTAL_ADDITIONS = 8
MAX_NOTE_TOTAL_CHARS = 12000
LIST_LINE_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+")

RetrievalScope = Literal["frontend", "backend"]


@dataclass(slots=True)
class RetrievalResult:
    chunk: dict
    semantic_score: float
    keyword_score: float
    bm25_score: float
    rerank_score: float
    final_score: float


@dataclass(slots=True)
class RepositoryBundle:
    repository: dict
    metadata: dict
    index: object
    chunks: list[dict]
    bm25: BM25Okapi | None


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RepositoryStore(settings)
        self.indexer = EmbeddingIndexer(settings)
        self.reranker = HeuristicReranker(settings)
        self._repo_cache: dict[str, RepositoryBundle] = {}

    def list_repositories(self) -> list[dict]:
        return self.store.list_repositories()

    def get_active_repository(self) -> dict | None:
        return self.store.get_repository()

    def is_ready(self, repo_id: str | None = None) -> bool:
        return self._get_repository_bundle(repo_id) is not None

    def refresh(self, repo_id: str | None = None) -> None:
        if repo_id is None:
            self._repo_cache.clear()
            return
        self._repo_cache.pop(repo_id, None)

    def retrieve(
        self,
        query: str,
        repo_id: str | None = None,
        top_k: int | None = None,
        scope: RetrievalScope | None = None,
        include_document_neighbors: bool = False,
        include_topic_note_aggregation: bool = False,
        prefer_full_note_file: bool = False,
    ) -> list[RetrievalResult]:
        bundle = self._get_repository_bundle(repo_id)
        if bundle is None:
            raise RuntimeError("No repository has been indexed yet. Run /ingest first.")

        requested_k = top_k or self.settings.top_k
        eligible_indices = self._eligible_chunk_indices(bundle.chunks, scope)
        if not eligible_indices:
            return []

        search_k = min(
            max(requested_k * 4, self.settings.rerank_candidate_count, requested_k),
            len(eligible_indices),
        )

        query_embedding = self.indexer.embed_texts([query])
        scores, indices = bundle.index.search(query_embedding, len(bundle.chunks))

        candidate_scores: dict[int, dict[str, float]] = {}
        for raw_score, raw_index in zip(scores[0], indices[0]):
            if raw_index < 0:
                continue
            if int(raw_index) not in eligible_indices:
                continue
            candidate_scores[int(raw_index)] = {
                "semantic": normalize_semantic_score(float(raw_score)),
                "keyword": 0.0,
                "bm25": 0.0,
            }
            if len(candidate_scores) >= search_k:
                break

        if bundle.bm25 is not None:
            query_tokens = tokenize(query)
            bm25_scores = bundle.bm25.get_scores(query_tokens)
            scoped_bm25_pairs = [
                (int(index), float(bm25_scores[index]))
                for index in eligible_indices
            ]
            scoped_bm25_pairs.sort(key=lambda item: item[1], reverse=True)
            top_bm25_pairs = scoped_bm25_pairs[:search_k]
            top_bm25_values = [score for _, score in top_bm25_pairs]
            max_bm25 = max(top_bm25_values) if top_bm25_values else 0.0

            for index, raw_bm25_score in top_bm25_pairs:
                normalized = (raw_bm25_score / max_bm25) if max_bm25 > 0 else 0.0
                candidate_scores.setdefault(index, {"semantic": 0.0, "keyword": 0.0, "bm25": 0.0})
                candidate_scores[index]["bm25"] = normalized

        ranked: list[RetrievalResult] = []
        query_tokens = set(tokenize(query))
        for chunk_index, score_map in candidate_scores.items():
            chunk = bundle.chunks[chunk_index]
            keyword_score = keyword_overlap_score(
                query,
                f"{chunk['repo_name']} {chunk['file_path']} {chunk['symbol']} {chunk['text']}",
            )
            score_map["keyword"] = keyword_score
            weighted_score = (
                self.settings.semantic_weight * score_map["semantic"]
                + self.settings.keyword_weight * score_map["keyword"]
                + self.settings.bm25_weight * score_map["bm25"]
            )
            final_score = weighted_score * self._path_priority_multiplier(chunk, query_tokens)
            ranked.append(
                RetrievalResult(
                    chunk=chunk,
                    semantic_score=score_map["semantic"],
                    keyword_score=score_map["keyword"],
                    bm25_score=score_map["bm25"],
                    rerank_score=0.0,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda result: result.final_score, reverse=True)
        if self.settings.rerank_enabled:
            ranked = self.reranker.rerank(query, ranked, requested_k)
        else:
            ranked = ranked[:requested_k]

        if include_document_neighbors:
            ranked = self._expand_document_neighbors(
                bundle.chunks,
                ranked,
                requested_k,
                query_tokens,
                prefer_full_note_file=prefer_full_note_file,
            )
        if include_topic_note_aggregation:
            ranked = self._expand_related_note_files(
                bundle.chunks,
                ranked,
                requested_k,
                query_tokens,
                prefer_full_note_file=prefer_full_note_file,
            )
        return ranked[:requested_k]

    def retrieve_note_documents(
        self,
        query: str,
        repo_id: str | None = None,
        max_files: int = 3,
    ) -> list[RetrievalResult]:
        bundle = self._get_repository_bundle(repo_id)
        if bundle is None:
            raise RuntimeError("No repository has been indexed yet. Run /ingest first.")

        query_tokens = set(tokenize(query))
        markdown_chunks = [chunk for chunk in bundle.chunks if self._is_document_chunk(chunk)]
        if not markdown_chunks:
            return []

        by_file = self._chunks_by_file(markdown_chunks)
        focus_tokens = query_tokens & NOTE_TOPIC_TERMS
        file_scores: list[tuple[str, float]] = []
        for file_path, file_chunks in by_file.items():
            combined_text = "\n".join(chunk["text"] for chunk in file_chunks[:3])
            searchable = f"{file_path}\n{combined_text}"
            searchable_lower = searchable.lower()
            if focus_tokens and not any(token in searchable_lower for token in focus_tokens):
                continue
            keyword_score = keyword_overlap_score(query, searchable)
            path_tokens = set(tokenize(file_path))
            text_tokens = set(tokenize(combined_text))
            token_overlap = len(query_tokens & (path_tokens | text_tokens))
            file_score = keyword_score + (0.08 * token_overlap)
            if any(token in file_path.lower() for token in query_tokens):
                file_score += 0.12
            if file_score > 0:
                file_scores.append((file_path, file_score))

        if not file_scores:
            return []

        file_scores.sort(key=lambda item: item[1], reverse=True)
        best_score = file_scores[0][1]
        selected_files = [
            file_path
            for file_path, score in file_scores
            if score >= max(0.18, best_score * 0.6)
        ][:max_files]

        results: list[RetrievalResult] = []
        for file_path in selected_files:
            file_score = next(score for candidate_path, score in file_scores if candidate_path == file_path)
            for position, chunk in enumerate(by_file[file_path]):
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        semantic_score=0.0,
                        keyword_score=file_score,
                        bm25_score=0.0,
                        rerank_score=0.0,
                        final_score=max(0.0, file_score - (0.01 * position)),
                    )
                )

        results.sort(key=lambda item: (-item.final_score, item.chunk["file_path"], item.chunk["start_line"]))
        return results

    def _eligible_chunk_indices(self, chunks: list[dict], scope: RetrievalScope | None) -> set[int]:
        eligible: set[int] = set()
        for index, chunk in enumerate(chunks):
            if scope is None or self._chunk_matches_scope(chunk, scope):
                eligible.add(index)
        return eligible

    def _chunk_matches_scope(self, chunk: dict, scope: RetrievalScope) -> bool:
        file_path = chunk["file_path"].lower()
        if scope == "frontend":
            return any(token in file_path for token in FRONTEND_PATH_TERMS)
        return any(token in file_path for token in BACKEND_PATH_TERMS)

    def _expand_document_neighbors(
        self,
        chunks: list[dict],
        results: list[RetrievalResult],
        top_k: int,
        query_tokens: set[str],
        prefer_full_note_file: bool = False,
    ) -> list[RetrievalResult]:
        if not results:
            return results

        by_file = self._chunks_by_file(chunks)

        expanded = list(results)
        seen = {self._chunk_ref(result.chunk) for result in results}
        current_chars = sum(len(result.chunk["text"]) for result in results)
        total_additions = 0

        for result in list(results[: min(3, len(results))]):
            chunk = result.chunk
            if not self._is_document_chunk(chunk):
                continue

            file_chunks = by_file.get(chunk["file_path"], [])
            if not file_chunks:
                continue

            chunk_index = next(
                (index for index, item in enumerate(file_chunks) if item["start_line"] == chunk["start_line"] and item["end_line"] == chunk["end_line"]),
                None,
            )
            if chunk_index is None:
                continue

            added = self._expand_file_section(
                file_chunks=file_chunks,
                seed_index=chunk_index,
                seed_result=result,
                query_tokens=query_tokens,
                expanded=expanded,
                seen=seen,
                current_chars=current_chars,
                max_total_results=max(top_k, len(results) + MAX_NOTE_TOTAL_ADDITIONS),
                remaining_total_additions=MAX_NOTE_TOTAL_ADDITIONS - total_additions,
                prefer_full_note_file=prefer_full_note_file,
            )
            current_chars += added["chars"]
            total_additions += added["count"]
            if total_additions >= MAX_NOTE_TOTAL_ADDITIONS:
                break

        expanded.sort(key=lambda item: item.final_score, reverse=True)
        return expanded

    def _expand_related_note_files(
        self,
        chunks: list[dict],
        results: list[RetrievalResult],
        top_k: int,
        query_tokens: set[str],
        prefer_full_note_file: bool = False,
    ) -> list[RetrievalResult]:
        markdown_results = [result for result in results if self._is_document_chunk(result.chunk)]
        if not markdown_results:
            return results

        seed_files = {result.chunk["file_path"] for result in markdown_results[:3]}
        folder_prefixes = {self._note_folder_prefix(path) for path in seed_files}
        topic_tokens = query_tokens & NOTE_TOPIC_TERMS
        if not topic_tokens:
            topic_tokens = {
                token
                for result in markdown_results[:2]
                for token in tokenize(result.chunk["file_path"])
                if token in NOTE_TOPIC_TERMS
            }
        if not topic_tokens:
            return results

        best_by_file: dict[str, dict] = {}
        for chunk in chunks:
            if not self._is_document_chunk(chunk):
                continue
            file_path = chunk["file_path"]
            if file_path in seed_files:
                continue
            if self._note_folder_prefix(file_path) not in folder_prefixes:
                continue
            searchable = f"{file_path} {chunk['text']}".lower()
            overlap = sum(1 for token in topic_tokens if token in searchable)
            if overlap <= 0:
                continue
            current = best_by_file.get(file_path)
            if current is None or overlap > current["overlap"]:
                best_by_file[file_path] = {"chunk": chunk, "overlap": overlap}

        if not best_by_file:
            return results

        expanded = list(results)
        seen = {self._chunk_ref(result.chunk) for result in results}
        seed_score = min((result.final_score for result in markdown_results[:2]), default=0.3)
        by_file = self._chunks_by_file(chunks)
        current_chars = sum(len(result.chunk["text"]) for result in results)
        total_additions = 0

        additions = sorted(
            best_by_file.values(),
            key=lambda item: (item["overlap"], len(item["chunk"]["text"])),
            reverse=True,
        )
        for item in additions:
            chunk = item["chunk"]
            ref = self._chunk_ref(chunk)
            if ref in seen:
                continue
            seen.add(ref)
            seed_result = RetrievalResult(
                chunk=chunk,
                semantic_score=0.0,
                keyword_score=0.0,
                bm25_score=0.0,
                rerank_score=0.0,
                final_score=max(0.0, seed_score - 0.05),
            )
            expanded.append(seed_result)
            total_additions += 1
            current_chars += len(chunk["text"])
            if len(expanded) >= max(top_k, len(results) + MAX_NOTE_TOTAL_ADDITIONS):
                break
            file_chunks = by_file.get(chunk["file_path"], [])
            chunk_index = next(
                (index for index, candidate in enumerate(file_chunks) if self._chunk_ref(candidate) == ref),
                None,
            )
            if chunk_index is None:
                continue
            added = self._expand_file_section(
                file_chunks=file_chunks,
                seed_index=chunk_index,
                seed_result=seed_result,
                query_tokens=query_tokens | topic_tokens,
                expanded=expanded,
                seen=seen,
                current_chars=current_chars,
                max_total_results=max(top_k, len(results) + MAX_NOTE_TOTAL_ADDITIONS),
                remaining_total_additions=MAX_NOTE_TOTAL_ADDITIONS - total_additions,
                prefer_full_note_file=prefer_full_note_file,
            )
            current_chars += added["chars"]
            total_additions += added["count"]
            if total_additions >= MAX_NOTE_TOTAL_ADDITIONS:
                break

        expanded.sort(key=lambda item: item.final_score, reverse=True)
        return expanded

    def _expand_file_section(
        self,
        file_chunks: list[dict],
        seed_index: int,
        seed_result: RetrievalResult,
        query_tokens: set[str],
        expanded: list[RetrievalResult],
        seen: set[tuple[str, int, int]],
        current_chars: int,
        max_total_results: int,
        remaining_total_additions: int,
        prefer_full_note_file: bool = False,
    ) -> dict[str, int]:
        if remaining_total_additions <= 0 or current_chars >= MAX_NOTE_TOTAL_CHARS:
            return {"count": 0, "chars": 0}

        seed_chunk = file_chunks[seed_index]
        seed_tokens = set(tokenize(f"{seed_chunk['file_path']} {seed_chunk['text']}")) & (NOTE_TOPIC_TERMS | query_tokens)
        additions = 0
        added_chars = 0

        for direction in (-1, 1):
            cursor = seed_index
            previous_chunk = seed_chunk
            max_file_additions = remaining_total_additions if prefer_full_note_file else min(
                MAX_NOTE_EXPANDED_CHUNKS_PER_FILE,
                remaining_total_additions,
            )
            while additions < max_file_additions:
                neighbor_index = cursor + direction
                if neighbor_index < 0 or neighbor_index >= len(file_chunks):
                    break
                neighbor = file_chunks[neighbor_index]
                ref = self._chunk_ref(neighbor)
                cursor = neighbor_index
                if ref in seen:
                    previous_chunk = neighbor
                    continue
                if current_chars + added_chars + len(neighbor["text"]) > MAX_NOTE_TOTAL_CHARS:
                    break
                if not self._should_continue_note_section(
                    previous_chunk,
                    neighbor,
                    query_tokens,
                    seed_tokens,
                    prefer_full_note_file=prefer_full_note_file,
                ):
                    break
                seen.add(ref)
                additions += 1
                added_chars += len(neighbor["text"])
                expanded.append(
                    RetrievalResult(
                        chunk=neighbor,
                        semantic_score=seed_result.semantic_score,
                        keyword_score=seed_result.keyword_score,
                        bm25_score=seed_result.bm25_score,
                        rerank_score=seed_result.rerank_score,
                        final_score=max(0.0, seed_result.final_score - (0.02 * additions)),
                    )
                )
                previous_chunk = neighbor
                if len(expanded) >= max_total_results:
                    return {"count": additions, "chars": added_chars}

        return {"count": additions, "chars": added_chars}

    def _should_continue_note_section(
        self,
        current_chunk: dict,
        candidate_chunk: dict,
        query_tokens: set[str],
        seed_tokens: set[str],
        prefer_full_note_file: bool = False,
    ) -> bool:
        candidate_text = candidate_chunk["text"]
        current_text = current_chunk["text"]
        candidate_lines = [line.strip() for line in candidate_text.splitlines() if line.strip()]
        current_lines = [line.strip() for line in current_text.splitlines() if line.strip()]
        candidate_first = candidate_lines[0] if candidate_lines else ""

        effective_tokens = set(query_tokens) | set(seed_tokens)
        candidate_tokens = set(tokenize(candidate_text))
        overlap = len(candidate_tokens & effective_tokens)
        current_has_list = any(LIST_LINE_PATTERN.match(line) for line in current_lines)
        candidate_has_list = any(LIST_LINE_PATTERN.match(line) for line in candidate_lines)

        if prefer_full_note_file:
            if overlap > 0 or candidate_has_list:
                return True
            if not candidate_lines:
                return False
            return not candidate_first.lower().startswith(("these ", "conclusion", "summary"))

        if HEADING_PATTERN.match(candidate_first) and overlap == 0 and not candidate_has_list:
            return False
        if overlap > 0:
            return True
        if current_has_list and candidate_has_list:
            return True
        if current_has_list and candidate_lines and not HEADING_PATTERN.match(candidate_first):
            return True
        return False

    def _chunks_by_file(self, chunks: list[dict]) -> dict[str, list[dict]]:
        by_file: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_file.setdefault(chunk["file_path"], []).append(chunk)
        for file_chunks in by_file.values():
            file_chunks.sort(key=lambda item: item["start_line"])
        return by_file

    def _chunk_ref(self, chunk: dict) -> tuple[str, int, int]:
        return (chunk["file_path"], chunk["start_line"], chunk["end_line"])

    def _note_folder_prefix(self, file_path: str) -> str:
        parts = Path(file_path).parts
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return file_path

    def _is_document_chunk(self, chunk: dict) -> bool:
        file_path = chunk["file_path"].lower()
        return chunk["chunk_type"] == "module" and file_path.endswith(".md")

    def _path_priority_multiplier(self, chunk: dict, query_tokens: set[str]) -> float:
        file_path = chunk["file_path"].lower()
        path_parts = [part for part in Path(file_path).parts if part not in {".", "/"}]
        file_name = path_parts[-1] if path_parts else file_path
        symbol_tokens = set(tokenize(chunk["symbol"]))
        multiplier = 1.0

        if chunk["chunk_type"] in {"class", "function", "method"}:
            multiplier += 0.12
        elif chunk["chunk_type"] == "module":
            multiplier -= 0.08

        if query_tokens & symbol_tokens:
            multiplier += 0.10

        if any(part in {"tests", "test", "docs", "doc", "migrations"} for part in path_parts):
            multiplier -= 0.12

        if file_name in {"settings.py", "config.py", "apps.py", "urls.py", "wsgi.py", "asgi.py"}:
            multiplier -= 0.18

        if any(token in file_path for token in ("middleware", "permission", "permissions")):
            multiplier -= 0.06

        if query_tokens & AUTH_QUERY_TERMS:
            if any(token in file_path for token in ("auth", "login", "jwt", "token", "permission", "guard", "session")):
                multiplier += 0.14
            if query_tokens & symbol_tokens:
                multiplier += 0.06

        if query_tokens & FRONTEND_QUERY_TERMS:
            if any(token in file_path for token in FRONTEND_PATH_TERMS):
                multiplier += 0.26
            if any(token in file_path for token in BACKEND_PATH_TERMS):
                multiplier -= 0.18

        if (query_tokens & FRONTEND_QUERY_TERMS) and (query_tokens & AUTH_QUERY_TERMS):
            if any(token in file_path for token in FRONTEND_AUTH_PATH_TERMS):
                multiplier += 0.32
            if "auth.service.ts" in file_path:
                multiplier += 0.16
            if "auth.guard.ts" in file_path:
                multiplier += 0.12
            if any(token in file_path for token in BACKEND_PATH_TERMS):
                multiplier -= 0.22

        if file_name.endswith((".min.js", ".bundle.js")):
            multiplier -= 0.18

        return max(0.55, min(1.45, multiplier))

    def _get_repository_bundle(self, repo_id: str | None = None) -> RepositoryBundle | None:
        repository = self.store.get_repository(repo_id)
        if repository is None:
            return None

        selected_repo_id = repository["repo_id"]
        if selected_repo_id in self._repo_cache:
            return self._repo_cache[selected_repo_id]

        metadata = self.indexer.load_metadata(selected_repo_id)
        index = self.indexer.load_index(selected_repo_id)
        chunks = self.indexer.load_chunks(selected_repo_id)
        if metadata is None or index is None or not chunks:
            return None

        bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks]) if self.settings.use_bm25 else None
        bundle = RepositoryBundle(
            repository=repository,
            metadata=metadata,
            index=index,
            chunks=chunks,
            bm25=bm25,
        )
        self._repo_cache[selected_repo_id] = bundle
        return bundle
