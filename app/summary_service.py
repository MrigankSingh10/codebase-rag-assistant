from __future__ import annotations

import json
import logging
from collections import Counter

from app.config import Settings
from app.indexer import EmbeddingIndexer
from app.prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from app.qa_service import BaseLLMClient, LocalGroundedClient, build_llm_client
from app.repository_store import RepositoryStore

logger = logging.getLogger(__name__)


class SummaryService:
    def __init__(self, settings: Settings, llm_client: BaseLLMClient | None = None) -> None:
        self.settings = settings
        self.store = RepositoryStore(settings)
        self.indexer = EmbeddingIndexer(settings)
        self.llm_client = llm_client or build_llm_client(settings)

    def summarize_repository(self, repo_id: str | None = None, refresh: bool = False) -> dict:
        repository = self.store.get_repository(repo_id)
        if repository is None:
            raise RuntimeError("No repository has been indexed yet. Run /ingest first.")

        paths = self.store.get_repo_artifact_paths(repository["repo_id"])
        if paths["summary_path"].exists() and not refresh:
            return json.loads(paths["summary_path"].read_text(encoding="utf-8"))

        metadata = self.indexer.load_metadata(repository["repo_id"])
        chunks = self.indexer.load_chunks(repository["repo_id"])
        if metadata is None or not chunks:
            raise RuntimeError("Summary could not be generated because repository artifacts are incomplete.")

        payload = self._build_summary_payload(repository=repository, metadata=metadata, chunks=chunks)
        paths["summary_path"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.store.update_summary(repository["repo_id"], payload["summary"])
        return payload

    def overview_confidence(self, payload: dict) -> str:
        summary = (payload.get("summary") or "").strip()
        highlights = payload.get("highlights") or []
        if summary and len(highlights) >= 3:
            return "high"
        if summary and highlights:
            return "medium"
        return "low"

    def _build_summary_payload(self, repository: dict, metadata: dict, chunks: list[dict]) -> dict:
        extension_counts = Counter()
        directory_counts = Counter()
        chunk_type_counts = Counter()
        for chunk in chunks:
            file_path = chunk["file_path"]
            extension_counts[file_path.rsplit(".", 1)[-1] if "." in file_path else "none"] += 1
            directory = file_path.split("/", 1)[0] if "/" in file_path else "."
            directory_counts[directory] += 1
            chunk_type_counts[chunk["chunk_type"]] += 1

        highlights = self._select_highlights(chunks)
        repo_metadata = {
            "repo_id": repository["repo_id"],
            "repo_name": repository["repo_name"],
            "repo_path": repository["repo_path"],
            "file_count": metadata.get("file_count", repository.get("file_count", 0)),
            "chunk_count": metadata.get("chunk_count", repository.get("chunk_count", 0)),
            "extensions": extension_counts.most_common(5),
            "directories": directory_counts.most_common(5),
            "chunk_types": chunk_type_counts.most_common(5),
        }

        summary = self._generate_summary(repo_metadata=repo_metadata, highlights=highlights)
        return {
            "repo_id": repository["repo_id"],
            "repo_name": repository["repo_name"],
            "repo_path": repository["repo_path"],
            "file_count": repo_metadata["file_count"],
            "chunk_count": repo_metadata["chunk_count"],
            "indexed_at": metadata.get("indexed_at"),
            "summary": summary,
            "highlights": [
                {
                    "repo_id": chunk["repo_id"],
                    "file_path": chunk["file_path"],
                    "symbol": chunk["symbol"],
                    "lines": f"{chunk['start_line']}-{chunk['end_line']}",
                    "chunk_type": chunk["chunk_type"],
                }
                for chunk in highlights
            ],
            "stats": {
                "extensions": extension_counts.most_common(5),
                "directories": directory_counts.most_common(5),
                "chunk_types": chunk_type_counts.most_common(5),
            },
        }

    def _generate_summary(self, repo_metadata: dict, highlights: list[dict]) -> str:
        if isinstance(self.llm_client, LocalGroundedClient):
            return self._build_local_summary(repo_metadata, highlights)

        prompt = build_summary_prompt(repo_metadata, highlights)
        try:
            return self.llm_client.generate(SUMMARY_SYSTEM_PROMPT, prompt)
        except Exception as exc:
            logger.warning("Falling back to local repository summary after LLM failure: %s", exc)
            return self._build_local_summary(repo_metadata, highlights)

    def _build_local_summary(self, repo_metadata: dict, highlights: list[dict]) -> str:
        extension_summary = ", ".join(f"{ext} ({count})" for ext, count in repo_metadata["extensions"]) or "none"
        directory_summary = ", ".join(f"{name} ({count})" for name, count in repo_metadata["directories"]) or "."
        highlight_summary = ", ".join(f"{chunk['file_path']}::{chunk['symbol']}" for chunk in highlights[:3]) or "no highlighted symbols"
        return (
            f"{repo_metadata['repo_name']} appears to be a codebase with {repo_metadata['file_count']} supported files "
            f"and {repo_metadata['chunk_count']} indexed chunks. Dominant file types: {extension_summary}. "
            f"Most represented areas: {directory_summary}. Key indexed symbols include {highlight_summary}."
        )

    def _select_highlights(self, chunks: list[dict]) -> list[dict]:
        def priority(chunk: dict) -> tuple[int, int, int]:
            chunk_type_rank = {"class": 3, "function": 2, "method": 2, "module": 1}
            path = chunk["file_path"].lower()
            score = 0
            if any(
                token in path
                for token in (
                    "package-lock.json",
                    "pnpm-lock.yaml",
                    "yarn.lock",
                    "poetry.lock",
                    "cargo.lock",
                    ".min.js",
                    ".bundle.js",
                    "/dist/",
                    "/build/",
                )
            ):
                score -= 4
            if any(
                token in path
                for token in (
                    "/migrations/",
                    "/__pycache__/",
                    "readme.md",
                    "angular.json",
                    "tsconfig.json",
                    "package.json",
                )
            ):
                score -= 2
            if any(token in path for token in ("/src/app/", "/products/", "/userauth/", "/pricing_optimizations/")):
                score += 2
            if any(token in path for token in (".component.", ".service.", ".guard.", ".views.py", "views.py", "models.py")):
                score += 1
            return (score, chunk_type_rank.get(chunk["chunk_type"], 0), len(chunk["text"]))

        ranked = sorted(chunks, key=priority, reverse=True)

        selected: list[dict] = []
        seen_refs: set[tuple[str, str, int, int]] = set()
        seen_top_dirs: set[str] = set()

        for chunk in ranked:
            top_dir = chunk["file_path"].split("/", 1)[0]
            ref = (chunk["file_path"], chunk["symbol"], chunk["start_line"], chunk["end_line"])
            if ref in seen_refs:
                continue
            if top_dir in seen_top_dirs:
                continue
            selected.append(chunk)
            seen_refs.add(ref)
            seen_top_dirs.add(top_dir)
            if len(selected) >= self.settings.summary_highlight_count:
                return selected

        for chunk in ranked:
            ref = (chunk["file_path"], chunk["symbol"], chunk["start_line"], chunk["end_line"])
            if ref in seen_refs:
                continue
            selected.append(chunk)
            seen_refs.add(ref)
            if len(selected) >= self.settings.summary_highlight_count:
                break

        return selected
