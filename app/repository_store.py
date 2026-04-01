from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.utils import dump_json, ensure_directory, load_json, stable_repo_id, utc_now_iso


class RepositoryStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        ensure_directory(self.settings.data_dir)
        ensure_directory(self.settings.repos_dir)
        if not self.settings.registry_path.exists():
            self._save_registry({"active_repo_id": None, "repositories": []})

    def build_repo_id(self, repo_path: Path) -> str:
        return stable_repo_id(repo_path)

    def get_repo_dir(self, repo_id: str) -> Path:
        return self.settings.repos_dir / repo_id

    def get_repo_artifact_paths(self, repo_id: str) -> dict[str, Path]:
        repo_dir = self.get_repo_dir(repo_id)
        return {
            "repo_dir": repo_dir,
            "faiss_index_path": repo_dir / "faiss.index",
            "metadata_path": repo_dir / "metadata.json",
            "chunks_path": repo_dir / "chunks.jsonl",
            "summary_path": repo_dir / "summary.json",
        }

    def list_repositories(self) -> list[dict]:
        registry = self._load_registry()
        active_repo_id = registry.get("active_repo_id")
        repositories = []
        for record in registry.get("repositories", []):
            enriched = dict(record)
            enriched["is_active"] = record.get("repo_id") == active_repo_id
            repositories.append(enriched)
        return sorted(repositories, key=lambda item: item.get("indexed_at", ""), reverse=True)

    def get_repository(self, repo_id: str | None = None) -> dict | None:
        registry = self._load_registry()
        repositories = registry.get("repositories", [])
        if repo_id:
            return next((record for record in repositories if record.get("repo_id") == repo_id), None)

        active_repo_id = registry.get("active_repo_id")
        if active_repo_id:
            active = next((record for record in repositories if record.get("repo_id") == active_repo_id), None)
            if active is not None:
                return active

        return repositories[0] if repositories else None

    def register_repository(
        self,
        repo_id: str,
        repo_name: str,
        repo_path: str,
        file_count: int,
        chunk_count: int,
        set_active: bool = True,
    ) -> dict:
        registry = self._load_registry()
        repositories = registry.get("repositories", [])
        record = {
            "repo_id": repo_id,
            "repo_name": repo_name,
            "repo_path": repo_path,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "indexed_at": utc_now_iso(),
            "summary_excerpt": None,
        }

        updated = False
        for index, existing in enumerate(repositories):
            if existing.get("repo_id") == repo_id:
                record["summary_excerpt"] = existing.get("summary_excerpt")
                repositories[index] = record
                updated = True
                break

        if not updated:
            repositories.append(record)

        if set_active:
            registry["active_repo_id"] = repo_id
        registry["repositories"] = repositories
        self._save_registry(registry)
        return record

    def set_active_repo(self, repo_id: str) -> None:
        registry = self._load_registry()
        registry["active_repo_id"] = repo_id
        self._save_registry(registry)

    def update_summary(self, repo_id: str, summary: str) -> None:
        registry = self._load_registry()
        repositories = registry.get("repositories", [])
        for record in repositories:
            if record.get("repo_id") == repo_id:
                record["summary_excerpt"] = summary[:240]
                break
        registry["repositories"] = repositories
        self._save_registry(registry)

    def _load_registry(self) -> dict:
        path = self.settings.registry_path
        if not path.exists() or path.stat().st_size == 0:
            return {"active_repo_id": None, "repositories": []}
        return load_json(path)

    def _save_registry(self, payload: dict) -> None:
        dump_json(self.settings.registry_path, payload)
