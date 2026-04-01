from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.chunker import CodeChunk, RepositoryChunker
from app.config import Settings
from app.indexer import EmbeddingIndexer
from app.repository_store import RepositoryStore
from app.utils import iter_repository_files


@dataclass(slots=True)
class IngestResult:
    repo_id: str
    repo_name: str
    repo_path: str
    file_count: int
    chunk_count: int
    indexed: bool
    active: bool


class RepositoryIngestionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RepositoryStore(settings)
        self.chunker = RepositoryChunker(
            context_lines=settings.chunk_context_lines,
            fallback_chunk_max_chars=settings.fallback_chunk_max_chars,
        )
        self.indexer = EmbeddingIndexer(settings)

    def ingest(self, repo_path: str, set_active: bool = True) -> IngestResult:
        root = Path(repo_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Repository path does not exist or is not a directory: {repo_path}")

        files = iter_repository_files(
            repo_path=root,
            supported_extensions=self.settings.supported_extensions,
            ignored_directories=self.settings.ignored_directories,
        )
        if not files:
            raise ValueError("No supported source files were found in the repository.")

        chunks: list[CodeChunk] = []
        for file_path in files:
            chunks.extend(self.chunker.chunk_file(file_path=file_path, repo_root=root))

        if not chunks:
            raise ValueError("No chunks were produced during ingestion.")

        repo_id = self.store.build_repo_id(root)
        index = self.indexer.build_index(chunks)
        self.indexer.save_artifacts(
            repo_id=repo_id,
            repo_path=root,
            chunks=chunks,
            index=index,
            file_count=len(files),
        )
        self.store.register_repository(
            repo_id=repo_id,
            repo_name=root.name,
            repo_path=str(root),
            file_count=len(files),
            chunk_count=len(chunks),
            set_active=set_active,
        )

        return IngestResult(
            repo_id=repo_id,
            repo_name=root.name,
            repo_path=str(root),
            file_count=len(files),
            chunk_count=len(chunks),
            indexed=True,
            active=set_active,
        )
