from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="local", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_name: str = Field(default="Codebase RAG Assistant", alias="APP_NAME")
    llm_provider: Literal["local", "openai", "gemini"] = Field(default="local", alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-flash-lite-latest", alias="GEMINI_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    top_k: int = Field(default=6, alias="TOP_K")
    semantic_weight: float = Field(default=0.65, alias="SEMANTIC_WEIGHT")
    keyword_weight: float = Field(default=0.20, alias="KEYWORD_WEIGHT")
    bm25_weight: float = Field(default=0.15, alias="BM25_WEIGHT")
    use_bm25: bool = Field(default=True, alias="USE_BM25")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_weight: float = Field(default=0.20, alias="RERANK_WEIGHT")
    rerank_candidate_count: int = Field(default=12, alias="RERANK_CANDIDATE_COUNT")

    chunk_context_lines: int = Field(default=3, alias="CHUNK_CONTEXT_LINES")
    fallback_chunk_max_chars: int = Field(default=2200, alias="FALLBACK_CHUNK_MAX_CHARS")
    embedding_batch_size: int = 32
    summary_highlight_count: int = 5

    supported_extensions: tuple[str, ...] = (".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml")
    ignored_directories: tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".direnv",
        ".venv",
        "venv",
        "env",
        "lib",
        ".cache",
        ".angular",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".parcel-cache",
        ".turbo",
        ".yarn",
        "__pypackages__",
        "site-packages",
        "dist-packages",
        "__pycache__",
        "node_modules",
        "coverage",
        "vendor",
        "vendors",
        "dist",
        "build",
        ".idea",
        ".vscode",
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "repositories.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def sample_repo_path(self) -> Path:
        return self.project_root / "sample_repo"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.repos_dir.mkdir(parents=True, exist_ok=True)
    return settings
