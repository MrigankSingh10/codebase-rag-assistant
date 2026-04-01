from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.chunker import CodeChunk
from app.config import Settings
from app.repository_store import RepositoryStore
from app.utils import dump_json, ensure_directory, utc_now_iso


class EmbeddingIndexer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RepositoryStore(settings)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_index(self, chunks: list[CodeChunk]) -> faiss.IndexFlatIP:
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks.")

        embeddings = self.embed_texts(chunk.text for chunk in chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def save_artifacts(
        self,
        repo_id: str,
        repo_path: Path,
        chunks: list[CodeChunk],
        index: faiss.IndexFlatIP,
        file_count: int,
    ) -> None:
        paths = self.store.get_repo_artifact_paths(repo_id)
        ensure_directory(paths["repo_dir"])
        faiss.write_index(index, str(paths["faiss_index_path"]))

        with paths["chunks_path"].open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")

        dump_json(
            paths["metadata_path"],
            {
                "repo_id": repo_id,
                "repo_name": repo_path.name,
                "repo_path": str(repo_path.resolve()),
                "file_count": file_count,
                "chunk_count": len(chunks),
                "embedding_model": self.settings.embedding_model,
                "indexed_at": utc_now_iso(),
                "top_k": self.settings.top_k,
            },
        )

    def load_index(self, repo_id: str) -> faiss.Index | None:
        path = self.store.get_repo_artifact_paths(repo_id)["faiss_index_path"]
        if not path.exists() or path.stat().st_size == 0:
            return None
        return faiss.read_index(str(path))

    def load_chunks(self, repo_id: str) -> list[dict]:
        path = self.store.get_repo_artifact_paths(repo_id)["chunks_path"]
        if not path.exists():
            return []

        metadata = self.load_metadata(repo_id) or {}
        loaded: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["repo_id"] = repo_id
                record["repo_name"] = metadata.get("repo_name", repo_id)
                loaded.append(record)
        return loaded

    def load_metadata(self, repo_id: str) -> dict | None:
        path = self.store.get_repo_artifact_paths(repo_id)["metadata_path"]
        if not path.exists() or path.stat().st_size == 0:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
