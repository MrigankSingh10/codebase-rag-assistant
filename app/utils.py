from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_chunk_id(file_path: str, symbol: str, start_line: int, end_line: int, text: str) -> str:
    payload = f"{file_path}|{symbol}|{start_line}|{end_line}|{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def stable_repo_id(path: Path) -> str:
    slug = slugify(path.name or "repo")
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode file: {path}")


def iter_repository_files(
    repo_path: Path,
    supported_extensions: Iterable[str],
    ignored_directories: Iterable[str],
) -> list[Path]:
    supported = {suffix.lower() for suffix in supported_extensions}
    ignored = set(ignored_directories)
    discovered: list[Path] = []

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue

        if any(part in ignored for part in path.parts):
            continue

        if path.suffix.lower() not in supported:
            continue

        discovered.append(path)

    return sorted(discovered)


def chunk_text_by_lines(
    text: str,
    max_chars: int,
    overlap_lines: int = 4,
) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[tuple[int, int, str]] = []
    current_lines: list[str] = []
    chunk_start = 1

    for index, line in enumerate(lines, start=1):
        projected = "\n".join(current_lines + [line]).strip("\n")
        if current_lines and len(projected) > max_chars:
            chunk_end = index - 1
            chunk_text = "\n".join(current_lines).strip("\n")
            if chunk_text:
                chunks.append((chunk_start, chunk_end, chunk_text))

            overlap = current_lines[-overlap_lines:] if overlap_lines > 0 else []
            current_lines = overlap + [line]
            chunk_start = max(1, index - len(overlap))
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append((chunk_start, len(lines), "\n".join(current_lines).strip("\n")))

    return chunks


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def slugify(value: str) -> str:
    normalized = SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized or "repo"


def keyword_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def normalize_semantic_score(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def unique_by_key(items: Iterable[dict], key: str) -> list[dict]:
    seen: set[str] = set()
    unique_items: list[dict] = []
    for item in items:
        value = item.get(key)
        if value in seen:
            continue
        seen.add(value)
        unique_items.append(item)
    return unique_items


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
