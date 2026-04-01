from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from app.utils import chunk_text_by_lines, stable_chunk_id


@dataclass(slots=True)
class CodeChunk:
    chunk_id: str
    file_path: str
    symbol: str
    chunk_type: str
    start_line: int
    end_line: int
    text: str

    def to_record(self) -> dict:
        return asdict(self)


class RepositoryChunker:
    def __init__(self, context_lines: int = 3, fallback_chunk_max_chars: int = 2200) -> None:
        self.context_lines = context_lines
        self.fallback_chunk_max_chars = fallback_chunk_max_chars

    def chunk_file(self, file_path: Path, repo_root: Path) -> list[CodeChunk]:
        relative_path = file_path.relative_to(repo_root).as_posix()
        source = file_path.read_text(encoding="utf-8", errors="ignore")

        if file_path.suffix.lower() == ".py":
            return self._chunk_python(relative_path=relative_path, source=source)

        return self._chunk_text_file(relative_path=relative_path, source=source, file_kind="module")

    def _chunk_python(self, relative_path: str, source: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._chunk_text_file(relative_path=relative_path, source=source, file_kind="module")

        lines = source.splitlines()
        total_lines = len(lines)
        chunks: list[CodeChunk] = []
        covered_ranges: list[tuple[int, int]] = []

        def add_chunk(symbol: str, chunk_type: str, start_line: int, end_line: int) -> None:
            bounded_start = max(1, start_line - self.context_lines)
            bounded_end = min(total_lines, end_line + self.context_lines)
            text = "\n".join(lines[bounded_start - 1 : bounded_end]).strip()
            if not text:
                return

            chunks.append(
                CodeChunk(
                    chunk_id=stable_chunk_id(relative_path, symbol, bounded_start, bounded_end, text),
                    file_path=relative_path,
                    symbol=symbol,
                    chunk_type=chunk_type,
                    start_line=bounded_start,
                    end_line=bounded_end,
                    text=text,
                )
            )

        def walk(node: ast.AST, scope: list[str], in_class: bool = False) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    if hasattr(child, "end_lineno") and child.end_lineno is not None:
                        symbol = ".".join([*scope, child.name]) if scope else child.name
                        covered_ranges.append((child.lineno, child.end_lineno))
                        add_chunk(symbol, "class", child.lineno, child.end_lineno)
                    walk(child, [*scope, child.name], in_class=True)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(child, "end_lineno") and child.end_lineno is not None:
                        symbol = ".".join([*scope, child.name]) if scope else child.name
                        chunk_type = "method" if in_class else "function"
                        covered_ranges.append((child.lineno, child.end_lineno))
                        add_chunk(symbol, chunk_type, child.lineno, child.end_lineno)
                    walk(child, [*scope, child.name], in_class=in_class)
                else:
                    walk(child, scope, in_class=in_class)

        walk(tree, [])

        if total_lines == 0:
            return chunks

        uncovered = self._uncovered_ranges(covered_ranges, total_lines)
        for start_line, end_line in uncovered:
            module_text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not module_text:
                continue
            module_chunks = chunk_text_by_lines(module_text, self.fallback_chunk_max_chars)
            for offset_start, offset_end, text in module_chunks:
                actual_start = start_line + offset_start - 1
                actual_end = start_line + offset_end - 1
                symbol = f"{relative_path}:module:{actual_start}"
                chunks.append(
                    CodeChunk(
                        chunk_id=stable_chunk_id(relative_path, symbol, actual_start, actual_end, text),
                        file_path=relative_path,
                        symbol=symbol,
                        chunk_type="module",
                        start_line=actual_start,
                        end_line=actual_end,
                        text=text,
                    )
                )

        if not chunks:
            return self._chunk_text_file(relative_path=relative_path, source=source, file_kind="module")

        return sorted(chunks, key=lambda chunk: (chunk.file_path, chunk.start_line, chunk.symbol))

    def _uncovered_ranges(
        self,
        covered_ranges: Iterable[tuple[int, int]],
        total_lines: int,
    ) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(covered_ranges):
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        if not merged:
            return [(1, total_lines)]

        uncovered: list[tuple[int, int]] = []
        cursor = 1
        for start, end in merged:
            if cursor < start:
                uncovered.append((cursor, start - 1))
            cursor = end + 1
        if cursor <= total_lines:
            uncovered.append((cursor, total_lines))
        return uncovered

    def _chunk_text_file(self, relative_path: str, source: str, file_kind: str) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for start_line, end_line, text in chunk_text_by_lines(source, self.fallback_chunk_max_chars):
            symbol = f"{relative_path}:{file_kind}:{start_line}"
            chunks.append(
                CodeChunk(
                    chunk_id=stable_chunk_id(relative_path, symbol, start_line, end_line, text),
                    file_path=relative_path,
                    symbol=symbol,
                    chunk_type=file_kind,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                )
            )
        return chunks
