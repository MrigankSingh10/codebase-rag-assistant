from __future__ import annotations


QA_SYSTEM_PROMPT = """You are a codebase question-answering assistant.
Answer only from the supplied repository context.
If the answer is not grounded in the context, say exactly: not found.
Always cite relevant file paths and line ranges.
Keep the answer concise and technically precise."""


DEBUG_SYSTEM_PROMPT = """You are a debugging assistant for source repositories.
Use only the provided code context and error details.
Identify likely failure points, explain why they are plausible, and suggest checks or fixes grounded in the code.
If the context is insufficient, say exactly: not found.
Always cite relevant file paths and line ranges."""


SUMMARY_SYSTEM_PROMPT = """You summarize software repositories.
Use only the supplied repository statistics and representative chunks.
Describe the repository's likely purpose, main areas, and notable implementation patterns.
Treat dependency lockfiles, generated manifests, and build artifacts as low-signal context unless the repository is explicitly about build tooling.
Do not invent components that are not in the context."""


def _render_context_block(chunk: dict) -> str:
    return (
        f"FILE: {chunk['file_path']}\n"
        f"SYMBOL: {chunk['symbol']}\n"
        f"TYPE: {chunk['chunk_type']}\n"
        f"LINES: {chunk['start_line']}-{chunk['end_line']}\n"
        f"CONTENT:\n{chunk['text']}\n"
    )


def build_qa_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(_render_context_block(chunk) for chunk in chunks)
    return (
        "Question:\n"
        f"{question}\n\n"
        "Repository context:\n"
        f"{context}\n\n"
        "Return a concise answer with citations."
    )


def build_debug_prompt(question: str | None, error_context: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(_render_context_block(chunk) for chunk in chunks)
    user_question = question or "Identify the likely cause of the failure."
    return (
        "Question:\n"
        f"{user_question}\n\n"
        "Error context:\n"
        f"{error_context}\n\n"
        "Repository context:\n"
        f"{context}\n\n"
        "Return likely cause, affected files, and suggested checks with citations."
    )


def build_summary_prompt(repo_metadata: dict, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(_render_context_block(chunk) for chunk in chunks)
    return (
        "Repository metadata:\n"
        f"{repo_metadata}\n\n"
        "Representative code context:\n"
        f"{context}\n\n"
        "Return a concise repository summary and the main implementation areas."
    )
