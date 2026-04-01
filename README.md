# AI-Powered Codebase Q&A and Debug Assistant using RAG

Production-oriented FastAPI service for repository ingestion, code-aware chunking, hybrid retrieval, reranking, grounded Q&A, debugging workflows, and repository summaries over one or more local codebases.

## Architecture

The system is split into focused modules:

- `app/chunker.py`: AST-based Python chunking for classes, functions, and methods, with module-level fallback chunks for imports and global code.
- `app/ingest.py`: repository crawling, filtering, and chunk generation.
- `app/indexer.py`: sentence-transformer embeddings and FAISS persistence.
- `app/repository_store.py`: multi-repo registry and per-repo artifact layout.
- `app/retriever.py`: hybrid retrieval that combines semantic similarity, keyword overlap, optional BM25, and reranking.
- `app/qa_service.py`: grounded Q&A with OpenAI, Gemini, or a deterministic local fallback.
- `app/debug_service.py`: stack-trace and error-context retrieval plus failure analysis.
- `app/summary_service.py`: repository-level summarization and highlights.
- `app/api.py` and `app/main.py`: FastAPI request models and endpoint wiring.

## Features

- Repository ingestion for `.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`, `.yml`
- Ignore rules for `.git`, `node_modules`, virtual environments, caches, and common build directories
- AST-aware Python chunking with metadata:
  - `file_path`
  - `symbol`
  - `chunk_type`
  - `start_line`
  - `end_line`
  - `text`
- FAISS vector index with sentence-transformer embeddings
- Multi-repo support with per-repository artifacts under `data/repos/<repo_id>/`
- Hybrid retrieval:
  - semantic similarity
  - keyword overlap boosting
  - optional BM25
- Second-stage heuristic reranking over the strongest candidates
- Structured JSON responses with citations and confidence
- Debug workflow for raw error messages and stack traces
- Repository summaries with indexed highlights and repo statistics
- Pluggable generation layer:
  - `LLM_PROVIDER=local` for deterministic offline-friendly answers
  - `LLM_PROVIDER=openai` for higher-quality generated responses
  - `LLM_PROVIDER=gemini` for Gemini-backed generated responses

## Project Structure

```text
codebase-rag-assistant/
├── app/
│   ├── api.py
│   ├── chunker.py
│   ├── config.py
│   ├── debug_service.py
│   ├── indexer.py
│   ├── ingest.py
│   ├── main.py
│   ├── prompts.py
│   ├── qa_service.py
│   ├── repository_store.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── summary_service.py
│   └── utils.py
├── data/
│   ├── chunks.jsonl
│   ├── faiss.index
│   └── metadata.json
├── sample_repo/
├── requirements.txt
├── README.md
└── .env.example
```

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment file:

```bash
cp .env.example .env
```

4. Start the API:

```bash
uvicorn app.main:app --reload
```

## Configuration

Key environment variables:

- `ENV=local`
- `LOG_LEVEL=INFO`
- `LLM_PROVIDER=local|openai|gemini`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini`
- `GEMINI_API_KEY=...`
- `GEMINI_MODEL=gemini-flash-lite-latest`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `TOP_K=6`
- `USE_BM25=true`
- `RERANK_ENABLED=true`
- `RERANK_CANDIDATE_COUNT=12`

## API Endpoints

### `GET /health`

Returns service health and whether an index is loaded.

### `GET /repos`

Lists all indexed repositories and indicates the active default repo.

### `POST /ingest`

```json
{
  "repo_path": "./sample_repo",
  "set_active": true
}
```

Example response:

```json
{
  "repo_id": "sample-repo-1234567890",
  "repo_name": "sample_repo",
  "repo_path": "/absolute/path/to/sample_repo",
  "file_count": 3,
  "chunk_count": 6,
  "indexed": true,
  "active": true,
  "summary": "..."
}
```

### `POST /ask`

```json
{
  "repo_id": "sample-repo-1234567890",
  "question": "How is completion_ratio computed?"
}
```

Example response:

```json
{
  "repo_id": "sample-repo-1234567890",
  "answer": "build_profile_response loads a user profile and computes completion_ratio by calling divide_numbers with completed_tasks and assigned_tasks. Citations: sample_repo/app.py:4-7, sample_repo/utils.py:11-12",
  "citations": [
    {
      "repo_id": "sample-repo-1234567890",
      "file_path": "sample_repo/app.py",
      "symbol": "build_profile_response",
      "lines": "4-7"
    },
    {
      "repo_id": "sample-repo-1234567890",
      "file_path": "sample_repo/utils.py",
      "symbol": "divide_numbers",
      "lines": "11-12"
    }
  ],
  "confidence": "high"
}
```

### `POST /debug`

```json
{
  "repo_id": "sample-repo-1234567890",
  "question": "Why does the sample repo crash at runtime?",
  "error_context": "ZeroDivisionError: division by zero\nTraceback (most recent call last):\n  File \"sample_repo/app.py\", line 10, in <module>\n    print(build_profile_response(\"alice\"))\n  File \"sample_repo/app.py\", line 6, in build_profile_response\n    completion = divide_numbers(profile[\"completed_tasks\"], profile[\"assigned_tasks\"])\n  File \"sample_repo/utils.py\", line 12, in divide_numbers\n    return numerator / denominator"
}
```

### `GET /repos/{repo_id}/summary`

Returns a persisted summary, highlights, and stats for the selected repository.

## Engineering Decisions

- Python uses AST chunking because function and class boundaries are retrieval-critical.
- Module chunks are still indexed so imports, constants, and top-level code remain searchable.
- Retrieval uses weighted fusion instead of raw vector search to avoid missing exact symbol and filename matches.
- Reranking adds a second pass that emphasizes symbol and file-path matches when the first-stage retrieval returns several close candidates.
- Multi-repo support isolates FAISS, chunks, metadata, and summaries per repository to avoid mixing unrelated codebases.
- The local generation path is deterministic and grounded, so the service remains runnable even without an API key.

## Local Workflow

1. Ingest a repository:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_path":"./sample_repo","set_active":true}'
```

2. Ask a codebase question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is divide_numbers used?"}'
```

3. Debug a failure:

```bash
curl -X POST http://127.0.0.1:8000/debug \
  -H "Content-Type: application/json" \
  -d '{"question":"Why does this crash?","error_context":"ZeroDivisionError: division by zero"}'
```

4. Read the repository summary:

```bash
curl http://127.0.0.1:8000/repos/<repo_id>/summary
```

## Notes

- The first ingestion downloads the embedding model from Hugging Face if it is not already cached locally.
- OpenAI generation is optional; local mode works without external LLM access.
- Each ingested repository gets its own artifact directory under `data/repos/`.
- If `repo_id` is omitted from `/ask` or `/debug`, the active repository is used.
