from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException


class IngestRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to the repository to index.")
    set_active: bool = True


class AskRequest(BaseModel):
    question: str
    repo_id: str | None = None


class DebugRequest(BaseModel):
    question: str | None = None
    error_context: str
    repo_id: str | None = None


class Citation(BaseModel):
    repo_id: str
    file_path: str
    symbol: str
    lines: str
    score: float | None = None


class AnswerResponse(BaseModel):
    repo_id: str | None = None
    answer: str
    citations: list[Citation]
    confidence: str


class IngestResponse(BaseModel):
    repo_id: str
    repo_name: str
    repo_path: str
    file_count: int
    chunk_count: int
    indexed: bool
    active: bool
    summary: str | None = None


class HealthResponse(BaseModel):
    status: str
    indexed: bool
    chunk_count: int
    repo_count: int
    active_repo_id: str | None = None


class RepositorySummaryResponse(BaseModel):
    repo_id: str
    repo_name: str
    repo_path: str
    file_count: int
    chunk_count: int
    indexed_at: str | None = None
    summary: str
    highlights: list[dict]
    stats: dict


class RepositoryListResponse(BaseModel):
    repositories: list[dict]


def create_router(service_container) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        retriever = service_container.retriever
        active = retriever.get_active_repository()
        repositories = retriever.list_repositories()
        return HealthResponse(
            status="ok",
            indexed=retriever.is_ready(),
            chunk_count=(active.get("chunk_count", 0) if active else 0),
            repo_count=len(repositories),
            active_repo_id=(active.get("repo_id") if active else None),
        )

    @router.get("/repos", response_model=RepositoryListResponse)
    def list_repositories() -> RepositoryListResponse:
        return RepositoryListResponse(repositories=service_container.retriever.list_repositories())

    @router.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        try:
            result = service_container.ingestion.ingest(request.repo_path, set_active=request.set_active)
            service_container.retriever.refresh(result.repo_id)
            summary_payload = service_container.summary.summarize_repository(result.repo_id, refresh=True)
            return IngestResponse(
                repo_id=result.repo_id,
                repo_name=result.repo_name,
                repo_path=result.repo_path,
                file_count=result.file_count,
                chunk_count=result.chunk_count,
                indexed=result.indexed,
                active=result.active,
                summary=summary_payload["summary"],
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    @router.post("/ask", response_model=AnswerResponse)
    def ask(request: AskRequest) -> AnswerResponse:
        try:
            return AnswerResponse(**service_container.qa.answer_question(request.question, repo_id=request.repo_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

    @router.post("/debug", response_model=AnswerResponse)
    def debug(request: DebugRequest) -> AnswerResponse:
        try:
            return AnswerResponse(**service_container.debug.debug(request.error_context, request.question, repo_id=request.repo_id))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Debug analysis failed: {exc}") from exc

    @router.get("/repos/{repo_id}/summary", response_model=RepositorySummaryResponse)
    def get_repository_summary(repo_id: str) -> RepositorySummaryResponse:
        try:
            payload = service_container.summary.summarize_repository(repo_id=repo_id, refresh=False)
            return RepositorySummaryResponse(**payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Summary retrieval failed: {exc}") from exc

    @router.post("/repos/{repo_id}/summary", response_model=RepositorySummaryResponse)
    def refresh_repository_summary(repo_id: str) -> RepositorySummaryResponse:
        try:
            payload = service_container.summary.summarize_repository(repo_id=repo_id, refresh=True)
            return RepositorySummaryResponse(**payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Summary refresh failed: {exc}") from exc

    return router
