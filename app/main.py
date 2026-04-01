from __future__ import annotations

from dataclasses import dataclass
import logging

from fastapi import FastAPI

from app.api import create_router
from app.config import get_settings
from app.debug_service import DebugService
from app.ingest import RepositoryIngestionService
from app.qa_service import QAService, build_llm_client
from app.retriever import HybridRetriever
from app.summary_service import SummaryService


@dataclass(slots=True)
class ServiceContainer:
    ingestion: RepositoryIngestionService
    retriever: HybridRetriever
    qa: QAService
    debug: DebugService
    summary: SummaryService


def build_service_container() -> ServiceContainer:
    settings = get_settings()
    retriever = HybridRetriever(settings)
    llm_client = build_llm_client(settings)
    summary = SummaryService(settings, llm_client=llm_client)
    return ServiceContainer(
        ingestion=RepositoryIngestionService(settings),
        retriever=retriever,
        qa=QAService(settings, retriever, llm_client=llm_client, summary_service=summary),
        debug=DebugService(settings, retriever, llm_client=llm_client),
        summary=summary,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger(__name__).info("Starting app with llm_provider=%s", settings.llm_provider)
    service_container = build_service_container()

    app = FastAPI(title=settings.app_name, version="1.0.0")
    app.state.services = service_container
    app.include_router(create_router(service_container))
    return app


app = create_app()
