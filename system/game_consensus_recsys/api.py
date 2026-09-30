"""Optional FastAPI surface. The domain package remains usable without FastAPI."""
from __future__ import annotations

from dataclasses import asdict

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional runtime dependency
    FastAPI = None

from .domain import SearchRequest
from .workflow import ConsensusWorkflow


if FastAPI is not None:
    class SearchBody(BaseModel):
        query: str
        top_k: int = Field(default=5, ge=1, le=20)
        preferred_genres: list[str] = []
        required_genres: list[str] = []
        excluded_genres: list[str] = []
        min_year: int | None = None
        max_year: int | None = None
        latency_budget_ms: int = Field(default=2000, gt=0)

    app = FastAPI(title="Game Consensus Recommender", version="0.1.0")
    workflow = ConsensusWorkflow.from_environment()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/search")
    def search(body: SearchBody):
        try:
            response = workflow.run(SearchRequest(**body.model_dump()))
            return asdict(response)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
else:
    app = None
