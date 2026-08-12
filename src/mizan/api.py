"""FastAPI application for the Mizan verifier."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .model import MizanClassifier
from .retriever import EvidenceRetriever
from .service import VerifierService


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=2000, description="Arabic claim to verify")
    top_k: int = Field(default=3, ge=1, le=10)


class VerifyResponse(BaseModel):
    claim: str
    verdict: str | None
    confidence: float
    evidence_status: str
    model_version: str
    model_verdict: str | None = None
    retrieval_score: float | None = None
    probabilities: dict[str, float] | None = None
    evidence: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
_SERVICE: VerifierService | None = None


def configure_service(service: VerifierService) -> None:
    global _SERVICE
    _SERVICE = service


def load_service_from_env() -> VerifierService:
    model_path = Path(os.getenv("MIZAN_MODEL_PATH", "models/classifier.joblib"))
    index_path = Path(os.getenv("MIZAN_INDEX_PATH", "models/retriever.joblib"))
    classifier = MizanClassifier.load(model_path)
    retriever = __import__("joblib").load(index_path)
    if not isinstance(retriever, EvidenceRetriever):
        raise TypeError(f"artifact at {index_path} is not an EvidenceRetriever")
    return VerifierService(classifier=classifier, retriever=retriever)


def load_artifacts_if_available() -> None:
    global _SERVICE
    if _SERVICE is not None:
        return
    model_path = Path(os.getenv("MIZAN_MODEL_PATH", "models/classifier.joblib"))
    index_path = Path(os.getenv("MIZAN_INDEX_PATH", "models/retriever.joblib"))
    if model_path.exists() and index_path.exists():
        _SERVICE = load_service_from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts_if_available()
    yield


app = FastAPI(
    title="Mizan Arabic Claim Verification API",
    version="0.2.0",
    description="Evidence-backed, reviewable Arabic claim verification.",
    lifespan=lifespan,
)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
@app.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    if not (FRONTEND_DIR / "index.html").exists():
        raise HTTPException(status_code=404, detail="frontend demo is not installed")
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mizan", "model_loaded": str(_SERVICE is not None).lower()}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if _SERVICE is None:
        raise HTTPException(status_code=503, detail="model artifacts are not loaded")
    return _SERVICE.metadata()


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest) -> dict[str, Any]:
    if _SERVICE is None:
        raise HTTPException(status_code=503, detail="model artifacts are not loaded")
    try:
        return _SERVICE.verify(payload.claim, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
