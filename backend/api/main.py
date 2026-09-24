"""FastAPI gateway. Run: uvicorn backend.api.main:app --reload"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from agents.governance import registry, service
from agents.orchestrator.workflow import HAS_LANGGRAPH, run_workflow
from agents.security.guardrails import scan
from monitoring.logging_config import setup_logging
from monitoring.metrics.recorder import metrics

setup_logging()
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.init_db()
    if os.getenv("AUTO_BOOTSTRAP", "false").lower() == "true" and not registry.list_profiles():
        service.bootstrap_demo()
    yield
    metrics.flush()


app = FastAPI(title="AI Governance & Observability Platform", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.record("api_errors", 1, "Count", {"path": request.url.path})
        raise
    ms = (time.perf_counter() - t0) * 1000
    metrics.record("api_latency_ms", ms, "Milliseconds", {"path": request.url.path})
    if response.status_code >= 500:
        metrics.record("api_errors", 1, "Count", {"path": request.url.path})
    return response


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    model_id: Optional[str] = None
    user: str = "anonymous"


class ScanRequest(BaseModel):
    text: str


class ReviewRequest(BaseModel):
    decision: Literal["APPROVE", "REJECT", "REQUEST_CHANGES"]
    reviewer: str
    comment: str = ""


class PredictRequest(BaseModel):
    row: dict


class TrainRequest(BaseModel):
    name: str = "CreditRisk"
    version: str = "v1"
    owner: str = "risk-team"
    bias: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "orchestrator": "langgraph" if HAS_LANGGRAPH else "sequential"}


@app.post("/demo/bootstrap")
def bootstrap():
    return service.bootstrap_demo()


@app.post("/models/train")
def train_model(req: TrainRequest):
    return service.train_and_register(req.name, req.version, req.owner, req.bias)


@app.get("/models")
def list_models():
    return registry.list_profiles()


def _get(model_id: str) -> dict:
    p = registry.get_profile(model_id)
    if not p:
        raise HTTPException(404, f"model {model_id} not found")
    return p


@app.get("/models/{model_id}")
def get_model(model_id: str):
    return _get(model_id)


@app.get("/models/{model_id}/timeline")
def timeline(model_id: str):
    _get(model_id)
    return registry.list_evaluations(model_id)


@app.post("/models/{model_id}/evaluate")
async def evaluate(model_id: str, scenario: str = "none", file: UploadFile | None = File(default=None)):
    """Evaluate against a synthetic scenario (none|drift|concept|severe) or an uploaded CSV
    (feature columns, optionally a `default` label column)."""
    _get(model_id)
    try:
        current = pd.read_csv(io.BytesIO(await file.read())) if file else None
        return service.evaluate_model(model_id, current=current, scenario=scenario)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/models/{model_id}/predict")
def predict(model_id: str, req: PredictRequest):
    _get(model_id)
    try:
        return service.predict(model_id, req.row)
    except KeyError as exc:
        raise HTTPException(400, f"missing feature {exc}")


@app.post("/models/{model_id}/review")
def review(model_id: str, req: ReviewRequest):
    try:
        return service.apply_review(model_id, req.decision, req.reviewer, req.comment)
    except FileNotFoundError:
        raise HTTPException(404, "model not found")


@app.get("/reviews/pending")
def pending():
    return [p for p in registry.list_profiles() if p["status"] == "PENDING_REVIEW"]


@app.get("/reviews/{model_id}")
def review_history(model_id: str):
    return registry.list_reviews(model_id)


@app.post("/guardrails/scan")
def guardrails(req: ScanRequest):
    r = scan(req.text)
    d = r.to_dict()
    registry.log_event("security", "HIGH" if r.blocked else "INFO",
                       {"blocked": r.blocked, "reasons": r.reasons, "pii_types": d["pii_types"]})
    return d


@app.post("/chat")
def chat(req: ChatRequest):
    s = run_workflow(req.query, req.model_id, req.user)
    return {k: s.get(k) for k in ("final_response", "blocked", "block_reason", "pii_types", "hallucination_score",
                                  "groundedness", "retrieved_docs", "risk_score", "risk_level", "request_risk",
                                  "requires_human_approval", "human_approval", "llm_usage", "trace", "engine")}


@app.get("/events")
def events(limit: int = 100, type: Optional[str] = None):
    return registry.list_events(limit, type)


@app.get("/observability")
def observability():
    return {"metrics": metrics.summary(), "latency_series": metrics.series("api_latency_ms")}


@app.get("/finops")
def finops():
    return registry.usage_summary()
