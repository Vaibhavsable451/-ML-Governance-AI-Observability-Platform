"""Central configuration. Everything is overridable through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = DATA_DIR / "artifacts"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "governance.db"))
POLICY_DIR = Path(os.getenv("POLICY_DIR", ROOT / "rag" / "policies"))

# LLM: "mock" works offline (extractive answers); set ANTHROPIC_API_KEY to use Claude.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
LLM_PRICE_IN_PER_1M = float(os.getenv("LLM_PRICE_IN_PER_1M", "3.0"))
LLM_PRICE_OUT_PER_1M = float(os.getenv("LLM_PRICE_OUT_PER_1M", "15.0"))
MONTHLY_BUDGET_USD = float(os.getenv("MONTHLY_BUDGET_USD", "50"))

# Observability
USE_CLOUDWATCH = os.getenv("USE_CLOUDWATCH", "false").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CLOUDWATCH_NAMESPACE = os.getenv("CLOUDWATCH_NAMESPACE", "AIGovernance")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")

# Retrieval
EMBEDDER = os.getenv("EMBEDDER", "tfidf")  # tfidf | sbert

# Risk thresholds
AUTO_APPROVE_MAX = 30
LEVELS = [(30, "LOW"), (60, "MEDIUM"), (80, "HIGH"), (100, "CRITICAL")]

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
