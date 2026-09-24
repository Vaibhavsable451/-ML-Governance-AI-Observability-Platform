"""Governance agent core: run all checks for a model and produce its governance profile."""
from __future__ import annotations

import json
import time
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from agents.governance import registry
from agents.risk.risk_engine import compute_risk, cost_risk, performance_risk
from core import config
from ml.bias.fairness import fairness_report
from ml.data import FEATURES, PROTECTED, TARGET, make_credit_data
from ml.drift.drift import concept_drift, detect_drift
from ml.explainability.explain import explain_instance, global_importance
from ml.models.train import train
from monitoring.alerts.notify import send_alert
from monitoring.metrics.recorder import metrics

SCENARIOS = {
    "none": dict(drift=0.0, concept=0.0),
    "drift": dict(drift=0.8, concept=0.0),
    "concept": dict(drift=0.1, concept=0.9),
    "severe": dict(drift=0.9, concept=0.9),
}


@lru_cache(maxsize=16)
def load_artifacts(model_id: str):
    d = config.ARTIFACT_DIR / model_id
    if not (d / "model.joblib").exists():
        raise FileNotFoundError(f"unknown model {model_id}")
    meta = json.loads((d / "meta.json").read_text())
    return joblib.load(d / "model.joblib"), pd.read_csv(d / "reference.csv"), meta


def scenario_data(scenario: str, n: int = 2000, seed: int = 7) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS)}")
    return make_credit_data(n, seed=seed, **SCENARIOS[scenario])


def evaluate_model(model_id: str, current: pd.DataFrame | None = None, scenario: str = "none") -> dict:
    model, ref, meta = load_artifacts(model_id)
    cur = current if current is not None else scenario_data(scenario)
    missing = [f for f in FEATURES if f not in cur.columns]
    if missing:
        raise ValueError(f"current data is missing columns: {missing}")
    X = cur[FEATURES].astype(float)
    baseline_auc = meta["metrics"]["roc_auc"]

    with metrics.timer("governance_eval_ms", {"model": model_id}):
        drift = detect_drift(ref, X, FEATURES)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        has_labels = TARGET in cur.columns
        cur_auc = float(roc_auc_score(cur[TARGET], proba)) if has_labels else None
        cur_acc = float(accuracy_score(cur[TARGET], pred)) if has_labels else None
        concept = concept_drift(baseline_auc, cur_auc)
        bias = fairness_report(pred, X[PROTECTED], cur[TARGET] if has_labels else None)
        expl = global_importance(model, X.sample(min(300, len(X)), random_state=0), ref[FEATURES], PROTECTED)

    usage = registry.usage_summary()
    components = {
        "drift": drift["drift_score"],
        "performance": performance_risk(baseline_auc, cur_auc),
        "bias": bias["bias_score"],
        "security": registry.security_risk(),
        "hallucination": registry.avg_hallucination(),
        "explainability": expl["explainability_risk"],
        "cost": cost_risk(usage["cost_usd"], usage["budget_usd"]),
    }
    risk = compute_risk(components)
    status = "APPROVED" if risk["decision"] == "AUTO_APPROVE" else "PENDING_REVIEW"

    profile = {
        "model_id": model_id, "name": meta["name"], "version": meta["version"], "owner": meta["owner"],
        "dataset": meta["dataset"], "trained_at": meta["trained_at"], "features": FEATURES,
        "baseline_metrics": meta["metrics"],
        "current_metrics": {"roc_auc": round(cur_auc, 4) if cur_auc is not None else None,
                            "accuracy": round(cur_acc, 4) if cur_acc is not None else None,
                            "rows": len(cur)},
        "drift": drift, "concept_drift": concept, "bias": bias, "explainability": expl,
        "cost": usage, "risk": risk, "status": status, "scenario": scenario if current is None else "uploaded",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    registry.save_profile(profile)
    metrics.record("risk_score", risk["score"], "None", {"model": model_id})
    metrics.record("drift_score", drift["drift_score"], "None", {"model": model_id})

    for f in drift["critical"]:
        registry.log_event("drift", "HIGH", {"feature": f, "psi": drift["features"][f]["psi"]}, model_id)
    if concept["status"] == "CRITICAL":
        registry.log_event("performance", "HIGH", concept, model_id)
    if bias["level"] == "HIGH":
        registry.log_event("bias", "HIGH", bias, model_id)
    if risk["decision"] == "HUMAN_REVIEW":
        registry.log_event("review_required", "CRITICAL" if risk["level"] == "CRITICAL" else "MEDIUM",
                           {"score": risk["score"], "level": risk["level"]}, model_id)
        if risk["escalated"]:
            send_alert(f"{model_id} risk {risk['score']} ({risk['level']})", {"model": model_id, "risk": risk})
    return profile


def predict(model_id: str, row: dict) -> dict:
    model, ref, _ = load_artifacts(model_id)
    with metrics.timer("inference_ms", {"model": model_id}):
        result = explain_instance(model, row, ref[FEATURES], FEATURES)
    metrics.record("predictions", 1, "Count", {"model": model_id})
    return result


def evidence_chunks(model_id: str) -> list[dict]:
    """Turn a stored profile into searchable text so the RAG assistant can cite real evidence."""
    p = registry.get_profile(model_id)
    if not p:
        return []
    src = f"governance-profile:{model_id}"
    r, d, b = p["risk"], p["drift"], p["bias"]
    comp = ", ".join(f"{k} {v}" for k, v in sorted(r["components"].items(), key=lambda kv: -kv[1]))
    chunks = [
        {"source": src, "text": f"Model {model_id} has an overall risk score of {r['score']} ({r['level']}). "
                                f"Status is {p['status']}. Risk components: {comp}."},
        {"source": src, "text": "Why model was flagged: " + ("; ".join(x["text"] for x in r["reasons"]) or "no risk reasons.")},
        {"source": src, "text": f"Drift evidence for {model_id}: critical features "
                                f"{', '.join(d['critical']) or 'none'}; warning features {', '.join(d['warning']) or 'none'}. "
                                + " ".join(f"{f} PSI {v['psi']}." for f, v in d["features"].items() if v["status"] != "STABLE")},
        {"source": src, "text": f"Bias evidence for {model_id}: demographic parity difference "
                                f"{b['demographic_parity_diff']}, equal opportunity difference {b['equal_opportunity_diff']}, "
                                f"disparate impact {b['disparate_impact']}, bias level {b['level']}."},
        {"source": src, "text": f"Performance for {model_id}: baseline ROC AUC {p['baseline_metrics']['roc_auc']}, "
                                f"current ROC AUC {p['current_metrics']['roc_auc']}, concept drift {p['concept_drift']['status']}."},
    ]
    for rv in registry.list_reviews(model_id)[:3]:
        chunks.append({"source": src, "text": f"Review of {model_id}: {rv['decision']} by {rv['reviewer']}. {rv['comment']}"})
    return chunks


def apply_review(model_id: str, decision: str, reviewer: str, comment: str) -> dict:
    mapping = {"APPROVE": "APPROVED", "REJECT": "REJECTED", "REQUEST_CHANGES": "CHANGES_REQUESTED"}
    if decision not in mapping:
        raise ValueError(f"decision must be one of {list(mapping)}")
    if not registry.get_profile(model_id):
        raise FileNotFoundError(model_id)
    registry.add_review(model_id, decision, reviewer, comment)
    registry.set_status(model_id, mapping[decision])
    registry.log_event("review", "INFO", {"decision": decision, "reviewer": reviewer}, model_id)
    return registry.get_profile(model_id)


def train_and_register(name="CreditRisk", version="v1", owner="risk-team", bias: float = 0.0) -> dict:
    meta = train(name=name, version=version, owner=owner, bias=bias)
    load_artifacts.cache_clear()
    return evaluate_model(meta["model_id"], scenario="none")


def bootstrap_demo() -> list[dict]:
    """Two models: a healthy baseline and a v2 that has been evaluated against drifted production data."""
    registry.init_db()
    out = [train_and_register("CreditRisk", "v1")]
    meta = train(name="FraudModel", version="v8", owner="fraud-team", bias=0.8)
    load_artifacts.cache_clear()
    out.append(evaluate_model(meta["model_id"], scenario="drift"))
    return out
