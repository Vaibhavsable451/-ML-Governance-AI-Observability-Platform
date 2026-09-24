"""Unified governance risk score (0-100, higher = riskier) and approval routing."""
from __future__ import annotations

from core.config import AUTO_APPROVE_MAX, LEVELS

WEIGHTS = {"drift": 0.20, "performance": 0.20, "bias": 0.15, "security": 0.15,
           "hallucination": 0.10, "explainability": 0.10, "cost": 0.10}
LABELS = {"drift": "Data/feature drift", "performance": "Model performance degradation",
          "bias": "Fairness/bias", "security": "Security incidents", "hallucination": "LLM hallucination rate",
          "explainability": "Explainability concentration", "cost": "Inference cost vs budget"}
HARD_GATE = 75  # any single component at/above this forces human review


def level_for(score: float) -> str:
    for upper, name in LEVELS:
        if score <= upper:
            return name
    return "CRITICAL"


def performance_risk(baseline_auc: float, current_auc: float | None) -> float:
    cur = baseline_auc if current_auc is None else current_auc
    decay = max(0.0, baseline_auc - cur) / 0.10 * 100
    floor = max(0.0, 0.80 - cur) / 0.30 * 100
    return round(min(100.0, max(decay, floor)), 1)


def cost_risk(spent_usd: float, budget_usd: float) -> float:
    return round(min(100.0, spent_usd / budget_usd * 100), 1) if budget_usd > 0 else 0.0


def compute_risk(components: dict, weights: dict | None = None) -> dict:
    w = weights or WEIGHTS
    comps = {k: round(float(min(100, max(0, components.get(k, 0)))), 1) for k in w}
    score = round(sum(comps[k] * w[k] for k in w) / sum(w.values()), 1)
    level = level_for(score)
    reasons = []
    for k, v in sorted(comps.items(), key=lambda kv: kv[1], reverse=True):
        if v >= 60:
            reasons.append({"severity": "HIGH", "component": k, "score": v, "text": f"{LABELS[k]} is high ({v}/100)"})
        elif v >= 30:
            reasons.append({"severity": "WARN", "component": k, "score": v, "text": f"{LABELS[k]} needs attention ({v}/100)"})
    gated = [k for k, v in comps.items() if v >= HARD_GATE]
    if score <= AUTO_APPROVE_MAX and not gated:
        decision = "AUTO_APPROVE"
    else:
        decision = "HUMAN_REVIEW"
    return {"score": score, "level": level, "components": comps, "weights": w, "decision": decision,
            "escalated": score > 60 or bool(gated), "reasons": reasons}
