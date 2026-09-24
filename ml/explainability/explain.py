"""SHAP explanations with an occlusion fallback if shap is unavailable."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    vals = shap.TreeExplainer(model).shap_values(X)
    if isinstance(vals, list):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return vals


def _occlusion_values(model, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    base = model.predict_proba(X)[:, 1]
    out = np.zeros(X.shape)
    means = background.mean()
    for j, c in enumerate(X.columns):
        Xo = X.copy()
        Xo[c] = means[c]
        out[:, j] = base - model.predict_proba(Xo)[:, 1]
    return out


def contributions(model, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    if HAS_SHAP:
        try:
            return _shap_values(model, X)
        except Exception:
            pass
    return _occlusion_values(model, X, background)


def global_importance(model, X: pd.DataFrame, background: pd.DataFrame, protected: str = "gender") -> dict:
    vals = np.abs(contributions(model, X, background)).mean(axis=0)
    total = vals.sum() or 1.0
    share = {c: float(v / total) for c, v in zip(X.columns, vals)}
    ranked = dict(sorted(share.items(), key=lambda kv: kv[1], reverse=True))
    top3 = list(ranked)[:3]
    # explainability risk: over-concentration in one feature + protected attribute among top drivers
    risk = min(100.0, max(ranked.values()) * 100 * 0.9 + (25 if protected in top3 else 0))
    return {
        "method": "shap" if HAS_SHAP else "occlusion",
        "importance": {k: round(v, 4) for k, v in ranked.items()},
        "protected_in_top3": protected in top3,
        "explainability_risk": round(risk, 1),
        "status": "PASS" if risk < 50 else "REVIEW",
    }


def explain_instance(model, row: dict, background: pd.DataFrame, features: list[str]) -> dict:
    X = pd.DataFrame([row])[features].astype(float)
    proba = float(model.predict_proba(X)[0, 1])
    vals = contributions(model, X, background)[0]
    items = sorted(({"feature": f, "value": float(row[f]), "contribution": round(float(v), 4)}
                    for f, v in zip(features, vals)), key=lambda d: abs(d["contribution"]), reverse=True)
    return {"probability": round(proba, 4), "prediction": "HIGH RISK" if proba >= 0.5 else "LOW RISK",
            "contributions": items, "method": "shap" if HAS_SHAP else "occlusion"}
