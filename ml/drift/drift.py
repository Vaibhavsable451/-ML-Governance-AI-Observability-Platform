"""Drift engine: PSI, KS, Jensen-Shannon, Wasserstein + concept drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

PSI_WARN, PSI_CRIT = 0.10, 0.25


def psi(expected, actual, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant or binary feature
        mid = float(np.median(expected))
        edges = np.array([-np.inf, mid + 1e-9, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_test(expected, actual) -> tuple[float, float]:
    r = stats.ks_2samp(expected, actual)
    return float(r.statistic), float(r.pvalue)


def js_divergence(expected, actual, bins: int = 20) -> float:
    lo = min(np.min(expected), np.min(actual))
    hi = max(np.max(expected), np.max(actual))
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(expected, edges)[0] + 1e-9
    q = np.histogram(actual, edges)[0] + 1e-9
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def wasserstein_norm(expected, actual) -> float:
    """Wasserstein distance scaled by the reference std so features are comparable."""
    sd = float(np.std(expected)) or 1.0
    return float(stats.wasserstein_distance(expected, actual) / sd)


def status_for(psi_value: float) -> str:
    if psi_value < PSI_WARN:
        return "STABLE"
    return "WARNING" if psi_value <= PSI_CRIT else "CRITICAL"


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict:
    per_feature = {}
    for f in features:
        ref, cur = reference[f].to_numpy(float), current[f].to_numpy(float)
        p = psi(ref, cur)
        ks_stat, ks_p = ks_test(ref, cur)
        per_feature[f] = {
            "psi": round(p, 4), "ks_stat": round(ks_stat, 4), "ks_pvalue": round(ks_p, 6),
            "js": round(js_divergence(ref, cur), 4), "wasserstein": round(wasserstein_norm(ref, cur), 4),
            "status": status_for(p),
            "drift_pct": round(min(p / 0.5, 1.0) * 100, 1),
        }
    scores = [v["drift_pct"] for v in per_feature.values()]
    # blend mean and worst feature so one badly drifted feature is not averaged away
    overall = 0.5 * float(np.mean(scores)) + 0.5 * float(np.max(scores))
    return {
        "features": per_feature,
        "drift_score": round(overall, 1),
        "critical": [f for f, v in per_feature.items() if v["status"] == "CRITICAL"],
        "warning": [f for f, v in per_feature.items() if v["status"] == "WARNING"],
    }


def concept_drift(baseline_auc: float, current_auc: float | None) -> dict:
    """Concept drift = the input->label relationship changed, visible as performance decay."""
    if current_auc is None:
        return {"available": False, "auc_drop": 0.0, "status": "UNKNOWN"}
    drop = baseline_auc - current_auc
    status = "STABLE" if drop < 0.02 else "WARNING" if drop < 0.05 else "CRITICAL"
    return {"available": True, "auc_drop": round(drop, 4), "status": status}
