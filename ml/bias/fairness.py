"""Group fairness metrics for a binary protected attribute."""
from __future__ import annotations

import numpy as np


def _rate(mask_pred, mask_group):
    n = mask_group.sum()
    return float(mask_pred[mask_group].mean()) if n else 0.0


def fairness_report(y_pred, protected, y_true=None) -> dict:
    y_pred, protected = np.asarray(y_pred).astype(int), np.asarray(protected).astype(int)
    g0, g1 = protected == 0, protected == 1
    sr0, sr1 = _rate(y_pred == 1, g0), _rate(y_pred == 1, g1)
    dpd = abs(sr1 - sr0)
    di = (min(sr0, sr1) / max(sr0, sr1)) if max(sr0, sr1) > 0 else 1.0
    eod = 0.0
    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)
        tpr0 = _rate(y_pred == 1, g0 & (y_true == 1))
        tpr1 = _rate(y_pred == 1, g1 & (y_true == 1))
        eod = abs(tpr1 - tpr0)
    worst = max(dpd, eod)
    score = min(100.0, worst / 0.25 * 100)
    if di < 0.8:  # four-fifths rule
        score = max(score, 60.0)
    level = "LOW" if score < 30 else "MEDIUM" if score < 60 else "HIGH"
    return {
        "selection_rate_group0": round(sr0, 4), "selection_rate_group1": round(sr1, 4),
        "demographic_parity_diff": round(dpd, 4), "equal_opportunity_diff": round(eod, 4),
        "disparate_impact": round(di, 4), "bias_score": round(score, 1), "level": level,
    }
