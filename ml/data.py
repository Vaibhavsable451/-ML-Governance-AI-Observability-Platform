"""Synthetic credit-risk data with knobs to simulate data drift, concept drift and bias."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["age", "income", "credit_score", "debt_ratio", "employment_years", "loan_amount", "gender"]
TARGET = "default"
PROTECTED = "gender"


def make_credit_data(n: int = 5000, seed: int = 42, drift: float = 0.0,
                     concept: float = 0.0, bias: float = 0.0) -> pd.DataFrame:
    """drift   0..1 shifts input distributions (data / feature drift)
    concept 0..1 weakens the credit_score -> default relationship (concept drift)
    bias    adds a direct gender effect to the label (fairness demo)"""
    rng = np.random.default_rng(seed)
    age = rng.normal(40 + 8 * drift, 11, n).clip(18, 85)
    income = rng.lognormal(10.8 + 0.5 * drift, 0.5, n)
    credit_score = rng.normal(680 - 45 * drift, 70, n).clip(300, 850)
    debt_ratio = rng.beta(2 + 2.5 * drift, 5, n)
    employment_years = rng.gamma(4, 2.0 * (1 - 0.5 * drift), n).clip(0, 40)
    loan_amount = rng.lognormal(9.8 + 0.25 * drift, 0.6, n)
    gender = rng.binomial(1, 0.5, n)

    logit = (
        -2.6
        + 5.0 * debt_ratio
        - 0.02 * (1 - concept) * (credit_score - 650)
        - 1.0 * (np.log(income) - 10.8)
        + 0.6 * (np.log(loan_amount) - 9.8)
        - 0.05 * employment_years
        + bias * gender
        + rng.normal(0, 0.2, n)
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({
        "age": age, "income": income, "credit_score": credit_score, "debt_ratio": debt_ratio,
        "employment_years": employment_years, "loan_amount": loan_amount, "gender": gender,
        TARGET: y,
    })
