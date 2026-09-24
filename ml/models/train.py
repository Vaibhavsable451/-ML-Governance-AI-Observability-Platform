"""Train the demo credit-risk model, log it (MLflow optional) and write artifacts."""
from __future__ import annotations

import json
import time

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from core import config
from ml.data import FEATURES, TARGET, make_credit_data


def train(name: str = "CreditRisk", version: str = "v1", owner: str = "risk-team",
          n: int = 6000, seed: int = 42, bias: float = 0.0) -> dict:
    df = make_credit_data(n, seed=seed, bias=bias)
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.25,
                                              random_state=seed, stratify=df[TARGET])
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed).fit(X_tr, y_tr)
    pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_te, pred), 4), "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4), "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }
    model_id = f"{name}-{version}"
    out = config.ARTIFACT_DIR / model_id
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    X_tr.sample(min(2000, len(X_tr)), random_state=seed).to_csv(out / "reference.csv", index=False)
    meta = {"model_id": model_id, "name": name, "version": version, "owner": owner,
            "dataset": f"synthetic-credit-{n}", "features": FEATURES, "target": TARGET,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metrics": metrics}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    if config.MLFLOW_TRACKING_URI:  # only when a tracking server is configured
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("ai-governance")
            with mlflow.start_run(run_name=model_id):
                mlflow.log_params({"n_estimators": 150, "max_depth": 3, "owner": owner})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
        except Exception as exc:  # never fail training because tracking is down
            meta["mlflow_error"] = str(exc)
    return meta


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
