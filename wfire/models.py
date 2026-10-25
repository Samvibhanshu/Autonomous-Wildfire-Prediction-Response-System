"""Model definitions and the trained-model bundle used at inference time."""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .aggregator import Assessment, aggregate
from .config import ARTIFACTS, FEATURES, SEED

MODEL_NAMES = ["random_forest", "gradient_boosting", "logistic_regression"]

DEFAULT_PARAMS = {
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 3, "max_features": "sqrt"},
    "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
    "logistic_regression": {"C": 1.0},
}

PARAM_GRIDS = {
    "random_forest": [{"n_estimators": n, "min_samples_leaf": l, "max_features": f}
                      for n in (200, 400) for l in (1, 3, 5) for f in ("sqrt", 0.5)],
    "gradient_boosting": [{"n_estimators": n, "learning_rate": lr, "max_depth": d}
                          for n in (100, 200) for lr in (0.03, 0.1) for d in (2, 3)],
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
}


def build(name: str, params: dict | None = None, seed: int = SEED):
    p = dict(DEFAULT_PARAMS[name], **(params or {}))
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **p)
    if name == "logistic_regression":                    # scaling matters here (section 8.1)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **p))
    raise KeyError(name)


class RiskEngine:
    """Loads trained models and turns feature dictionaries into aggregated risk assessments."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.models = bundle["models"]
        self.version = bundle["version"]
        self.weights = bundle.get("weights")
        self.train_ranges = bundle["train_ranges"]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "RiskEngine":
        path = Path(path or ARTIFACTS / "model_bundle.joblib")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python train.py` first.")
        return cls(joblib.load(path))

    def per_model_proba(self, X: pd.DataFrame) -> dict:
        return {n: m.predict_proba(X[FEATURES])[:, 1] for n, m in self.models.items()}

    def assess(self, features: dict, warnings=None) -> Assessment:
        X = pd.DataFrame([{k: float(features[k]) for k in FEATURES}])
        probs = {n: float(v[0]) for n, v in self.per_model_proba(X).items()}
        return aggregate(probs, self.weights, warnings=warnings)

    def assess_many(self, rows: list[dict], warnings=None) -> list[Assessment]:
        if not rows:
            return []
        X = pd.DataFrame([{k: float(r[k]) for k in FEATURES} for r in rows])
        P = self.per_model_proba(X)
        return [aggregate({n: float(P[n][i]) for n in P}, self.weights,
                          warnings=(warnings[i] if warnings else None)) for i in range(len(rows))]

    def feature_importance(self) -> dict:
        rf = self.models["random_forest"]
        return dict(sorted(zip(FEATURES, map(float, rf.feature_importances_)), key=lambda kv: -kv[1]))


def make_bundle(models: dict, version: str, train_ranges: dict, weights=None, meta=None) -> dict:
    return {"models": models, "version": version, "train_ranges": train_ranges, "weights": weights,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "meta": meta or {}}
