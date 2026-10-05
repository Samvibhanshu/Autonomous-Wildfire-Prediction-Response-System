import json

import numpy as np
import pytest

from wfire import data as D
from wfire import training
from wfire.config import FEATURES


def test_dataset_matches_the_disclosure_description():
    raw = D.load_raw()
    assert raw.shape == (517, 13) and raw.isna().sum().sum() == 0
    X, y = D.encode(raw)
    assert list(X.columns) == FEATURES and int(y.sum()) == 270 and int((y == 0).sum()) == 247      # fire = area > 0
    assert X.month.min() == 1 and X.month.max() == 12 and X.day.min() == 0 and X.day.max() == 6
    assert X.loc[raw.month == "dec", "month"].eq(12).all() and X.loc[raw.day == "mon", "day"].eq(0).all()


def test_split_is_70_15_15_stratified_and_disjoint():
    X, y = D.encode(D.load_raw())
    Xtr, Xva, Xte, ytr, yva, yte = D.split_70_15_15(X, y)
    assert (len(Xtr), len(Xva), len(Xte)) == (361, 78, 78)
    assert not (set(Xtr.index) & set(Xva.index)) and not (set(Xtr.index) & set(Xte.index)) and not (set(Xva.index) & set(Xte.index))
    for part in (ytr, yva, yte):
        assert abs(part.mean() - y.mean()) < 0.02


def test_quick_training_writes_bundle_metrics_and_is_internally_consistent(artifacts):
    m = json.loads((artifacts / "metrics.json").read_text())
    assert (artifacts / "model_bundle.joblib").exists() and m["split"] == {"train": 361, "validation": 78, "test": 78}
    for name in ("random_forest", "gradient_boosting", "logistic_regression", "ensemble"):
        r = m["test"][name]
        assert 0 <= r["accuracy"] <= 1 and 0.3 < r["auc"] < 1 and r["auc_ci95"][0] <= r["auc"] <= r["auc_ci95"][1]
        cv = m["cross_validation"]["results"][name]
        assert 0.3 < cv["auc"]["mean"] < 0.9 and cv["accuracy"]["std"] >= 0
    g = m["confidence_gating"]
    assert sum(v["n"] for v in g["by_level"].values()) == 517
    assert g["alerts"]["confident HIGH/CRITICAL alerts only"]["n"] <= g["alerts"]["all HIGH/CRITICAL alerts"]["n"]
    assert "PROJECTED" in m["disclosure_projection_note"]
    assert abs(sum(m["rf_feature_importance"].values()) - 1) < 1e-6


def test_bootstrap_ci_brackets_the_point_estimate_and_gating_analysis_arithmetic():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); p = np.clip(y * .3 + rng.random(300) * .7, 0, 1)
    lo, hi = training.bootstrap_ci(y, p, roc_auc_score, 300)
    assert lo < roc_auc_score(y, p) < hi
    oof = {"a": np.array([.9, .9, .1, .5]), "b": np.array([.8, .2, .2, .5]), "c": np.array([.9, .8, .1, .5])}
    g = training.gating_analysis(oof, np.array([1, 0, 0, 1]))
    assert sum(v["n"] for v in g["by_level"].values()) == 4 and g["base_rate"] == .5
