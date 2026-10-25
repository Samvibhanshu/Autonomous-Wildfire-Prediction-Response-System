"""The tree models are constrained so hotter / drier / windier conditions can never lower the risk score.

The old sanity check ("heatwaves score higher than cool and wet weather") only passed in August, because the
simulator draws its baseline from the *current* calendar month and the model had learned almost nothing about
dryness (cross-validated AUC is about 0.6 on 517 records). These tests pin the property itself, in every month,
instead of hoping a noisy model happens to agree.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from wfire import data as D
from wfire import sensors
from wfire.config import FEATURES, MONOTONIC_DOWN, MONOTONIC_UP
from wfire.models import RiskEngine

CONSTRAINED_MODELS = ("random_forest", "gradient_boosting")        # logistic regression is linear and unconstrained


def _real_rows(n=200):
    X, _ = D.encode(D.load_raw())
    return X.sample(n, random_state=0).reset_index(drop=True)


def _shipped_engine():
    try:
        return RiskEngine.load()
    except Exception as e:                                           # e.g. a different scikit-learn than the pickle needs
        pytest.skip(f"shipped model bundle not loadable here: {e}")


@pytest.mark.parametrize("which", ["quick_trained", "shipped_bundle"])
def test_raising_a_fire_weather_feature_never_lowers_the_score(engine, which):
    eng = engine if which == "quick_trained" else _shipped_engine()
    base = _real_rows()
    ranges = eng.train_ranges
    for feat in MONOTONIC_UP + MONOTONIC_DOWN:
        lo, hi = ranges[feat]
        bumped = base.copy()
        bumped[feat] = np.minimum(bumped[feat] + 0.25 * (hi - lo), hi) if feat in MONOTONIC_UP else np.maximum(bumped[feat] - 0.25 * (hi - lo), lo)
        # "more fire-prone" means higher for UP features and lower for DOWN features, so risk must not drop either way.
        before, after = eng.per_model_proba(base), eng.per_model_proba(bumped)
        for name in CONSTRAINED_MODELS:
            assert (after[name] - before[name] >= -1e-9).all(), f"{name} lowered the score when {feat} became more fire-prone"


def test_the_constraint_is_what_makes_it_true(artifacts):
    """Guard against the test above passing vacuously: an UNconstrained forest does violate monotonicity on this data."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = D.encode(D.load_raw())
    free = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1).fit(X, y)
    violations = 0
    base = _real_rows()
    for feat in MONOTONIC_UP:
        lo, hi = X[feat].min(), X[feat].max()
        bumped = base.copy()
        bumped[feat] = np.minimum(bumped[feat] + 0.25 * (hi - lo), hi)
        violations += int((free.predict_proba(bumped[FEATURES])[:, 1] < free.predict_proba(base[FEATURES])[:, 1] - 1e-9).sum())
    assert violations > 0, "an unconstrained forest happened to be monotone here; this test no longer proves anything"


@pytest.mark.parametrize("which", ["quick_trained", "shipped_bundle"])
def test_heatwave_scores_at_least_cool_wet_for_the_same_zone_in_every_month(engine, which):
    eng = engine if which == "quick_trained" else _shipped_engine()
    raw = D.load_raw()
    for month in range(1, 13):
        now = datetime(2026, month, 15, 12, tzinfo=timezone.utc)
        for seed in range(3):
            hot = sensors.generate(raw, "heatwave", 30, seed, now)
            cool = sensors.generate(raw, "cool_wet", 30, seed, now)     # same seed => same zones and same baseline rows
            for r in hot + cool:
                r["month"], r["day"] = month, now.weekday()
            P_hot = eng.per_model_proba(pd.DataFrame(hot))
            P_cool = eng.per_model_proba(pd.DataFrame(cool))
            for name in CONSTRAINED_MODELS:
                assert (P_hot[name] >= P_cool[name] - 1e-9).all(), f"{name}: a cool, wet day outscored a heatwave in month {month}"
