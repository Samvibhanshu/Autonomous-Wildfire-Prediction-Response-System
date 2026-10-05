"""Reproduces the prototype experiment of disclosure section 8 and evaluates it honestly.

Beyond the single 70/15/15 split the disclosure describes, this also reports repeated cross-validation, a
majority-class baseline, bootstrap confidence intervals, and what the confidence gate actually does. With only
517 records (78 in the test split), a single split is a noisy estimate of real performance.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from . import data as D
from .aggregator import aggregate, risk_level
from .config import ARTIFACTS, CONFIDENCE_GATE, LEVELS, SEED
from .models import MODEL_NAMES, PARAM_GRIDS, DEFAULT_PARAMS, build, make_bundle


def _metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p)}


def tune(name, Xtr, ytr, Xva, yva, quick=False):
    """Pick hyper-parameters by validation AUC (the validation split's role in section 8.1)."""
    grid = PARAM_GRIDS[name][:2] if quick else PARAM_GRIDS[name]
    best, best_auc = DEFAULT_PARAMS[name], -1
    for params in grid:
        m = build(name, params).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if auc > best_auc:
            best, best_auc = params, auc
    return best, best_auc


def bootstrap_ci(y, p, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p, vals = np.asarray(y), np.asarray(p), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def oof_predictions(params, X, y, n_splits=5, n_repeats=10, seed=SEED):
    """Out-of-fold probabilities for every model, averaged over repeats. Also per-fold accuracy/AUC for mean +/- std."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    sums = {n: np.zeros(len(y)) for n in MODEL_NAMES}
    counts, folds = np.zeros(len(y)), {n: {"accuracy": [], "auc": []} for n in MODEL_NAMES + ["ensemble"]}
    for tr, te in cv.split(X, y):
        fold_p = {}
        for n in MODEL_NAMES:
            m = build(n, params[n]).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            sums[n][te] += p
            fold_p[n] = p
            folds[n]["accuracy"].append(accuracy_score(y.iloc[te], p >= .5))
            folds[n]["auc"].append(roc_auc_score(y.iloc[te], p))
        ens = np.mean([fold_p[n] for n in MODEL_NAMES], axis=0)
        folds["ensemble"]["accuracy"].append(accuracy_score(y.iloc[te], ens >= .5))
        folds["ensemble"]["auc"].append(roc_auc_score(y.iloc[te], ens))
        counts[te] += 1
    oof = {n: sums[n] / counts for n in MODEL_NAMES}
    summary = {n: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in d.items()} for n, d in folds.items()}
    return oof, summary


def gating_analysis(oof, y):
    """What the confidence gate does to alert precision, and the fire rate actually seen at each risk level."""
    y = np.asarray(y)
    rows = pd.DataFrame(oof)
    ens = rows.mean(axis=1).values
    spread = (rows.max(axis=1) - rows.min(axis=1)).values
    unanimous = ((rows >= .5).all(axis=1) | (rows < .5).all(axis=1)).values
    confident = (1 - spread >= CONFIDENCE_GATE) & unanimous
    levels = np.array([risk_level(p) for p in ens])
    out = {"by_level": {}, "alerts": {}}
    for lv in LEVELS:
        mask = levels == lv
        out["by_level"][lv] = {"n": int(mask.sum()), "observed_fire_rate": float(y[mask].mean()) if mask.any() else None}
    alert = np.isin(levels, ["HIGH", "CRITICAL"])
    for label, mask in (("all HIGH/CRITICAL alerts", alert), ("confident HIGH/CRITICAL alerts only", alert & confident)):
        n = int(mask.sum())
        out["alerts"][label] = {"n": n, "precision": float(y[mask].mean()) if n else None,
                                "false_alarm_rate": float(1 - y[mask].mean()) if n else None,
                                "recall_of_all_fires": float(y[mask].sum() / y.sum()) if n else 0.0}
    out["base_rate"] = float(y.mean())
    return out


def run(quick=False, out_dir=ARTIFACTS, make_figs=True, seed=SEED, path=D.DATA_PATH):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = D.load_raw(path)
    X, y = D.encode(raw)
    Xtr, Xva, Xte, ytr, yva, yte = D.split_70_15_15(X, y, seed)

    # 1) tune on validation, 2) evaluate the train-only fit on the untouched test split
    params, val_auc, test_p, fitted = {}, {}, {}, {}
    for n in MODEL_NAMES:
        params[n], val_auc[n] = tune(n, Xtr, ytr, Xva, yva, quick)
        fitted[n] = build(n, params[n]).fit(Xtr, ytr)
        test_p[n] = fitted[n].predict_proba(Xte)[:, 1]
    test_p["ensemble"] = np.mean([test_p[n] for n in MODEL_NAMES], axis=0)
    baseline = float(max(yte.mean(), 1 - yte.mean()))
    test = {n: _metrics(yte, p) for n, p in test_p.items()}
    for n in test:
        test[n]["auc_ci95"] = bootstrap_ci(yte, test_p[n], roc_auc_score, 200 if quick else 1000)
        test[n]["accuracy_ci95"] = bootstrap_ci(yte, test_p[n], lambda a, b: accuracy_score(a, b >= .5), 200 if quick else 1000)

    # repeated CV over all 517 records: a far steadier estimate than one 78-row split
    oof, cv = oof_predictions(params, X, y, n_repeats=2 if quick else 10, seed=seed)
    gating = gating_analysis(oof, y)

    # deployment models: refit on train+validation with the chosen hyper-parameters
    Xall = pd.concat([Xtr, Xva]); yall = pd.concat([ytr, yva])
    final = {n: build(n, params[n]).fit(Xall, yall) for n in MODEL_NAMES}
    version = "v1-" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
    bundle = make_bundle(final, version, D.training_ranges(X), meta={"data_hash": D.data_hash(path), "params": params, "seed": seed})
    joblib.dump(bundle, out_dir / "model_bundle.joblib")

    rf_imp = dict(sorted(zip(X.columns, map(float, final["random_forest"].feature_importances_)), key=lambda kv: -kv[1]))
    metrics = {
        "version": version, "trained_at": bundle["trained_at"], "dataset": {"records": int(len(X)), "positives": int(y.sum()),
            "positive_rate": float(y.mean()), "duplicates": int(raw.duplicated().sum()), "hash": D.data_hash(path)},
        "split": {"train": len(Xtr), "validation": len(Xva), "test": len(Xte)},
        "chosen_params": params, "validation_auc": val_auc,
        "test": test, "test_majority_baseline_accuracy": baseline,
        "cross_validation": {"scheme": "5-fold x %d repeats, stratified" % (2 if quick else 10), "results": cv,
                             "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean()))},
        "confidence_gating": gating, "rf_feature_importance": rf_imp,
        "thresholds": {"cutoffs": [0.35, 0.55, 0.70], "confidence_gate": CONFIDENCE_GATE},
        "disclosure_projection_note": ("Section 8.5 of the disclosure lists PROJECTED figures (90-95% accuracy, precision ~0.90, "
                                       "recall ~0.92). They are not measured results and are not supported by this prototype."),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if make_figs:
        from .figures import make_all
        make_all(raw, X, y, Xte, yte, test_p, final["random_forest"], cv, baseline, out_dir / "figures")
    return metrics
