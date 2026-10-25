"""Adaptive feedback and retraining (claim 8) with a champion/challenger guard.

A challenger trained on the original data plus logged real outcomes only replaces the current model if it does not do
worse on a hold-out set it has never seen. Otherwise the current model stays and the report says why.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import data as D
from .config import ARTIFACTS, FEATURES, SEED
from .models import MODEL_NAMES, RiskEngine, build, make_bundle


def _ens_auc(models, X, y):
    p = np.mean([m.predict_proba(X[FEATURES])[:, 1] for m in models.values()], axis=0)
    return float(roc_auc_score(y, p))


def retrain(store, engine: RiskEngine, artifacts: Path = ARTIFACTS, min_new: int = 30, path=D.DATA_PATH, seed=SEED) -> dict:
    new = store.labelled_examples()
    if len(new) < min_new:
        return {"status": "skipped", "reason": f"need at least {min_new} logged outcomes, have {len(new)}", "new_outcomes": len(new)}
    nd = pd.DataFrame(new)
    ny = nd["fire"].astype(int)
    if ny.nunique() < 2:
        return {"status": "skipped", "reason": "logged outcomes contain only one class", "new_outcomes": len(new)}
    X, y = D.encode(D.load_raw(path))
    Xtr, Xva, Xte, ytr, yva, yte = D.split_70_15_15(X, y, seed)
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(nd[FEATURES].astype(float), ny, test_size=0.3, stratify=ny, random_state=seed)

    hold_X, hold_y = pd.concat([Xte, Xn_te]), pd.concat([yte, yn_te])
    fit_X, fit_y = pd.concat([Xtr, Xva, Xn_tr]), pd.concat([ytr, yva, yn_tr])
    params = engine.bundle["meta"].get("params", {})
    challenger = {n: build(n, params.get(n)).fit(fit_X, fit_y) for n in MODEL_NAMES}
    champ_auc, chal_auc = _ens_auc(engine.models, hold_X, hold_y), _ens_auc(challenger, hold_X, hold_y)
    report = {"new_outcomes": len(new), "holdout_size": int(len(hold_y)), "champion_auc": round(champ_auc, 4),
              "challenger_auc": round(chal_auc, 4), "champion_version": engine.version}
    if chal_auc < champ_auc:
        return {**report, "status": "rejected", "reason": "challenger did not beat the current model on the hold-out set"}
    version = f"v{int(engine.version.split('-')[0][1:] or 1) + 1}-" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
    bundle = make_bundle(challenger, version, D.training_ranges(pd.concat([X, nd[FEATURES]])),
                         meta={**engine.bundle["meta"], "feedback_outcomes": len(new), "previous": engine.version})
    joblib.dump(bundle, artifacts / f"model_bundle_{version}.joblib")
    joblib.dump(bundle, artifacts / "model_bundle.joblib")
    return {**report, "status": "promoted", "new_version": version}
