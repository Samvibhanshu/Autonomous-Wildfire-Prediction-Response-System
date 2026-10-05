"""Dataset loading and the preprocessing described in disclosure section 8.1."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, DAYS, FEATURES, MONTHS, SEED


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def data_hash(path=DATA_PATH) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def encode(df: pd.DataFrame):
    """Fire = 1 if burned area > 0 (section 8.1). month/day become integers.

    Month/day are encoded in calendar order (jan=1, mon=0) rather than alphabetically. Both are 'label encoding',
    but calendar order keeps the numbers meaningful (dec is next to jan, not next to aug).
    """
    d = df.copy()
    d["month"] = d["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    d["day"] = d["day"].map({m: i for i, m in enumerate(DAYS)})
    if d[["month", "day"]].isna().any().any():
        raise ValueError("unrecognised month/day value in dataset")
    y = (d["area"] > 0).astype(int)
    return d[FEATURES].astype(float), y


def split_70_15_15(X, y, seed=SEED):
    """Stratified 70% train / 15% validation / 15% test (section 8.1)."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=seed)
    return Xtr, Xva, Xte, ytr, yva, yte


def training_ranges(X: pd.DataFrame) -> dict:
    return {c: [float(X[c].min()), float(X[c].max())] for c in X.columns}
