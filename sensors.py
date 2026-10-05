"""Simulated sensor layer (disclosure step 1) for demos and the section 8.7 conceptual scenarios.

Baseline conditions are bootstrapped from real historical records for the same calendar month; a scenario then perturbs
them (heatwave, dry spell, wind event...). Everything produced here is labelled source='simulated'.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import DAYS, GRID_X, GRID_Y, MONTHS

SCENARIOS = {
    "normal": {"label": "Normal conditions", "desc": "Unmodified historical conditions for the current month."},
    "heatwave": {"label": "Heatwave", "desc": "Sudden temperature rise, humidity drop, drier fuels."},
    "dry_spell": {"label": "Dry spell", "desc": "Low humidity and drought indices, no rain."},
    "wind_event": {"label": "Strong wind event", "desc": "Rapid wind speed change; spread index rises."},
    "cool_wet": {"label": "Cool and wet", "desc": "Low temperature, high humidity, some rain."},
}


def _perturb(f: dict, scenario: str) -> dict:
    f = dict(f)
    if scenario == "heatwave":
        f["temp"] = min(f["temp"] + 8, 40); f["RH"] = max(f["RH"] * 0.55, 10); f["wind"] += 2
        f["FFMC"] = min(f["FFMC"] + 6, 96.5); f["DMC"] *= 1.3; f["DC"] *= 1.15; f["rain"] = 0
    elif scenario == "dry_spell":
        f["RH"] = max(f["RH"] * 0.6, 10); f["rain"] = 0; f["DMC"] *= 1.4; f["DC"] *= 1.2; f["FFMC"] = min(f["FFMC"] + 4, 96.5)
    elif scenario == "wind_event":
        f["wind"] = min(f["wind"] + 4.5, 9.4); f["ISI"] = min(f["ISI"] * 1.8 + 2, 56)
    elif scenario == "cool_wet":
        f["temp"] = max(f["temp"] - 6, 2.5); f["RH"] = min(f["RH"] + 25, 100); f["rain"] = max(f["rain"], 2.0)
        f["FFMC"] = max(f["FFMC"] - 12, 20); f["ISI"] = max(f["ISI"] * 0.5, 0.1)
    return f


def generate(raw: pd.DataFrame, scenario="normal", n_zones=12, seed=None, now: datetime | None = None) -> list[dict]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}; choose from {', '.join(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    now = now or datetime.now(timezone.utc)
    cells = [(x, y) for x in GRID_X for y in GRID_Y]
    picks = rng.choice(len(cells), size=min(n_zones, len(cells)), replace=False)
    month_name = MONTHS[now.month - 1]
    pool = raw[raw.month == month_name]
    pool = pool if len(pool) >= 5 else raw
    out = []
    for i in picks:
        x, y = cells[i]
        row = pool.iloc[int(rng.integers(0, len(pool)))]
        base = {k: float(row[k]) for k in ("FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain")}
        f = _perturb(base, scenario)
        out.append({"timestamp": now.isoformat().replace("+00:00", "Z"), "X": x, "Y": y,
                    **{k: round(float(v), 2) for k, v in f.items()}})
    return out
