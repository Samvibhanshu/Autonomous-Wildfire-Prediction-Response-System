"""All tunable design parameters live here, not scattered through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "forestfires.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# Model input order. X/Y are the park-grid cell, month/day come from the timestamp, the rest are sensor/FWI values.
FEATURES = ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]

# Risk levels (disclosure: LOW / MODERATE / HIGH / CRITICAL). Probability cut-offs are design parameters,
# not tuned to flatter the metrics; the model page reports the fire rate actually observed at each level.
LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
LEVEL_CUTOFFS = (0.35, 0.55, 0.70)      # p < 0.35 LOW, < 0.55 MODERATE, < 0.70 HIGH, else CRITICAL

# Risk Confidence Aggregator: autonomous action needs the models to agree.
CONFIDENCE_GATE = 0.60                   # confidence = 1 - (max_p - min_p) across models

SEED = 42
GRID_X = range(1, 10)                    # Montesinho park map cells
GRID_Y = range(2, 10)
