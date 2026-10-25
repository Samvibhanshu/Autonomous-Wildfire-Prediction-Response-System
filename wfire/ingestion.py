"""Data Ingestion microservice logic (disclosure step 2): schema + range validation, timestamp validation and
synchronisation to UTC, noise filtering / anomaly removal, and conversion to a model-ready feature row."""
import math
import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .config import DAYS, FEATURES, MONTHS

# Physically possible values. Outside these the reading is rejected as a sensor fault.
HARD_LIMITS = {"FFMC": (0, 101), "DMC": (0, 500), "DC": (0, 1500), "ISI": (0, 100),
               "temp": (-30, 60), "RH": (0, 100), "wind": (0, 80), "rain": (0, 500)}
SENSOR_FIELDS = list(HARD_LIMITS)
SPIKE_FLOOR = {"temp": 1.0, "RH": 3.0, "wind": 0.8}      # minimum spread so a flat signal isn't hypersensitive
SPIKE_K = 4.0


class ValidationError(Exception):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


@dataclass
class Reading:
    zone_id: str
    timestamp: datetime
    features: dict
    warnings: list = field(default_factory=list)
    filtered: list = field(default_factory=list)


def zone_id(x: int, y: int) -> str:
    return f"X{x}-Y{y}"


def _num(payload, key, errors):
    v = payload.get(key)
    if v is None or isinstance(v, bool) or isinstance(v, str) and not v.strip():
        errors.append({"field": key, "message": "required"})
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        errors.append({"field": key, "message": "must be a number"})
        return None
    if not math.isfinite(f):
        errors.append({"field": key, "message": "must be finite"})
        return None
    return f


def parse_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("must be ISO-8601, e.g. 2026-08-14T13:05:00Z")
    if dt.tzinfo is None:
        raise ValueError("must include a timezone offset (or Z)")
    return dt.astimezone(timezone.utc)


def month_index(v):
    if isinstance(v, str) and v.strip().lower()[:3] in MONTHS:
        return MONTHS.index(v.strip().lower()[:3]) + 1
    return int(v)


def day_index(v):
    if isinstance(v, str) and v.strip().lower()[:3] in DAYS:
        return DAYS.index(v.strip().lower()[:3])
    return int(v)


class Ingestor:
    def __init__(self, train_ranges: dict | None = None, window: int = 5, max_age_s: int = 3600, max_skew_s: int = 300):
        self.train_ranges = train_ranges or {}
        self.window, self.max_age, self.max_skew = window, max_age_s, max_skew_s
        self._hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window)))
        self._lock = threading.Lock()

    def validate(self, payload: dict, *, require_timestamp=True, now: datetime | None = None,
                 allow_stale=False) -> tuple[dict, datetime, list]:
        """Returns (features, utc_timestamp, warnings) or raises ValidationError listing every problem."""
        if not isinstance(payload, dict):
            raise ValidationError([{"field": "body", "message": "must be a JSON object"}])
        errors, now = [], now or datetime.now(timezone.utc)
        ts = None
        if payload.get("timestamp") in (None, ""):
            if require_timestamp:
                errors.append({"field": "timestamp", "message": "required"})
            ts = now
        else:
            try:
                ts = parse_timestamp(payload["timestamp"])
                if (ts - now).total_seconds() > self.max_skew:
                    errors.append({"field": "timestamp", "message": "is in the future (clock skew)"})
                elif not allow_stale and (now - ts).total_seconds() > self.max_age:
                    errors.append({"field": "timestamp", "message": f"is stale (older than {self.max_age // 60} min)"})
            except ValueError as e:
                errors.append({"field": "timestamp", "message": str(e)})

        feats = {}
        for k in ("X", "Y"):
            v = _num(payload, k, errors)
            if v is not None:
                if v != int(v):
                    errors.append({"field": k, "message": "must be a whole grid coordinate"})
                lo, hi = (1, 9) if k == "X" else (2, 9)
                if not lo <= v <= hi:
                    errors.append({"field": k, "message": f"must be between {lo} and {hi}"})
                feats[k] = float(int(v))
        for k in SENSOR_FIELDS:
            v = _num(payload, k, errors)
            if v is not None:
                lo, hi = HARD_LIMITS[k]
                if not lo <= v <= hi:
                    errors.append({"field": k, "message": f"{v} is outside the possible range {lo}-{hi} (sensor fault?)"})
                feats[k] = v
        try:
            feats["month"] = float(month_index(payload["month"])) if payload.get("month") not in (None, "") else float(ts.month if ts else 1)
            feats["day"] = float(day_index(payload["day"])) if payload.get("day") not in (None, "") else float(ts.weekday() if ts else 0)
            if not 1 <= feats["month"] <= 12 or not 0 <= feats["day"] <= 6:
                errors.append({"field": "month/day", "message": "month must be 1-12 (or jan-dec) and day 0-6 (or mon-sun)"})
        except (ValueError, TypeError):
            errors.append({"field": "month/day", "message": "unrecognised month or day"})
        if errors:
            raise ValidationError(errors)

        warnings = []
        for k, (lo, hi) in self.train_ranges.items():
            if k in feats and k not in ("X", "Y") and not lo <= feats[k] <= hi:
                warnings.append(f"{k}={feats[k]:g} is outside the training range {lo:g}-{hi:g}: the model is extrapolating")
        return {k: feats[k] for k in FEATURES}, ts, warnings

    def process(self, payload: dict, **kw) -> Reading:
        """Validate, then remove spikes using a per-zone rolling median (Hampel filter)."""
        feats, ts, warnings = self.validate(payload, **kw)
        zid, filtered = zone_id(int(feats["X"]), int(feats["Y"])), []
        with self._lock:
            hist = self._hist[zid]
            for k, floor in SPIKE_FLOOR.items():
                h = hist[k]
                if len(h) >= 4:
                    med = statistics.median(h)
                    mad = statistics.median(abs(v - med) for v in h) * 1.4826
                    if abs(feats[k] - med) > SPIKE_K * max(mad, floor):
                        filtered.append({"field": k, "raw": feats[k], "replaced_with": round(med, 2)})
                        feats[k] = float(med)
                h.append(feats[k])                              # the filtered value, so a spike can't poison the window
        for f in filtered:
            warnings.append(f"{f['field']} spike {f['raw']:g} replaced by rolling median {f['replaced_with']:g}")
        return Reading(zid, ts, feats, warnings, filtered)
