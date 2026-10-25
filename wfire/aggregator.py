"""Risk Confidence Aggregator (disclosure step 4, claim 2).

Combines the per-model probabilities into one score, a risk level, and a confidence value.
Confidence = 1 - spread, where spread is max(p) - min(p) across models. Models that disagree give low confidence,
and low-confidence alerts are not acted on autonomously (they go to a human).
"""
from dataclasses import asdict, dataclass, field

from .config import CONFIDENCE_GATE, LEVEL_CUTOFFS, LEVELS


def risk_level(p: float, cutoffs=LEVEL_CUTOFFS) -> str:
    for level, cut in zip(LEVELS, cutoffs):
        if p < cut:
            return level
    return LEVELS[-1]


@dataclass
class Assessment:
    probability: float
    level: str
    confidence: float
    unanimous: bool                  # every model is on the same side of 0.5
    confident: bool                  # confidence >= gate AND unanimous
    per_model: dict
    weights: dict
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def aggregate(per_model: dict, weights: dict | None = None, gate: float = CONFIDENCE_GATE,
              cutoffs=LEVEL_CUTOFFS, warnings=None) -> Assessment:
    if not per_model:
        raise ValueError("no model outputs to aggregate")
    if any(not 0.0 <= p <= 1.0 for p in per_model.values()):
        raise ValueError("model probabilities must be within [0, 1]")
    w = {k: (weights or {}).get(k, 1.0) for k in per_model}
    total = sum(w.values())
    if total <= 0:
        w, total = {k: 1.0 for k in per_model}, float(len(per_model))
    w = {k: v / total for k, v in w.items()}
    p = sum(per_model[k] * w[k] for k in per_model)
    spread = max(per_model.values()) - min(per_model.values())
    confidence = 1.0 - spread
    sides = {v >= 0.5 for v in per_model.values()}
    unanimous = len(sides) == 1
    return Assessment(probability=round(p, 4), level=risk_level(p, cutoffs), confidence=round(confidence, 4),
                      unanimous=unanimous, confident=(confidence >= gate and unanimous),
                      per_model={k: round(v, 4) for k, v in per_model.items()},
                      weights={k: round(v, 4) for k, v in w.items()}, warnings=list(warnings or []))
