"""Wires the services together: ingest -> assess -> orchestrate -> log. One cycle per call."""
import threading
from collections import Counter

from .ingestion import Ingestor, ValidationError
from .models import RiskEngine
from .orchestrator import Orchestrator, ResourceManager
from .store import Store


class Pipeline:
    def __init__(self, engine: RiskEngine, store: Store, resources: ResourceManager | None = None):
        self.engine, self.store = engine, store
        self.resources = resources or ResourceManager()
        self.orchestrator = Orchestrator(self.resources)
        self.ingestor = Ingestor(engine.train_ranges)
        self.lock = threading.RLock()

    def resources_view(self) -> dict:
        """Fleet status plus what the latest cycle committed (planning is per cycle: each cycle re-plans against the fleet)."""
        committed = self.store.last_cycle_allocations()
        snap = self.resources.snapshot()
        for k, v in snap.items():
            v["committed"] = committed.get(k, 0)
            v["free"] = v["available"] - v["committed"]
        return snap

    def set_engine(self, engine: RiskEngine):
        with self.lock:
            self.engine = engine
            self.ingestor.train_ranges = engine.train_ranges

    @staticmethod
    def _item(zone_id, a):
        return {"zone_id": zone_id, "level": a.level, "probability": a.probability, "confidence": a.confidence, "confident": a.confident}

    def run(self, payloads: list[dict], *, source: str, scenario: str | None = None, require_timestamp=True,
            allow_stale=False, execute=True) -> dict:
        with self.lock:
            readings, rejected = [], []
            for i, p in enumerate(payloads):
                try:
                    readings.append(self.ingestor.process(p, require_timestamp=require_timestamp, allow_stale=allow_stale))
                except ValidationError as e:
                    rejected.append({"index": i, "errors": e.errors})
            assessments = self.engine.assess_many([r.features for r in readings])
            summary = {"received": len(payloads), "accepted": len(readings), "rejected": len(rejected),
                       "levels": dict(Counter(a.level for a in assessments))}
            cycle_id = self.store.log_cycle(scenario, source, self.resources.snapshot(), summary)

            pred_ids, latest = {}, {}
            for r, a in zip(readings, assessments):
                pid = self.store.log_prediction(r, a, source, self.engine.version, cycle_id)
                pred_ids[r.zone_id] = pid                   # if a zone repeats in a batch, the last reading wins
                latest[r.zone_id] = (r, a)
            decisions = self.orchestrator.plan([self._item(z, a) for z, (_, a) in latest.items()], execute=execute)
            for d in decisions:
                self.store.log_decision(cycle_id, pred_ids[d.zone_id], d)
            summary["modes"] = dict(Counter(d.mode for d in decisions))
            return {"cycle_id": cycle_id, "summary": summary, "rejected": rejected,
                    "results": [{"prediction_id": pred_ids[z], "zone_id": z, "timestamp": r.timestamp.isoformat(),
                                 "features": r.features, "filtered": r.filtered, "assessment": a.to_dict(),
                                 "warnings": a.warnings + r.warnings} for z, (r, a) in latest.items()],
                    "decisions": [d.to_dict() for d in decisions], "resources": self.resources_view()}

    def what_if(self, payload: dict) -> dict:
        """Score one hypothetical condition set. Nothing is logged, and no history filter is applied."""
        feats, ts, warnings = self.ingestor.validate(payload, require_timestamp=False, allow_stale=True)
        a = self.engine.assess(feats, warnings=warnings)
        d = self.orchestrator.plan([self._item("what-if", a)], execute=False)[0]
        return {"features": feats, "assessment": a.to_dict(), "decision": d.to_dict(), "resources": self.resources_view()}
