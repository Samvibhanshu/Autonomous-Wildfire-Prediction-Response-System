"""Resource Availability Manager (step 6) and Autonomous Response Orchestration Agent, AROA (step 5).

Everything here is a SIMULATION: actions are recorded by a SimulatedGateway. A real deployment would implement
ActuatorGateway against drone, sprinkler and alerting systems.

Decision rules
  LOW                              monitor only
  MODERATE                         advisory: early warning to authorities
  HIGH / CRITICAL, models agree    autonomous: compete for resources by priority
  HIGH / CRITICAL, models disagree human review: early warning only, no resources committed
Allocation runs in two passes so scarcity is handled fairly: pass 1 gives every actionable zone its MINIMUM package in
priority order (a CRITICAL zone gets its drone and truck before any HIGH zone gets a top-up); pass 2 tops zones up to
their FULL package in priority order. Priority = risk level, then probability, then confidence.
"""
from dataclasses import asdict, dataclass, field

from .config import LEVELS

RESOURCE_TYPES = ["drones", "fire_trucks", "water_tankers", "personnel_teams", "sprinkler_systems"]
DEFAULT_FLEET = {"drones": 5, "fire_trucks": 6, "water_tankers": 4, "personnel_teams": 8, "sprinkler_systems": 6}

PLAN = {
    "HIGH": {"minimum": {"drones": 1},
             "full": {"drones": 1, "fire_trucks": 1, "sprinkler_systems": 1}},
    "CRITICAL": {"minimum": {"drones": 1, "fire_trucks": 1},
                 "full": {"drones": 1, "fire_trucks": 2, "water_tankers": 1, "personnel_teams": 2, "sprinkler_systems": 1}},
}
# which resource makes which action possible
ACTION_FOR = {"drones": "DRONE_RECON", "fire_trucks": "FIRE_UNIT_PREPOSITION", "water_tankers": "TANKER_STAGING",
              "personnel_teams": "PERSONNEL_STANDBY", "sprinkler_systems": "SPRINKLER_ACTIVATION"}
NEEDS_APPROVAL = {"EVACUATION_ADVISORY"}                # drafted automatically, sent only after an authority signs off


class ResourceManager:
    def __init__(self, fleet: dict | None = None):
        self.total = dict(fleet or DEFAULT_FLEET)
        self.out_of_service = {k: 0 for k in self.total}

    def available(self) -> dict:
        return {k: self.total[k] - self.out_of_service[k] for k in self.total}

    def set_out_of_service(self, kind: str, n: int):
        if kind not in self.total:
            raise KeyError(f"unknown resource type {kind}")
        if not 0 <= n <= self.total[kind]:
            raise ValueError(f"out_of_service must be between 0 and {self.total[kind]}")
        self.out_of_service[kind] = n

    def snapshot(self) -> dict:
        av = self.available()
        return {k: {"total": self.total[k], "out_of_service": self.out_of_service[k], "available": av[k]} for k in self.total}


@dataclass
class Decision:
    zone_id: str
    level: str
    probability: float
    confidence: float
    mode: str                                   # MONITOR | ADVISORY | AUTONOMOUS | HUMAN_REVIEW
    allocated: dict = field(default_factory=dict)
    shortfall: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)  # [{type, status}]
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def _rank(level):
    return LEVELS.index(level)


class SimulatedGateway:
    """Stands in for real actuators. Marks each action as executed in simulation, or pending approval."""
    def execute(self, decision: Decision) -> Decision:
        for a in decision.actions:
            if a["status"] == "PLANNED":
                a["status"] = "PENDING_APPROVAL" if a["type"] in NEEDS_APPROVAL else "SIMULATED_EXECUTED"
        return decision


class Orchestrator:
    def __init__(self, resources: ResourceManager, gateway=None):
        self.resources, self.gateway = resources, gateway or SimulatedGateway()

    def plan(self, items: list[dict], execute=True) -> list[Decision]:
        """items: [{zone_id, level, probability, confidence, confident}]. Pure w.r.t. resources: it plans against the
        currently available fleet each cycle and does not consume it permanently."""
        left = self.resources.available()
        decisions, contenders = [], []
        for it in items:
            lvl = it["level"]
            d = Decision(it["zone_id"], lvl, it["probability"], it["confidence"], "MONITOR")
            if lvl == "LOW":
                d.reason = "Low risk: continue monitoring."
            elif lvl == "MODERATE":
                d.mode, d.reason = "ADVISORY", "Elevated risk: early warning to authorities."
                d.actions.append({"type": "EARLY_WARNING", "status": "PLANNED"})
            elif not it["confident"]:
                d.mode = "HUMAN_REVIEW"
                d.reason = "High risk but the models disagree (low confidence): no autonomous deployment, escalated to a human."
                d.actions.append({"type": "EARLY_WARNING", "status": "PLANNED"})
            else:
                d.mode, d.reason = "AUTONOMOUS", "High risk and the models agree: deploying resources by priority."
                contenders.append(d)
            decisions.append(d)

        contenders.sort(key=lambda d: (-_rank(d.level), -d.probability, -d.confidence, d.zone_id))
        for d in contenders:                                  # pass 1: minimum package
            for res, n in PLAN[d.level]["minimum"].items():
                got = min(n, left[res])
                left[res] -= got
                if got:
                    d.allocated[res] = d.allocated.get(res, 0) + got
        for d in contenders:                                  # pass 2: top up to the full package
            for res, n in PLAN[d.level]["full"].items():
                need = n - d.allocated.get(res, 0)
                got = min(need, left[res])
                left[res] -= got
                if got:
                    d.allocated[res] = d.allocated.get(res, 0) + got
            d.shortfall = {r: n - d.allocated.get(r, 0) for r, n in PLAN[d.level]["full"].items() if n - d.allocated.get(r, 0) > 0}
            for res in d.allocated:
                d.actions.append({"type": ACTION_FOR[res], "status": "PLANNED"})
            d.actions.append({"type": "EARLY_WARNING", "status": "PLANNED"})
            if d.level == "CRITICAL":
                d.actions.append({"type": "EVACUATION_ADVISORY", "status": "PLANNED"})
            if d.shortfall:
                d.reason += " Shortfall: " + ", ".join(f"{n} {r}" for r, n in d.shortfall.items()) + "."
        if execute:
            for d in decisions:
                self.gateway.execute(d)
        return sorted(decisions, key=lambda d: (-_rank(d.level), -d.probability, d.zone_id))
