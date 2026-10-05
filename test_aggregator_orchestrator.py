import random

import pytest

from wfire.aggregator import aggregate, risk_level
from wfire.orchestrator import DEFAULT_FLEET, PLAN, Orchestrator, ResourceManager


# ---------------------------------------------------------------- aggregator
@pytest.mark.parametrize("p,lvl", [(0.0, "LOW"), (0.3499, "LOW"), (0.35, "MODERATE"), (0.5499, "MODERATE"),
                                   (0.55, "HIGH"), (0.6999, "HIGH"), (0.70, "CRITICAL"), (1.0, "CRITICAL")])
def test_level_boundaries(p, lvl):
    assert risk_level(p) == lvl


def test_confidence_is_one_minus_spread_and_gate_needs_agreement():
    a = aggregate({"a": 0.70, "b": 0.72, "c": 0.68})
    assert a.confidence == pytest.approx(0.96) and a.unanimous and a.confident and a.level == "CRITICAL"
    b = aggregate({"a": 0.90, "b": 0.30, "c": 0.60})
    assert b.confidence == pytest.approx(0.40) and not b.unanimous and not b.confident
    c = aggregate({"a": 0.52, "b": 0.48, "c": 0.51})              # tiny spread but models straddle 0.5
    assert c.confidence > 0.9 and not c.unanimous and not c.confident


def test_weights_are_normalised_and_missing_weights_default_to_equal():
    a = aggregate({"a": 0.2, "b": 0.8}, {"a": 3, "b": 1})
    assert a.probability == pytest.approx(0.35) and sum(a.weights.values()) == pytest.approx(1)
    assert aggregate({"a": 0.2, "b": 0.8}).probability == pytest.approx(0.5)
    assert aggregate({"a": 0.2, "b": 0.8}, {"a": 0, "b": 0}).probability == pytest.approx(0.5)


def test_bad_inputs_are_rejected():
    for bad in ({}, {"a": 1.2}, {"a": -0.1}):
        with pytest.raises(ValueError):
            aggregate(bad)


# ---------------------------------------------------------------- orchestrator
def item(zone, level, p=0.75, conf=0.9, confident=True):
    return {"zone_id": zone, "level": level, "probability": p, "confidence": conf, "confident": confident}


def orch(**fleet):
    return Orchestrator(ResourceManager({**DEFAULT_FLEET, **fleet}))


def test_modes_by_risk_level_and_confidence():
    ds = {d.zone_id: d for d in orch().plan([item("a", "LOW"), item("b", "MODERATE"), item("c", "HIGH"),
                                             item("d", "HIGH", confident=False), item("e", "CRITICAL")])}
    assert ds["a"].mode == "MONITOR" and ds["a"].actions == []
    assert ds["b"].mode == "ADVISORY" and [a["type"] for a in ds["b"].actions] == ["EARLY_WARNING"] and ds["b"].allocated == {}
    assert ds["c"].mode == "AUTONOMOUS" and ds["c"].allocated == PLAN["HIGH"]["full"]
    assert ds["d"].mode == "HUMAN_REVIEW" and ds["d"].allocated == {} and [a["type"] for a in ds["d"].actions] == ["EARLY_WARNING"]
    assert ds["e"].mode == "AUTONOMOUS" and ds["e"].allocated == PLAN["CRITICAL"]["full"]


def test_actions_reflect_only_what_was_actually_allocated():
    d = orch(drones=0).plan([item("z", "HIGH")])[0]
    types = [a["type"] for a in d.actions]
    assert "DRONE_RECON" not in types and "FIRE_UNIT_PREPOSITION" in types
    assert d.shortfall == {"drones": 1} and "Shortfall" in d.reason


def test_evacuation_advisory_needs_human_approval_and_dry_runs_do_not_execute():
    d = orch().plan([item("z", "CRITICAL")])[0]
    status = {a["type"]: a["status"] for a in d.actions}
    assert status["EVACUATION_ADVISORY"] == "PENDING_APPROVAL" and status["DRONE_RECON"] == "SIMULATED_EXECUTED"
    dry = orch().plan([item("z", "CRITICAL")], execute=False)[0]
    assert {a["status"] for a in dry.actions} == {"PLANNED"}


def test_scarce_resources_go_to_the_highest_priority_zone_first():
    ds = {d.zone_id: d for d in orch(drones=1).plan([item("h", "HIGH", 0.66), item("c", "CRITICAL", 0.72)])}
    assert ds["c"].allocated.get("drones") == 1 and "drones" not in ds["h"].allocated
    assert ds["h"].shortfall == {"drones": 1}


def test_minimum_packages_are_served_before_any_top_up():
    # 2 trucks: CRITICAL wants 2 (min 1), HIGH wants 1 (min 0). Pass 1 gives CRITICAL its minimum truck; pass 2 then tops up
    # CRITICAL (2nd truck) before HIGH gets its truck, because CRITICAL outranks HIGH.
    ds = {d.zone_id: d for d in orch(fire_trucks=2).plan([item("h", "HIGH", 0.66), item("c", "CRITICAL", 0.72)])}
    assert ds["c"].allocated["fire_trucks"] == 2 and "fire_trucks" not in ds["h"].allocated
    # with 1 truck the CRITICAL zone still gets it (its minimum) ahead of HIGH
    ds = {d.zone_id: d for d in orch(fire_trucks=1).plan([item("h", "HIGH", 0.66), item("c", "CRITICAL", 0.72)])}
    assert ds["c"].allocated["fire_trucks"] == 1 and "fire_trucks" not in ds["h"].allocated


def test_ties_break_by_probability_then_confidence_deterministically():
    a = orch(drones=1).plan([item("a", "HIGH", 0.60), item("b", "HIGH", 0.68)])
    assert {d.zone_id: d.allocated.get("drones", 0) for d in a} == {"a": 0, "b": 1}
    b = orch(drones=1).plan([item("a", "HIGH", 0.6, 0.7), item("b", "HIGH", 0.6, 0.9)])
    assert {d.zone_id: d.allocated.get("drones", 0) for d in b} == {"a": 0, "b": 1}
    x = orch(drones=1).plan([item("y", "HIGH"), item("x", "HIGH")])
    assert {d.zone_id: d.allocated.get("drones", 0) for d in x} == {"x": 1, "y": 0}


def test_allocation_never_exceeds_the_fleet_and_never_goes_negative():
    rnd = random.Random(7)
    for _ in range(300):
        fleet = {k: rnd.randint(0, 6) for k in DEFAULT_FLEET}
        o = Orchestrator(ResourceManager(fleet))
        for k in fleet:
            o.resources.set_out_of_service(k, rnd.randint(0, fleet[k]))
        items = [item(f"z{i}", rnd.choice(["LOW", "MODERATE", "HIGH", "CRITICAL"]), rnd.random(), rnd.random(), rnd.random() < .7) for i in range(rnd.randint(0, 25))]
        used = {}
        for d in o.plan(items):
            for k, v in d.allocated.items():
                assert v > 0
                used[k] = used.get(k, 0) + v
            assert all(v > 0 for v in d.shortfall.values())
            if d.mode != "AUTONOMOUS":
                assert d.allocated == {}
        for k, v in used.items():
            assert v <= o.resources.available()[k]


def test_out_of_service_equipment_is_not_allocated_and_input_is_validated():
    o = orch(drones=2)
    o.resources.set_out_of_service("drones", 2)
    assert o.plan([item("z", "HIGH")])[0].allocated.get("drones") is None
    for kind, n in (("drones", 3), ("drones", -1), ("hovercraft", 0)):
        with pytest.raises((ValueError, KeyError)):
            o.resources.set_out_of_service(kind, n)
