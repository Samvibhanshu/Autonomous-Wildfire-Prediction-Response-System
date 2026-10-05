import json
import time

import pandas as pd
import pytest

from conftest import now_z, payload
from wfire import data as D
from wfire.sensors import SCENARIOS

W = {"X": 5, "Y": 5, "month": "aug", "day": "sat", "temp": 30, "RH": 22, "wind": 4.5, "rain": 0, "FFMC": 93, "DMC": 170, "DC": 700, "ISI": 12}


def test_pages_health_and_static_files(client):
    for path in ("/", "/predict", "/model"):
        r = client.get(path)
        assert r.status_code == 200 and b"Prototype, simulation mode" in r.data
    assert client.get("/health").get_json()["ok"] is True
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/nope").get_json()["error"]["code"] == "NOT_FOUND"
    assert client.get("/figures/fig08_roc.png").status_code == 200
    assert client.get("/figures/../metrics.json").status_code == 404
    assert client.get("/figures/secret.png").status_code == 404


def test_model_page_reports_measured_numbers_not_the_projections(client, app):
    html = client.get("/model").get_data(as_text=True)
    assert "not measured here" in html and "cross-validated" in html
    m = client.get("/api/model/metrics").get_json()
    assert f"{m['cross_validation']['results']['ensemble']['accuracy']['mean'] * 100:.1f}%" in html


def test_what_if_prediction(client):
    j = client.post("/api/predict", json=W).get_json()
    a = j["assessment"]
    assert 0 <= a["probability"] <= 1 and a["level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL") and set(a["per_model"]) == {"random_forest", "gradient_boosting", "logistic_regression"}
    assert j["decision"]["zone_id"] == "what-if" and client.get("/api/logs/predictions").get_json() == []      # not logged


def test_what_if_rejects_bad_input_with_field_details(client):
    r = client.post("/api/predict", json={**W, "temp": 95, "RH": None})
    assert r.status_code == 400 and {d["field"] for d in r.get_json()["error"]["details"]} == {"temp", "RH"}
    assert client.post("/api/predict", data="{bad", content_type="application/json").status_code == 400
    assert client.post("/api/predict").status_code == 400
    assert client.post("/api/predict", json=[1, 2]).status_code == 400


def test_ingest_single_and_batch_with_partial_rejection(client):
    ok = client.post("/api/ingest", json=payload()).get_json()
    assert ok["summary"]["accepted"] == 1 and ok["results"][0]["zone_id"] == "X5-Y5" and ok["decisions"]
    batch = client.post("/api/ingest", json={"readings": [payload(X=1, Y=2), payload(temp=999), payload(X=2, Y=3, timestamp="2020-01-01T00:00:00Z")]}).get_json()
    assert batch["summary"]["accepted"] == 1 and batch["summary"]["rejected"] == 2
    assert {r["index"] for r in batch["rejected"]} == {1, 2}
    assert client.post("/api/ingest", json={"readings": []}).status_code == 400
    assert client.post("/api/ingest", json={"readings": [payload()] * 501}).status_code == 400


def test_a_repeated_zone_in_one_batch_uses_the_last_reading(client):
    r = client.post("/api/ingest", json={"readings": [payload(temp=10, RH=90), payload(temp=32, RH=20)]}).get_json()
    assert len(r["results"]) == 1 and r["results"][0]["features"]["temp"] == 32 and r["summary"]["accepted"] == 2


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_scenario_runs_and_logs_everything(client, scenario):
    r = client.post("/api/simulate", json={"scenario": scenario, "zones": 10, "seed": 1}).get_json()
    assert r["summary"]["accepted"] == 10 and r["summary"]["rejected"] == 0 and len(r["decisions"]) == 10
    counts = client.get("/api/zones").get_json()["counts"]
    assert counts["predictions"] == 10 and counts["decisions"] == 10 and counts["cycles"] == 1
    for d in r["decisions"]:
        assert d["mode"] in ("MONITOR", "ADVISORY", "AUTONOMOUS", "HUMAN_REVIEW")
    for k, v in r["resources"].items():
        assert 0 <= v["free"] <= v["total"] and v["committed"] <= v["available"]


def test_simulation_validation_and_seed_determinism(client):
    assert client.post("/api/simulate", json={"scenario": "meteor"}).status_code == 400
    for z in (0, 73, "9", True):
        assert client.post("/api/simulate", json={"zones": z}).status_code == 400
    a = client.post("/api/simulate", json={"scenario": "heatwave", "zones": 8, "seed": 5}).get_json()
    b = client.post("/api/simulate", json={"scenario": "heatwave", "zones": 8, "seed": 5}).get_json()
    assert [(x["zone_id"], x["assessment"]["probability"]) for x in a["results"]] == [(x["zone_id"], x["assessment"]["probability"]) for x in b["results"]]


def test_heatwaves_score_higher_than_cool_wet_weather(client):
    """Sanity check that the model responds to conditions in the right direction (not a claim about accuracy)."""
    def mean_p(sc):
        ps = []
        for seed in range(4):
            r = client.post("/api/simulate", json={"scenario": sc, "zones": 30, "seed": seed}).get_json()
            ps += [x["assessment"]["probability"] for x in r["results"]]
        return sum(ps) / len(ps)
    assert mean_p("heatwave") > mean_p("cool_wet")


def test_zone_map_returns_latest_reading_per_zone_with_its_decision(client):
    client.post("/api/ingest", json=payload(temp=15, RH=80))
    client.post("/api/ingest", json=payload(temp=30, RH=20))
    zs = client.get("/api/zones").get_json()["zones"]
    assert len(zs) == 1 and zs[0]["features"]["temp"] == 30 and zs[0]["mode"] and isinstance(zs[0]["actions"], list)


def test_resource_availability_is_managed_and_shown(client):
    assert client.post("/api/resources", json={"type": "drones", "out_of_service": 5}).get_json()["drones"]["free"] == 0
    r = client.post("/api/simulate", json={"scenario": "heatwave", "zones": 20, "seed": 3}).get_json()
    assert all("drones" not in d["allocated"] for d in r["decisions"])
    assert client.post("/api/resources", json={"type": "drones", "out_of_service": 99}).status_code == 400
    assert client.post("/api/resources", json={"type": "tank", "out_of_service": 1}).status_code == 400


def test_logs_endpoints(client):
    client.post("/api/simulate", json={"zones": 5, "seed": 1})
    assert len(client.get("/api/logs/predictions?limit=3").get_json()) == 3
    assert len(client.get("/api/logs/predictions?limit=99999").get_json()) == 5
    assert client.get("/api/logs/decisions").get_json()[0]["mode"]
    assert client.get("/api/logs/cycles").get_json()[0]["summary"]["accepted"] == 5
    assert client.get("/api/logs/secrets").status_code == 404
    assert client.get("/api/logs/predictions?limit=abc").status_code == 400


def test_api_key_protects_writes_but_not_reads(artifacts, tmp_path):
    import shutil
    from wfire.webapp import create_app
    shutil.copytree(artifacts, tmp_path / "a")
    c = create_app(artifacts_dir=tmp_path / "a", api_key="s3cret").test_client()
    assert c.get("/api/resources").status_code == 200 and c.get("/health").status_code == 200
    assert c.post("/api/simulate", json={}).status_code == 401
    assert c.post("/api/simulate", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/api/simulate", json={"zones": 2}, headers={"X-API-Key": "s3cret"}).status_code == 200


# ---------------------------------------------------------------- feedback and retraining
def _ingest_history(client, n, fire_from_data=True):
    raw = D.load_raw().sample(n, random_state=1)
    ids = []
    for _, row in raw.iterrows():
        p = {"timestamp": now_z(), "X": int(row.X), "Y": int(row.Y), "month": row.month, "day": row.day,
             **{k: float(row[k]) for k in ("FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain")}}
        r = client.post("/api/ingest", json=p).get_json()
        ids.append((r["results"][0]["prediction_id"], int(row.area > 0), float(row.area)))
    return ids


def test_outcome_validation(client):
    pid = client.post("/api/ingest", json=payload()).get_json()["results"][0]["prediction_id"]
    assert client.post("/api/outcomes", json={"prediction_id": 9999, "observed_fire": 1}).status_code == 400
    assert client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": 2}).status_code == 400
    assert client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": 1, "area_ha": -4}).status_code == 400
    assert client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": 1, "area_ha": 3.2}).get_json()["labelled_examples"] == 1
    assert client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": 0}).get_json()["labelled_examples"] == 1     # correction, not a duplicate


def test_simulation_alone_can_never_retrain_the_model(client):
    for seed in range(5):
        client.post("/api/simulate", json={"zones": 20, "seed": seed})
    r = client.post("/api/retrain", json={"min_new": 1}).get_json()
    assert r["status"] == "skipped" and r["new_outcomes"] == 0


def test_retraining_is_guarded_by_volume_and_class_balance(client):
    ids = _ingest_history(client, 40)
    for pid, fire, area in ids[:10]:
        client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": fire})
    assert client.post("/api/retrain", json={}).get_json()["status"] == "skipped"                   # default needs 30
    one_class = [i for i in ids if i[1] == 1]
    app_store = client.application.store
    app_store.db.execute("DELETE FROM outcomes"); app_store.db.commit()
    for pid, fire, _ in one_class[:12]:
        client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": 1})
    r = client.post("/api/retrain", json={"min_new": 5}).get_json()
    assert r["status"] == "skipped" and "one class" in r["reason"]


def test_champion_challenger_retraining_promotes_or_rejects_with_evidence(client):
    ids = _ingest_history(client, 90)
    for pid, fire, area in ids:
        client.post("/api/outcomes", json={"prediction_id": pid, "observed_fire": fire, "area_ha": area})
    before = client.get("/health").get_json()["model_version"]
    r = client.post("/api/retrain", json={"min_new": 30}).get_json()
    assert r["status"] in ("promoted", "rejected") and r["new_outcomes"] == 90 and r["holdout_size"] > 78
    assert 0 <= r["champion_auc"] <= 1 and 0 <= r["challenger_auc"] <= 1
    after = client.get("/health").get_json()["model_version"]
    if r["status"] == "promoted":
        assert after == r["new_version"] != before and r["challenger_auc"] >= r["champion_auc"]
        assert client.post("/api/predict", json=W).status_code == 200                                # new model serves requests
    else:
        assert after == before and r["challenger_auc"] < r["champion_auc"]
