"""Flask dashboard and JSON API (disclosure: 'UI/Dashboard, Flask querying microservices for live data and logs')."""
import hmac
import json
import os

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from . import data as D
from . import retrain as R
from . import sensors
from .config import ARTIFACTS, GRID_X, GRID_Y, LEVELS, ROOT
from .ingestion import ValidationError
from .models import RiskEngine
from .pipeline import Pipeline
from .store import Store

FIGURE_TITLES = {
    "fig08_roc.png": "Fig 8. ROC curves (test split)", "fig09_accuracy.png": "Fig 9. Model accuracy vs baseline",
    "fig10_feature_importance.png": "Fig 10. Feature importance", "fig11_temp_vs_area.png": "Fig 11. Temperature vs burned area",
    "fig12_rh_vs_area.png": "Fig 12. Humidity vs burned area", "fig13_risk_heatmap.png": "Fig 13. Fire events per grid cell",
    "fig14_temp_distribution.png": "Fig 14. Temperature distribution", "fig15_monthly_fires.png": "Fig 15. Monthly fire occurrence"}


def create_app(artifacts_dir=None, db_path=None, engine: RiskEngine | None = None, api_key: str | None = None):
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    art = artifacts_dir or ARTIFACTS
    engine = engine or RiskEngine.load(art / "model_bundle.joblib")
    store = Store(db_path or ":memory:")
    pipe = Pipeline(engine, store)
    raw = D.load_raw()
    key = api_key if api_key is not None else os.environ.get("WILDFIRE_API_KEY")
    app.pipeline, app.store = pipe, store

    def metrics():
        p = art / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    def err(code, message, status, **extra):
        return jsonify(error={"code": code, "message": message, **extra}), status

    @app.before_request
    def guard():
        if key and request.method in ("POST", "PUT", "DELETE") and request.path.startswith("/api/"):
            if not hmac.compare_digest(request.headers.get("X-API-Key", ""), key):
                return err("UNAUTHORIZED", "Missing or wrong X-API-Key", 401)

    @app.errorhandler(ValidationError)
    def _v(e):
        return err("VALIDATION_ERROR", "Input failed validation", 400, details=e.errors)

    for code, name in ((404, "NOT_FOUND"), (405, "METHOD_NOT_ALLOWED"), (413, "TOO_LARGE")):
        app.register_error_handler(code, lambda e, c=code, n=name: err(n, str(getattr(e, "description", n)), c))

    def body():
        b = request.get_json(silent=True)
        if b is None:
            raise ValidationError([{"field": "body", "message": "send a JSON body"}])
        return b

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", scenarios=sensors.SCENARIOS, levels=LEVELS, metrics=metrics(),
                               gx=list(GRID_X), gy=list(GRID_Y), version=pipe.engine.version)

    @app.get("/predict")
    def predict_page():
        return render_template("predict.html", metrics=metrics(), ranges=pipe.engine.train_ranges, version=pipe.engine.version)

    @app.get("/model")
    def model_page():
        return render_template("model.html", m=metrics(), figures=FIGURE_TITLES, version=pipe.engine.version)

    @app.get("/figures/<path:name>")
    def figure(name):
        if name not in FIGURE_TITLES:
            abort(404)
        return send_from_directory(art / "figures", name, max_age=0)

    @app.get("/health")
    def health():
        return {"ok": True, "model_version": pipe.engine.version}

    # ---------------------------------------------------------------- API
    @app.post("/api/predict")
    def predict():
        return jsonify(pipe.what_if(body()))

    @app.post("/api/ingest")
    def ingest():
        b = body()
        readings = b.get("readings") if isinstance(b, dict) and "readings" in b else [b]
        if not isinstance(readings, list) or not readings or len(readings) > 500:
            raise ValidationError([{"field": "readings", "message": "send 1-500 readings"}])
        return jsonify(pipe.run(readings, source="sensor"))

    @app.post("/api/simulate")
    def simulate():
        b = request.get_json(silent=True) or {}
        scenario, n = b.get("scenario", "normal"), b.get("zones", 12)
        if scenario not in sensors.SCENARIOS:
            raise ValidationError([{"field": "scenario", "message": f"must be one of {', '.join(sensors.SCENARIOS)}"}])
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 72:
            raise ValidationError([{"field": "zones", "message": "must be a whole number 1-72"}])
        payloads = sensors.generate(raw, scenario, n, seed=b.get("seed"))
        return jsonify(pipe.run(payloads, source="simulated", scenario=scenario))

    @app.get("/api/zones")
    def zones():
        return jsonify({"zones": store.latest_by_zone(), "counts": store.counts()})

    @app.get("/api/resources")
    def resources():
        return jsonify(pipe.resources_view())

    @app.post("/api/resources")
    def set_resources():
        b = body()
        try:
            pipe.resources.set_out_of_service(b.get("type"), b.get("out_of_service"))
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError([{"field": "resources", "message": str(e)}])
        return jsonify(pipe.resources_view())

    @app.get("/api/logs/<table>")
    def logs(table):
        try:
            limit = min(max(int(request.args.get("limit", 50)), 1), 500)
        except ValueError:
            raise ValidationError([{"field": "limit", "message": "must be a whole number"}])
        try:
            return jsonify(store.recent(table, limit))
        except KeyError:
            abort(404)

    @app.post("/api/outcomes")
    def outcomes():
        b = body()
        pid, fire = b.get("prediction_id"), b.get("observed_fire")
        if not isinstance(pid, int) or store.prediction(pid) is None:
            raise ValidationError([{"field": "prediction_id", "message": "unknown prediction"}])
        if fire not in (0, 1, True, False):
            raise ValidationError([{"field": "observed_fire", "message": "must be 0 or 1"}])
        area = b.get("area_ha")
        if area is not None and (not isinstance(area, (int, float)) or area < 0):
            raise ValidationError([{"field": "area_ha", "message": "must be a non-negative number"}])
        store.add_outcome(pid, int(fire), area, b.get("note"))
        return jsonify({"recorded": True, "labelled_examples": len(store.labelled_examples())})

    @app.post("/api/retrain")
    def retrain():
        result = R.retrain(store, pipe.engine, art, min_new=int((request.get_json(silent=True) or {}).get("min_new", 30)))
        if result["status"] == "promoted":
            pipe.set_engine(RiskEngine.load(art / "model_bundle.joblib"))
        return jsonify(result)

    @app.get("/api/model/metrics")
    def model_metrics():
        m = metrics()
        return jsonify(m) if m else err("NOT_FOUND", "metrics.json missing; run python train.py", 404)

    return app
