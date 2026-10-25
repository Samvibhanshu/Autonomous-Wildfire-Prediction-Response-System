# Autonomous Wildfire Risk Prediction and Resource-Orchestrated Response: prototype

[![tests](https://github.com/Samvibhanshu/Autonomous-Wildfire-Prediction-Response-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Samvibhanshu/Autonomous-Wildfire-Prediction-Response-System/actions/workflows/ci.yml)

ML pipeline plus Flask dashboard for the IDF-B disclosure *"Autonomous Wildfire Risk Prediction and Resource-Orchestrated
Response Pattern"* (TRL 4). Python, scikit-learn, Flask, SQLite. 71 tests.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                       # about 30 s; trains a small model on the real data
python train.py                                           # ~1 min: tuning, cross-validation, figures 8-15, artifacts/
python run.py                                             # dashboard at http://127.0.0.1:5000
```
`artifacts/` ships with a trained model and all figures, so `python run.py` works immediately.

## Read this first: what the numbers actually are

The disclosure's section 8.5 lists **projected** figures (90-95% accuracy, precision ~0.90, recall ~0.92). This code measures
the real thing on the disclosure's own experiment (UCI Forest Fires, fire = area > 0, RF / GB / LR, stratified 70/15/15):

| | Single 70/15/15 split (test n = 78) | 5-fold x 10 repeats, all 517 records |
|---|---|---|
| Random Forest | acc 0.679, AUC 0.67 | acc 0.580 ± 0.039, AUC 0.596 ± 0.052 |
| Gradient Boosting | acc 0.641, AUC 0.70 | acc 0.564 ± 0.035, AUC 0.593 ± 0.045 |
| Logistic Regression | acc 0.603, AUC 0.61 | acc 0.547 ± 0.043, AUC 0.551 ± 0.051 |
| Ensemble (mean) | acc 0.641, AUC 0.70 (95% CI 0.58-0.82) | acc 0.572 ± 0.039, AUC 0.595 ± 0.047 |
| Always predict majority class | 0.513 | 0.522 |

The models beat chance, but modestly. The 78-row test split is optimistic and very noisy; cross-validation is the steadier
estimate. This dataset is known to be hard (the original authors also found weak predictability). Other claims in the disclosure, tested:

* *Ensemble/confidence gating reduces false alarms:* not visible here. HIGH/CRITICAL alert precision is 0.62 for all alerts and 0.61 for
  confident-only alerts. The gate still guarantees that disagreeing models never trigger autonomous deployment.
* *Ensemble beats single models:* in cross-validation the mean ensemble is about equal to Random Forest or Gradient Boosting alone (AUC 0.595 vs 0.596 / 0.593; the spread between folds is about 0.05).
* *Feature importance = temperature, wind, DMC:* not supported. In the Random Forest the largest impurity importances belong to park-grid position (X, Y) and calendar (day, month), which together carry about 0.80 of the total; temperature is 5th and DMC 6th (about 0.04 and 0.03 each), wind 8th, and rain last. Most of what little the model predicts comes from where and when, not from the weather. Impurity importance is biased toward features with many distinct values, so treat the ranking loosely.
* Fire rate observed at each risk level (out-of-fold): LOW 27%, MODERATE 49%, HIGH 63%, CRITICAL 47%. Fire rate rises from LOW to HIGH,
  but CRITICAL (only 15 cases) is not shown to be riskier than HIGH.

`metrics.json` holds all of this and the dashboard's Model page and banner read from it, so the app cannot show stale or projected numbers.
If you present this work, quote the measured figures and describe 8.5 as projections for a future sensor deployment.

## Architecture (disclosure section 6 -> code)

| Disclosure step | Module | Notes |
|---|---|---|
| Sensing layer | `sensors.py` | **Simulated**: real historical records for the month, perturbed by a scenario (heatwave, dry spell, wind event, cool/wet). Labelled `simulated`. |
| Data ingestion | `ingestion.py` | Schema and physical-range validation, timestamp validation and UTC sync, per-zone rolling-median spike filter, out-of-training-range warnings. |
| ML prediction engine | `models.py`, `training.py`, `data.py` | RF, GB, LR as in section 8 (GB is scikit-learn's histogram gradient boosting so it can take monotonic constraints). Hyper-parameters tuned on the validation split. |
| Risk Confidence Aggregator (claim 2) | `aggregator.py` | Mean probability -> LOW/MODERATE/HIGH/CRITICAL; confidence = 1 - spread between models. |
| Resource Availability Manager | `orchestrator.py` | Fleet (drones, trucks, tankers, personnel, sprinklers), out-of-service tracking. |
| AROA orchestration (claims 3-4) | `orchestrator.py` | Two-pass priority allocation under scarcity, shortfall reporting, human review when models disagree. |
| Logging & evidence (claim 8) | `store.py` | SQLite: predictions, cycles, decisions, real outcomes. |
| Adaptive retraining (claim 8) | `retrain.py` | Champion/challenger: a retrained model replaces the current one only if it does not lose on a hold-out set. |
| Heatmap (claim 7) | `figures.py`, dashboard | Fire-event heatmap (Fig 13) and a live zone-risk map on the 9x8 park grid. |
| Dashboard | `webapp.py`, `templates/` | Flask UI plus JSON API. |

## Design choices worth knowing

* **Hotter / drier / windier can never mean lower risk.** RF and GB are trained with monotonic constraints on the fire-weather features
  (`config.MONOTONIC_UP/DOWN`: FFMC, DMC, DC, ISI, temp, wind up; RH, rain down). The earlier unconstrained models had learned so little about dryness
  that a simulated heatwave scored *lower than a cool, wet day* in 11 of 12 calendar months (the test passed only in August), which a fire-risk tool should not do. Constraining them costs nothing measurable
  (cross-validated ensemble AUC 0.595 vs 0.583 before, well within the 0.05 fold-to-fold spread) and makes the property a guarantee that `tests/test_monotonicity.py` checks,
  rather than something a noisy model happens to satisfy. Logistic regression is not constrained; it is a weak third voice in the ensemble.

* **Simulation only.** Actions go through `SimulatedGateway`. Nothing controls real equipment. `ActuatorGateway` is the seam for real integrations.
* **Evacuation advisories need sign-off.** The disclosure describes fully autonomous action; here `EVACUATION_ADVISORY` is drafted automatically but stays `PENDING_APPROVAL`. Edit `NEEDS_APPROVAL` in `orchestrator.py` to change that.
* **Simulated data can never retrain the model.** Only outcomes a person records against a logged prediction (`POST /api/outcomes`) count, and there must be at least 30 with both classes present.
* **Planning is per cycle.** Each cycle re-plans against the current fleet; it does not track units in the field over time.
* **Month/day encoding** uses calendar order (jan=1, mon=0) rather than alphabetical label encoding, so the numbers keep their meaning.
* **Deployed models** are refit on train+validation with the chosen hyper-parameters; the reported test numbers come from the train-only fit.

## API

`POST /api/predict` what-if for one zone (nothing logged) · `POST /api/ingest` one reading or `{"readings":[...]}` (up to 500; bad ones are rejected individually with reasons) ·
`POST /api/simulate` `{scenario, zones, seed?}` · `GET /api/zones` · `GET/POST /api/resources` · `GET /api/logs/{predictions|decisions|cycles|outcomes}?limit=` ·
`POST /api/outcomes` · `POST /api/retrain` · `GET /api/model/metrics` · `GET /health`.
Set `WILDFIRE_API_KEY` to require an `X-API-Key` header on every POST.

## Not implemented (and why)

* **CNN on satellite imagery and LSTM on time series.** The prototype dataset has neither images nor sequences; the disclosure itself defers them. `models.py` is where they would plug in.
* **MQTT ingestion, Redis, cloud microservice deployment, edge (Jetson/Raspberry Pi) packaging, SMS/email alerts.** Ingestion is HTTP only, and all services run in one process.
* **NDVI, gas sensors, satellite thermal data.** None are in the dataset. The model inputs are the 12 UCI columns.
* **Multi-worker deployment.** Ingestor history is in memory and the SQLite store is single-node. Run one process.

## Limits of the evidence

517 records, one park, 2000-2003, weather-index inputs only, a label that treats a 0.1 ha and a 1,000 ha fire the same, and 4 exact duplicate rows
that a random split can put on both sides. Nothing here has been validated on other regions or on live sensors.
