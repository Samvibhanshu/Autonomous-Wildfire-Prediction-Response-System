"""Logging & Evidence Storage (disclosure step 8): predictions, decisions, resource state, real outcomes."""
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, zone_id TEXT NOT NULL, source TEXT NOT NULL,
  features TEXT NOT NULL, per_model TEXT NOT NULL, probability REAL NOT NULL, confidence REAL NOT NULL,
  level TEXT NOT NULL, confident INTEGER NOT NULL, model_version TEXT NOT NULL, warnings TEXT, cycle_id INTEGER);
CREATE INDEX IF NOT EXISTS idx_pred_zone ON predictions(zone_id, id);
CREATE TABLE IF NOT EXISTS cycles(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, scenario TEXT, source TEXT, resources TEXT NOT NULL, summary TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY, cycle_id INTEGER NOT NULL, prediction_id INTEGER, zone_id TEXT NOT NULL, level TEXT NOT NULL,
  mode TEXT NOT NULL, allocated TEXT NOT NULL, shortfall TEXT NOT NULL, actions TEXT NOT NULL, reason TEXT, ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outcomes(
  id INTEGER PRIMARY KEY, prediction_id INTEGER NOT NULL UNIQUE REFERENCES predictions(id),
  observed_fire INTEGER NOT NULL CHECK (observed_fire IN (0,1)), area_ha REAL, note TEXT, recorded_at TEXT NOT NULL);
"""


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)
            if path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")

    def _q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def log_cycle(self, scenario, source, resources, summary) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO cycles(ts,scenario,source,resources,summary) VALUES(?,?,?,?,?)",
                                  (now_iso(), scenario, source, json.dumps(resources), json.dumps(summary)))
            self.db.commit()
            return cur.lastrowid

    def log_prediction(self, reading, assessment, source, model_version, cycle_id=None) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO predictions(ts,zone_id,source,features,per_model,probability,confidence,level,confident,model_version,warnings,cycle_id)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (reading.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), reading.zone_id, source, json.dumps(reading.features),
                 json.dumps(assessment.per_model), assessment.probability, assessment.confidence, assessment.level,
                 int(assessment.confident), model_version, json.dumps(assessment.warnings + reading.warnings), cycle_id))
            self.db.commit()
            return cur.lastrowid

    def log_decision(self, cycle_id, prediction_id, d):
        with self.lock:
            self.db.execute(
                "INSERT INTO decisions(cycle_id,prediction_id,zone_id,level,mode,allocated,shortfall,actions,reason,ts) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (cycle_id, prediction_id, d.zone_id, d.level, d.mode, json.dumps(d.allocated), json.dumps(d.shortfall),
                 json.dumps(d.actions), d.reason, now_iso()))
            self.db.commit()

    def latest_by_zone(self) -> list[dict]:
        rows = self._q("""SELECT p.*, d.mode, d.allocated, d.shortfall, d.actions, d.reason FROM predictions p
                          LEFT JOIN decisions d ON d.prediction_id = p.id
                          WHERE p.id IN (SELECT MAX(id) FROM predictions GROUP BY zone_id) ORDER BY p.zone_id""")
        for r in rows:
            for k in ("features", "per_model", "warnings", "allocated", "shortfall", "actions"):
                r[k] = json.loads(r[k]) if r.get(k) else ({} if k != "warnings" and k != "actions" else [])
        return rows

    def recent(self, table: str, limit=50) -> list[dict]:
        if table not in ("predictions", "decisions", "cycles", "outcomes"):
            raise KeyError(table)
        rows = self._q(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (int(limit),))
        for r in rows:
            for k in ("features", "per_model", "warnings", "allocated", "shortfall", "actions", "resources", "summary"):
                if r.get(k):
                    r[k] = json.loads(r[k])
        return rows

    def prediction(self, pid: int):
        r = self._q("SELECT * FROM predictions WHERE id=?", (pid,))
        return r[0] if r else None

    def add_outcome(self, prediction_id, observed_fire, area_ha=None, note=None):
        with self.lock:
            self.db.execute("INSERT INTO outcomes(prediction_id,observed_fire,area_ha,note,recorded_at) VALUES(?,?,?,?,?)"
                            " ON CONFLICT(prediction_id) DO UPDATE SET observed_fire=excluded.observed_fire, area_ha=excluded.area_ha,"
                            " note=excluded.note, recorded_at=excluded.recorded_at",
                            (prediction_id, int(observed_fire), area_ha, note, now_iso()))
            self.db.commit()

    def labelled_examples(self) -> list[dict]:
        """Logged predictions whose real outcome was recorded: the retraining feedback set. Simulated readings only
        count if a person recorded an outcome for them, so pure simulation can never retrain the model on its own."""
        rows = self._q("""SELECT p.features, o.observed_fire FROM outcomes o JOIN predictions p ON p.id=o.prediction_id""")
        return [{**json.loads(r["features"]), "fire": r["observed_fire"]} for r in rows]

    def last_cycle_allocations(self) -> dict:
        """Total resources committed by the most recent cycle's decisions."""
        rows = self._q("SELECT allocated FROM decisions WHERE cycle_id=(SELECT MAX(id) FROM cycles)")
        total: dict = {}
        for r in rows:
            for k, v in json.loads(r["allocated"]).items():
                total[k] = total.get(k, 0) + v
        return total

    def counts(self) -> dict:
        return {t: self._q(f"SELECT COUNT(*) c FROM {t}")[0]["c"] for t in ("predictions", "decisions", "cycles", "outcomes")}
