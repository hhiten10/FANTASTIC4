"""
serving.py - step 8: turn the saved model into per-machine predictions.

Used by api.py (POST /predict, GET /explain) and predict.py (predictions file).
For every machine it returns the failure probability, risk band, days-to-failure estimate,
SHAP drivers and one of the status values from guide section 6.2:
    scored, scored_low_confidence, sensor_fault_suspected, insufficient_history, invalid_input
"""
import joblib
import numpy as np
import pandas as pd

from .config import DAY, HORIZON_DAYS, LOW_CONF_HISTORY_DAYS, MIN_HISTORY_DAYS, MODEL_PATH, SENSORS
from .features import FEATURES, build_features
from .ingest import clean, load_all
from .models import shap_values

# Readings older than this are not needed: the longest feature window is 7 days and the drift
# check looks back 2 x 7 days plus a 7-day hold.
WINDOW = pd.Timedelta(days=21)


class Predictor:
    """Holds the saved model bundle (models/model.joblib) and the loaded CSV data."""

    def __init__(self, model_path=MODEL_PATH):
        # bundle = {"model": CalibratedModel, "threshold", "ttf": {quantile: regressor}, "features", "model_card"}
        self.bundle = joblib.load(model_path)
        self.reload()

    def reload(self) -> None:
        """Re-read the CSVs (called at start-up and after POST /ingest)."""
        self.data = load_all()
        # First reading per machine = how much history it really has (the 21-day window cannot tell).
        self.first_seen = self.data["sensor_readings"].groupby("machine_id").ts.min()

    @property
    def version(self) -> str:
        return self.bundle["model_card"]["version"]

    def score(self, machine_ids=None, as_of=None, horizon_days=HORIZON_DAYS, rows=None, n_drivers=3) -> dict:
        """Build the POST /predict response (guide section 6.1).

        machine_ids  machines to score (default: all known machines)
        as_of        "YYYY-MM-DD HH:MM" UTC (default: just after the latest reading)
        rows         optional extra sensor_readings rows, used for this request only
        n_drivers    number of SHAP drivers to return per machine
        """
        b, data = self.bundle, self.data
        readings, first_seen = data["sensor_readings"], self.first_seen
        if rows:   # extra rows are used for this request only (use POST /ingest to store data)
            extra, _ = clean(pd.DataFrame(rows), "sensor_readings")
            readings = pd.concat([readings, extra]).drop_duplicates(["machine_id", "ts"]).sort_values(["machine_id", "ts"])
            first_seen = readings.groupby("machine_id").ts.min()
        known = sorted(set(data["production_schedule"].machine_id) | set(readings.machine_id))
        machines = [m.strip().upper() for m in machine_ids] if machine_ids else known
        head = {"model_version": self.version, "horizon_days": horizon_days, "threshold": b["threshold"]}

        def invalid(msg, t=None):
            """Whole request is invalid: every machine gets status invalid_input."""
            return {**head, "as_of": t, "predictions": [{"machine_id": m, "status": "invalid_input",
                                                         "failure_prob": None, "data_quality": msg} for m in machines]}

        # --- check the request
        try:
            t = pd.Timestamp(as_of, tz="UTC") if as_of else (readings.ts.max() + pd.Timedelta("10min")).floor("min")
        except ValueError:
            return invalid(f"as_of {as_of!r} is not 'YYYY-MM-DD HH:MM'")
        t_str = t.strftime("%Y-%m-%d %H:%M")
        if horizon_days != HORIZON_DAYS:
            return invalid(f"model is trained for horizon_days={HORIZON_DAYS} only", t_str)
        if t <= readings.ts.min():
            return invalid("as_of is earlier than the data", t_str)

        # --- features (same function as training), probability, SHAP, days to failure
        ok_ids = [m for m in machines if m in known]
        window = readings[(readings.ts >= t - WINDOW) & (readings.ts < t) & readings.machine_id.isin(ok_ids)]
        f = build_features(window, data["work_orders"], data["production_schedule"],
                           pd.DataFrame({"machine_id": ok_ids, "as_of": t}))
        shap, ttf = None, None
        if len(f):
            f["history_days"] = ((t - f.machine_id.map(first_seen)) / DAY).fillna(0).astype(int)
            f["prob"] = b["model"].predict_proba(f)
            shap = shap_values(b["model"], f)
            ttf = {q: m.predict(f[FEATURES]) for q, m in b["ttf"].items()}
        f = f.set_index("machine_id")
        preds = [self._one(m, known, f, shap, ttf, n_drivers) for m in (ok_ids if machine_ids is None else machines)]
        return {**head, "as_of": t_str, "predictions": preds}

    def _one(self, m, known, f, shap, ttf, n_drivers) -> dict:
        """Evaluates one machine according to the 5 official status rules (guide section 6.2).
        
        Order of evaluation:
          1. 'invalid_input': Machine ID does not exist in plant registry.
          2. 'insufficient_history': < 7 days of telemetry (e.g. newly commissioned unit M-040).
          3. 'sensor_fault_suspected': Sensor is frozen (0 variance) or drifting (vibration rising
             without heat, e.g. M-022). Prob is set to NULL to prevent false teardowns!
          4. 'scored_low_confidence': Valid score, but history is short (<21 days) or coverage is patchy.
          5. 'scored': Fully reliable prediction with calibrated probability and SHAP drivers.
        """
        thr = self.bundle["threshold"]
        # Rule 1: Unknown asset
        if m not in known:
            return {"machine_id": m, "status": "invalid_input", "failure_prob": None, "data_quality": "unknown machine"}
        
        # Rule 2: Too short history (< 7 days)
        if m not in f.index or f.loc[m, "history_days"] < MIN_HISTORY_DAYS:
            days = int(f.loc[m, "history_days"]) if m in f.index else 0
            return {"machine_id": m, "status": "insufficient_history", "failure_prob": None,
                    "days_to_failure_est": None, "days_to_failure_note": "not enough history",
                    "data_quality": f"only {days} days of data"}
        
        row, k = f.loc[m], f.index.get_loc(m)       # k = row position in the shap / ttf arrays
        
        # Rule 3: Drifting or frozen transducer detected
        if isinstance(row.sensor_fault, str):
            return {"machine_id": m, "status": "sensor_fault_suspected", "failure_prob": None,
                    "days_to_failure_est": None, "days_to_failure_note": "not estimated: sensor fault suspected",
                    "sensor_fault_flag": True, "data_quality": row.sensor_fault}
        
        # Rule 4 & 5: Scored (Low confidence warning or Normal)
        notes = []
        if row.history_days < LOW_CONF_HISTORY_DAYS:
            notes.append(f"short history: {row.history_days} days")
        if row.coverage_1d < 0.5:
            notes.append(f"only {row.coverage_1d:.0%} of readings in the last 24h")
        prob = float(row.prob)
        flagged = prob >= thr                       # above the threshold -> alarm
        order = np.argsort(-np.abs(shap[k]))[:n_drivers]    # strongest drivers first (either direction)
        return {
            "machine_id": m, "status": "scored_low_confidence" if notes else "scored",
            "failure_prob": round(prob, 3),
            "risk_band": "high" if flagged else "medium" if prob >= thr / 2 else "low",
            # Days to failure only makes sense for machines expected to fail; clipped to 0..7.
            "days_to_failure_est": int(np.clip(round(ttf[0.5][k]), 0, 7)) if flagged else None,
            "days_to_failure_range": [int(np.clip(np.floor(ttf[0.1][k]), 0, 7)),
                                      int(np.clip(np.ceil(ttf[0.9][k]), 0, 7))] if flagged else None,
            "days_to_failure_note": "estimate with 10-90% range" if flagged else
                                    f"low risk: no failure expected within {HORIZON_DAYS} days",
            "sensor_fault_flag": False,
            # contribution > 0 pushes risk up, < 0 pushes it down
            "top_drivers": [{"feature": FEATURES[j],
                             "value": None if pd.isna(row[FEATURES[j]]) else round(float(row[FEATURES[j]]), 3),
                             "contribution": round(float(shap[k][j]), 3)} for j in order],
            "data_quality": "; ".join(notes) or "ok",
        }

    def trends(self, machine_id: str, as_of: str, days: int = 14) -> list[dict]:
        """Daily mean of every sensor over the last `days` days before as_of (evidence for GET /explain)."""
        r = self.data["sensor_readings"]
        r = r[(r.machine_id == machine_id.upper()) & (r.ts < pd.Timestamp(as_of, tz="UTC"))]
        r = r[r.ts >= r.ts.max() - pd.Timedelta(days=days)]
        return (r.set_index("ts")[SENSORS].resample("1D").mean().round(3)
                .reset_index().assign(ts=lambda d: d.ts.dt.strftime("%Y-%m-%d")).to_dict("records"))
