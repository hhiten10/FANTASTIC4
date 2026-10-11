"""
api.py - FastAPI service implementing the contract in guide section 6.

    uvicorn api:app --port 8000          (interactive docs at http://localhost:8000/docs)

Endpoints:
  POST /predict             score machines (probability, risk band, days to failure, drivers)
  POST /ingest              add a CSV to the data while running
  GET  /model/info          model card
  GET  /explain/{machine}   SHAP drivers + recent sensor trends for one machine
  GET  /health              API, data and model status

The prediction logic lives in src/serving.py; this file only defines the HTTP endpoints.
"""
import io
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import HORIZON_DAYS
from src.ingest import read_csv, save_new_file
from src.serving import Predictor

STATE: dict = {}      # holds the Predictor once the app has started


@asynccontextmanager
async def lifespan(_):
    """Runs once at start-up: load the saved model and the CSV data."""
    STATE["predictor"] = Predictor()
    yield


app = FastAPI(title="Predictive Maintenance Command Center", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Body of POST /predict. Every field is optional."""
    machine_ids: list[str] | None = None
    as_of: str | None = None          # "YYYY-MM-DD HH:MM", UTC
    horizon_days: int = HORIZON_DAYS
    rows: list[dict] | None = None    # optional sensor_readings rows (Annex B) to score with


@app.post("/predict")
def predict(req: PredictRequest):
    """POST /predict (Guide Section 6.1):
    Scores machines for 7-day failure risk.
    Payload:
      - machine_ids: list of machine IDs (e.g. ["M-014"]) or None (scores all 40 machines)
      - as_of: snapshot timestamp (e.g. "2026-09-15 08:00")
      - horizon_days: default 7 (returns invalid_input if != 7)
      - rows: optional raw telemetry batch for live in-memory testing
    Returns:
      model_version, threshold, as_of, and per-machine predictions array.
    """
    return STATE["predictor"].score(req.machine_ids, req.as_of, req.horizon_days, req.rows)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), meta: str = Form('{"table": "sensor_readings"}')):
    """POST /ingest:
    Live data ingestion without restarting the server.
    Accepts multipart CSV upload + table name.
    Cleans, validates physical limits, drops duplicates, saves to data/raw,
    and calls reload() so subsequent /predict calls instantly reflect new data.
    """
    try:
        table = json.loads(meta)["table"]
        info = save_new_file(read_csv(io.BytesIO(await file.read())), table, file.filename or "upload.csv")
    except (ValueError, KeyError) as e:        # bad JSON, unknown table or missing columns
        return {"rows_loaded": 0, "rows_rejected": None, "status": f"error: {e}"}
    STATE["predictor"].reload()
    return info


@app.get("/model/info")
def model_info():
    """Model card: algorithm, features, training window, validation, metrics, threshold, version."""
    return {"model_card": STATE["predictor"].bundle["model_card"]}


@app.get("/explain/{machine_id}")
def explain(machine_id: str, as_of: str | None = None):
    """Why is this machine scored the way it is? Top 8 SHAP drivers plus 14 days of sensor trends."""
    p = STATE["predictor"]
    res = p.score([machine_id], as_of, n_drivers=8)
    pred = res["predictions"][0]
    if pred["status"] == "invalid_input" and pred["data_quality"] == "unknown machine":
        raise HTTPException(404, "unknown machine")
    return {**res, "prediction": pred, "sensor_trends_daily": p.trends(machine_id, res["as_of"]),
            "note": "contribution = SHAP value in the model's raw score space; positive pushes risk up"}


@app.get("/health")
def health():
    """Readiness: is the API up, is the model loaded, how many rows of each table are loaded."""
    p = STATE.get("predictor")
    return {"api": "ok", "model_loaded": p is not None, "model_version": p.version if p else None,
            "data_store": {t: len(v) for t, v in p.data.items()} if p else {}}
