# Predictive Maintenance Command Center

Predicts which machines will fail in the next 7 days, estimates days left, and explains each prediction.

```
CSV files (4) → src/ingest → src/labels → src/features → src/training (LR / RF / XGBoost, time-based CV)
  → calibrated model (models/model.joblib) → src/serving → api.py (FastAPI) → app.py (Streamlit)
```

## Files
Scripts you run (top level):

| File | What it does |
|---|---|
| `generate_data.py` | Synthetic data pack (Annex B schema), late data drop, data card |
| `load_data.py` | Loader CLI for new CSVs (judge data, late drop) + prints validation checks |
| `train.py` | Runs the whole training flow and saves model + reports |
| `predict.py` | Writes the predictions file (same code path as `/predict`) |
| `api.py` | FastAPI: `POST /predict`, `POST /ingest`, `GET /model/info`, `GET /explain/{id}`, `GET /health` |
| `app.py` | Streamlit: watchlist, machine page, model quality |

Building blocks (`src/`), one per step:

| File | Step |
|---|---|
| `config.py` | All settings: paths, seed, horizon, cost, windows, schema |
| `ingest.py` | 1–2. Load the 4 CSVs; clean (column names, UTC, bad rows, duplicates) |
| `validation.py` | Data checks from guide 4.5 |
| `labels.py` | 3. Failure events (changeovers excluded) and the 7-day label |
| `features.py` | 4. **The one feature function** `build_features` (training and API) |
| `sensor_check.py` | Stuck / drifting sensor rule |
| `models.py` | 5–7. Classifiers, days-to-failure regressors, calibrated model, SHAP |
| `training.py` | 5–6. Training table, time split, CV, model choice, calibration, threshold |
| `evaluation.py` | Held-out metrics, comparison, slices, error analysis, leakage demo, plot |
| `report.py` | Writes `reports/evaluation_report.md` |
| `serving.py` | 8. Saved model → per-machine predictions with status values |

Also: `tests/` (leakage, labels, time split, schema/timezone handling, API contract) and `reports/` (evaluation
report, model card, calibration plot, experiment log, late-drop predictions).

## Run
```bash
pip install -r requirements.txt
python generate_data.py          # data/raw, data/late_drop (seed 42)
python load_data.py              # validation checks (all must PASS)
python train.py                  # -> models/model.joblib, reports/
uvicorn api:app --port 8000
streamlit run app.py             # http://localhost:8501
pytest
```
Docker: `docker compose up` (API on :8000, UI on :8501; generates data on first start, uses the saved model).

### Late data drop / judge data
```bash
python load_data.py data/late_drop/sensor_readings_2026-10.csv     # renamed column is handled
python predict.py --as-of "2026-10-11 00:00"                       # -> reports/late_drop_predictions.json
```
Or while the API runs: `curl -F file=@my.csv -F 'meta={"table":"sensor_readings"}' localhost:8000/ingest`.

### API examples
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"machine_ids": ["M-014"], "as_of": "2026-09-15 08:00"}'
curl localhost:8000/explain/M-003
curl localhost:8000/model/info
```
`/predict` also accepts `"rows": [...]` (sensor_readings rows); they are used for that request only, not stored.

## Reproducing the headline numbers
`python generate_data.py && python train.py` → `reports/evaluation_report.md`. Seed 42 everywhere; versions pinned in
`requirements.txt`. Each run appends to `reports/experiments.jsonl` (model, params, data window, CV folds, metrics).

Current result (held-out 2026-08-27 to 2026-09-23): random forest, PR-AUC 0.391 (95% CI 0.10–0.75),
4 of 8 failures caught, median lead time 4.8 days, 0.93 false alarms per machine-month, Brier 0.029.

## Key decisions
- **Target**: unplanned failure in the next 7 days, one snapshot per machine per day. Corrective work orders opened
  inside a planned changeover/maintenance window are not failures.
- **No leakage**: features use only readings before `as_of` and repairs closed before `as_of`; note text is never used.
- **Features** (1, 3 and 7-day windows only): mean, std and slope over 1, 3 and 7 days per sensor; change versus the machine's own baseline
  (its last-7-day mean); vibration per 1000 rpm trend (removes load changes); days since last repair; criticality.
- **Sensor fault vs wear**: stuck sensor = no change in 24h. Drift = vibration per rpm rising for two 7-day windows in a
  row while temperature does not rise with it; once seen, the machine stays flagged for 7 days (production speed
  changes make the pattern switch on and off). Those machines get `sensor_fault_suspected` and no score.
- **Validation**: last 4 weeks held out and used once; 4 rolling time folds with a 7-day gap; class weights inside folds.
- **Model choice**: keep logistic regression unless another model beats it by > 0.02 CV PR-AUC (RF won on CV).
- **Calibration**: Platt scaling on out-of-fold scores. **Threshold**: minimises 10 × misses + 1 × false alarms on
  out-of-fold data.
- **Explanations**: SHAP (TreeExplainer / LinearExplainer).
- **Days to failure**: quantile gradient boosting (10/50/90%).

## Limits
- Synthetic data; only 8 failures in the held-out period, so metrics have wide intervals (on the held-out period
  LR tied the chosen RF at 0.391 and XGBoost scored 0.422; we did not re-pick on held-out data).
- Sudden failures with no sensor precursor are missed. A new fault type (pressure leak, in the late drop) is not
  learned yet — retrain once labelled examples exist.
- Drift rule only catches a drift after about two weeks, and can drop the flag if the pattern is not seen for 7 days
  (catches ~97% of drift days in our data); unflagged drift can cause false alarms.
- Model is for a 7-day horizon only; other `horizon_days` return `invalid_input`.
- Parts/roster table is loaded and validated but not used by the model.

## Data card (`data/raw/data_card.json`)
Fully synthetic, made by `generate_data.py` (numpy, seed 42), no LLM, no real company or personal data.
40 machines, 2026-04-01 to 2026-09-30, ~986k readings, 256 work orders (55 corrective, 201 preventive; 45% have notes),
2,854 schedule rows. Traps added on purpose: 1% missing readings and gaps during stops, 0.3% duplicate timestamps,
25 negative-RPM rows, drifting vibration sensors (M-022, M-031), stuck pressure sensor (M-017), changeover logged as
a failure (M-005), cause-naming repair notes (leakage trap), short history (M-040), zero-stock part with 45-day lead
time, work orders in IST (+05:30) vs sensors in UTC. Late drop: 2026-10-01 to 10-10, `vibration_mm_s` renamed to
`vibration_mms`, new `pressure_leak` fault type.

## Model card
See `reports/model_card.json` (also served by `GET /model/info`).

## AI-usage disclosure
Code was written with an AI coding assistant (Claude Code). Verified by: running the full pipeline, unit tests,
calling every API endpoint (including bad inputs), and reading every reported number back from the generated report.
<!-- Team: add who reviewed which part. -->

## Team contribution statement
<!-- Fill in: member — component they built — what they can explain. Each member signs the declaration of original work. -->
