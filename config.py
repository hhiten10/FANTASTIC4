"""
config.py - all project settings in one place.

Every other module imports its constants from here, so a judge's request such as
"change the cost of a miss" or "use a different horizon" is a one-line change.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent      # project root (the folder above src/)
RAW_DIR = ROOT / "data" / "raw"          # every CSV here is part of the data (loader/ingest add files)
LATE_DIR = ROOT / "data" / "late_drop"   # the "new month" file released on the day
HIDDEN_DIR = ROOT / "data" / "hidden"    # answer key of our synthetic data; never read by training
MODEL_PATH = ROOT / "models" / "model.joblib"      # saved model loaded by the API
REPORTS = ROOT / "reports"               # evaluation report, model card, plots, experiment log

# ---------------------------------------------------------------- problem definition
SEED = 42                                # fixed random seed for every model and sampler
HORIZON_DAYS = 7                         # predict "failure within the next 7 days"
DAY = pd.Timedelta("1D")
COST_MISS, COST_FALSE_ALARM = 10, 1      # a missed failure costs 10x a false alarm (sets the threshold)
MIN_HISTORY_DAYS = 7                     # fewer days of data -> status insufficient_history
LOW_CONF_HISTORY_DAYS = 21               # fewer days of data -> status scored_low_confidence

# ---------------------------------------------------------------- validation / model choice
GAP = pd.Timedelta(days=HORIZON_DAYS)    # labels look 7 days ahead -> 7-day gap between train and test
HOLDOUT_DAYS = 28                        # last 4 weeks are held out and used only once
N_FOLDS = 4                              # number of time-based cross-validation folds
RULE_FEATURE = "vib_slope_7d"            # baseline 1: "alarm when vibration rises fast"
SIMPLER_WINS_BY = 0.02                   # keep logistic regression unless beaten by more than this (CV PR-AUC)

# ---------------------------------------------------------------- data schema (guide Annex B)
SENSORS = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm"]
# Short sensor names used inside feature names, e.g. "vib_slope_7d".
SHORT = {"temperature_c": "temp", "vibration_mm_s": "vib", "pressure_bar": "pres", "rpm": "rpm"}
# Physical limits: readings outside these are rejected while loading (e.g. negative RPM).
LIMITS = {"temperature_c": (-20, 200), "vibration_mm_s": (0, 50), "pressure_bar": (0, 30), "rpm": (0, 5000)}
TABLES = {  # required columns of each table
    "sensor_readings": ["machine_id", "ts", *SENSORS],
    "work_orders": ["wo_id", "machine_id", "opened_at", "closed_at", "type", "note"],
    "production_schedule": ["machine_id", "start", "end", "kind", "criticality"],
    "parts_and_roster": ["item", "kind", "qty_or_skill", "lead_time_days", "shift"],
}
EXTRA = {"production_schedule": ["machine_type"]}   # optional extra columns we keep (used for slices)
TIME_COLS = {"sensor_readings": ["ts"], "work_orders": ["opened_at", "closed_at"],
             "production_schedule": ["start", "end"], "parts_and_roster": []}
# Columns that identify a row; a second row with the same key is a duplicate.
KEYS = {"sensor_readings": ["machine_id", "ts"], "work_orders": ["wo_id"],
        "production_schedule": ["machine_id", "start", "kind"], "parts_and_roster": ["item"]}
# Columns that may legitimately be empty (all others must have a value).
NULLABLE = {"sensor_readings": SENSORS, "work_orders": ["note"], "production_schedule": [],
            "parts_and_roster": ["qty_or_skill", "lead_time_days", "shift"]}
# Known alternative column names. Names that differ only in case or punctuation
# (e.g. "vibration_mms" vs "vibration_mm_s") are matched automatically in ingest.py.
ALIASES = {"timestamp": "ts", "time": "ts", "machine": "machine_id", "temperature": "temperature_c",
           "temp_c": "temperature_c", "vibration": "vibration_mm_s", "pressure": "pressure_bar"}
