"""
ingest.py - steps 1-2 of the pipeline: load the 4 CSV files and clean them.

Cleaning does four things:
  1. fixes column names (so a renamed column in the late data drop still loads),
  2. converts every timestamp to UTC (sensors are UTC, work orders are in IST +05:30),
  3. rejects invalid rows (missing required values, future dates, values outside physical limits),
  4. drops duplicate rows (e.g. the same machine and timestamp twice).

The same `clean` function is used for the data pack, the CLI loader and POST /ingest.
"""
import re
from pathlib import Path

import pandas as pd

from .config import (ALIASES, DAY, EXTRA, KEYS, LIMITS, NULLABLE, RAW_DIR, SENSORS, TABLES, TIME_COLS)


def _norm(name) -> str:
    """Lower-case a column name and strip everything except letters/digits: 'Vibration_MM-S' -> 'vibrationmms'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def clean(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict]:
    """Clean one raw table.

    Returns the cleaned frame and a small report:
    {"rows_in", "rows_rejected", "duplicates_dropped"}.
    Raises ValueError if a required column is missing.
    """
    # --- 1. column names: match on the normalised name, then on known aliases
    wanted = TABLES[table] + EXTRA.get(table, [])
    lookup = {_norm(c): c for c in wanted} | {_norm(k): v for k, v in ALIASES.items() if v in wanted}
    df = df.rename(columns={c: lookup.get(_norm(c), c) for c in df.columns})
    missing = [c for c in TABLES[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}; got {list(df.columns)}")
    df = df[[c for c in wanted if c in df.columns]].copy()     # drop columns we do not use
    n_in = len(df)

    # --- 2. types: times to UTC, machine ids in one format
    for c in TIME_COLS[table]:   # UTC sensors and IST (+05:30) work orders both end up in UTC
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601")
    if "machine_id" in df:
        df["machine_id"] = df["machine_id"].astype(str).str.strip().str.upper()   # " m-001" -> "M-001"

    # --- 3. invalid rows
    bad = pd.Series(False, index=df.index)
    for c in TABLES[table]:
        if c not in NULLABLE[table]:
            bad |= df[c].isna()                                   # required value missing / unparsable
    for c in TIME_COLS[table]:
        if table != "production_schedule":                       # the schedule is planned ahead
            bad |= df[c] > pd.Timestamp.now(tz="UTC") + DAY      # no future dates
    if table == "sensor_readings":
        for s in SENSORS:
            df[s] = pd.to_numeric(df[s], errors="coerce")
            bad |= (df[s] < LIMITS[s][0]) | (df[s] > LIMITS[s][1])  # e.g. negative RPM
            # A missing (NaN) sensor value is NOT rejected: it is a gap, handled later by imputation.
    if table == "production_schedule":
        df["criticality"] = pd.to_numeric(df["criticality"], errors="coerce")
    df = df[~bad]
    n_valid = len(df)

    # --- 4. duplicates (keep the first copy) and a stable sort order
    df = df.drop_duplicates(subset=KEYS[table]).sort_values(KEYS[table]).reset_index(drop=True)
    return df, {"rows_in": n_in, "rows_rejected": int(bad.sum()), "duplicates_dropped": n_valid - len(df)}


def read_csv(path) -> pd.DataFrame:
    """Read a CSV with every column as text; `clean` does the type conversion."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def load_table(table: str) -> pd.DataFrame:
    """All CSVs in data/raw whose name starts with the table name, cleaned and merged.

    e.g. sensor_readings.csv (the pack) + sensor_readings__late_drop.csv (added by the loader).
    """
    files = sorted(RAW_DIR.glob(f"{table}*.csv"))
    if not files:
        raise FileNotFoundError(f"no {table}*.csv in {RAW_DIR}; run generate_data.py first")
    df = pd.concat([clean(read_csv(f), table)[0] for f in files], ignore_index=True)
    # Files may overlap, so de-duplicate again after merging them.
    return df.drop_duplicates(subset=KEYS[table]).sort_values(KEYS[table]).reset_index(drop=True)


def load_all() -> dict[str, pd.DataFrame]:
    """Load all 4 tables: {"sensor_readings": df, "work_orders": df, ...}."""
    return {t: load_table(t) for t in TABLES}


def save_new_file(df: pd.DataFrame, table: str, name: str) -> dict:
    """Clean a new file and store it in data/raw as <table>__<name>.csv.

    Used by load_data.py and POST /ingest. Returns the cleaning report plus rows_loaded/status.
    """
    clean_df, info = clean(df, table)
    out = RAW_DIR / f"{table}__{Path(name).stem}.csv"
    clean_df.to_csv(out, index=False)
    info.update(rows_loaded=len(clean_df), status="ok", stored_as=out.name)
    return info
