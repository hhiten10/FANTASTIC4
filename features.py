"""
features.py - step 4: feature engineering.

`build_features` is THE feature function. Training (src/training.py) and serving
(src/serving.py) both call it, so the model always sees features computed the same way.

How it works:
  1. 10-minute readings are grouped into 1-day bins per machine (mean, std, count, slope).
  2. Rolling windows of 1, 3 and 7 days are computed over those daily values.
  3. The feature row for prediction time `as_of` uses only the bins that END at or before as_of.

Leakage rule: a feature row for (machine, as_of) only uses readings with ts < as_of and
work orders closed before as_of. Work-order note text is never used.
"""
import numpy as np
import pandas as pd

from .config import DAY, SENSORS, SHORT
from .sensor_check import drift_pattern, sensor_fault

# The model's input columns. Per sensor (temp, vib, pres, rpm):
#   mean_1d/3d/7d    average level over the last 1, 3, 7 days
#   std_1d/3d/7d     how much the readings jump around
#   slope_1d/3d/7d   trend (units per day): rising or falling
#   vs_base_1d/3d    change versus the machine's own baseline (its last-7-day mean)
# Plus: vibration-per-rpm trend (load context), data coverage, days since repair, criticality.
FEATURES = [f"{k}_{x}" for k in SHORT.values() for x in
            ["mean_1d", "mean_3d", "mean_7d", "std_1d", "std_3d", "std_7d",
             "slope_1d", "slope_3d", "slope_7d", "vs_base_1d", "vs_base_3d"]] + \
           ["vib_per_krpm_slope_7d", "coverage_1d", "days_since_repair", "criticality"]


def _slope(y: np.ndarray) -> float:
    """Least-squares slope (per day) of a series of daily values, ignoring missing days."""
    ok = ~np.isnan(y)
    if ok.sum() < 2:                      # need at least 2 points for a line
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    x, y = x - x.mean(), y[ok]
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _daily(r: pd.DataFrame, offset: pd.Timedelta) -> pd.DataFrame:
    """Per machine and day: mean, std, count and within-day slope of every sensor.

    Days are shifted by `offset` so a prediction time like 13:50 falls exactly on a day edge
    (training uses midnight, so offset = 0 there).
    """
    day = (r.ts - offset).dt.floor("D") + offset          # start of the 1-day bin of each reading
    g = r.groupby([r.machine_id, day])[SENSORS]
    daily = pd.concat([g.mean().add_suffix("_mean"), g.std().add_suffix("_std"), g.size().rename("n")], axis=1)
    # Slope inside each 1-day bin (units per day) = cov(t, y) / var(t), with t = time in days.
    # Computed from group means of t, y, t*y and t*t so it stays fast on ~1M rows.
    t = (r.ts - day) / DAY
    for s in SENSORS:
        ts_ = t.where(r[s].notna())       # ignore times where this sensor has no value
        mom = pd.DataFrame({"t": ts_, "y": r[s], "ty": ts_ * r[s], "tt": ts_ * ts_}).groupby([r.machine_id, day]).mean()
        daily[f"{s}_slope"] = (mom.ty - mom.t * mom.y) / (mom.tt - mom.t ** 2)
    return daily


def _machine_features(d: pd.DataFrame) -> pd.DataFrame:
    """Turn one machine's daily aggregates into feature rows.

    d is indexed by day on a complete daily grid (missing days are NaN). The bin starting at
    day D becomes the feature row for as_of = D + 1 day, so only past data is ever used.
    min_periods lets a window work with a few missing days.
    """
    f = pd.DataFrame(index=d.index + DAY)
    for s, k in SHORT.items():
        m, sd = d[f"{s}_mean"], d[f"{s}_std"]
        f[f"{k}_mean_1d"] = m.values
        f[f"{k}_mean_3d"] = m.rolling(3, min_periods=2).mean().values
        f[f"{k}_mean_7d"] = m.rolling(7, min_periods=4).mean().values
        f[f"{k}_std_1d"] = sd.values
        f[f"{k}_std_3d"] = sd.rolling(3, min_periods=2).mean().values
        f[f"{k}_std_7d"] = sd.rolling(7, min_periods=4).mean().values
        f[f"{k}_slope_1d"] = d[f"{s}_slope"].values      # trend within the last day
        f[f"{k}_slope_3d"] = m.rolling(3, min_periods=2).apply(_slope, raw=True).values
        f[f"{k}_slope_7d"] = m.rolling(7, min_periods=4).apply(_slope, raw=True).values
        # Change versus the machine's own baseline (= its last-7-day mean): machines have
        # different normal levels, so "2.6 mm/s" means different things on different machines.
        f[f"{k}_vs_base_1d"] = f[f"{k}_mean_1d"] - f[f"{k}_mean_7d"]
        f[f"{k}_vs_base_3d"] = f[f"{k}_mean_3d"] - f[f"{k}_mean_7d"]
    # Vibration per 1000 rpm: removes the effect of load changes between product runs.
    vpr = 1000 * d["vibration_mm_s_mean"] / d["rpm_mean"]
    f["vib_per_krpm_slope_7d"] = vpr.rolling(7, min_periods=4).apply(_slope, raw=True).values
    f["vib_drift_pattern"] = drift_pattern(f)                   # used by the sensor-fault rule
    f["coverage_1d"] = (d["n"].fillna(0) / 144).values          # share of the 144 expected 10-min readings
    f["history_days"] = d["n"].gt(0).cumsum().values            # days with any data so far
    return f


def build_features(readings: pd.DataFrame, work_orders: pd.DataFrame, schedule: pd.DataFrame,
                   as_of: pd.DataFrame) -> pd.DataFrame:
    """Compute features for every (machine_id, as_of) pair in `as_of`.

    Returns one row per pair that has any reading history, with FEATURES plus
    history_days, machine_type and sensor_fault (reason text or None).
    """
    as_of = as_of.copy()
    # Use one timestamp precision everywhere (user-supplied times can differ from the data's).
    as_of["as_of"] = pd.to_datetime(as_of.as_of, utc=True).astype("datetime64[ns, UTC]")
    readings = readings.assign(ts=readings.ts.astype("datetime64[ns, UTC]"))
    as_of["offset"] = as_of.as_of - as_of.as_of.dt.floor("D")   # time of day of each as_of
    parts = []
    for offset, grp in as_of.groupby("offset"):
        # Only readings strictly before the latest as_of of this group (no future data).
        r = readings[(readings.ts < grp.as_of.max()) & readings.machine_id.isin(grp.machine_id)]
        if r.empty:
            continue
        for mid, d in _daily(r, offset).groupby(level=0):
            d = d.droplevel(0)
            # Complete daily grid up to the last full day, so missing days appear as NaN.
            d = d.reindex(pd.date_range(d.index.min(), grp.as_of.max() - DAY, freq="D"))
            f = _machine_features(d).rename_axis("as_of").reset_index()
            f["machine_id"] = mid
            parts.append(f)
    if not parts:
        return pd.DataFrame(columns=["machine_id", "as_of", *FEATURES, "history_days", "sensor_fault"])
    feats = as_of[["machine_id", "as_of"]].merge(pd.concat(parts), on=["machine_id", "as_of"])

    # Days since the last repair that was CLOSED before as_of (strictly before: no exact matches).
    rep = work_orders[work_orders["type"].isin(["corrective", "preventive"])][["machine_id", "closed_at"]]
    rep = rep.assign(closed_at=rep.closed_at.astype("datetime64[ns, UTC]"))
    feats = pd.merge_asof(feats.sort_values("as_of"), rep.sort_values("closed_at"), left_on="as_of",
                          right_on="closed_at", by="machine_id", allow_exact_matches=False)
    feats["days_since_repair"] = (feats.as_of - feats.closed_at) / DAY
    # Machine info from the production schedule (criticality is a feature, type is for slices).
    info = schedule.groupby("machine_id").agg(criticality=("criticality", "max"),
                                              machine_type=("machine_type", "first"))
    feats = feats.drop(columns="closed_at").merge(info, left_on="machine_id", right_index=True, how="left")
    feats["sensor_fault"] = sensor_fault(feats)
    return feats.sort_values(["machine_id", "as_of"]).reset_index(drop=True)
