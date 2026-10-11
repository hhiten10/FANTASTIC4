"""
sensor_check.py - is it a bad sensor or a bad machine? (requirement R5)

Two rules:
  stuck  a sensor's readings did not change at all over the last day.
  drift  vibration (per 1000 rpm) keeps rising while temperature does not rise with it.
         Real wear heats the machine up within days; a drifting sensor rises alone for weeks.

A machine flagged here gets status "sensor_fault_suspected" and no failure score, because its
data cannot be trusted.
"""
import pandas as pd

from .config import SHORT


def drift_pattern(f: pd.DataFrame) -> pd.Series:
    """True on days where the vibration-sensor drift pattern is present.

    f = one machine's daily feature rows (built in features._machine_features).
    """
    # Pattern of one 7-day window: vibration per rpm rising (> 0.01 per day) and temperature
    # rising less than 2x that. Both thresholds were tuned on our synthetic data.
    pattern = pd.Series((f.vib_per_krpm_slope_7d > 0.01) & (f.temp_slope_7d < 2 * f.vib_per_krpm_slope_7d),
                        index=f.index)
    # Must hold in this 7-day window AND in the 7-day window before it: early wear can look like
    # drift for a few days, but it does not last two weeks without heating the machine.
    seen = (pattern & pattern.shift(7, fill_value=False)).astype(float)
    # Production runs at different speeds make the pattern switch on and off, so once it is seen
    # the machine stays flagged for the next 7 days.
    return seen.rolling(7, min_periods=1).max().astype(bool)


def sensor_fault(f: pd.DataFrame) -> pd.Series:
    """Per feature row: a text reason if a sensor looks faulty, otherwise None."""
    reason = pd.Series(None, index=f.index, dtype=object)
    for s, k in SHORT.items():
        # std of the last day's readings is 0 -> the sensor is frozen on one value
        reason[(f[f"{k}_std_1d"] < 1e-6) & reason.isna()] = f"{s} sensor stuck (no change in 24h)"
    reason[f.vib_drift_pattern.astype(bool) & reason.isna()] = "vibration sensor drift: vibration up without temperature rise"
    return reason
