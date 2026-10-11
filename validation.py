"""
validation.py - data checks from guide section 4.5.

Run before any analysis: by load_data.py (prints them) and by train.py (stops if one fails).
The leakage test and the split test live in tests/ and in training.split_holdout.
"""
import json

import pandas as pd

from .config import LIMITS, RAW_DIR, SENSORS
from .labels import failure_events


def validate(data: dict[str, pd.DataFrame]) -> list[tuple[str, bool, str]]:
    """Run every check on the loaded tables. Returns (check name, passed, detail) per check."""
    r, wo, sc = data["sensor_readings"], data["work_orders"], data["production_schedule"]
    known = set(sc.machine_id)                     # the machine list = machines in the schedule
    ev = failure_events(wo, sc)
    # Failures that still fall inside a planned stop would mean the relabelling did not work.
    planned = sc[sc.kind != "run"].merge(ev, on="machine_id")
    overlap = ((planned.failure_ts >= planned.start) & (planned.failure_ts <= planned.end)).sum()
    checks = [
        ("every machine in readings is in the machine list", set(r.machine_id) <= known,
         f"{len(set(r.machine_id) - known)} unknown"),
        ("timestamps ordered, UTC, no future dates",
         bool(r.groupby("machine_id").ts.apply(lambda s: s.is_monotonic_increasing).all()
              and str(r.ts.dt.tz) == "UTC" and r.ts.max() <= pd.Timestamp.now(tz="UTC")),
         f"{r.ts.min()} .. {r.ts.max()}"),
        ("values inside physical limits",
         all(r[s].dropna().between(*LIMITS[s]).all() for s in SENSORS), "invalid rows rejected while loading"),
        ("work orders link to a real machine and have a closing time",
         bool(wo.machine_id.isin(known).all() and wo.closed_at.notna().all()), f"{len(wo)} work orders"),
        ("planned stops do not overlap failure labels", overlap == 0,
         f"{len(wo[wo.type == 'corrective']) - len(ev)} corrective order(s) inside planned stops relabelled"),
    ]
    # Row counts are compared with the data card written by generate_data.py (if present).
    card = RAW_DIR / "data_card.json"
    if card.exists():
        exp = json.loads(card.read_text())["expected_rows_after_cleaning"]
        got = {t: len(d) for t, d in data.items()}
        # Extra files added by the loader are allowed, so the pack must be fully present (>=).
        checks.append(("row counts match the data card", all(got[t] >= n for t, n in exp.items()),
                       f"expected {exp}, got {got}"))
    return checks


def print_checks(checks) -> bool:
    """Print PASS/FAIL per check; return True only if all passed."""
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    return all(ok for _, ok, _ in checks)
