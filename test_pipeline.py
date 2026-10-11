"""
Tests for the key functions: leakage, labels, time split, schema handling and the API contract.

    pytest
"""
import numpy as np
import pandas as pd
import pytest

from src.config import GAP, MODEL_PATH
from src.features import FEATURES, build_features
from src.ingest import clean
from src.labels import add_labels, failure_events
from src.training import time_folds

T0 = pd.Timestamp("2026-05-01", tz="UTC")      # the prediction time used in the tests


def fake_readings(days=20, machine="M-001"):
    """Normal 10-minute readings from `days` days before T0 until 5 days after it."""
    ts = pd.date_range(T0 - pd.Timedelta(days=days), T0 + pd.Timedelta(days=5), freq="10min", inclusive="left")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"machine_id": machine, "ts": ts, "temperature_c": 60 + rng.normal(0, 1, len(ts)),
                         "vibration_mm_s": 2 + rng.normal(0, 0.1, len(ts)), "pressure_bar": 6.0 + rng.normal(0, .1, len(ts)),
                         "rpm": 1500 + rng.normal(0, 5, len(ts))})


# Minimal schedule / work orders / prediction request shared by the tests.
SCHEDULE = pd.DataFrame({"machine_id": ["M-001"], "start": [T0 - pd.Timedelta(days=30)], "end": [T0 + pd.Timedelta(days=30)],
                         "kind": ["run"], "criticality": [2], "machine_type": ["A"]})
NO_WO = pd.DataFrame({"wo_id": pd.Series([], dtype=object), "machine_id": pd.Series([], dtype=object), "opened_at": pd.to_datetime([], utc=True),
                      "closed_at": pd.to_datetime([], utc=True), "type": [], "note": []})
AS_OF = pd.DataFrame({"machine_id": ["M-001"], "as_of": [T0]})


def test_no_future_leakage():
    """Changing readings after as_of must not change the features at as_of."""
    r = fake_readings()
    before = build_features(r, NO_WO, SCHEDULE, AS_OF)
    r2 = r.copy()
    r2.loc[r2.ts >= T0, "vibration_mm_s"] = 40.0
    after = build_features(r2, NO_WO, SCHEDULE, AS_OF)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])


def test_changeover_logged_as_corrective_is_not_a_failure():
    """A corrective order opened during a planned changeover is not a failure; one 3 days later is."""
    sched = pd.DataFrame({"machine_id": ["M-001"], "start": [T0], "end": [T0 + pd.Timedelta(hours=3)],
                          "kind": ["changeover"], "criticality": [1], "machine_type": ["A"]})
    wo = pd.DataFrame({"wo_id": ["W1", "W2"], "machine_id": ["M-001", "M-001"],
                       "opened_at": [T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(days=3)],
                       "closed_at": [T0 + pd.Timedelta(hours=2), T0 + pd.Timedelta(days=3, hours=5)],
                       "type": ["corrective", "corrective"], "note": ["changeover", "bearing"]})
    ev = failure_events(wo, sched)
    assert list(ev.failure_ts) == [T0 + pd.Timedelta(days=3)]


def test_label_horizon():
    """Label is 1 only when the failure is less than 7 days after as_of."""
    ev = pd.DataFrame({"machine_id": ["M-001"], "failure_ts": [T0 + pd.Timedelta(days=6, hours=23)]})
    df = pd.DataFrame({"machine_id": ["M-001"] * 2, "as_of": [T0, T0 - pd.Timedelta(days=1)]})
    out = add_labels(df, ev).sort_values("as_of")
    assert out.label.tolist() == [0, 1]       # 7d23h away -> 0, 6d23h away -> 1


def test_time_folds_train_always_before_validation():
    """Split test: in every fold, all training days are more than 7 days before validation days."""
    days = pd.date_range("2026-04-01", periods=120, freq="D", tz="UTC")
    dev = pd.DataFrame({"as_of": np.repeat(days, 3)})
    for tr, va in time_folds(dev):
        assert dev.loc[tr, "as_of"].max() + GAP < dev.loc[va, "as_of"].min()


def test_renamed_column_and_timezone_and_bad_rows():
    """Cleaning handles a renamed column, an IST timestamp and a negative-RPM row."""
    raw = pd.DataFrame({"Machine_ID": ["m-001", "M-001", "M-001"],
                        "ts": ["2026-05-01T05:30:00+05:30", "2026-05-01T00:10:00Z", "2026-05-01T00:20:00Z"],
                        "temperature_c": ["60", "61", "62"], "vibration_mms": ["2", "2", "2"],
                        "pressure_bar": ["6", "6", "6"], "rpm": ["1500", "1500", "-5"]})
    df, info = clean(raw, "sensor_readings")
    assert list(df.ts) == [T0, T0 + pd.Timedelta("10min")]   # IST converted to UTC, negative rpm dropped
    assert info["rows_rejected"] == 1 and "vibration_mm_s" in df


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train.py first")
def test_api_predict_contract():
    """POST /predict returns the guide's fields and status values; unknown machines are invalid_input."""
    from fastapi.testclient import TestClient
    import api
    with TestClient(api.app) as c:
        r = c.post("/predict", json={"machine_ids": ["M-001", "M-999"]}).json()
        assert {"model_version", "as_of", "horizon_days", "threshold", "predictions"} <= set(r)
        statuses = {x["machine_id"]: x["status"] for x in r["predictions"]}
        assert statuses["M-999"] == "invalid_input"
        assert statuses["M-001"] in {"scored", "scored_low_confidence", "sensor_fault_suspected", "insufficient_history"}
        assert c.get("/health").json()["model_loaded"]
