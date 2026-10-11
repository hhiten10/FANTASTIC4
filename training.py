"""
training.py - steps 5-6: build the training table, validate by time, calibrate, pick a threshold.

Rules followed (guide Annex A):
  - time-based splits only (never random), with a 7-day gap so labels cannot overlap,
  - class weights / imputation / scaling fitted inside each training fold only,
  - calibration and threshold chosen on validation (out-of-fold) data, never on the held-out period.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .config import (COST_FALSE_ALARM, COST_MISS, DAY, GAP, HOLDOUT_DAYS, MIN_HISTORY_DAYS, N_FOLDS,
                     REPORTS, RULE_FEATURE, SEED, SIMPLER_WINS_BY)
from .features import FEATURES, build_features
from .labels import add_labels, failure_events
from .models import CLASSIFIERS, TTF_QUANTILES, CalibratedModel, logit, make_classifier, make_ttf_regressor


# ---------------------------------------------------------------- training table + split
def make_dataset(data: dict) -> pd.DataFrame:
    """One row per machine per day (as_of = 00:00 UTC) with features and the 7-day label."""
    r, wo, sc = data["sensor_readings"], data["work_orders"], data["production_schedule"]
    # Outcomes are only known up to the last logged work order; later days could still fail
    # without us knowing, so the last 7 days before that point cannot be labelled.
    label_end = wo.opened_at.max().floor("D")
    days = pd.date_range(r.ts.min().floor("D") + DAY, label_end - GAP, freq="D")
    grid = pd.MultiIndex.from_product([sorted(r.machine_id.unique()), days],
                                      names=["machine_id", "as_of"]).to_frame(index=False)
    df = add_labels(build_features(r, wo, sc, grid), failure_events(wo, sc))
    df = df[df.history_days >= MIN_HISTORY_DAYS]          # too little history to say anything
    return df.sort_values(["as_of", "machine_id"]).reset_index(drop=True)


def split_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into development data and the held-out last 4 weeks (used only once, at the end).

    A 7-day gap is left before the held-out period so no training label looks into it.
    """
    hold_start = df.as_of.max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    dev = df[df.as_of < hold_start - GAP].reset_index(drop=True)
    hold = df[df.as_of >= hold_start].reset_index(drop=True)
    assert dev.as_of.max() + GAP < hold.as_of.min(), "split test failed"   # every train row is earlier
    return dev, hold


def time_folds(dev: pd.DataFrame):
    """Rolling-origin folds: train on everything before the block (minus a 7-day gap), validate on the block.

    The dev days are cut into N_FOLDS + 1 blocks; block 0 is only ever used for training.
    Yields (train index, validation index) pairs.
    """
    blocks = np.array_split(np.sort(dev.as_of.unique()), N_FOLDS + 1)
    for b in blocks[1:]:
        yield dev.index[dev.as_of < b[0] - GAP], dev.index[dev.as_of.isin(b)]


# ---------------------------------------------------------------- scoring helpers
def raw_scores(name: str, train: pd.DataFrame, test: pd.DataFrame):
    """Fit model `name` on train and score test. Returns (fitted model, scores).

    The threshold rule needs no fitting: its score is simply the 7-day vibration slope.
    """
    if name == "threshold_rule":
        return None, test[RULE_FEATURE].fillna(test[RULE_FEATURE].min()).to_numpy()
    m = make_classifier(name, train).fit(train[FEATURES], train.label)
    return m, m.predict_proba(test[FEATURES])[:, 1]


def suppress(df: pd.DataFrame, s: np.ndarray, handle_faults: bool) -> np.ndarray:
    """Sensor-fault handling: a machine with a suspected bad sensor gets no failure score (no alarm).

    Its score is set to the lowest score so it never raises an alarm. With handle_faults=False
    (the ablation) scores are left unchanged.
    """
    return np.where(df.sensor_fault.notna(), s.min() if len(s) else 0, s) if handle_faults else s


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Threshold with the lowest cost: COST_MISS x missed failures + COST_FALSE_ALARM x false alarms."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)            # try 0.01, 0.02, ..., 0.99
    cost = [COST_MISS * ((prob < t) & (y == 1)).sum() + COST_FALSE_ALARM * ((prob >= t) & (y == 0)).sum()
            for t in grid]
    return float(grid[int(np.argmin(cost))])


# ---------------------------------------------------------------- CV, model choice, final fit
def cross_validate(dev: pd.DataFrame, name: str, handle_faults: bool = True):
    """Time-based CV of one model. Returns out-of-fold raw scores and the PR-AUC of each fold.

    Out-of-fold = every score comes from a model trained only on earlier data.
    """
    oof, folds = pd.Series(np.nan, index=dev.index), []
    for tr, va in time_folds(dev):
        train = dev.loc[tr]
        if handle_faults:
            train = train[train.sensor_fault.isna()]          # do not learn from bad-sensor rows
        _, s = raw_scores(name, train, dev.loc[va])
        oof[va] = s
        folds.append(average_precision_score(dev.loc[va, "label"], suppress(dev.loc[va], s, handle_faults)))
    return oof, folds


def compare_models(dev: pd.DataFrame, run_id: str, window: str) -> dict:
    """CV of 2 baselines + 2 model families. Every model is logged to reports/experiments.jsonl.

    Returns {model name: [PR-AUC per fold]}.
    """
    cv = {}
    with open(REPORTS / "experiments.jsonl", "a") as log:
        for name in ["threshold_rule"] + CLASSIFIERS:
            _, folds = cross_validate(dev, name)
            cv[name] = folds
            params = "" if name == "threshold_rule" else str(make_classifier(name, dev).get_params())[:500]
            log.write(json.dumps({"run_id": run_id, "model": name, "data_window": window, "seed": SEED,
                                  "validation": f"{N_FOLDS} rolling time folds, {GAP.days}-day gap",
                                  "cv_pr_auc_folds": [round(f, 3) for f in folds],
                                  "cv_pr_auc_mean": round(np.mean(folds), 3), "params": params}) + "\n")
            print(f"{name:<20} CV PR-AUC {np.mean(folds):.3f} +/- {np.std(folds):.3f}")
    return cv


def pick_model(cv: dict) -> str:
    """Keep logistic regression unless another model beats it by more than SIMPLER_WINS_BY.

    "A complex model must earn its place" (guide R6).
    """
    best = max(CLASSIFIERS, key=lambda n: np.mean(cv[n]))
    return best if np.mean(cv[best]) > np.mean(cv["logistic_regression"]) + SIMPLER_WINS_BY else "logistic_regression"


def fit_final(dev: pd.DataFrame, name: str, handle_faults: bool = True):
    """Build the model that gets saved.

    1. time-based CV -> out-of-fold scores
    2. Platt calibrator fitted on those out-of-fold scores
    3. decision threshold chosen on the calibrated out-of-fold probabilities (cost-based)
    4. classifier refitted on all development data
    Returns (CalibratedModel, threshold, CV fold scores).
    """
    oof, folds = cross_validate(dev, name, handle_faults)
    seen = oof.dropna().index                                 # rows that were in some validation block
    cal_rows = seen if not handle_faults else seen[dev.loc[seen, "sensor_fault"].isna()]
    calibrator = LogisticRegression().fit(logit(oof[cal_rows].to_numpy()), dev.loc[cal_rows, "label"])
    oof_prob = calibrator.predict_proba(logit(oof[seen].to_numpy()))[:, 1]
    threshold = best_threshold(dev.loc[seen, "label"].to_numpy(),
                               suppress(dev.loc[seen], oof_prob, handle_faults))
    train = dev[dev.sensor_fault.isna()] if handle_faults else dev
    model, _ = raw_scores(name, train, train.head(1))         # refit on all dev data
    background = train[FEATURES].sample(min(200, len(train)), random_state=SEED)   # for SHAP
    return CalibratedModel(model, calibrator, background), threshold, folds


def train_ttf(dev: pd.DataFrame) -> dict:
    """Days-to-failure regressors {0.1: model, 0.5: model, 0.9: model}.

    Trained only on rows where the machine did fail within 7 days (days_to_failure is known).
    """
    pos = dev[(dev.label == 1) & dev.sensor_fault.isna()]
    return {q: make_ttf_regressor(q).fit(pos[FEATURES], pos.days_to_failure) for q in TTF_QUANTILES}
