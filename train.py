"""
train.py - train, validate, calibrate and save the model; write the evaluation report.

    python train.py

Flow:
  validate data -> daily snapshots + labels -> time split (last 4 weeks held out)
  -> time-based CV of 2 baselines + 2 model families -> pick model -> Platt calibration
  -> cost-based threshold -> ONE evaluation on the held-out period -> save model + reports.

Outputs: models/model.joblib, reports/evaluation_report.md, reports/model_card.json,
         reports/calibration.png, reports/experiments.jsonl (appended).
The work is done by the modules in src/; this file only runs them in order.
"""
import json
import time

import joblib
import pandas as pd

from src import evaluation as E
from src.config import COST_MISS, GAP, HORIZON_DAYS, MODEL_PATH, N_FOLDS, REPORTS, SEED
from src.features import FEATURES, build_features
from src.ingest import load_all
from src.labels import failure_events
from src.report import write_report
from src.training import compare_models, fit_final, make_dataset, pick_model, split_holdout, suppress, train_ttf
from src.validation import print_checks, validate

SHORT_NAMES = {"logistic_regression": "lr", "random_forest": "rf", "xgboost": "xgb"}   # for the model version


def main():
    """Run the full training flow end to end."""
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # 1) load + validate
    data = load_all()
    if not print_checks(validate(data)):
        raise SystemExit("validation failed - fix the data first")

    # 2) training table + time split
    df = make_dataset(data)
    dev, hold = split_holdout(df)
    window = f"{dev.as_of.min().date()} to {dev.as_of.max().date()}"
    print(f"dev {len(dev)} rows ({window}), held-out {len(hold)} rows from {hold.as_of.min().date()}")

    # 3) compare models with time-based CV, pick one
    run_id = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    cv = compare_models(dev, run_id, window)
    chosen = pick_model(cv)
    print("chosen:", chosen)

    # 4) calibrate + threshold + final fit; days-to-failure model
    model, threshold, _ = fit_final(dev, chosen)
    ttf = train_ttf(dev)
    t_train = time.time() - t0

    # 5) ONE evaluation on the held-out period (nothing below changes the model or threshold)
    hold_prob = suppress(hold, model.predict_proba(hold), True)   # bad-sensor machines get no alarm
    final = E.evaluate(hold, hold_prob, threshold)
    final["pr_auc_95ci"] = E.bootstrap_pr_auc(hold, hold_prob)
    # Ablation: the same model trained and scored WITHOUT sensor-fault handling.
    abl_model, abl_thr, _ = fit_final(dev, chosen, handle_faults=False)
    ablation = E.evaluate(hold, abl_model.predict_proba(hold), abl_thr)
    comparison = E.heldout_comparison(dev, hold, cv, chosen)
    importance = E.shap_importance(model, hold)
    hold = hold.assign(prob=hold_prob, alarm=hold_prob >= threshold)   # needed by slices / error analysis
    leak_cv, leak_live = E.leakage_demo(dev, hold, data["work_orders"])
    E.reliability_plot(hold, chosen, final["brier"], REPORTS / "calibration.png")

    # 6) time to score one machine
    one = data["sensor_readings"].query("machine_id == 'M-001'")
    t1 = time.time()
    model.predict_proba(build_features(one, data["work_orders"], data["production_schedule"],
                                       pd.DataFrame({"machine_id": ["M-001"], "as_of": [one.ts.max().ceil("D")]})))
    score_ms = round((time.time() - t1) * 1000)

    # 7) save model + model card + experiment log
    card = {   # model card (guide Annex C); also served by GET /model/info
        "name": f"Machine failure within {HORIZON_DAYS} days", "version": f"{SHORT_NAMES[chosen]}-{run_id}",
        "algorithm": f"{chosen}, class-weighted, Platt calibration",
        "training_window": window, "validation": f"Rolling-origin time split, {N_FOLDS} folds, {GAP.days}-day gap",
        "heldout_period": f"{hold.as_of.min().date()} to {hold.as_of.max().date()}",
        "baselines": {r["model"]: {"heldout_pr_auc": r["heldout_pr_auc"]} for r in comparison[:2]},
        "final": final, "threshold": threshold,
        "cost_assumption": f"A miss costs {COST_MISS}x a false alarm",
        "top_features": list(importance.index[:5]), "time_to_failure": E.ttf_metrics(ttf, hold),
        "known_limits": "Synthetic data; few failures in the held-out period; sensor-fault handling is rule-based; "
                        "trained for a 7-day horizon only; an unseen fault type (e.g. pressure-only) may be missed.",
        "random_seed": SEED, "train_seconds": round(t_train, 1), "score_ms_one_machine": score_ms,
    }
    # Everything the API needs, saved as one file.
    joblib.dump({"model": model, "threshold": threshold, "ttf": ttf, "features": FEATURES, "model_card": card},
                MODEL_PATH)
    (REPORTS / "model_card.json").write_text(json.dumps({"model_card": card}, indent=2, default=str))
    with open(REPORTS / "experiments.jsonl", "a") as log:
        log.write(json.dumps({"run_id": run_id, "model": f"FINAL {chosen}", "data_window": window, "seed": SEED,
                              "threshold": threshold, "heldout": final, "ablation_no_fault_handling": ablation},
                             default=str) + "\n")

    # 8) report: collect every result into one dict for src/report.py
    r = data["sensor_readings"]
    write_report({
        "card": card, "chosen": chosen, "final": final, "ablation": ablation, "comparison": comparison,
        "importance": importance, "slices": E.slices(hold), "missed": E.missed_failures(hold),
        "false_alarms": E.top_false_alarms(hold), "leak_cv": leak_cv, "leak_live": leak_live,
        "eda": E.eda_table(df), "n_machines": r.machine_id.nunique(), "n_readings": len(r),
        "reading_range": f"{r.ts.min().date()} to {r.ts.max().date()}",
        "relabelled": (data["work_orders"].type == "corrective").sum()
                      - len(failure_events(data["work_orders"], data["production_schedule"])),
        "n_rows": len(df), "n_pos": int(df.label.sum()), "pos_rate": df.label.mean(),
        "n_fault": int(df.sensor_fault.notna().sum()), "n_hold": len(hold),
    })
    print(json.dumps(card, indent=2, default=str))


if __name__ == "__main__":
    main()
