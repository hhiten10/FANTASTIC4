"""
evaluation.py - everything measured on the held-out period (guide section 7).

Nothing here influences a modelling choice: the model, calibration and threshold are fixed
before these functions run. They only measure and explain.
"""
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from .config import DAY, GAP, SEED
from .features import FEATURES
from .models import CLASSIFIERS, make_classifier, shap_values
from .training import raw_scores, suppress, time_folds

matplotlib.use("Agg")                     # draw plots to files, no window
import matplotlib.pyplot as plt  # noqa: E402


def evaluate(df: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict:
    """Row-level and failure-level metrics.

    pr_auc                          ranking quality on a rare event
    recall_at_precision_0.5         share of failure rows caught while half the alarms are right
    brier                           calibration error (lower is better)
    failures / failures_caught      failure events, and how many got an alarm in the 7 days before
    median_lead_time_days           days between the first alarm and the failure
    false_alarms_per_machine_month  workload created by wrong alarms
    """
    y, alarm = df.label.to_numpy(), prob >= threshold
    prec, rec, _ = precision_recall_curve(y, prob)
    # One failure event = one (machine, next_failure) pair; its rows are the 7 days before it.
    events = df[df.label == 1].assign(alarm=alarm[df.label.to_numpy() == 1])
    lead = []
    for _, e in events.groupby(["machine_id", "next_failure"]):
        if e.alarm.any():
            lead.append((e.next_failure.iloc[0] - e.loc[e.alarm, "as_of"].min()) / DAY)
    n_events = events.groupby(["machine_id", "next_failure"]).ngroups
    return {
        "pr_auc": round(average_precision_score(y, prob), 3),
        "recall_at_precision_0.5": round(float(rec[prec >= 0.5].max()) if (prec >= 0.5).any() else 0.0, 3),
        "brier": round(brier_score_loss(y, prob), 4),
        "precision_at_threshold": round(float((y[alarm] == 1).mean()) if alarm.any() else 0.0, 3),
        "row_recall_at_threshold": round(float(alarm[y == 1].mean()), 3),
        "failures": n_events,
        "failures_caught": len(lead),
        "median_lead_time_days": round(float(np.median(lead)), 1) if lead else None,
        "false_alarms_per_machine_month": round(float((alarm & (y == 0)).sum() / (len(df) / 30)), 2),
    }


def bootstrap_pr_auc(df: pd.DataFrame, prob: np.ndarray, n: int = 200) -> list:
    """95% confidence interval of PR-AUC, by resampling whole machines n times."""
    rng, groups = np.random.default_rng(SEED), df.groupby("machine_id").indices
    ids, y, vals = list(groups), df.label.to_numpy(), []
    for _ in range(n):
        idx = np.concatenate([groups[m] for m in rng.choice(ids, len(ids))])
        if y[idx].sum():                          # PR-AUC needs at least one failure
            vals.append(average_precision_score(y[idx], prob[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def heldout_comparison(dev, hold, cv: dict, chosen: str) -> list[dict]:
    """Model comparison table: CV score and held-out score of every baseline/model.

    Each model is trained on dev and scored on the held-out period. For reporting only;
    the choice was already made from the CV scores.
    """
    rows = []
    for name in ["threshold_rule"] + CLASSIFIERS:
        if name == "threshold_rule":
            _, s = raw_scores(name, dev, hold)
        else:
            tr = dev[dev.sensor_fault.isna()]
            s = make_classifier(name, tr).fit(tr[FEATURES], tr.label).predict_proba(hold[FEATURES])[:, 1]
        s = suppress(hold, s, True)
        rows.append({"model": name, "cv_pr_auc": f"{np.mean(cv[name]):.3f} ± {np.std(cv[name]):.3f}",
                     "heldout_pr_auc": round(average_precision_score(hold.label, s), 3),
                     "chosen": "✔" if name == chosen else ""})
    return rows


def ttf_metrics(ttf: dict, hold: pd.DataFrame) -> dict:
    """Days-to-failure quality on held-out failure rows: mean absolute error and range coverage."""
    pos = hold[hold.label == 1]
    lo, mid, hi = (ttf[q].predict(pos[FEATURES]) for q in (0.1, 0.5, 0.9))
    return {"mae_days": round(float(np.abs(mid - pos.days_to_failure).mean()), 2),
            # share of true values inside the predicted 10-90% range (target 0.8)
            "coverage_10_90": round(float(((pos.days_to_failure >= lo) & (pos.days_to_failure <= hi)).mean()), 2)}


def shap_importance(model, hold: pd.DataFrame) -> pd.Series:
    """Global feature importance = mean |SHAP value| over the held-out rows, largest first."""
    return pd.Series(np.abs(shap_values(model, hold)).mean(0), index=FEATURES).sort_values(ascending=False)


def slices(hold: pd.DataFrame) -> list[dict]:
    """Model quality per machine type and per criticality. `hold` needs `prob` and `alarm` columns."""
    out = []
    for col in ["machine_type", "criticality"]:
        for v, g in hold.groupby(col):
            out.append({"slice": f"{col}={v}", "rows": len(g), "failure_rows": int(g.label.sum()),
                        "pr_auc": round(average_precision_score(g.label, g.prob), 3) if g.label.any() else "n/a",
                        "recall_at_threshold": round(g.loc[g.label == 1, "alarm"].mean(), 2) if g.label.any() else "n/a"})
    return out


def missed_failures(hold: pd.DataFrame) -> list[dict]:
    """Error analysis: failures with no alarm in the 7 days before them, with their sensor signals."""
    ev = hold[hold.label == 1].groupby(["machine_id", "next_failure"])
    return [{"machine": m, "failure": str(f.date()), "max_prob": round(g.prob.max(), 3),
             "vib_vs_base_3d": round(g.vib_vs_base_3d.iloc[-1], 2), "temp_vs_base_3d": round(g.temp_vs_base_3d.iloc[-1], 2),
             "sensor_fault": g.sensor_fault.dropna().iloc[0] if g.sensor_fault.notna().any() else ""}
            for (m, f), g in ev if not g.alarm.any()]


def top_false_alarms(hold: pd.DataFrame, n: int = 5) -> pd.Series:
    """Machines with the most false alarms (alarm raised, no failure within 7 days)."""
    return hold[hold.alarm & (hold.label == 0)].groupby("machine_id").size().sort_values(ascending=False).head(n)


def leakage_demo(dev, hold, work_orders) -> tuple[float, float]:
    """Show the leakage trap from guide Annex A. Returns (CV PR-AUC, held-out PR-AUC).

    We add a forbidden feature "a repair note naming the cause exists within 7 days". In training
    it predicts failure almost perfectly, because the note is written after the failure. At real
    prediction time the note does not exist yet, so on the held-out period the score collapses.
    This model is only for the report - it is never saved or used.
    """
    notes = work_orders.query("type == 'corrective' and note.notna()")[["machine_id", "opened_at"]].sort_values("opened_at")

    def note_flag(df):
        # 1 if the machine has a corrective order with a note within the next 7 days (future data!)
        m = pd.merge_asof(df.sort_values("as_of")[["machine_id", "as_of"]].reset_index(), notes,
                          left_on="as_of", right_on="opened_at", by="machine_id", direction="forward")
        return ((m.opened_at - m.as_of) < GAP).set_axis(m["index"]).reindex(df.index).astype(float)

    feats = FEATURES + ["note_names_cause"]
    tr = dev.assign(note_names_cause=note_flag(dev))
    _, va = list(time_folds(dev))[-1]                         # use the last CV fold
    fit_rows = tr.loc[tr.as_of < tr.loc[va, "as_of"].min() - GAP]
    m = make_classifier("logistic_regression", fit_rows).fit(fit_rows[feats], fit_rows.label)
    cv = average_precision_score(tr.loc[va, "label"], m.predict_proba(tr.loc[va, feats])[:, 1])
    live = hold.assign(note_names_cause=0.0)     # at prediction time the note has not been written
    return round(cv, 3), round(average_precision_score(live.label, m.predict_proba(live[feats])[:, 1]), 3)


def eda_table(df: pd.DataFrame) -> list[dict]:
    """EDA: average sensor signals by days before failure, compared with normal days."""
    cols = ["vib_vs_base_3d", "temp_vs_base_3d", "vib_slope_7d"]
    pre = df[df.label == 1].assign(days_before=lambda x: np.floor(x.days_to_failure).astype(int))
    rows = [{"days_before_failure": d, "rows": len(g), **{c: round(g[c].mean(), 3) for c in cols}}
            for d, g in pre.groupby("days_before")]
    normal = df[df.label == 0]
    rows.append({"days_before_failure": "normal days", "rows": len(normal), **{c: round(normal[c].mean(), 3) for c in cols}})
    return rows


def reliability_plot(hold: pd.DataFrame, label: str, brier: float, path) -> None:
    """Calibration plot: predicted probability vs observed failure rate (diagonal = perfect)."""
    frac, mean_pred = calibration_curve(hold.label, hold.prob, n_bins=8, strategy="quantile")
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    plt.plot(mean_pred, frac, "o-", label=label)
    plt.xlabel("predicted probability"); plt.ylabel("observed failure rate")
    plt.title(f"Reliability (held-out), Brier {brier}"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
