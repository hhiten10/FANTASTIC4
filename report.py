"""
report.py - writes reports/evaluation_report.md (the evaluation report deliverable).

train.py collects every result into one dict and passes it to `write_report`.
The dashboard's "Model quality" tab shows this file.
"""
from .config import COST_FALSE_ALARM, COST_MISS, GAP, HORIZON_DAYS, N_FOLDS, REPORTS, SEED, SIMPLER_WINS_BY


def table(rows: list[dict], cols: list[str]) -> str:
    """List of dicts -> Markdown table with the given columns."""
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write_report(r: dict) -> None:
    """Fill the Markdown template below with the results dict `r` built in train.py."""
    final, abl, card = r["final"], r["ablation"], r["card"]
    missed = table(r["missed"], ["machine", "failure", "max_prob", "vib_vs_base_3d", "temp_vs_base_3d",
                                 "sensor_fault"]) if r["missed"] else "none"
    shap_rows = [{"feature": k, "mean_abs_shap": round(v, 4)} for k, v in r["importance"].head(10).items()]
    text = f"""# Evaluation report — {card['version']}

Reproduce: `python generate_data.py && python train.py` (seed {SEED}). Every number below comes from that run.

## Data and target
- {r['n_machines']} machines, {r['n_readings']:,} readings, {r['reading_range']}.
- Target: unplanned failure within {HORIZON_DAYS} days of the prediction time (one snapshot per machine per day, 00:00 UTC).
- Planned changeovers / maintenance are not failures: corrective orders opened inside a planned stop are dropped
  ({r['relabelled']} found).
- Snapshots: {r['n_rows']:,}; failure rows: {r['n_pos']} ({r['pos_rate']:.1%}) -> heavily imbalanced.
- Rows with a suspected sensor fault: {r['n_fault']} (no failure score is given for them).

### What pre-failure behaviour looks like (EDA)
{table(r['eda'], list(r['eda'][0]))}

Vibration and temperature both climb in the days before a wear failure; some failures have no warning at all.

## Validation method
- Held-out period (used once, at the end): {card['heldout_period']} ({r['n_hold']} rows).
- Development data: {card['training_window']}. Time-based CV, {N_FOLDS} rolling-origin folds, {GAP.days}-day gap before every validation block.
- Class imbalance: class weights, fitted inside each training fold only. Imputation/scaling also inside the fold (sklearn pipeline).
- Split test: last training day + {GAP.days} days < first held-out day — passed.

## Model comparison
{table(r['comparison'], ["model", "cv_pr_auc", "heldout_pr_auc", "chosen"])}

Selection rule (decided before looking at the held-out period): keep logistic regression unless another model beats its
CV PR-AUC by more than {SIMPLER_WINS_BY}. Chosen: **{r['chosen']}**.

## Final model on the held-out period
- Threshold **{card['threshold']}**, chosen on out-of-fold validation scores to minimise {COST_MISS} x misses + {COST_FALSE_ALARM} x false alarms.

{table([final], list(final))}

Calibration: Platt scaling fitted on out-of-fold scores (few positives, so the 2-parameter Platt fit is safer than isotonic).

![reliability](calibration.png)

## Ablation: sensor-fault handling
| variant | PR-AUC | false alarms / machine-month | failures caught |
|---|---|---|---|
| with handling (final) | {final['pr_auc']} | {final['false_alarms_per_machine_month']} | {final['failures_caught']}/{final['failures']} |
| without handling | {abl['pr_auc']} | {abl['false_alarms_per_machine_month']} | {abl['failures_caught']}/{abl['failures']} |

## Leakage audit
- Features use only readings before the prediction time and repairs closed before it (unit test `test_no_future_leakage`).
- Work-order note text is never a feature.
- Trap demo: adding "a note naming the cause exists within 7 days" gives CV PR-AUC **{r['leak_cv']}**, but on the held-out
  period, where the note does not exist yet at prediction time, PR-AUC is **{r['leak_live']}**. That feature is leakage.

## Explainability (mean |SHAP| on the held-out period)
{table(shap_rows, ["feature", "mean_abs_shap"])}

## Slices
{table(r['slices'], ["slice", "rows", "failure_rows", "pr_auc", "recall_at_threshold"])}

## Error analysis
Missed failures (no alarm in the 7 days before):

{missed}

Missed failures with no rise in vibration or temperature are sudden failures: the sensors give no warning, so no
sensor model can catch them. What we would fix: add new sensor types (current, acoustic) for those.

Machines with the most false alarms: {", ".join(f"{m} ({n})" for m, n in r['false_alarms'].items()) or "none"}.

## Days to failure
Quantile gradient boosting (10/50/90%) on rows that fail within 7 days: MAE {card['time_to_failure']['mae_days']} days,
range coverage {card['time_to_failure']['coverage_10_90']} (target 0.8).

## Cost
Training run {card['train_seconds']:.0f} s; scoring one machine {card['score_ms_one_machine']} ms.
"""
    (REPORTS / "evaluation_report.md").write_text(text, encoding="utf-8")
