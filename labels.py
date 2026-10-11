"""
labels.py - step 3: the target "will this machine have an unplanned failure in the next 7 days?".

A failure is a corrective work order. Corrective orders opened during a planned changeover or
planned maintenance are mislogged planned stops (guide 4.4 trap), so they are NOT failures.
Only the opening time of the order is used - never the note text, which is written afterwards.
"""
import pandas as pd

from .config import DAY, HORIZON_DAYS


def failure_events(work_orders: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unplanned failure: [machine_id, failure_ts]."""
    wo = work_orders[work_orders["type"].str.lower() == "corrective"]
    planned = schedule[schedule["kind"].isin(["changeover", "planned_maintenance"])]
    # Pair every corrective order with every planned stop of the same machine and find
    # the orders that were opened inside a planned stop window.
    m = wo.merge(planned[["machine_id", "start", "end"]], on="machine_id")
    planned_ids = m.loc[(m.opened_at >= m.start) & (m.opened_at <= m.end), "wo_id"]
    ev = wo[~wo.wo_id.isin(planned_ids)]
    return ev[["machine_id", "opened_at"]].rename(columns={"opened_at": "failure_ts"}).sort_values("failure_ts")


def add_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add the target to rows of [machine_id, as_of].

    New columns:
      next_failure     time of the machine's next failure at or after as_of
      days_to_failure  days from as_of to that failure (used by the days-to-failure model)
      label            1 if the failure is within HORIZON_DAYS (7) days, else 0
    """
    # merge_asof "forward" finds, for each row, the first failure at or after as_of.
    out = pd.merge_asof(df.sort_values("as_of"), events.rename(columns={"failure_ts": "next_failure"}),
                        left_on="as_of", right_on="next_failure", by="machine_id", direction="forward")
    out["days_to_failure"] = (out.next_failure - out.as_of) / DAY
    out["label"] = (out.days_to_failure < HORIZON_DAYS).astype(int)   # no next failure (NaN) -> 0
    return out
