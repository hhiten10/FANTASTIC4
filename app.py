"""
app.py - Streamlit dashboard with three tabs:

  Watchlist      all machines ranked by failure risk
  Machine        one machine: probability, days to failure, SHAP drivers, sensor trends
  Model quality  model card, calibration plot and the evaluation report

    streamlit run app.py        (needs the API running; set API_URL if it is not on localhost:8000)

The dashboard only talks to the API, so it shows exactly what a judge gets from the endpoints.
"""
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
REPORTS = Path(__file__).parent / "reports"

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Command Center")
# Optional prediction time; blank = score "now" (just after the latest reading).
as_of = st.sidebar.text_input("as_of (YYYY-MM-DD HH:MM, UTC; blank = latest data)", "")
watch, machine, quality = st.tabs(["Watchlist", "Machine", "Model quality"])


@st.cache_data(ttl=60)
def predict(as_of):
    """Score all machines via POST /predict (cached for 60 s so switching tabs is fast)."""
    return requests.post(f"{API}/predict", json={"as_of": as_of or None}, timeout=120).json()


try:
    res = predict(as_of)
except requests.RequestException as e:
    st.error(f"API not reachable at {API}: {e}")
    st.stop()

# ---------------------------------------------------------------- tab 1: watchlist
with watch:
    st.caption(f"as_of {res['as_of']} · model {res['model_version']} · threshold {res['threshold']} · "
               f"horizon {res['horizon_days']} days")
    df = pd.DataFrame(res["predictions"])
    # Show the driver names as one readable column.
    df["drivers"] = df.get("top_drivers", pd.Series(dtype=object)).apply(
        lambda d: ", ".join(x["feature"] for x in d) if isinstance(d, list) else "")
    df = df.sort_values("failure_prob", ascending=False, na_position="last")   # highest risk first
    cols = ["machine_id", "status", "failure_prob", "risk_band", "days_to_failure_est", "days_to_failure_note", "drivers", "data_quality"]
    st.metric("Machines flagged (high risk)", int((df.get("risk_band") == "high").sum()))
    st.dataframe(df[[c for c in cols if c in df]], hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 2: one machine
with machine:
    mid = st.selectbox("Machine", df.machine_id.tolist())
    e = requests.get(f"{API}/explain/{mid}", params={"as_of": as_of or None}, timeout=120).json()
    pr = e["prediction"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability (7 days)", "-" if pr.get("failure_prob") is None else f"{pr['failure_prob']:.0%}")
    c2.metric("Status", pr["status"])
    c3.metric("Days to failure", "-" if pr.get("days_to_failure_est") is None else
              f"{pr['days_to_failure_est']} ({pr['days_to_failure_range'][0]}-{pr['days_to_failure_range'][1]})")
    if pr.get("days_to_failure_note"):
        c3.caption(pr["days_to_failure_note"])
    st.write("Data quality:", pr.get("data_quality"))
    if pr.get("top_drivers"):
        # Positive bars push the risk up, negative bars push it down.
        st.subheader("Why (SHAP drivers)")
        st.bar_chart(pd.DataFrame(pr["top_drivers"]).set_index("feature")["contribution"])
    st.subheader("Sensor trends (daily mean, last 14 days)")
    tr = pd.DataFrame(e["sensor_trends_daily"])
    if len(tr):
        tr = tr.set_index("ts")
        for col in tr.columns:                 # one small chart per sensor
            st.line_chart(tr[[col]], height=160)

# ---------------------------------------------------------------- tab 3: model quality
with quality:
    info = requests.get(f"{API}/model/info", timeout=30).json()["model_card"]
    st.json(info, expanded=False)
    if (REPORTS / "calibration.png").exists():
        st.image(str(REPORTS / "calibration.png"), width=420)
    if (REPORTS / "evaluation_report.md").exists():
        # The report links the plot as an image file; it is already shown above, so drop the link.
        st.markdown((REPORTS / "evaluation_report.md").read_text(encoding="utf-8").replace("![reliability](calibration.png)", ""))
