"""
predict.py - score every machine with the saved model and save the result as JSON.

Uses the same code path as POST /predict (src/serving.py), so the file matches what the API returns.
Used for the "predictions file for the late data drop" deliverable.

    python predict.py                                   # as_of = just after the latest reading
    python predict.py --as-of "2026-10-11 00:00" --out reports/late_drop_predictions.json
"""
import argparse
import json

from src.config import REPORTS
from src.serving import Predictor


def main():
    """Parse arguments, score all machines, write the JSON file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of")
    ap.add_argument("--out", default=str(REPORTS / "late_drop_predictions.json"))
    a = ap.parse_args()
    res = Predictor().score(as_of=a.as_of)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    # Print the 10 highest-risk machines as a quick check.
    for x in sorted(res["predictions"], key=lambda x: -(x["failure_prob"] or 0))[:10]:
        print(x["machine_id"], x["status"], x["failure_prob"], x.get("risk_band"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
