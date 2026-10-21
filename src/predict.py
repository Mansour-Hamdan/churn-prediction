"""Score customers with the saved model.

Usage:
    python src/predict.py path/to/customers.csv

The CSV needs the same columns as the training data (Churn is optional).
Prints each customer's churn probability, highest risk first.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"


def main(csv_path: str) -> None:
    if not MODEL_PATH.exists():
        sys.exit("No saved model found. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(csv_path)
    ids = df["customerID"] if "customerID" in df else df.index
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    out = pd.DataFrame(
        {"customerID": ids, "churn_probability": model.predict_proba(X)[:, 1].round(3)}
    ).sort_values("churn_probability", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
