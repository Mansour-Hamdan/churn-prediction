"""Loading and cleaning the Telco customer churn dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
TARGET = "Churn"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a cleaned DataFrame.

    Cleaning steps:
    - TotalCharges is stored as text and is blank for 11 customers with
      tenure 0 (they have not been billed yet), so those become 0.
    - customerID is an identifier, not a feature, so it is dropped.
    - Churn is mapped from Yes/No to 1/0.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["customerID"])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y)."""
    return df.drop(columns=[TARGET]), df[TARGET]
