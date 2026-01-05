# src/utils.py
import pandas as pd
import json
from pathlib import Path

def load_csv(path: Path):
    return pd.read_csv(path)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Try to normalize common structures
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    return pd.json_normalize(data)

def safe_to_datetime(df: pd.DataFrame, col_candidates=("timestamp", "time", "datetime")):
    for col in col_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return df, col
    # If none found, raise
    raise ValueError(f"No timestamp column found. Candidates tried: {col_candidates}")

def drop_and_report_duplicates(df: pd.DataFrame, subset=None):
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    return df, before - after
