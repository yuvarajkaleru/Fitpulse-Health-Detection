# preprocess_single_output.py
import pandas as pd
import numpy as np
import os

# -------------------------
# Config — YOUR EXACT PATHS
# -------------------------
INPUT_PATH = r"C:\Users\yuvar\Desktop\fit pulse\fitpulse_one_day_sample.csv"
OUTPUT_PATH = r"C:\Users\yuvar\Desktop\fit pulse\fitpulse_one_day_sample_preprocessed.csv"

FINAL_COLS = ["timestamp", "heart_rate", "hrv", "stress_score", "skin_temp"]

def parse_timestamps(df):
    df = df.copy()
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp_parsed"]).reset_index(drop=True)
    return df

def coerce_numeric(df):
    df = df.copy()
    for col in ["heart_rate", "hrv", "stress_score", "skin_temp"]:
        df[col] = df[col].replace({"": np.nan, "NaN": np.nan, "nan": np.nan, "ERROR": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def remove_domain_outliers(df):
    df = df.copy()

    # heart rate realistic range
    df.loc[(df["heart_rate"] < 30) | (df["heart_rate"] > 220), "heart_rate"] = np.nan

    # hrv realistic
    df.loc[(df["hrv"] < 0) | (df["hrv"] > 300), "hrv"] = np.nan

    # stress should not be negative or >150
    df.loc[(df["stress_score"] < 0) | (df["stress_score"] > 150), "stress_score"] = np.nan

    # skin temp realistic range + remove 99.9 spikes
    df.loc[(df["skin_temp"] < 25) | (df["skin_temp"] > 45) | (df["skin_temp"] == 99.9), "skin_temp"] = np.nan

    return df

def impute_existing_rows(df):
    df = df.copy()

    df = df.set_index("timestamp_parsed")

    for col in ["heart_rate", "hrv", "stress_score", "skin_temp"]:
        df[col] = df[col].interpolate(method="time", limit_direction="both")
        df[col] = df[col].fillna(df[col].median())

    df = df.reset_index()
    df["timestamp"] = df["timestamp_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

def smooth_heart_rate(df):
    df = df.copy()
    df = df.set_index("timestamp_parsed")
    df["heart_rate"] = df["heart_rate"].rolling(window=3, center=True, min_periods=1).median()
    df = df.reset_index()
    return df

def preprocess():
    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    # Ensure all required columns exist
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Pipeline
    df = parse_timestamps(df)
    df = coerce_numeric(df)
    df = remove_domain_outliers(df)
    df = impute_existing_rows(df)
    df = smooth_heart_rate(df)

    # Final output containing ONLY your 5 natural columns
    df_out = df[FINAL_COLS].copy()

    df_out.to_csv(OUTPUT_PATH, index=False)
    print("Saved cleaned dataset to:")
    print(OUTPUT_PATH)
    print(df_out.head())

if __name__ == "__main__":
    preprocess()
