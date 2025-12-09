# generate_fitpulse_sample.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_fitpulse_sample(
    out_path: str = "./fitpulse_one_day_sample.csv",
    start_time: datetime = None,
    minutes: int = 1440,
    seed: int = 42
):
    """
    Generate a 1-day (default 1440 minutes) FitPulse sample CSV with columns:
      - timestamp (YYYY-MM-DD HH:MM:SS)
      - heart_rate (BPM)
      - hrv (RMSSD-like units)
      - stress_score (0-100)
      - skin_temp (°C)
    The generator injects a few realistic variations and some intentional errors
    (missing values, wrong types, outliers) for testing preprocessing.
    """
    np.random.seed(seed)
    if start_time is None:
        # Use today's date at 10:00:00 (mirrors your original)
        now = datetime.now()
        start_time = now.replace(hour=10, minute=0, second=0, microsecond=0)

    timestamps = []
    heart_rates = []
    hrvs = []
    stress_scores = []
    skin_temps = []

    for i in range(minutes):
        ts = start_time + timedelta(minutes=i)
        timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))

        # --- Heart Rate (BPM) ---
        # baseline sinusoidal daily pattern + random noise
        base_hr = 65 + 10 * np.sin(i / 144.0 * 2 * np.pi * 3)  # slower multi-cycle wiggles
        activity_bump = 0
        # simulate short activity bursts
        if (i % 180) in range(30):  # small activity window every 3 hours
            activity_bump = 15 * np.exp(-((i % 30) / 10.0))
        noise = np.random.normal(0, 3)
        hr = base_hr + activity_bump + noise
        hr = float(np.clip(hr, 40, 180))
        heart_rates.append(round(hr))

        # --- HRV (RMSSD-like) ---
        # HRV inversely correlated with HR: higher HR -> lower HRV
        hrv_base = 60 - (hr - 60) * 0.5 + np.random.normal(0, 5)
        hrv = float(max(5, min(120, hrv_base)))
        hrvs.append(round(hrv, 1))

        # --- Stress score (0-100) ---
        # Higher when HR is high and HRV low; scaled and clipped
        stress = (hr - 50) * 0.6 + (80 - hrv) * 0.4 + np.random.normal(0, 5)
        stress = float(np.clip(stress, 0, 100))
        stress_scores.append(round(stress, 1))

        # --- Skin temperature (°C) ---
        # base human skin temp ~32.5-34, slight drift + occasional spikes (fever)
        temp = 33.0 + 0.3 * np.sin(i / 200.0 * 2 * np.pi) + np.random.normal(0, 0.2)
        # small elevation during activity windows
        if activity_bump > 5:
            temp += 0.4
        skin_temps.append(round(temp, 1))

    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": heart_rates,
        "hrv": hrvs,
        "stress_score": stress_scores,
        "skin_temp": skin_temps
    })

    # ----- Inject intentional preprocessing errors / anomalies -----
    # 1) Some non-numeric text values in heart_rate
    for idx in range(0, minutes, 240):  # every 4 hours
        df.loc[idx, "heart_rate"] = "ERROR"

    # 2) Missing HRV values occasionally
    for idx in range(50, minutes, 300):
        df.loc[idx: idx+2, "hrv"] = ""   # empty string

    # 3) String "NaN" in stress_score sometimes
    for idx in range(0, minutes, 500):
        df.loc[idx, "stress_score"] = "NaN"

    # 4) Unrealistic skin temp spikes (outliers)
    for idx in range(125, minutes, 375):
        df.loc[idx, "skin_temp"] = 99.9

    # 5) Negative values in stress_score occasionally (bad sensor)
    for idx in range(200, minutes, 450):
        df.loc[idx, "stress_score"] = -10

    # 6) Randomly drop a couple of timestamps to create missing rows (simulate packet loss)
    drop_idxs = np.random.choice(np.arange(10, minutes-10), size=6, replace=False)
    df = df.drop(index=drop_idxs).reset_index(drop=True)

    # Save CSV
    df.to_csv(out_path, index=False)
    print(f"Created sample dataset at: {out_path} (rows: {len(df)})")
    return df


if __name__ == "__main__":
    df = generate_fitpulse_sample()
    print(df.head(10))
