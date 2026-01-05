import pandas as pd

def daily_comparison(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    summary = df.groupby("date").agg({
        "heart_rate": "mean",
        "steps": "sum",
        "sleep_hours": "mean"
    }).reset_index()

    summary.columns = [
        "date",
        "avg_heart_rate",
        "total_steps",
        "avg_sleep_hours"
    ]

    return summary
