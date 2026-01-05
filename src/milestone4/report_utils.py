import pandas as pd

def generate_summary_report(df):
    report = {
        "Average Heart Rate": round(df["heart_rate"].mean(), 2),
        "Max Heart Rate": int(df["heart_rate"].max()),
        "Average Steps": int(df["steps"].mean()),
        "Average Sleep Hours": round(df["sleep_hours"].mean(), 2),
        "Total Records": len(df)
    }
    return pd.DataFrame(report.items(), columns=["Metric", "Value"])
