import pandas as pd
from tsfresh import extract_features

def extract_tsfresh_features(df, signal):
    data = df[["timestamp", signal]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["id"] = 1

    data = data.rename(columns={
        "timestamp": "time",
        signal: "value"
    })

    features = extract_features(
        data,
        column_id="id",
        column_sort="time",
        disable_progressbar=True
    )

    return features
