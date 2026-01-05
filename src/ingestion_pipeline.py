import pandas as pd
import json
import os

class DataIngestionEngine:

    def __init__(self):
        self.supported = [".csv", ".json"]

    def detect_format(self, file):
        return os.path.splitext(file)[1].lower()

    def load(self, file_path):
        ext = self.detect_format(file_path)

        if ext not in self.supported:
            raise ValueError(f"Unsupported format {ext}")

        print(f"📥 Loading {file_path}")

        if ext == ".csv":
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=["timestamp"])  # remove bad timestamps
            return df

        if ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            return pd.json_normalize(data, record_path="cycles")

