import pandas as pd
import numpy as np
import os

def run_preprocessing():
    """
    Preprocess unified fitness dataset:
    1. Load the unified raw dataset (with missing values)
    2. Detect and report missing values
    3. Handle missing values via interpolation or forward fill
    4. Validate data
    5. Save cleaned dataset
    """
    
    # Ensure output folder exists
    os.makedirs("data", exist_ok=True)

    # Load unified dataset (with missing values)
    try:
        df = pd.read_csv("data/raw/unified_fitness_data.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except FileNotFoundError:
        raise FileNotFoundError("Unified dataset not found. Run data generation first.")
    
    # ---- DETECT AND REPORT MISSING VALUES ----
    print("\n📊 Missing Value Report:")
    print("=" * 50)
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100
    
    for col in df.columns:
        if missing_summary[col] > 0:
            print(f"  {col}: {missing_summary[col]} missing ({missing_pct[col]:.2f}%)")
    
    total_missing = missing_summary.sum()
    total_cells = len(df) * len(df.columns)
    overall_pct = (total_missing / total_cells) * 100
    print(f"\n  Total missing: {total_missing} cells out of {total_cells} ({overall_pct:.2f}%)")
    print("=" * 50 + "\n")
    
    # ---- HANDLE MISSING VALUES ----
    # Strategy: For time-series data, use interpolation
    # For steps/calories: interpolate (continuous values)
    # For sleep_minutes/sleep_hours: interpolate (distributed sleep data)
    # For stress_level: forward fill (categorical tendency)
    
    # Interpolate steps (linear interpolation for activity data)
    df["steps"] = df["steps"].interpolate(method='linear', limit_direction='both')
    
    # Interpolate calories_burned
    df["calories_burned"] = df["calories_burned"].interpolate(method='linear', limit_direction='both')
    
    # Interpolate sleep_minutes and sleep_hours
    df["sleep_minutes"] = df["sleep_minutes"].interpolate(method='linear', limit_direction='both')
    df["sleep_hours"] = df["sleep_hours"].interpolate(method='linear', limit_direction='both')
    
    # Forward fill stress_level (maintain physiological state)
    df["stress_level"] = df["stress_level"].ffill().bfill()
    
    # Final safety check: fill any remaining NaN with 0 (shouldn't happen)
    df = df.fillna(0)
    
    # ---- VALIDATE CLEANED DATA ----
    print("✅ Data Cleaning Complete:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  No missing values: {df.isnull().sum().sum() == 0}")
    print()
    
    # Save final cleaned dataset
    df.to_csv("data/cleaned_fitness_data.csv", index=False)
    
    print("✔ Preprocessing complete — cleaned_fitness_data.csv generated")
