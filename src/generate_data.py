import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir("data/raw")

# ======================== GENERATE UNIFIED FITNESS DATASET ========================
def generate_realistic_sample_data():
    """
    Generate unified fitness data for 30 days with natural patterns:
    - ONE unified dataset with all metrics (timestamp, heart_rate, stress_level, steps, calories_burned, sleep_minutes, sleep_hours)
    - Heart rate: Varies by activity, rest, stress
    - Steps: Daily activity patterns
    - Sleep: Realistic sleep duration (non-zero only during sleep hours, 0 during day)
    - ANOMALIES: Controlled synthetic anomalies in heart rate
    - MISSING VALUES: Inject 2-5% random NaN in steps, calories_burned, sleep_minutes, stress_level (NOT in timestamp or heart_rate)
    """
    
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=29)
    
    unified_data = []
    
    # ---- PLAN ANOMALY INJECTION ----
    anomaly_windows = [
        {'day': 5, 'hour': 12, 'duration_min': 35, 'type': 'spike'},    # Heart rate spike
        {'day': 15, 'hour': 14, 'duration_min': 35, 'type': 'drop'},    # Heart rate drop
        {'day': 25, 'hour': 16, 'duration_min': 35, 'type': 'spike'},   # Another spike
    ]
    
    # Generate daily step and sleep values first
    daily_steps_map = {}
    daily_sleep_minutes_map = {}
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Daily steps (will be applied to all timestamps for that day)
        activity_level = np.random.choice(['low', 'moderate', 'high'], p=[0.3, 0.5, 0.2])
        if activity_level == 'low':
            daily_steps = int(np.random.normal(5000, 800))
        elif activity_level == 'moderate':
            daily_steps = int(np.random.normal(8500, 1500))
        else:
            daily_steps = int(np.random.normal(12000, 2000))
        daily_steps = max(1000, min(daily_steps, 20000))
        daily_steps_map[date_str] = daily_steps
        
        # Daily sleep (realistic 6-9 hours per night)
        sleep_duration_hours = np.random.normal(7.5, 0.8)
        sleep_duration_hours = max(6, min(sleep_duration_hours, 9))
        daily_sleep_minutes_map[date_str] = sleep_duration_hours * 60
    
    # Generate heart rate data at 15-min intervals for 30 days
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Get daily stats
        daily_steps = daily_steps_map[date_str]
        daily_sleep_minutes = daily_sleep_minutes_map[date_str]
        calories_burned = round(daily_steps * 0.04, 2)
        
        # Check for anomaly windows on this day
        anomalies_on_this_day = [a for a in anomaly_windows if a['day'] == day]
        
        # Morning (6 AM - 9 AM - resting)
        base_hr_morning = np.random.normal(62, 3)
        for hour in range(6, 9):
            for minute in [0, 15, 30, 45]:
                timestamp = current_date.replace(hour=hour, minute=minute)
                hr = int(base_hr_morning + np.random.normal(0, 2))
                hr = max(50, min(hr, 120))
                
                # Check anomaly windows
                is_anomaly_time = False
                for anomaly in anomalies_on_this_day:
                    anomaly_start_min = anomaly['hour'] * 60
                    anomaly_end_min = anomaly_start_min + anomaly['duration_min']
                    current_min = hour * 60 + minute
                    if anomaly_start_min <= current_min < anomaly_end_min:
                        is_anomaly_time = True
                        if anomaly['type'] == 'spike':
                            hr = int(np.random.uniform(140, 160))
                        else:
                            hr = int(np.random.uniform(35, 40))
                        break
                
                stress_level = int(np.interp(hr, [35, 160], [5, 95])) if is_anomaly_time else int(np.interp(hr, [50, 120], [10, 80]))
                
                # Sleep: 0 during non-sleep hours (6 AM - 10 PM)
                sleep_minutes = 0
                
                unified_data.append({
                    "timestamp": timestamp,
                    "heart_rate": hr,
                    "stress_level": stress_level,
                    "steps": daily_steps,
                    "calories_burned": calories_burned,
                    "sleep_minutes": sleep_minutes,
                    "sleep_hours": 0.0
                })
        
        # Daytime (9 AM - 6 PM - active)
        base_hr_day = np.random.normal(75, 5)
        for hour in range(9, 18):
            for minute in [0, 15, 30, 45]:
                timestamp = current_date.replace(hour=hour, minute=minute)
                activity_boost = np.random.normal(0, 8)
                hr = int(base_hr_day + activity_boost)
                hr = max(55, min(hr, 140))
                
                is_anomaly_time = False
                for anomaly in anomalies_on_this_day:
                    anomaly_start_min = anomaly['hour'] * 60
                    anomaly_end_min = anomaly_start_min + anomaly['duration_min']
                    current_min = hour * 60 + minute
                    if anomaly_start_min <= current_min < anomaly_end_min:
                        is_anomaly_time = True
                        if anomaly['type'] == 'spike':
                            hr = int(np.random.uniform(140, 160))
                        else:
                            hr = int(np.random.uniform(35, 40))
                        break
                
                stress_level = int(np.interp(hr, [35, 160], [5, 95])) if is_anomaly_time else int(np.interp(hr, [55, 140], [15, 85]))
                
                # Sleep: 0 during day (9 AM - 6 PM)
                sleep_minutes = 0
                
                unified_data.append({
                    "timestamp": timestamp,
                    "heart_rate": hr,
                    "stress_level": stress_level,
                    "steps": daily_steps,
                    "calories_burned": calories_burned,
                    "sleep_minutes": sleep_minutes,
                    "sleep_hours": 0.0
                })
        
        # Evening & Night (6 PM - 6 AM next day - resting + sleeping)
        base_hr_night = np.random.normal(60, 3)
        for hour in list(range(18, 24)) + list(range(0, 6)):
            for minute in [0, 15, 30, 45]:
                if hour < 6 or hour >= 18:
                    timestamp = (current_date + timedelta(days=1 if hour < 6 else 0)).replace(hour=hour % 24, minute=minute)
                    hr = int(base_hr_night + np.random.normal(0, 2))
                    hr = max(45, min(hr, 100))
                    
                    stress_level = int(np.interp(hr, [45, 100], [5, 60]))
                    
                    # Sleep: Non-zero during sleep hours (10 PM - 6 AM typical sleep window)
                    # Sleep minutes distributed across the sleep period
                    if (hour >= 22 or hour < 6):
                        # During sleep period - distribute daily sleep minutes across readings
                        # Sleep period is roughly 8 hours = 32 readings of 15 min each
                        sleep_minutes = daily_sleep_minutes / 32  # Distribute evenly
                        sleep_minutes = max(0, sleep_minutes + np.random.normal(0, 2))  # Add variation
                    else:
                        # During evening (6 PM - 10 PM) - no sleep
                        sleep_minutes = 0
                    
                    sleep_hours = sleep_minutes / 60
                    
                    unified_data.append({
                        "timestamp": timestamp,
                        "heart_rate": hr,
                        "stress_level": stress_level,
                        "steps": daily_steps,
                        "calories_burned": calories_burned,
                        "sleep_minutes": sleep_minutes,
                        "sleep_hours": sleep_hours
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(unified_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # ---- INJECT CONTROLLED MISSING VALUES (2-5% random NaN) ----
    # NEVER in timestamp or heart_rate
    # ONLY in: steps, calories_burned, sleep_minutes, stress_level
    missing_rate = np.random.uniform(0.02, 0.05)  # 2-5% missing
    
    np.random.seed(None)  # Random seed for variety
    n_rows = len(df)
    
    # Missing values in steps
    missing_indices = np.random.choice(n_rows, size=int(n_rows * missing_rate), replace=False)
    df.loc[missing_indices, 'steps'] = np.nan
    
    # Missing values in calories_burned
    missing_indices = np.random.choice(n_rows, size=int(n_rows * missing_rate), replace=False)
    df.loc[missing_indices, 'calories_burned'] = np.nan
    
    # Missing values in sleep_minutes
    missing_indices = np.random.choice(n_rows, size=int(n_rows * missing_rate), replace=False)
    df.loc[missing_indices, 'sleep_minutes'] = np.nan
    df.loc[missing_indices, 'sleep_hours'] = np.nan  # Also nullify sleep_hours when sleep_minutes is missing
    
    # Missing values in stress_level
    missing_indices = np.random.choice(n_rows, size=int(n_rows * missing_rate), replace=False)
    df.loc[missing_indices, 'stress_level'] = np.nan
    
    # Save unified dataset (with missing values) to CSV
    df.to_csv("data/raw/unified_fitness_data.csv", index=False)
    
    return df


# ======================== LEGACY FUNCTIONS (kept for compatibility) ========================
def generate_heart_rate_csv():
    """Legacy function - returns unified dataset"""
    return generate_realistic_sample_data()

def generate_steps_csv():
    """Legacy function - returns unified dataset"""
    return generate_realistic_sample_data()

def generate_sleep_json():
    """Legacy function - returns unified dataset"""
    return generate_realistic_sample_data()
    return result["sleep"]
