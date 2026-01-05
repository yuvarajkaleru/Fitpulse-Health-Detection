import pandas as pd
import numpy as np


def detect_anomalies(df):
    """
    Detect heart rate anomalies using rolling statistics with robust baselines.
    
    Algorithm:
    1. Compute global mean and std (baseline) excluding obvious outliers
    2. Compute rolling mean and std over a small 15-30 min window
    3. Flag anomaly if |hr - rolling_mean| > 3 * rolling_std OR |hr - global_mean| > 3 * global_std
    4. Require minimum 2 consecutive readings to confirm anomaly
    5. Assign severity levels based on std deviations
    
    Returns:
    --------
    pd.DataFrame with columns:
        - anomaly: 0 (normal) or 1 (anomaly detected)
        - severity: 'Normal', 'Mild', 'Moderate', 'Critical'
        - rolling_mean, rolling_std: context statistics
        - std_deviations: distance from rolling mean in std units
    """
    
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Step 1: Calculate global statistics (baseline)
    # Use robust estimation - exclude extreme outliers (>4σ from median)
    q1 = df['heart_rate'].quantile(0.25)
    q3 = df['heart_rate'].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Filter outliers for baseline calculation
    baseline_data = df[(df['heart_rate'] >= lower_fence) & (df['heart_rate'] <= upper_fence)]['heart_rate']
    global_mean = baseline_data.mean()
    global_std = baseline_data.std()
    
    # Step 2: Calculate rolling statistics (smaller window for sensitivity)
    # Window size: 2-4 readings (30-60 minutes for 15-min intervals)
    window_size = max(2, min(4, len(df) // 12))
    
    # Use trailing window (not centered) so rolling mean/std don't include future anomalous values
    df['rolling_mean'] = df['heart_rate'].rolling(
        window=window_size, 
        center=False,
        min_periods=1
    ).mean()
    
    df['rolling_std'] = df['heart_rate'].rolling(
        window=window_size, 
        center=False,
        min_periods=1
    ).std()
    
    # Handle NaN std (happens when all values are same)
    df['rolling_std'] = df['rolling_std'].fillna(global_std)
    
    # Step 2b: Calculate deviations from BOTH rolling and global baselines
    df['std_deviations_rolling'] = np.abs(df['heart_rate'] - df['rolling_mean']) / (df['rolling_std'] + 0.1)
    df['std_deviations_global'] = np.abs(df['heart_rate'] - global_mean) / (global_std + 0.1)
    
    # Use maximum deviation (more sensitive)
    df['std_deviations'] = np.maximum(df['std_deviations_rolling'], df['std_deviations_global'])
    
    # Flag potential anomalies (> 2.5 std deviations - more sensitive than 3σ)
    df['is_anomaly_point'] = df['std_deviations'] > 2.5
    
    # Step 3: Apply minimum duration rule (2+ consecutive anomalies to reduce false negatives)
    # Use a rolling window to count consecutive anomalies
    df['anomaly_group'] = (df['is_anomaly_point'] != df['is_anomaly_point'].shift()).cumsum()
    anomaly_counts = df.groupby('anomaly_group')['is_anomaly_point'].sum()
    
    # Get groups with at least 2 consecutive anomalies
    valid_groups = anomaly_counts[anomaly_counts >= 2].index
    df['is_anomaly_confirmed'] = df['anomaly_group'].isin(valid_groups) & df['is_anomaly_point']
    
    # Step 4: Group consecutive anomalies into single events
    # Create event IDs by grouping anomalies within 15-minute windows
    df['event_gap'] = (df['is_anomaly_confirmed'] != df['is_anomaly_confirmed'].shift()).cumsum()
    df['event_id'] = df['event_gap']
    
    # Filter to only anomaly events
    anomaly_events = df[df['is_anomaly_confirmed']].copy()
    
    # Step 5: Assign severity levels based on std deviations
    def get_severity(std_dev):
        if std_dev < 2:
            return 'Normal'
        elif std_dev < 2.5:
            return 'Mild'
        elif std_dev < 3.5:
            return 'Moderate'
        else:
            return 'Critical'
    
    df['severity'] = df['std_deviations'].apply(get_severity)
    
    # Final anomaly column: 1 if confirmed anomaly, 0 otherwise
    df['anomaly'] = df['is_anomaly_confirmed'].astype(int)
    
    return df
