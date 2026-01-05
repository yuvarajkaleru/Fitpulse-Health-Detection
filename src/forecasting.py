import pandas as pd
from prophet import Prophet

# -------------------------------------------------
# HEART RATE FORECAST
# -------------------------------------------------
def forecast_heart_rate(df):
    data = df[["timestamp", "heart_rate"]].dropna()

    if len(data) < 2:
        raise ValueError("Not enough heart rate data to forecast.")

    data = data.rename(columns={"timestamp": "ds", "heart_rate": "y"})

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)

    return model, forecast


# -------------------------------------------------
# SLEEP FORECAST
# -------------------------------------------------
def forecast_sleep(df):
    """
    Forecast sleep duration.
    
    IMPORTANT: Sleep data comes as per-stage rows (REM, deep, light).
    We must aggregate by DATE to get daily totals before forecasting.
    
    Steps:
    1. Extract sleep data and remove rows where sleep_hours is 0 or NaN
    2. Group by DATE (not timestamp) and sum sleep_hours
    3. Convert to hours
    4. Run Prophet on daily totals
    """
    
    # Extract sleep data and drop zeros/NaN
    sleep_data = df[["timestamp", "sleep_hours"]].copy()
    sleep_data = sleep_data[sleep_data["sleep_hours"] > 0].dropna()
    
    if len(sleep_data) < 2:
        raise ValueError("Not enough sleep data to forecast.")
    
    # Convert timestamp to date only (remove time component)
    sleep_data["date"] = pd.to_datetime(sleep_data["timestamp"]).dt.date
    
    # Group by date and sum sleep hours per day
    daily_sleep = sleep_data.groupby("date")["sleep_hours"].sum().reset_index()
    
    # Rename columns for Prophet
    daily_sleep["ds"] = pd.to_datetime(daily_sleep["date"])
    daily_sleep["y"] = daily_sleep["sleep_hours"]
    daily_sleep = daily_sleep[["ds", "y"]]
    
    if len(daily_sleep) < 2:
        raise ValueError("Not enough daily sleep data to forecast after aggregation.")
    
    model = Prophet(weekly_seasonality=True)
    model.fit(daily_sleep)
    
    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)
    
    return model, forecast


# -------------------------------------------------
# STEPS FORECAST WITH EVENTS
# -------------------------------------------------
def forecast_steps_with_events(df):
    """
    Forecast daily steps.
    
    IMPORTANT: Steps data comes as daily records already aggregated.
    However, we need to ensure we're using DAILY totals, not per-row data.
    
    Steps:
    1. Extract steps data and remove zero/NaN rows (missing data)
    2. Group by DATE to get daily totals
    3. Run Prophet on daily step totals
    """
    
    # Extract steps data and drop zeros/NaN (these are missing data)
    steps_data = df[["timestamp", "steps"]].copy()
    steps_data = steps_data[steps_data["steps"] > 0].dropna()
    
    if len(steps_data) < 2:
        raise ValueError("Not enough steps data to forecast.")
    
    # Convert timestamp to date only (remove time component)
    steps_data["date"] = pd.to_datetime(steps_data["timestamp"]).dt.date
    
    # Group by date and sum steps per day
    daily_steps = steps_data.groupby("date")["steps"].sum().reset_index()
    
    # Rename columns for Prophet
    daily_steps["ds"] = pd.to_datetime(daily_steps["date"])
    daily_steps["y"] = daily_steps["steps"]
    daily_steps = daily_steps[["ds", "y"]]
    
    if len(daily_steps) < 2:
        raise ValueError("Not enough daily steps data to forecast after aggregation.")
    
    model = Prophet()
    model.fit(daily_steps)
    
    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)
    
    return model, forecast
