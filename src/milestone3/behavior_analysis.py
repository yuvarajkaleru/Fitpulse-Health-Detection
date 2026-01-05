def analyze_behavior(df):
    avg_steps = df["steps"].mean()
    avg_sleep = df["sleep_hours"].mean()

    if avg_steps > 8000 and avg_sleep >= 7:
        behavior = "Active & Healthy"
    elif avg_steps > 5000:
        behavior = "Moderately Active"
    else:
        behavior = "Sedentary"

    return {
        "average_steps": int(avg_steps),
        "average_sleep_hours": round(avg_sleep, 2),
        "behavior_label": behavior
    }
