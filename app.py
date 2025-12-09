# app.py
# FitPulse Lab – Milestone 1 (preprocessing) + Milestone 2 (Prophet forecasting)
# Run with:
#   streamlit run app.py

import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet

# -----------------------
# Configuration
# -----------------------
SAMPLE_DATA_PATH = r"C:\Users\yuvar\Desktop\fit pulse\fitpulse_one_day_sample.csv"
DEFAULT_OUTPUT_FILENAME = "fitpulse_one_day_sample_preprocessed.csv"
FINAL_COLS = ["timestamp", "heart_rate", "hrv", "stress_score", "skin_temp"]

RNG = np.random.default_rng(seed=42)

# =====================================================
#  GLOBAL UI STYLING  (custom look, not default theme)
# =====================================================
st.set_page_config(
    page_title="FitPulse Lab",
    layout="wide",
    page_icon="❤️",
)

st.markdown(
    """
<style>
/* overall background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #000000 100%);
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #111827 50%, #020617 100%);
    border-right: 1px solid #1f2937;
}

.sidebar-title {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #a5b4fc;
}

/* cards */
.fp-card {
    background: rgba(15,23,42,0.92);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(148,163,184,0.22);
    box-shadow: 0 18px 45px rgba(15,23,42,0.9);
    backdrop-filter: blur(12px);
}

.fp-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: .13em;
    color: #a5b4fc;
}

.fp-heading-main {
    font-size: 2.2rem;
    font-weight: 750;
    letter-spacing: .03em;
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}

.fp-sub {
    color: #9ca3af;
    font-size: .92rem;
}

/* tables */
thead tr th {
    background-color: #020617 !important;
}

/* buttons */
.stButton>button {
    border-radius: 999px;
    border: 1px solid rgba(129,140,248,0.6);
    background: radial-gradient(circle at top left, #4f46e5 0, #1d4ed8 40%, #0f172a 100%);
    color: white;
    padding: 0.45rem 1.2rem;
    font-weight: 600;
}
.stButton>button:hover {
    filter: brightness(1.1);
}

/* metrics */
[data-testid="stMetricValue"] {
    font-size: 1.4rem;
}

/* tabs */
[data-baseweb="tab-list"] {
    gap: 0.25rem;
}
[data-baseweb="tab"] {
    border-radius: 999px !important;
}

/* plots */
.js-plotly-plot, .stPlotlyChart {
    border-radius: 18px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================
# MILESTONE 1 – PREPROCESSING
# ==============================

def parse_timestamps_keep_rows(df, ts_col="timestamp"):
    df = df.copy()
    df["timestamp_parsed"] = pd.to_datetime(df[ts_col], errors="coerce")
    n_invalid = int(df["timestamp_parsed"].isna().sum())
    if n_invalid > 0:
        st.warning(f"Dropping {n_invalid} rows with invalid timestamps.")
    df = df.dropna(subset=["timestamp_parsed"]).reset_index(drop=True)
    return df

def coerce_numeric(df, numeric_cols):
    df = df.copy()
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].replace({"NaN": np.nan, "nan": np.nan, "": np.nan, "ERROR": np.nan, "error": np.nan})
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df

def mark_domain_errors(df):
    df = df.copy()
    if "heart_rate" in df.columns:
        df.loc[(df["heart_rate"] < 30) | (df["heart_rate"] > 220), "heart_rate"] = np.nan
    if "hrv" in df.columns:
        df.loc[(df["hrv"] < 0) | (df["hrv"] > 300), "hrv"] = np.nan
    if "stress_score" in df.columns:
        df.loc[(df["stress_score"] < 0) | (df["stress_score"] > 150), "stress_score"] = np.nan
    if "skin_temp" in df.columns:
        df.loc[(df["skin_temp"] < 25) | (df["skin_temp"] > 45) | (df["skin_temp"] == 99.9), "skin_temp"] = np.nan
    return df

def remove_statistical_outliers(df, numeric_cols, z_thresh=4.0):
    df = df.copy()
    for c in numeric_cols:
        if c not in df.columns:
            continue
        ser = df[c].dropna()
        if len(ser) < 10:
            continue
        med = ser.median()
        mad = np.median(np.abs(ser - med))
        if mad == 0:
            mean = ser.mean()
            std = ser.std()
            if std == 0:
                continue
            mask = (df[c] - mean).abs() > (z_thresh * std)
        else:
            robust_z = 0.6745 * (df[c] - med) / mad
            mask = robust_z.abs() > z_thresh
        df.loc[mask, c] = np.nan
    return df

def impute_on_existing_rows(df, numeric_cols):
    df = df.copy()
    if "timestamp_parsed" not in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp_parsed")
    for c in numeric_cols:
        if c not in df.columns:
            continue
        try:
            df[c] = df[c].interpolate(method="time", limit_direction="both")
        except Exception:
            df[c] = df[c].interpolate(method="linear", limit_direction="both")
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df = df.reset_index(drop=False)
    df["timestamp"] = df["timestamp_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

def smooth_heart_rate(df, window=3):
    df = df.copy()
    if "timestamp_parsed" in df.columns:
        df = df.set_index("timestamp_parsed")
        if "heart_rate" in df.columns:
            df["heart_rate"] = df["heart_rate"].rolling(window=window, center=True, min_periods=1).median()
        df = df.reset_index(drop=False)
    else:
        if "heart_rate" in df.columns:
            df["heart_rate"] = df["heart_rate"].rolling(window=window, center=True, min_periods=1).median()
    return df

def preprocess_df(df):
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = parse_timestamps_keep_rows(df, ts_col="timestamp")
    numeric_cols = ["heart_rate", "hrv", "stress_score", "skin_temp"]
    df = coerce_numeric(df, numeric_cols)
    df = mark_domain_errors(df)
    df = remove_statistical_outliers(df, numeric_cols, z_thresh=4.0)
    df = impute_on_existing_rows(df, numeric_cols)
    df = smooth_heart_rate(df, window=3)

    out_df = df[FINAL_COLS].copy()
    return out_df

# ---- plotting helpers ----
def plot_hr_raw_vs_clean(raw_df, clean_df):
    plt.figure(figsize=(10, 3.5))
    try:
        raw_dt = pd.to_datetime(raw_df["timestamp"])
        plt.plot(raw_dt, pd.to_numeric(raw_df.get("heart_rate", pd.Series([]))), label="raw heart_rate", linewidth=0.7)
    except Exception:
        pass
    try:
        clean_dt = pd.to_datetime(clean_df["timestamp"])
        plt.plot(clean_dt, pd.to_numeric(clean_df.get("heart_rate", pd.Series([]))), label="clean heart_rate", linewidth=1.0)
    except Exception:
        pass
    plt.xlabel("Time")
    plt.ylabel("BPM")
    plt.legend()
    plt.title("Heart Rate: raw vs cleaned")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_hrv_trend(clean_df):
    if "hrv" in clean_df.columns:
        plt.figure(figsize=(10, 3.5))
        dt = pd.to_datetime(clean_df["timestamp"])
        plt.plot(dt, clean_df["hrv"], linewidth=1.0)
        plt.xlabel("Time")
        plt.ylabel("HRV")
        plt.title("HRV (cleaned)")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    else:
        st.info("No HRV column to plot.")

def plot_missing_bar(df):
    na_counts = df.isna().sum()
    plt.figure(figsize=(8, 3))
    plt.bar(na_counts.index.astype(str), na_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing values per column")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()


def milestone1_page():
    st.markdown(
        """
<div class="fp-card">
  <div class="fp-pill">Milestone 1 · Data Hygiene</div>
  <div class="fp-heading-main">Preprocess raw FitPulse signals</div>
  <p class="fp-sub">
    Upload a CSV or JSON file, clean sensor artefacts, impute gaps and export a single tidy CSV
    with <code>timestamp, heart_rate, hrv, stress_score, skin_temp</code>.
  </p>
</div>
<br>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<p class="sidebar-title">Milestone 1</p>', unsafe_allow_html=True)
    st.sidebar.caption("Upload CSV/JSON or use the sample day.")

    # ---- data source selection ----
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
    st.sidebar.markdown(f"**Sample path (local):** `{SAMPLE_DATA_PATH}`")

    uploaded = st.sidebar.file_uploader(
        "Or upload your own data",
        type=["csv", "json"],
        help="Supported: CSV or JSON with timestamp & sensor columns.",
    )

    input_df = None
    input_path_used = None

    # priority: uploaded file
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                input_df = pd.read_csv(uploaded)
            else:  # JSON
                try:
                    input_df = pd.read_json(uploaded)
                except Exception:
                    uploaded.seek(0)
                    input_df = pd.read_json(uploaded, lines=True)
            input_path_used = uploaded.name
            st.sidebar.success(f"Loaded: {uploaded.name}")
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded file: {e}")

    elif use_sample:
        if os.path.exists(SAMPLE_DATA_PATH):
            input_df = pd.read_csv(SAMPLE_DATA_PATH)
            input_path_used = SAMPLE_DATA_PATH
        else:
            st.sidebar.error(f"Sample file not found at: {SAMPLE_DATA_PATH}")

    if input_df is None:
        st.info("➡ Upload a CSV/JSON or enable the sample dataset in the sidebar.")
        return

    tabs = st.tabs(["📄 Raw preview", "🧼 Cleaned output & plots"])

    with tabs[0]:
        st.markdown("#### Raw dataset (first 200 rows)")
        st.caption(f"Source: **{input_path_used}**")
        st.dataframe(input_df.head(200), use_container_width=True)

    with tabs[1]:
        if st.button("Run preprocessing", key="preproc_btn"):
            with st.spinner("Cleaning and imputing sensor data..."):
                cleaned = preprocess_df(input_df)

                buf = io.StringIO()
                cleaned.to_csv(buf, index=False)
                buf.seek(0)
                bts = buf.getvalue().encode("utf-8")

                if input_path_used == SAMPLE_DATA_PATH and os.path.exists(SAMPLE_DATA_PATH):
                    out_path = os.path.join(os.path.dirname(SAMPLE_DATA_PATH), DEFAULT_OUTPUT_FILENAME)
                    cleaned.to_csv(out_path, index=False)
                    st.success(f"Saved cleaned CSV to: `{out_path}`")
                else:
                    st.success("Preprocessing finished for the uploaded file.")

                c1, c2 = st.columns([2, 1.2])
                with c1:
                    st.markdown("##### Cleaned dataset (first 200 rows)")
                    st.dataframe(cleaned.head(200), use_container_width=True)
                with c2:
                    st.markdown("##### Export")
                    st.download_button(
                        label="⬇️ Download cleaned CSV",
                        data=bts,
                        file_name=DEFAULT_OUTPUT_FILENAME,
                        mime="text/csv",
                    )
                    st.caption("Single tidy CSV with exactly five columns.")

                st.markdown("##### Visual diagnostics")
                c3, c4 = st.columns(2)
                with c3:
                    st.markdown("**Heart rate: raw vs cleaned**")
                    plot_hr_raw_vs_clean(input_df, cleaned)
                with c4:
                    st.markdown("**HRV trend (cleaned)**")
                    plot_hrv_trend(cleaned)

                st.markdown("**Missing values per column (cleaned)**")
                plot_missing_bar(cleaned)

                st.info(
                    "- Row count is preserved (no synthetic timestamps added).\n"
                    "- Missing segments are filled using time-based interpolation, then median.\n"
                    "- Obvious domain violations and statistical outliers are converted to NaN before imputation."
                )
        else:
            st.info("Click **Run preprocessing** to generate the cleaned CSV and plots.")


# ==============================
# MILESTONE 2 – PROPHET TASKS
# ==============================

# ---- data generators ----
def generate_heart_rate_data():
    dates = pd.date_range("2024-01-01", periods=74, freq="D")
    base = 72 + 3 * np.sin(2 * np.pi * np.arange(74) / 7)
    noise = RNG.normal(0, 2, size=74)
    hr = base + noise
    return pd.DataFrame({"ds": dates, "y": hr})

def generate_sleep_data():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    weekday = dates.weekday
    base = np.where(weekday < 5, 7.0, 8.5)
    trend = np.linspace(0, 0.5, 90)
    noise = RNG.normal(0, 0.3, size=90)
    hours = base + trend + noise
    return pd.DataFrame({"ds": dates, "y": hours})

def generate_steps_data():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    base = 8000 + 1000 * np.sin(2 * np.pi * np.arange(120) / 7)
    noise = RNG.normal(0, 500, size=120)
    steps = base + noise
    df = pd.DataFrame({"ds": dates, "y": steps})
    df.loc[29:36, "y"] += 3000     # vacation
    df.loc[59:61, "y"] -= 4000     # sick
    df.loc[89, "y"] += 10000       # marathon
    return df

def build_holidays_df(df):
    start_date = df["ds"].iloc[0]
    records = []
    for day in range(30, 38):
        records.append({"holiday": "vacation", "ds": start_date + pd.Timedelta(days=day - 1)})
    for day in range(60, 63):
        records.append({"holiday": "sick", "ds": start_date + pd.Timedelta(days=day - 1)})
    records.append({"holiday": "marathon", "ds": start_date + pd.Timedelta(days=90 - 1)})
    return pd.DataFrame(records)

# ---- UI for each task ----
def task1_ui():
    st.markdown("#### Task 1 · Heart rate forecasting (60 days train → 14 days forecast)")
    df_all = generate_heart_rate_data()
    train = df_all.iloc[:60].copy()
    test = df_all.iloc[60:].copy()

    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=14, freq="D")
    forecast = m.predict(future)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train["ds"], train["y"], label="Train (actual)", linewidth=1)
    ax.plot(test["ds"], test["y"], label="Test (actual)", linewidth=1)
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", linewidth=2)
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2,
        label="95% CI",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("BPM")
    ax.legend()
    ax.set_title("Heart rate forecast")
    st.pyplot(fig)

    forecast_test = forecast.iloc[-14:].copy()
    mae = float(np.mean(np.abs(test["y"].values - forecast_test["yhat"].values)))

    day_67_date = df_all["ds"].iloc[66]
    day_67_val = float(
        forecast.loc[forecast["ds"] == day_67_date, "yhat"].iloc[0]
    )

    c1, c2 = st.columns(2)
    c1.metric("MAE on days 61–74", f"{mae:.2f} bpm")
    c2.metric(f"Predicted HR on Day 67 ({day_67_date.date()})", f"{day_67_val:.1f} bpm")

    st.caption(
        "The shaded band is Prophet's 95% confidence interval. "
        "Given the model and past data, each future day's true heart rate is expected to fall "
        "inside this band about 95% of the time."
    )

def task2_ui():
    st.markdown("#### Task 2 · Sleep duration with weekly seasonality")
    df = generate_sleep_data()
    m = Prophet(weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7, freq="D")
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    plt.title("Sleep duration forecast (next 7 days)")
    st.pyplot(fig1)

    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    df["day_name"] = df["ds"].dt.day_name()
    avg_sleep_by_day = df.groupby("day_name")["y"].mean().sort_values()
    worst_day = avg_sleep_by_day.index[0]
    best_day = avg_sleep_by_day.index[-1]

    first_day = df["ds"].iloc[0]
    last_day = df["ds"].iloc[-1]
    trend_start = float(forecast.loc[forecast["ds"] == first_day, "trend"].iloc[0])
    trend_end = float(forecast.loc[forecast["ds"] == last_day, "trend"].iloc[0])
    if trend_end > trend_start:
        direction = "increasing"
    elif trend_end < trend_start:
        direction = "decreasing"
    else:
        direction = "approximately flat"

    c1, c2, c3 = st.columns(3)
    c1.metric("Sleeps least on", worst_day)
    c2.metric("Sleeps most on", best_day)
    c3.metric("Overall trend", direction.capitalize())

def task3_ui():
    st.markdown("#### Task 3 · Step counts with holidays / events")
    df = generate_steps_data()
    holidays = build_holidays_df(df)

    m_no = Prophet()
    m_no.fit(df)
    fut_no = m_no.make_future_dataframe(periods=30, freq="D")
    fc_no = m_no.predict(fut_no)

    m_h = Prophet(holidays=holidays)
    m_h.fit(df)
    fut_h = m_h.make_future_dataframe(periods=30, freq="D")
    fc_h = m_h.predict(fut_h)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = m_no.plot(fc_no)
        plt.title("Forecast WITHOUT holidays")
        st.pyplot(fig1)
    with c2:
        fig2 = m_h.plot(fc_h)
        plt.title("Forecast WITH holidays")
        st.pyplot(fig2)

    event_dates = holidays["ds"].unique()
    fc_no_events = fc_no[fc_no["ds"].isin(event_dates)]
    fc_h_events = fc_h[fc_h["ds"].isin(event_dates)]
    avg_no = fc_no_events["yhat"].mean()
    avg_with = fc_h_events["yhat"].mean()
    overall_impact = float(avg_with - avg_no)

    impacts = {}
    for event in holidays["holiday"].unique():
        dates = holidays.loc[holidays["holiday"] == event, "ds"]
        no_e = fc_no[fc_no["ds"].isin(dates)]
        h_e = fc_h[fc_h["ds"].isin(dates)]
        impacts[event] = float(h_e["yhat"].mean() - no_e["yhat"].mean())

    biggest_event = max(impacts, key=lambda k: abs(impacts[k]))

    c3, c4 = st.columns(2)
    c3.metric(
        "Average change on event days (WITH − WITHOUT)",
        f"{overall_impact:+.0f} steps",
    )
    c4.metric("Largest effect event", biggest_event.capitalize())

    st.markdown("##### Per-event impact (WITH − WITHOUT)")
    per_event_rows = [
        {"event": k, "delta_steps": round(v, 0)} for k, v in impacts.items()
    ]
    st.dataframe(pd.DataFrame(per_event_rows), use_container_width=True)


def milestone2_page():
    st.markdown(
        """
<div class="fp-card">
  <div class="fp-pill">Milestone 2 · Time-series modelling</div>
  <div class="fp-heading-main">Forecast physiology and behaviour with Prophet</div>
  <p class="fp-sub">
    Three mini-experiments: (1) heart-rate forecasting with a 14-day horizon,
    (2) weekly sleep patterns, and (3) step counts influenced by vacations,
    sickness and a marathon day.
  </p>
</div>
<br>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<p class="sidebar-title">Milestone 2</p>', unsafe_allow_html=True)
    st.sidebar.caption("Run Prophet models on synthetic FitPulse-style data.")

    tab1, tab2, tab3 = st.tabs(
        ["❤️ Heart rate", "🛌 Sleep duration", "👟 Steps & holidays"]
    )

    with tab1:
        task1_ui()
    with tab2:
        task2_ui()
    with tab3:
        task3_ui()


# ==============================
# ROUTER
# ==============================

with st.sidebar:
    st.markdown("### FitPulse Lab")
    page = st.radio(
        "Choose workspace",
        ("Milestone 1 – Preprocessing", "Milestone 2 – Forecasts"),
        index=0,
    )

if page.startswith("Milestone 1"):
    milestone1_page()
else:
    milestone2_page()
