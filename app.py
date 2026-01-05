import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go

from src.generate_data import generate_realistic_sample_data
from src.preprocess import run_preprocessing
from src.forecasting import (
    forecast_heart_rate,
    forecast_sleep,
    forecast_steps_with_events
)

from src.visualization import (
    create_interactive_forecast_chart,
    create_forecast_summary_stats
)

from src.milestone3.comparison import daily_comparison
from src.milestone3.anomaly_detection import detect_anomalies
from src.milestone3.behavior_analysis import analyze_behavior
from src.milestone4.report_utils import generate_summary_report


# --------------------------------------------------
# PAGE CONFIG & THEME
# --------------------------------------------------
st.set_page_config(
    page_title="FitPulse – Health Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# GLOBAL THEME & BACKGROUND SETUP
# --------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme = st.session_state.theme

# Theme colors
if theme == "dark":
    page_bg = "#020617"
    nav_bg = "#020617"
    nav_border = "#1e293b"
    text_color = "#e5e7eb"
    btn_bg = "#020617"
    btn_hover = "#1e293b"
    btn_active = "#2563eb"
else:
    page_bg = "#f8fafc"
    nav_bg = "#ffffff"
    nav_border = "#e5e7eb"
    text_color = "#0f172a"
    btn_bg = "#ffffff"
    btn_hover = "#f1f5f9"
    btn_active = "#2563eb"

st.markdown(f"""
<style>

/* GLOBAL BACKGROUND */
.stApp {{
    background-color: {page_bg};
}}

html, body {{
    background-color: {page_bg};
    margin: 0;
    padding: 0;
}}

/* REMOVE STREAMLIT HEADER SPACE */
header {{
    display: none;
}}

.block-container {{
    padding-top: 90px !important;
}}

/* NAVBAR */
.fitpulse-navbar {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 64px;
    background: {nav_bg};
    border-bottom: 1px solid {nav_border};
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    z-index: 10000;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}}

.nav-left {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.logo {{
    font-size: 18px;
    font-weight: 700;
    color: {text_color};
    margin-right: 16px;
}}

.nav-btn {{
    padding: 8px 14px;
    border-radius: 8px;
    background: {btn_bg};
    color: {text_color};
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}}

.nav-btn:hover {{
    background: {btn_hover};
}}

.nav-btn.active {{
    background: {btn_active};
    color: #ffffff;
}}

.nav-right {{
    display: flex;
    align-items: center;
    gap: 12px;
    color: {text_color};
    font-size: 14px;
}}

</style>

<div class="fitpulse-navbar">
  <div class="nav-left">
    <div class="logo">FitPulse</div>
    <button class="nav-btn active">Home</button>
    <button class="nav-btn">Data Import</button>
    <button class="nav-btn">Data Cleaning</button>
    <button class="nav-btn">Forecasting</button>
    <button class="nav-btn">Anomaly Detection</button>
    <button class="nav-btn">Intelligence</button>
  </div>

  <div class="nav-right">
    <span>Theme</span>
    <strong>{theme.capitalize()}</strong>
  </div>
</div>
""", unsafe_allow_html=True)



# --------------------------------------------------
# LOGO BACKGROUND WATERMARK
# --------------------------------------------------
import base64
import os

def set_logo_background(theme):
    """Add FitPulse logo as subtle background watermark on all pages"""
    logo_path = "logo.png"
    
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as image_file:
                encoded_logo = base64.b64encode(image_file.read()).decode()
            
            # Adjust opacity for theme
            opacity = "0.06" if theme == "dark" else "0.08"
            
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{encoded_logo}");
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 420px;
                    background-attachment: fixed;
                    background-color: transparent;
                }}
                
                .stApp::before {{
                    content: '';
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-image: url("data:image/png;base64,{encoded_logo}");
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 420px;
                    background-attachment: fixed;
                    opacity: {opacity};
                    pointer-events: none;
                    z-index: 0;
                }}
                
                section[data-testid="stSidebar"],
                div[data-testid="stVerticalBlock"] {{
                    position: relative;
                    z-index: 1;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            pass  # Silently fail if logo not found or can't be encoded

set_logo_background(st.session_state.theme)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
for key in ["unified_df", "clean_df", "active_tab"]:
    if key not in st.session_state:
        if key == "active_tab":
            st.session_state[key] = "Home"
        else:
            st.session_state[key] = None


# --------------------------------------------------
# PROFESSIONAL WEBSITE-STYLE NAVBAR
# --------------------------------------------------
# --------------------------------------------------
# THEME TOGGLE & NAVIGATION
# --------------------------------------------------
col_left, col_right = st.columns([10, 2])

with col_left:
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)
    
    with nav_col1:
        if st.button("Home", use_container_width=True, key="nav_home"):
            st.session_state.active_tab = "Home"
            st.rerun()
    
    with nav_col2:
        if st.button("Data Import", use_container_width=True, key="nav_data_import"):
            st.session_state.active_tab = "Data Import"
            st.rerun()
    
    with nav_col3:
        if st.button("Data Cleaning", use_container_width=True, key="nav_data_cleaning"):
            st.session_state.active_tab = "Data Cleaning"
            st.rerun()
    
    with nav_col4:
        if st.button("Forecasting", use_container_width=True, key="nav_forecasting"):
            st.session_state.active_tab = "Forecasting"
            st.rerun()
    
    with nav_col5:
        if st.button("Anomaly Detection", use_container_width=True, key="nav_anomaly"):
            st.session_state.active_tab = "Anomaly Detection"
            st.rerun()
    
    with nav_col6:
        if st.button("Intelligence", use_container_width=True, key="nav_intelligence"):
            st.session_state.active_tab = "Intelligence & Summary"
            st.rerun()

with col_right:
    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
        label_visibility="collapsed",
        key="theme_toggle"
    )
    if theme_choice == "Light" and st.session_state.theme == "dark":
        st.session_state.theme = "light"
        st.rerun()
    elif theme_choice == "Dark" and st.session_state.theme == "light":
        st.session_state.theme = "dark"
        st.rerun()

st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)



# ==================================================
# TAB 0: HOME PAGE
# ==================================================
if st.session_state.active_tab == "Home":
    # Hero section
    st.markdown("""
        <h1 style='text-align: center; color: var(--primary-color, #58a6ff); margin: 1rem 0 0.5rem 0; font-size: 2.2rem; font-weight: 700;'>
            Monitor • Forecast • Detect • Act
        </h1>
    """, unsafe_allow_html=True)
    
    # Welcome Section
    st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(88, 166, 255, 0.08) 0%, rgba(63, 185, 80, 0.08) 100%); 
                    border-radius: 12px; padding: 2rem; margin: 1.5rem 0 2rem 0; border-left: 4px solid var(--primary-color, #58a6ff);'>
            <h2 style='color: var(--primary-color, #58a6ff); margin: 0 0 1rem 0; font-size: 1.5rem;'>Welcome to FitPulse</h2>
            <p style='font-size: 1rem; line-height: 1.6; color: var(--text-color, #e6edf3); margin: 0;'>
                FitPulse is an end-to-end health intelligence system that collects, cleans, forecasts, 
                and analyzes fitness data to detect anomalies and generate actionable insights. 
                Track your wellness journey with powerful analytics and AI-driven insights.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Workflow Guidance Section
    st.markdown("<h3 style='color: var(--primary-color, #58a6ff); margin: 2rem 0 1.5rem 0; font-size: 1.3rem;'>Your Health Intelligence Workflow</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])

    
    with col1:
        st.markdown("""
            <div style='background: #f8f9fa; border-radius: 12px; padding: 1.5rem; border-left: 4px solid #1f77b4;'>
                <h4 style='color: #1f77b4; margin-top: 0;'>📋 5-Step Process</h4>
                <ol style='line-height: 1.8; color: #333;'>
                    <li><strong>1️⃣ Data Import</strong> – Import or generate fitness data</li>
                    <li><strong>2️⃣ Data Cleaning</strong> – Handle missing values & validation</li>
                    <li><strong>3️⃣ Forecasting</strong> – Predict future health trends</li>
                    <li><strong>4️⃣ Anomaly Detection</strong> – Detect abnormal patterns</li>
                    <li><strong>5️⃣ Intelligence</strong> – Summarize insights & wellness</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: #e8f4f8; border-radius: 12px; padding: 1.5rem; border-left: 4px solid #2ca02c;'>
                <h4 style='color: #2ca02c; margin-top: 0;'>🚀 Get Started</h4>
                <p style='color: #333; line-height: 1.6;'>
                    <strong>New to FitPulse?</strong><br>
                    Start by importing your data or generating a realistic sample dataset. 
                    Once you have data loaded, the entire pipeline—from cleaning to analysis—
                    is available at your fingertips.
                </p>
                <p style='color: #555; font-size: 0.95rem; margin-bottom: 0;'>
                    💡 <strong>Tip:</strong> Use the sample data generator to explore all features 
                    without preparing your own dataset.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Call-to-action
    st.markdown("<hr style='margin: 2rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f0f7ff; border-radius: 8px;'>
            <h4 style='color: #1f77b4; margin-top: 0;'>👉 Ready to Begin?</h4>
            <p style='color: #333; font-size: 1.05rem;'>
                Click the <strong>📊 Data Import</strong> tab above to start your health analytics journey.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("<h3 style='color: #1f77b4; margin-top: 2rem; margin-bottom: 1.5rem;'>Key Features</h3>", unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
            <div style='background: #f8f9fa; border-radius: 8px; padding: 1.25rem; text-align: center;'>
                <h4 style='font-size: 2rem; margin: 0;'>📊</h4>
                <h5 style='color: #1f77b4; margin: 0.5rem 0;'>Data Integration</h5>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                    Seamlessly import and merge multiple fitness data sources
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
            <div style='background: #f8f9fa; border-radius: 8px; padding: 1.25rem; text-align: center;'>
                <h4 style='font-size: 2rem; margin: 0;'>🔍</h4>
                <h5 style='color: #1f77b4; margin: 0.5rem 0;'>Smart Cleaning</h5>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                    Intelligent handling of missing values and data validation
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
            <div style='background: #f8f9fa; border-radius: 8px; padding: 1.25rem; text-align: center;'>
                <h4 style='font-size: 2rem; margin: 0;'>🤖</h4>
                <h5 style='color: #1f77b4; margin: 0.5rem 0;'>AI Insights</h5>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                    Advanced anomaly detection and predictive analytics
                </p>
            </div>
        """, unsafe_allow_html=True)


# ==================================================
# TAB 1: DATA IMPORT & GENERATION
# ==================================================
elif st.session_state.active_tab == "Data Import":
    st.markdown("<h2 class='section-header'>Data Import & Generation</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Get started with realistic fitness data for your health analytics.</p>", unsafe_allow_html=True)
    
    # Two Options
    st.markdown("<h3 style='color: #1f77b4;'>Choose an Option</h3>", unsafe_allow_html=True)
    
    option_col1, option_col2 = st.columns(2)
    
    with option_col1:
        if st.button("📤 Import Your Own Data", use_container_width=True, key="import_data"):
            st.info("""
            **To import your own fitness data:**
            
            1. Prepare CSV files with columns:
               - `heart_rate.csv`: timestamp, heart_rate, stress_level
               - `steps.csv`: timestamp, steps, calories_burned
            
            2. Place files in the `data_raw/` folder
            
            3. Run the Data Cleaning pipeline to merge and validate
            
            **Supported file formats:** CSV, JSON
            """)
    
    with option_col2:
        if st.button("🔄 Generate Realistic Sample Data", use_container_width=True, key="gen_sample"):
            with st.spinner("Generating 30 days of realistic fitness data with missing values..."):
                unified_df = generate_realistic_sample_data()
                st.session_state.unified_df = unified_df
            st.success("✓ Unified fitness dataset generated (30 days, 2,880 readings, missing values injected)")
            st.balloons()
    
    st.markdown("---")
    
    # Data Preview
    st.markdown("<h3 style='color: #1f77b4;'>Generated Data Preview</h3>", unsafe_allow_html=True)
    
    if st.session_state.unified_df is not None:
        # Show unified dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Readings", len(st.session_state.unified_df))
        with col2:
            missing_vals = st.session_state.unified_df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_vals} ({missing_vals/(len(st.session_state.unified_df)*7)*100:.1f}%)")
        with col3:
            st.metric("Timestamp Range", f"{st.session_state.unified_df['timestamp'].min().strftime('%Y-%m-%d')} to {st.session_state.unified_df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        st.markdown("<h5>Unified Fitness Dataset (with missing values)</h5>", unsafe_allow_html=True)
        st.dataframe(st.session_state.unified_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("💡 Generate or import data to see a preview")



# ==================================================
# TAB 2: DATA CLEANING & INTEGRATION
# ==================================================
elif st.session_state.active_tab == "Data Cleaning":
    st.markdown("<h2 class='section-header'>Data Cleaning & Integration</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Process raw data: clean, validate, and merge into a unified dataset.</p>", unsafe_allow_html=True)
    
    # Preprocessing Pipeline
    if st.button("Run Preprocessing Pipeline", use_container_width=True, key="run_preprocess"):
        with st.spinner("Running preprocessing pipeline..."):
            run_preprocessing()
            cleaned_path = "data/cleaned_fitness_data.csv"
            
            if os.path.exists(cleaned_path):
                df = pd.read_csv(cleaned_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.session_state.clean_df = df
                st.success("✓ Data cleaning & integration completed successfully")
    
    st.markdown("---")
    
    # Cleaned Data Display
    if st.session_state.clean_df is not None:
        st.markdown("<h3 style='color: #1f77b4;'>Cleaned & Merged Dataset</h3>", unsafe_allow_html=True)
        
        # Summary Stats
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric(label="Total Records", value=len(st.session_state.clean_df))
        
        with stats_col2:
            st.metric(label="Columns", value=len(st.session_state.clean_df.columns))
        
        with stats_col3:
            st.metric(label="Date Range", value=f"{len(st.session_state.clean_df)} days")
        
        with stats_col4:
            st.metric(label="Missing Values", value=st.session_state.clean_df.isnull().sum().sum())
        
        st.markdown("<hr style='margin: 1rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)
        
        st.dataframe(st.session_state.clean_df, use_container_width=True, height=400)
    else:
        st.info("Run the preprocessing pipeline to view cleaned data")


# ==================================================
# TAB 3: TIME SERIES FORECASTING
# ==================================================
elif st.session_state.active_tab == "Forecasting":
    st.markdown("<h2 class='section-header'>Time Series Forecasting</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Predict future health metrics using advanced forecasting models.</p>", unsafe_allow_html=True)
    
    if st.session_state.clean_df is not None:
        # Forecast Selection
        forecast_col1, forecast_col2 = st.columns([2, 1])
        
        with forecast_col1:
            forecast_task = st.selectbox(
                "Select Metric to Forecast",
                ["Heart Rate Forecast", "Sleep Duration Forecast", "Steps Forecast"],
                key="forecast_select"
            )
        
        with forecast_col2:
            run_forecast = st.button("Run Forecast", use_container_width=True, key="run_forecast")
        
        st.markdown("---")
        
        if run_forecast:
            if len(st.session_state.clean_df) < 10:
                st.error("⚠️ Not enough data for forecasting. Please run the preprocessing pipeline first.")
            else:
                with st.spinner(f"Running {forecast_task}..."):
                    df = st.session_state.clean_df.copy()
                    
                    if forecast_task == "Heart Rate Forecast":
                        model, forecast = forecast_heart_rate(df)
                        metric_name = "Heart Rate"
                        unit = "bpm"
                        forecast_data = df[["timestamp", "heart_rate"]].dropna().rename(columns={"timestamp": "ds", "heart_rate": "y"})
                    elif forecast_task == "Sleep Duration Forecast":
                        model, forecast = forecast_sleep(df)
                        metric_name = "Sleep Duration"
                        unit = "hours"
                        # For sleep visualization, aggregate to daily totals (same as forecast does internally)
                        sleep_data = df[["timestamp", "sleep_hours"]].copy()
                        sleep_data = sleep_data[sleep_data["sleep_hours"] > 0].dropna()
                        sleep_data["date"] = pd.to_datetime(sleep_data["timestamp"]).dt.date
                        daily_sleep = sleep_data.groupby("date")["sleep_hours"].sum().reset_index()
                        daily_sleep["ds"] = pd.to_datetime(daily_sleep["date"])
                        forecast_data = daily_sleep[["ds", "sleep_hours"]].rename(columns={"sleep_hours": "y"})
                    else:
                        model, forecast = forecast_steps_with_events(df)
                        metric_name = "Steps"
                        unit = "steps"
                        # For steps visualization, aggregate to daily totals (same as forecast does internally)
                        steps_data = df[["timestamp", "steps"]].copy()
                        steps_data = steps_data[steps_data["steps"] > 0].dropna()
                        steps_data["date"] = pd.to_datetime(steps_data["timestamp"]).dt.date
                        daily_steps = steps_data.groupby("date")["steps"].sum().reset_index()
                        daily_steps["ds"] = pd.to_datetime(daily_steps["date"])
                        forecast_data = daily_steps[["ds", "steps"]].rename(columns={"steps": "y"})
                    
                    st.success(f"✓ {forecast_task} completed")
                    
                    st.markdown("---")
                    
                    # Create and display interactive chart
                    fig = create_interactive_forecast_chart(forecast_data, forecast, metric_name, unit)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary statistics
                    st.markdown("<h3 style='color: #1f77b4;'>Forecast Summary</h3>", unsafe_allow_html=True)
                    
                    stats = create_forecast_summary_stats(forecast_data, forecast, metric_name, unit)
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric(
                            label="Historical Average",
                            value=f"{stats['actual_mean']:.1f} {unit}",
                            delta=f"±{stats['actual_std']:.1f} {unit}"
                        )
                    
                    with summary_col2:
                        st.metric(
                            label="Forecasted Average",
                            value=f"{stats['forecast_mean']:.1f} {unit}",
                            delta=f"{stats['forecast_change']:+.1f}% change"
                        )
                    
                    with summary_col3:
                        st.metric(
                            label="Forecast Period",
                            value="14 Days",
                            delta="Next 2 weeks"
                        )
    else:
        st.info("Please run the preprocessing pipeline to enable forecasting")


# ==================================================
# TAB 4: ANOMALY DETECTION & ALERTS
# ==================================================
elif st.session_state.active_tab == "Anomaly Detection":
    st.markdown("<h2 class='section-header'>Anomaly Detection & Alerts</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Identify abnormal patterns in your heart rate data with rolling statistics.</p>", unsafe_allow_html=True)
    
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
        
        # Clip unrealistic values for clearer analysis
        df["sleep_hours"] = df["sleep_hours"].clip(lower=5.5, upper=8.5)
        df["steps"] = df["steps"].clip(lower=800, upper=12000)
        
        # Anomaly Detection Analysis
        st.markdown("<h3 style='color: #1f77b4;'>Heart Rate Anomalies Analysis</h3>", unsafe_allow_html=True)
        
        anomaly_df = detect_anomalies(df)
        anomaly_count = int(anomaly_df["anomaly"].sum())
        
        # Severity breakdown
        severity_counts = anomaly_df[anomaly_df["anomaly"] == 1]["severity"].value_counts()
        critical_count = severity_counts.get("Critical", 0)
        moderate_count = severity_counts.get("Moderate", 0)
        mild_count = severity_counts.get("Mild", 0)
        
        # Display Key Metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="Total Anomalies",
                value=anomaly_count,
                delta="⚠️ Events" if anomaly_count > 0 else "✓ Normal"
            )
        
        with metric_col2:
            st.metric(
                label="Critical",
                value=int(critical_count),
                delta="Urgent" if critical_count > 0 else "None"
            )
        
        with metric_col3:
            st.metric(
                label="Moderate",
                value=int(moderate_count),
                delta="Monitor"
            )
        
        with metric_col4:
            st.metric(
                label="Mild",
                value=int(mild_count),
                delta="Normal variation"
            )
        
        st.markdown("---")
        
        # Detailed Anomalies Table
        st.markdown("<h3 style='color: #1f77b4;'>Detected Anomalies by Severity</h3>", unsafe_allow_html=True)
        anomalies_only = anomaly_df[anomaly_df["anomaly"] == 1].copy()
        
        if len(anomalies_only) > 0:
            # Format the display
            display_cols = ["timestamp", "heart_rate", "rolling_mean", "std_deviations", "severity"]
            display_df = anomalies_only[display_cols].copy()
            display_df["rolling_mean"] = display_df["rolling_mean"].round(1)
            display_df["std_deviations"] = display_df["std_deviations"].round(2)
            display_df["heart_rate"] = display_df["heart_rate"].astype(int)
            
            # Color code by severity
            def severity_color(severity):
                if severity == "Critical":
                    return "🔴 Critical"
                elif severity == "Moderate":
                    return "🟠 Moderate"
                else:
                    return "🟡 Mild"
            
            display_df["severity"] = display_df["severity"].apply(severity_color)
            
            st.dataframe(
                display_df.rename(columns={
                    "timestamp": "Time",
                    "heart_rate": "HR (bpm)",
                    "rolling_mean": "Avg (bpm)",
                    "std_deviations": "Std Dev",
                    "severity": "Severity"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✓ No anomalies detected! Your heart rate data looks healthy.")
        
        st.markdown("---")
        
        # Heart Rate Trend with Anomaly Highlighting
        st.markdown("<h3 style='color: #1f77b4;'>Heart Rate Trend with Anomaly Detection</h3>", unsafe_allow_html=True)
        
        # Create interactive Plotly chart showing trend with anomalies
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add normal heart rate line
        normal_df = anomaly_df[anomaly_df["anomaly"] == 0]
        fig.add_trace(go.Scatter(
            x=normal_df["timestamp"],
            y=normal_df["heart_rate"],
            mode="lines+markers",
            name="Normal",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="<b>Normal</b><br>HR: %{y} bpm<br>Time: %{x|%Y-%m-%d %H:%M}<extra></extra>"
        ))
        
        # Add anomalies colored by severity
        for severity, color, symbol in [("Critical", "#d62728", "diamond"), ("Moderate", "#ff7f0e", "star"), ("Mild", "#ffbb78", "circle")]:
            severity_df = anomaly_df[(anomaly_df["anomaly"] == 1) & (anomaly_df["severity"] == severity)]
            if len(severity_df) > 0:
                fig.add_trace(go.Scatter(
                    x=severity_df["timestamp"],
                    y=severity_df["heart_rate"],
                    mode="markers",
                    name=f"{severity} Anomaly",
                    marker=dict(size=10, color=color, symbol=symbol),
                    hovertemplate="<b>" + severity + "</b><br>HR: %{y} bpm<br>Std Dev: %{customdata:.2f}<br>Time: %{x|%Y-%m-%d %H:%M}<extra></extra>",
                    customdata=severity_df["std_deviations"]
                ))
        
        # Add rolling mean line
        fig.add_trace(go.Scatter(
            x=anomaly_df["timestamp"],
            y=anomaly_df["rolling_mean"],
            mode="lines",
            name="Rolling Average",
            line=dict(color="#2ca02c", width=1, dash="dash"),
            hovertemplate="<b>Rolling Avg</b><br>HR: %{y:.1f} bpm<extra></extra>"
        ))
        
        # Add upper and lower bounds (3 std dev)
        upper_bound = anomaly_df["rolling_mean"] + 3 * anomaly_df["rolling_std"]
        lower_bound = anomaly_df["rolling_mean"] - 3 * anomaly_df["rolling_std"]
        
        fig.add_trace(go.Scatter(
            x=anomaly_df["timestamp"],
            y=upper_bound,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.add_trace(go.Scatter(
            x=anomaly_df["timestamp"],
            y=lower_bound,
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            name="Normal Range (±3σ)",
            fillcolor="rgba(44, 160, 44, 0.15)",
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            title="Heart Rate with Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Heart Rate (bpm)",
            hovermode="x unified",
            template="plotly_white",
            height=450,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Algorithm explanation
        st.markdown("---")
        st.markdown("<h4 style='color: #666;'>Detection Method</h4>", unsafe_allow_html=True)
        st.markdown("""
        **Rolling Statistics Approach:**
        - Computes rolling mean & standard deviation over dynamic window
        - Flags readings >3σ from rolling mean as anomalies
        - Requires 3+ consecutive readings to confirm anomaly
        - Assigns severity based on deviation distance:
          - **Mild (2-3σ):** Normal daily variation
          - **Moderate (3-4σ):** Unusual pattern, monitor
          - **Critical (>4σ):** Significant deviation, review recommended
        """)
    else:
        st.info("Please run the preprocessing pipeline to enable anomaly detection")


# ==================================================
# TAB 5: HEALTH INTELLIGENCE DASHBOARD
# ==================================================
elif st.session_state.active_tab == "Intelligence & Summary":
    st.markdown("<h2 class='section-header'>Health Intelligence Dashboard</h2>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Comprehensive health insights and AI-powered wellness recommendations.</p>", unsafe_allow_html=True)
    
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df.copy()
        
        # Clip unrealistic values for clearer analysis
        df["sleep_hours"] = df["sleep_hours"].clip(lower=5.5, upper=8.5)
        df["steps"] = df["steps"].clip(lower=800, upper=12000)
        
        # Daily Comparative Analytics
        st.markdown("<h3 style='color: #1f77b4;'>Daily Comparative Analytics</h3>", unsafe_allow_html=True)
        daily_df = daily_comparison(df)
        st.dataframe(daily_df, use_container_width=True)
        
        st.markdown("---")
        
        # Behavior Analysis
        st.markdown("<h3 style='color: #1f77b4;'>Behavior Analysis Summary</h3>", unsafe_allow_html=True)
        behavior = analyze_behavior(df)
        
        behavior_col1, behavior_col2, behavior_col3 = st.columns(3)
        
        with behavior_col1:
            st.metric(
                label="Average Daily Steps",
                value=f"{int(behavior['average_steps']):,}",
                delta="Target: 10,000 steps"
            )
        
        with behavior_col2:
            st.metric(
                label="Average Sleep Duration",
                value=f"{round(behavior['average_sleep_hours'], 1)} hrs",
                delta="Target: 7-9 hours"
            )
        
        with behavior_col3:
            lifestyle_emoji = {
                "Sedentary": "🛋️",
                "Moderately Active": "🚶",
                "Active": "🏃"
            }
            emoji = lifestyle_emoji.get(behavior["behavior_label"], "")
            st.metric(
                label="Lifestyle Classification",
                value=f"{emoji} {behavior['behavior_label']}"
            )
        
        st.markdown("---")
        
        # Wellness Score Calculation
        st.markdown("<h3 style='color: #1f77b4;'>Overall Wellness Score</h3>", unsafe_allow_html=True)
        
        avg_hr = daily_df["avg_heart_rate"].mean()
        avg_steps = behavior["average_steps"]
        avg_sleep = behavior["average_sleep_hours"]
        
        score = (
            (100 - abs(avg_hr - 70)) * 0.4 +
            min(avg_steps / 100, 100) * 0.3 +
            (avg_sleep / 8 * 100) * 0.3
        )
        
        score = int(min(max(score, 0), 100))
        
        if score >= 75:
            status = "Good 😊"
            status_color = "#2ca02c"
        elif score >= 50:
            status = "Moderate 🙂"
            status_color = "#ff7f0e"
        else:
            status = "Needs Improvement ⚠️"
            status_color = "#d62728"
        
        wellness_col1, wellness_col2 = st.columns(2)
        
        with wellness_col1:
            st.markdown(f"<div style='background: white; padding: 1.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'><h2 style='color: #1f77b4; margin: 0;'>{score}</h2><p style='color: #666; margin: 0.5rem 0 0 0;'>Wellness Score</p></div>", unsafe_allow_html=True)
        
        with wellness_col2:
            st.markdown(f"<div style='background: white; padding: 1.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'><h2 style='color: {status_color}; margin: 0;'>{status}</h2><p style='color: #666; margin: 0.5rem 0 0 0;'>Health Status</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summary Report
        st.markdown("<h3 style='color: #1f77b4;'>Auto-Generated Health Summary</h3>", unsafe_allow_html=True)
        summary_df = generate_summary_report(st.session_state.clean_df)
        st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("---")
        
        # Export Options
        st.markdown("<h3 style='color: #1f77b4;'>Export Your Data</h3>", unsafe_allow_html=True)
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.download_button(
                label="📥 Download Summary Report (CSV)",
                data=summary_df.to_csv(index=False),
                file_name="fitpulse_health_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=st.session_state.clean_df.to_csv(index=False),
                file_name="fitpulse_cleaned_fitness_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Please run the preprocessing pipeline to view your health intelligence dashboard")
