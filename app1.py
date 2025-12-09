# app.py
# FitPulse – Milestone 2: TSFresh Feature Extraction + Clustering (KMeans / DBSCAN)
#
# Run with:
#   streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# 1. Streamlit page + custom styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="FitPulse – Milestone 2",
    page_icon="🧬",
    layout="wide",
)

# Simple custom CSS to make it look different from default
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617 55%, #000000 100%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont;
    }
    section.main > div {
        padding-top: 1rem;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 24px 60px rgba(0,0,0,0.6);
    }
    .accent {
        color: #22d3ee;
        font-weight: 600;
    }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        border: 1px solid #22d3ee55;
        font-size: 0.75rem;
        color: #a5b4fc;
        margin-right: 0.4rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        border: 1px solid rgba(55, 65, 81, 0.8);
        font-size: 0.8rem;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #e5e7eb;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .cluster-legend {
        font-size: 0.8rem;
        color: #cbd5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="glass-card">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
        <div>
          <h1 style="margin-bottom:0.2rem;">🧬 FitPulse – Milestone 2</h1>
          <p style="margin:0; color:#9ca3af; font-size:0.9rem;">
            TSFresh feature extraction + clustering (KMeans / DBSCAN) on preprocessed fitness signals.
          </p>
        </div>
        <div style="text-align:right; font-size:0.8rem;">
          <span class="pill">Milestone 1 → Cleaned CSV</span>
          <span class="pill">Milestone 2 → Features & Patterns</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. Helper – sample data (if you want to test quickly)
# ---------------------------------------------------------
def create_sample_heart_rate():
    """Create 1-day, 1-minute heart-rate sample data."""
    timestamps = pd.date_range("2024-01-15 08:00:00", "2024-01-15 16:00:00", freq="1min")
    base_hr = 70
    values = []
    for ts in timestamps:
        tod = ts.hour + ts.minute / 60
        factor = 1.0
        if 9 <= tod < 10:
            factor = 1.5
        elif 14 <= tod < 15:
            factor = 1.3
        noise = np.random.normal(0, 3)
        hr = base_hr * factor + noise
        values.append(max(50, min(150, hr)))
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": values})
    return df


# ---------------------------------------------------------
# 3. Sidebar – data + algorithm configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙ Configuration")

    data_mode = st.radio(
        "Data source",
        options=["Upload preprocessed CSV", "Use sample heart-rate data"],
        index=0,
    )

    uploaded_file = None
    if data_mode == "Upload preprocessed CSV":
        uploaded_file = st.file_uploader(
            "Cleaned CSV from Milestone 1",
            type=["csv"],
            help="File must contain a 'timestamp' column + one or more numeric columns.",
        )

    # Feature extraction config
    st.subheader("📐 Feature extraction")
    window_size = st.slider(
        "Window size (number of points per window)",
        min_value=20,
        max_value=300,
        value=60,
        step=10,
    )
    overlap = st.slider(
        "Window overlap (%)",
        min_value=0,
        max_value=90,
        value=50,
        step=10,
    )

    # Clustering config
    st.subheader("🧩 Clustering")
    cluster_algo = st.selectbox(
        "Clustering algorithm",
        options=["None (just extract features)", "KMeans", "DBSCAN"],
        index=1,
    )

    if cluster_algo == "KMeans":
        n_clusters = st.slider(
            "Number of clusters (K)",
            min_value=2,
            max_value=10,
            value=3,
        )
        eps = None
        min_samples = None
    elif cluster_algo == "DBSCAN":
        eps = st.slider("eps (neighbourhood radius)", 0.1, 5.0, 1.5, 0.1)
        min_samples = st.slider("min_samples", 2, 30, 5, 1)
        n_clusters = None
    else:
        n_clusters = None
        eps = None
        min_samples = None


# ---------------------------------------------------------
# 4. Load data
# ---------------------------------------------------------
if data_mode == "Use sample heart-rate data":
    df_raw = create_sample_heart_rate()
    st.info("Using internally generated sample heart-rate data (1-minute resolution).")
else:
    if uploaded_file is None:
        st.warning("📂 Please upload a cleaned CSV from Milestone 1 in the sidebar.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# Basic checks & timestamp parsing
if "timestamp" not in df_raw.columns:
    st.error("CSV must contain a 'timestamp' column.")
    st.stop()

df = df_raw.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

numeric_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric columns found besides 'timestamp'. At least one signal column is required.")
    st.stop()

# ---------------------------------------------------------
# 5. Top layout: data preview + metrics
# ---------------------------------------------------------
top_col1, top_col2 = st.columns([2, 1], gap="large")

with top_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Input data preview")
    st.dataframe(df.head(200), use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Snapshot")
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Rows</div>
            <div class="metric-value">{len(df)}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Signals</div>
            <div class="metric-value">{len(numeric_cols)}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Start</div>
            <div class="metric-value">{start_ts}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">End</div>
            <div class="metric-value">{end_ts}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 6. Choose which signal to analyse (heart_rate, etc.)
# ---------------------------------------------------------
st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
st.markdown("### 🎯 Choose target signal & run Milestone 2")

signal_col = st.selectbox(
    "Signal to analyse (from numeric columns)",
    options=numeric_cols,
    index=0,
    help="For example: heart_rate, hrv, stress_score, skin_temp…",
)

run_button = st.button("🚀 Run TSFresh feature extraction & clustering", type="primary")
st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 7. Core logic – TSFresh + clustering
# ---------------------------------------------------------
def build_tsfresh_input(df_signal: pd.DataFrame, window_size: int, overlap_pct: int):
    """
    Convert a single-column time-series into TSFresh format:
    columns: window_id, timestamp, value
    """
    df_sorted = df_signal.sort_values("timestamp").reset_index(drop=True)
    step_size = max(1, int(window_size * (1 - overlap_pct / 100.0)))
    prepared = []
    window_id = 0

    for start in range(0, len(df_sorted) - window_size + 1, step_size):
        chunk = df_sorted.iloc[start:start + window_size].copy()
        chunk["window_id"] = window_id
        prepared.append(chunk[["window_id", "timestamp", signal_col]])
        window_id += 1

    if not prepared:
        return pd.DataFrame()

    df_prep = pd.concat(prepared, ignore_index=True)
    df_prep = df_prep.rename(columns={signal_col: "value"})
    return df_prep


def get_effient_fc_parameters():
    """
    Small but meaningful feature set (mean, std, min, max, quantiles, trend, autocorr, entropy).
    This is a custom subset, not the full comprehensive set.
    """
    return {
        "mean": None,
        "median": None,
        "standard_deviation": None,
        "variance": None,
        "minimum": None,
        "maximum": None,
        "skewness": None,
        "kurtosis": None,
        "quantile": [{"q": 0.25}, {"q": 0.75}],
        "abs_energy": None,
        "absolute_sum_of_changes": None,
        "count_above_mean": None,
        "count_below_mean": None,
        "autocorrelation": [{"lag": 1}, {"lag": 2}],
        "linear_trend": [{"attr": "slope"}],
        "approximate_entropy": [{"m": 2, "r": 0.2}],
    }


def run_tsfresh_and_clustering(df, signal_col, window_size, overlap_pct,
                               cluster_algo, n_clusters=None, eps=None, min_samples=None):
    # Prepare TSFresh input
    df_signal = df[["timestamp", signal_col]].dropna()
    tsfresh_input = build_tsfresh_input(df_signal, window_size, overlap_pct)

    if tsfresh_input.empty:
        st.error("Not enough data to create windows for feature extraction. Try reducing the window size.")
        return None, None, None

    st.info(f"Extracting TSFresh features from *{signal_col}* "
            f"using windows of {window_size} points (overlap {overlap_pct}%).")

    fc_params = get_effient_fc_parameters()

    feature_matrix = extract_features(
        tsfresh_input,
        column_id="window_id",
        column_sort="timestamp",
        default_fc_parameters=fc_params,
        disable_progressbar=True,
        n_jobs=1,
    )

    feature_matrix = impute(feature_matrix)   # handle NaNs

    # Remove constant columns
    constant_cols = [c for c in feature_matrix.columns if feature_matrix[c].std() == 0]
    if constant_cols:
        feature_matrix = feature_matrix.drop(columns=constant_cols)

    if feature_matrix.empty:
        st.error("Feature matrix is empty after removing constant features.")
        return None, None, None

    # Clustering
    labels = None
    cluster_info = None

    if cluster_algo != "None (just extract features)":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix.values)

        if cluster_algo == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            cluster_info = {
                "algorithm": "KMeans",
                "inertia": float(model.inertia_),
                "iterations": int(model.n_iter_),
            }
        elif cluster_algo == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            n_noise = int(np.sum(labels == -1))
            cluster_info = {
                "algorithm": "DBSCAN",
                "eps": float(eps),
                "min_samples": int(min_samples),
                "n_noise_points": n_noise,
                "noise_percentage": float(n_noise / len(labels) * 100),
            }

    return feature_matrix, labels, cluster_info


# ---------------------------------------------------------
# 8. Run + Visualisation
# ---------------------------------------------------------
if run_button:
    with st.spinner("Running TSFresh feature extraction and clustering..."):
        feat_matrix, labels, cluster_info = run_tsfresh_and_clustering(
            df=df,
            signal_col=signal_col,
            window_size=window_size,
            overlap_pct=overlap,
            cluster_algo=cluster_algo,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples,
        )

    if feat_matrix is not None:
        st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown(f"### 📦 Feature matrix for {signal_col}")
        st.write(f"Windows: *{feat_matrix.shape[0]}, Features per window: **{feat_matrix.shape[1]}*")
        st.dataframe(feat_matrix.head(), use_container_width=True, height=260)

        # Show basic stats of top-variance features
        variances = feat_matrix.var().sort_values(ascending=False)
        top_feats = variances.head(10).index.tolist()
        st.markdown("*Top 10 most variable features* (by variance):")
        st.dataframe(feat_matrix[top_feats].describe().T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # If clustering was applied, show cluster summary + PCA plot
        if labels is not None:
            st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
            st.markdown("### 🧭 Clustering results")

            # Summary table
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_df = pd.DataFrame(
                {
                    "Cluster label": unique_labels,
                    "Count": counts,
                    "Percentage": (counts / len(labels) * 100).round(2),
                }
            ).sort_values("Count", ascending=False)
            st.dataframe(cluster_df, use_container_width=True)

            if cluster_info is not None:
                st.markdown("*Algorithm details*")
                st.json(cluster_info)

            # 2D PCA scatter
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(feat_matrix.values)
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)

                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(
                    X_pca[:, 0],
                    X_pca[:, 1],
                    c=labels,
                    cmap="tab10",
                    alpha=0.85,
                    edgecolor="k",
                    linewidths=0.3,
                )
                ax.set_xlabel("PC 1")
                ax.set_ylabel("PC 2")
                ax.set_title(f"PCA view of TSFresh feature space – {signal_col}")
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown(
                    '<p class="cluster-legend">Each point = one time window of the '
                    f'<span class="accent">{signal_col}</span> signal, coloured by cluster label.</p>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"Could not generate PCA scatter plot: {e}")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success("✅ TSFresh feature extraction done (no clustering selected).")