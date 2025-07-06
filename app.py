import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy import signal
from enum import Enum
import colorednoise

# --- Enums ---
class SeriesType(str, Enum):
    RANDOM_WALK = "Random Walk (Brown Noise)"
    WHITE_NOISE = "White Noise"
    PINK_NOISE = "Pink Noise"
    OU_PROCESS = "Ornstein-Uhlenbeck"
    CUSTOM = "Custom"

class FillMethod(str, Enum):
    NONE = "None"
    FORWARD = "Forward Fill"
    ZERO = "Fill with Zero"

class TrendType(str, Enum):
    NONE = "None"
    LINEAR = "Linear"
    QUADRATIC = "Quadratic"
    EXPONENTIAL = "Exponential"

class SeasonalityType(str, Enum):
    NONE = "None"
    SINE = "Sine"
    SAWTOOTH = "Sawtooth"
    TRIANGLE = "Triangle"

# --- Functions ---
def ornstein_uhlenbeck_process(num_points, theta, mu, sigma):
    dt = 1
    ou = np.zeros(num_points)
    for t in range(1, num_points):
        ou[t] = ou[t-1] + theta * (mu - ou[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return ou

def custom_time_series(num_points, cfg):
    t = np.arange(num_points)

    trend_type = cfg["trend_type"]
    seas_type = cfg["seas_type"]
    lin_slope = cfg.get("lin_slope", 0.5)
    lin_intercept = cfg.get("lin_intercept", 0.0)
    amplitude = cfg.get("amplitude", 1.0)
    period = cfg.get("period", 50)
    noise_std = cfg.get("noise_std", 0.0)

    if trend_type == TrendType.LINEAR.value:
        y = lin_slope * t + lin_intercept
    elif trend_type == TrendType.QUADRATIC.value:
        y = 0.01 * t**2
    elif trend_type == TrendType.EXPONENTIAL.value:
        y = np.exp(0.01 * t)
    else:
        y = np.zeros(num_points)

    if seas_type == SeasonalityType.SINE.value:
        seasonal = amplitude * np.sin(2 * np.pi * t / period)
    elif seas_type == SeasonalityType.SAWTOOTH.value:
        seasonal = amplitude * signal.sawtooth(2 * np.pi * t / period)
    elif seas_type == SeasonalityType.TRIANGLE.value:
        seasonal = amplitude * signal.sawtooth(2 * np.pi * t / period, width=0.5)
    else:
        seasonal = 0

    y += seasonal

    if noise_std > 0:
        y += np.random.normal(0, noise_std, num_points)

    return y

def generate_ts(config):
    np.random.seed(config["global"]["rand_seed"])
    num_points = config["global"]["num_points"]

    series_type = config["global"]["series_type"]

    if series_type == SeriesType.WHITE_NOISE.value:
        data = np.random.normal(0, 1, num_points)
    elif series_type == SeriesType.RANDOM_WALK.value:
        drift = config["random_walk"].get("rw_drift", 0.0)
        steps = np.random.normal(0, 1, num_points) + drift
        data = np.cumsum(steps)
    elif series_type == SeriesType.PINK_NOISE.value:
        data = colorednoise.powerlaw_psd_gaussian(1, num_points)
    elif series_type == SeriesType.OU_PROCESS.value:
        p = config["ou"]
        data = ornstein_uhlenbeck_process(num_points, p["theta"], p["mu"], p["sigma"])
    elif series_type == SeriesType.CUSTOM.value:
        data = custom_time_series(num_points, config["custom"])

    if not config["global"]["allow_negative"]:
        min_val = np.min(data)
        if min_val < 0:
            data = data - min_val

    if config["global"]["missing_pct"] > 0:
        np.random.seed(config["global"]["missing_seed"])
        mask = np.random.rand(num_points) < (config["global"]["missing_pct"] / 100)
        data[mask] = np.nan

    fill_method = config["global"]["fill_method"]
    if fill_method == FillMethod.FORWARD.value:
        data = pd.Series(data).ffill().to_numpy()
    elif fill_method == FillMethod.ZERO.value:
        data = pd.Series(data).fillna(0).to_numpy()

    start = pd.to_datetime(config["global"]["start_time"])
    interval = config["global"]["time_interval"]
    timestamps = pd.date_range(start=start, periods=num_points, freq=pd.to_timedelta(interval, unit='s'))

    return pd.DataFrame({"timestamp": timestamps, "value": data})

def summarize_ts(series: pd.Series) -> pd.DataFrame:
    stats = {
        "Mean": series.mean(),
        "Std Dev": series.std(),
        "Min": series.min(),
        "Max": series.max(),
        "Skewness": skew(series, nan_policy='omit'),
        "Kurtosis": kurtosis(series, nan_policy='omit'),
    }
    return pd.DataFrame([stats]).round(3)

# --- Layout and Page Config ---
st.set_page_config(layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .stSlider > label {
        font-size: 0.55rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .block-container {
        padding-top: 1rem !important;
    }

    div.stDownloadButton > button {
        background-color: #4B9DAF;
        color: white;
        border-radius: 0.4rem;
        padding: 0.4rem 1rem;
        font-weight: 600;
    }
    div.stDownloadButton > button:hover {
        background-color: #3A8A99;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Time Series Data Generator")

left_col, spacer, right_col = st.columns([5, 0.5, 5])

with left_col:
    st.markdown("Generate univariate time series data. Optionally save in .csv format.")

    config = {"global":{}, "ou":{}, "random_walk":{}, "custom":{}}


    st.subheader("Series Settings")
    col1, col2 = st.columns([1, 2])
    with col1:
        series_type = st.selectbox("Time Series Type", options=[s.value for s in SeriesType])
    config["global"]["series_type"] = series_type
    with col2:
        series_col1, series_col2, series_col3 = st.columns(3)
        if series_type == SeriesType.OU_PROCESS.value:
            with series_col1:
                config["ou"]["theta"] = st.slider("θ (mean reversion)", 0.0, 1.0, 0.2, 0.001)
            with series_col2:
                config["ou"]["mu"] = st.slider("μ (long-term mean)", -10.0, 10.0, 0.0, 0.1)
            with series_col3:
                config["ou"]["sigma"] = st.slider("σ (volatility)", 0.01, 2.0, 0.3, 0.01)

        elif series_type == SeriesType.RANDOM_WALK.value:
            with series_col1:
                rw_drift = st.slider("Drift", -0.2, 0.2, 0.0, 0.01)
            config["random_walk"] = {"rw_drift": rw_drift}

        elif series_type == SeriesType.CUSTOM.value:
            config["custom"] = {}
            with series_col1:
                trend_type = st.selectbox("Trend Component", options=[t.value for t in TrendType])
                config["custom"]["trend_type"] = trend_type
                if trend_type == TrendType.LINEAR.value:
                    with series_col2:
                        config["custom"]["lin_slope"] = st.slider("Slope", -5.0, 5.0, 0.5, step=0.1)
                    with series_col3:
                        config["custom"]["lin_intercept"] = st.slider("Intercept", -100.0, 100.0, 0.0, step=1.0)
                config["custom"]["seas_type"] = st.selectbox("Seasonality", options=[s.value for s in SeasonalityType])


    col_data, col_time = st.columns([3, 2])
    with col_data:
        st.markdown("### Data Settings")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            num_points = st.slider("Number of Points", 100, 1000, 300, step=50)
        with col2:
            rand_seed = st.slider("Rand Seed", 0, 42, 100, step=1)
        with col3:
            allow_negative = st.checkbox("Allow Negatives", value=True)

    with col_time:
        st.markdown("### Time Settings")
        col4, col5 = st.columns([2, 1])
        with col4:
            start_time = st.text_input("Starting Timestamp", value="2000-01-01 00:00:00")
        with col5:
            time_interval = st.number_input("Interval (sec)", min_value=1, value=60, step=1)

    st.subheader("Anomalies and Missing Values")
    col6, col7, col8 = st.columns([1, 1, 1])
    with col6:
        missing_pct = st.slider("Missing Data (%)", 0.0, 40.0, 0.0, step=0.5)
    with col7:
        missing_seed = st.slider("MV Rand Seed", 0, 42, 100, step=1)
    with col8:
        fill_method = st.selectbox("Fill Method", options=[f.value for f in FillMethod])

    config["global"].update({
        "num_points": num_points,
        "rand_seed": rand_seed,
        "allow_negative": allow_negative,
        "start_time": start_time,
        "time_interval": time_interval,
        "missing_pct": missing_pct,
        "missing_seed": missing_seed,
        "fill_method": fill_method,
    })

with right_col:
    df = generate_ts(config)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["value"],
        mode="lines",
        name=series_type,
        line=dict(width=1, color="#D2671A"),
    ))
    fig.update_layout(
        title="Generated Series",
        height=500,
        xaxis_title="Time",
        yaxis_title="Value",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary_df = summarize_ts(df["value"])
    st.dataframe(summary_df, use_container_width=False)

    csv = df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"time_series_{timestamp}.csv"
    st.download_button("Download CSV", data=csv, file_name=file_name, mime="text/csv")
