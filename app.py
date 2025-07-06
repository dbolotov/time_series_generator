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
    RANDOM_WALK = "Random Walk"
    NOISE = "Noise"
    OU_PROCESS = "OU Process"
    CUSTOM = "Custom"
    # Temporarily keep:
    WHITE_NOISE = "White Noise"
    PINK_NOISE = "Pink Noise"

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

# --- Functions ---
def ornstein_uhlenbeck_process(num_points, theta, mu, sigma):
    dt = 1
    ou = np.zeros(num_points)
    for t in range(1, num_points):
        ou[t] = ou[t-1] + theta * (mu - ou[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return ou

def generate_cycle_component(num_points, amp=0.0, freq_base=0.00, freq_var=0.00, decay_rate=0.0):
    t = np.arange(num_points)

    # Slowly changing amplitude
    mod_amp = 1 + 0.5 * np.sin(2 * np.pi * t / (num_points * 0.8))

    # Slowly changing frequency
    cycle_freq = freq_base + freq_var * np.sin(2 * np.pi * t / (num_points * 0.6))
    cycle_phase = np.cumsum(cycle_freq)

    cycle = amp * mod_amp * np.sin(2 * np.pi * cycle_phase)

    # decay
    decay = np.exp(-decay_rate * t)
    cycle *= decay

    return cycle

def custom_time_series(num_points, cfg):
    t = np.arange(num_points)

    trend_type = cfg["trend_type"]
    seas_type = cfg["seas_type"]
    lin_slope = cfg.get("lin_slope", 0.0)
    lin_intercept = cfg.get("lin_intercept", 0.0)
    seas_amp = cfg.get("seas_amp", 1.0)
    seas_period = cfg.get("seas_period", 50)
    seas_width = cfg.get("seas_width", 1.0)
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
        seasonal = seas_amp * np.sin(2 * np.pi * t / seas_period)
    elif seas_type == SeasonalityType.SAWTOOTH.value:
        seasonal = seas_amp * signal.sawtooth(2 * np.pi * t / seas_period, width=seas_width)
    # elif seas_type == SeasonalityType.TRIANGLE.value:
    #     seasonal = seas_amp * signal.sawtooth(2 * np.pi * t / seas_period, width=0.5)
    else:
        seasonal = 0

    y += seasonal

    # --- Cycle ---
    if cfg.get("cycle_enabled"):
        cyc_amp = cfg.get("cyc_amp", 1.0)
        cyc_freq = cfg.get("cyc_freq", 0.03)
        cyc_var = cfg.get("cyc_var", 0.01)
        cyc_decay = cfg.get("cyc_decay", 0.0)
        cycle = generate_cycle_component(num_points, cyc_amp, cyc_freq, cyc_var, cyc_decay)
        y += cycle

    if noise_std > 0:
        y += np.random.normal(0, noise_std, num_points)

    return y


def generate_noise(num_points, config):
    cfg = config["noise"]
    beta = cfg.get("beta", 1.0)
    mean = cfg.get("mean", 0.0)
    std = cfg.get("std", 1.0)
    n_drift = cfg.get("n_drift", 0.0)
    seed = config["global"].get("rand_seed", None)

    noise = colorednoise.powerlaw_psd_gaussian(beta, num_points, random_state=seed)
    noise = noise * std + mean

    if n_drift != 0:
        noise += np.linspace(0, n_drift * num_points, num_points)

    return noise


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
    elif series_type == SeriesType.NOISE.value:
        data = generate_noise(num_points, config)

    if not config["global"]["allow_negative"]:
        min_val = np.min(data)
        if min_val < 0:
            data = data - min_val

    if config["global"]["missing_pct"] > 0:
        np.random.seed(config["global"]["missing_seed"])
        mask = np.random.rand(num_points) < (config["global"]["missing_pct"] / 100)
        data[mask] = np.nan

    missing_fill_method = config["global"]["missing_fill_method"]
    if missing_fill_method == FillMethod.FORWARD.value:
        data = pd.Series(data).ffill().to_numpy()
    elif missing_fill_method == FillMethod.ZERO.value:
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

    config = {"global": {}, "ou": {}, "random_walk": {}, "custom": {}, "noise": {}}


    st.subheader("Series Settings")
    col1, col2 = st.columns([1, 3.0])
    with col1:
        series_type = st.selectbox("Time Series Type", options=[s.value for s in SeriesType])
    config["global"]["series_type"] = series_type
    with col2:
        
        if series_type == SeriesType.OU_PROCESS.value:
            series_col1, series_col2, series_col3 = st.columns(3)
            with series_col1:
                config["ou"]["theta"] = st.slider("θ (mean reversion)", 0.0, 1.0, 0.2, 0.001)
            with series_col2:
                config["ou"]["mu"] = st.slider("μ (long-term mean)", -10.0, 10.0, 0.0, 0.1)
            with series_col3:
                config["ou"]["sigma"] = st.slider("σ (volatility)", 0.01, 2.0, 0.3, 0.01)

        elif series_type == SeriesType.RANDOM_WALK.value:
            series_col1, series_col2, series_col3 = st.columns(3)
            with series_col1:
                rw_drift = st.slider("Drift", -0.2, 0.2, 0.0, 0.01)
            config["random_walk"] = {"rw_drift": rw_drift}

        elif series_type == SeriesType.CUSTOM.value:
            config["custom"] = {}

            # First row: Trend settings
            trend_col1, trend_col2, trend_col3, trend_col4 = st.columns(4)
            with trend_col1:
                trend_type = st.selectbox("Trend Component", options=[t.value for t in TrendType])
                config["custom"]["trend_type"] = trend_type
            if trend_type == TrendType.LINEAR.value:
                with trend_col2:
                    config["custom"]["lin_slope"] = st.slider("Slope", -1.0, 1.0, 0.0, step=0.01)
                with trend_col3:
                    config["custom"]["lin_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0)

            # Second row: Seasonality settings
            seas_col1, seas_col2, seas_col3, seas_col4 = st.columns(4)
            with seas_col1:
                seas_type = st.selectbox("Seasonality", options=[s.value for s in SeasonalityType])
                config["custom"]["seas_type"] = seas_type

            if seas_type != SeasonalityType.NONE.value:
                with seas_col2:
                    config["custom"]["seas_amp"] = st.slider("Amplitude", 0.1, 10.0, 1.0, step=0.1)
                with seas_col3:
                    config["custom"]["seas_period"] = st.slider("Period", 5, 200, 50, step=5)
                with seas_col4:
                    if seas_type == SeasonalityType.SAWTOOTH.value:
                        config["custom"]["seas_width"] = st.slider("Wave Shape", 0.0, 1.0, 0.5, step=0.1)

            # Third row: Cycle settings
            cycle_enabled = st.checkbox("Enable Cycle", value=False)
            config["custom"]["cycle_enabled"] = cycle_enabled

            if cycle_enabled:
                cyc_col1, cyc_col2, cyc_col3, cyc_col4 = st.columns(4)
                with cyc_col1:
                    config["custom"]["cyc_amp"] = st.slider("Amplitude", 0.0, 5.0, 1.0, 0.1, format="%.2f")
                with cyc_col2:
                    config["custom"]["cyc_freq"] = st.slider("Base Freq", 0.000, 0.1, 0.03, 0.005, format="%.3f")
                with cyc_col3:
                    config["custom"]["cyc_var"] = st.slider("Freq Variability", 0.0, 0.1, 0.01, 0.005, format="%.3f")
                with cyc_col4:
                    config["custom"]["cyc_decay"] = st.slider("Decay Rate", -0.01, 0.01, 0.0, 0.0005, format="%.3f")

        elif series_type == SeriesType.NOISE.value:
            noise_col1, noise_col2, noise_col3, noise_col4 = st.columns(4)
            with noise_col1:
                config["noise"]["beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f")
            with noise_col2:
                config["noise"]["mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
            with noise_col3:
                config["noise"]["std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1)
            with noise_col4:
                config["noise"]["n_drift"] = st.slider("Drift", -0.2, 0.2, 0.0, 0.01)


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
        missing_fill_method = st.selectbox("Fill Method", options=[f.value for f in FillMethod])

    config["global"].update({
        "num_points": num_points,
        "rand_seed": rand_seed,
        "allow_negative": allow_negative,
        "start_time": start_time,
        "time_interval": time_interval,
        "missing_pct": missing_pct,
        "missing_seed": missing_seed,
        "missing_fill_method": missing_fill_method,
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
