import streamlit as st
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy import signal
import colorednoise

from enums import SeriesType, FillMethod, TrendType, SeasonalityType

from functions import (
    generate_ts,
    summarize_series,
    plot_series
)

# --- Functions ---
def render_ou_controls() -> dict[str, any]:
    cfg = {}
    series_col1, series_col2, series_col3 = st.columns(3)
    with series_col1:
        cfg["theta"] = st.slider("θ (Mean reversion)", 0.0, 1.0, 0.2, 0.001)
    with series_col2:
        cfg["mu"] = st.slider("μ (Long-term mean)", -10.0, 10.0, 0.0, 0.1)
    with series_col3:
        cfg["sigma"] = st.slider("σ (Volatility)", 0.01, 2.0, 0.3, 0.01)
    return cfg

def render_noise_controls() -> dict[str, any]:
    cfg = {}
    noise_col1, noise_col2, noise_col3, noise_col4 = st.columns(4)
    with noise_col1:
        cfg["beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f")
    with noise_col2:
        cfg["mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
    with noise_col3:
        cfg["std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1)
    with noise_col4:
        cfg["drift"] = st.slider("Drift", -0.2, 0.2, 0.0, 0.01)
    return cfg

def render_custom_series_controls() -> dict[str, any]:
    cfg = {}

    # Trend settings
    trend_col1, trend_col2, trend_col3, trend_col4 = st.columns(4)
    with trend_col1:
        trend_type = st.selectbox("Trend", options=[t.value for t in TrendType])
        cfg["trend_type"] = trend_type
    if trend_type == TrendType.LINEAR.value:
        with trend_col2:
            cfg["lin_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0)
        with trend_col3:
            cfg["lin_slope"] = st.slider("Slope", -1.0, 1.0, 0.0, step=0.01)
    elif trend_type == TrendType.QUADRATIC.value:
        with trend_col2:
            cfg["quad_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0)
        with trend_col3:
            cfg["quad_coef"] = st.slider("Quadratic Coef", -0.002, 0.002, 0.001, step=0.0001, format="%.4f")
    elif trend_type == TrendType.EXPONENTIAL.value:
        with trend_col2:
            cfg["exp_base"] = st.slider("Base", 1.000, 1.005, 1.001, step=0.001, format="%.3f")
        with trend_col3:
            cfg["exp_scale"] = st.slider("Scale", 0.0, 10.0, 1.0, step=0.1)

    # Seasonality settings
    seas_col1, seas_col2, seas_col3, seas_col4 = st.columns(4)
    with seas_col1:
        seas_type = st.selectbox("Seasonality", options=[s.value for s in SeasonalityType])
        cfg["seas_type"] = seas_type
    if seas_type != SeasonalityType.NONE.value:
        with seas_col2:
            cfg["seas_amp"] = st.slider("Amplitude", 0.1, 10.0, 1.0, step=0.1)
        with seas_col3:
            cfg["seas_period"] = st.slider("Period", 5, 200, 50, step=5)
        with seas_col4:
            if seas_type == SeasonalityType.SAWTOOTH.value:
                cfg["seas_width"] = st.slider("Wave Shape", 0.0, 1.0, 0.5, step=0.1)

    # Cycle settings
    cycle_enabled = st.checkbox("Cycle", value=False)
    cfg["cycle_enabled"] = cycle_enabled
    if cycle_enabled:
        cyc_col1, cyc_col2, cyc_col3, cyc_col4 = st.columns(4)
        with cyc_col1:
            cfg["cyc_amp"] = st.slider("Amplitude", 0.0, 5.0, 1.0, 0.1, format="%.2f")
        with cyc_col2:
            cfg["cyc_freq"] = st.slider("Base Freq", 0.000, 0.1, 0.03, 0.005, format="%.3f")
        with cyc_col3:
            cfg["cyc_var"] = st.slider("Freq Variability", 0.0, 0.1, 0.01, 0.005, format="%.3f")
        with cyc_col4:
            cfg["cyc_decay"] = st.slider("Decay Rate", -0.01, 0.01, 0.0, 0.0005, format="%.3f")

    # Noise settings
    noise_enabled = st.checkbox("Noise", value=False)
    cfg["noise_enabled"] = noise_enabled
    if noise_enabled:
        noise_col1, noise_col2, noise_col3 = st.columns(3)
        with noise_col1:
            cfg["noise_beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f")
        with noise_col2:
            cfg["noise_mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
        with noise_col3:
            cfg["noise_std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1)

    return cfg

def render_data_controls() -> dict[str, any]:
    st.markdown('<div class="section-header">Data Settings</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        num_points = st.slider("Number of Points", 20, 1000, 300, step=20)
    with col2:
        rand_seed = st.slider("Rand Seed", 0, 100, 42, step=1)
    with col3:
        allow_negative = st.checkbox("Allow Negatives", value=True)
    return {
        "num_points": num_points,
        "rand_seed": rand_seed,
        "allow_negative": allow_negative,
    }

def render_time_controls() -> dict[str, any]:
    st.markdown('<div class="section-header">Time Settings</div>', unsafe_allow_html=True)
    col4, col5 = st.columns([2, 1])
    with col4:
        start_time = st.text_input("Starting Timestamp", value="2000-01-01 00:00:00")
    with col5:
        time_interval = st.number_input("Interval (sec)", min_value=1, value=60, step=1)
    return {
        "start_time": start_time,
        "time_interval": time_interval,
    }

def render_missing_data_controls() -> dict[str, any]:
    st.markdown('<div class="section-header">Missing Value Settings</div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns([1, 1, 1])
    with col6:
        missing_pct = st.slider("Missing Data (%)", 0.0, 40.0, 0.0, step=0.5)
    with col7:
        missing_seed = st.slider("MV Rand Seed", 0, 100, 42, step=1)
    with col8:
        missing_fill_method = st.selectbox("Fill Method", options=[f.value for f in FillMethod])
    return {
        "missing_pct": missing_pct,
        "missing_seed": missing_seed,
        "missing_fill_method": missing_fill_method,
    }



# --- Styling ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Layout and Page Config ---
st.set_page_config(layout="wide")

st.markdown('<div class="boxed-title">Visual Time Series Generator</div>', unsafe_allow_html=True)

left_col, spacer, right_col = st.columns([5, 0.5, 5])

with left_col:
    st.markdown("Generate univariate time series data. Optionally save in .csv format.")
    st.markdown('<div class="section-header">Series Settings</div>', unsafe_allow_html=True)

    config = {"global": {}, "ou": {}, "custom": {}, "noise": {}}

    col1, col2 = st.columns([1, 3.0])

    with col1:
        series_type = st.selectbox("Time Series Type", options=[s.value for s in SeriesType])
    config["global"]["series_type"] = series_type

    with col2:

        if series_type == SeriesType.OU_PROCESS.value:
            config["ou"] = render_ou_controls()

        elif series_type == SeriesType.NOISE.value:
            config["noise"] = render_noise_controls()

        elif series_type == SeriesType.CUSTOM.value:
            config["custom"] = render_custom_series_controls()

    col_data, col_time = st.columns([3, 2])
    with col_data:
        data_cfg = render_data_controls()
    with col_time:
        time_cfg = render_time_controls()

    missing_cfg = render_missing_data_controls()

    config["global"].update(data_cfg)
    config["global"].update(time_cfg)
    config["global"].update(missing_cfg)

    # reusing rand_seed from global settings for the custom series.
    # do this here because config["global"]["rand_seed"] needs to be set first.
    if config["global"]["series_type"] == SeriesType.CUSTOM.value:
        config["custom"]["rand_seed"] = config["global"]["rand_seed"]

with right_col:
    df = generate_ts(config)

    fig = plot_series(df, series_type)

    st.plotly_chart(fig, use_container_width=True)

    summary_df = summarize_series(df["value"])
    st.dataframe(summary_df, hide_index=True, use_container_width=False)
    

    csv = df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"time_series_{timestamp}.csv"
    st.download_button("Download CSV", data=csv, file_name=file_name, mime="text/csv")
