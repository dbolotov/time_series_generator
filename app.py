import streamlit as st
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy import signal
import colorednoise

from enums import SeriesType, FillMethod, TrendType, SeasonalityType, AnomalyType

from functions import (
    generate_ts,
    summarize_series,
    plot_series
)

# --- Functions ---
def render_ou_controls() -> dict[str, any]:
    cfg = {}
    cols = st.columns(3)
    with cols[0]:
        cfg["theta"] = st.slider(
            "θ (Mean reversion)", 
            0.0, 1.0, 0.2, 0.001,
            help="Speed at which values return to the long-term mean (μ). Higher = quicker reversion."
        )
    with cols[1]:
        cfg["mu"] = st.slider(
            "μ (Long-term mean)", 
            -10.0, 10.0, 0.0, 0.1,
            help="The average value the series tends to revert toward."
        )
    with cols[2]:
        cfg["sigma"] = st.slider(
            "σ (Volatility)", 
            0.01, 2.0, 0.3, 0.01,
            help="Controls how noisy or volatile the process is."
        )
    return cfg


def render_noise_controls() -> dict[str, any]:
    cfg = {}
    cols = st.columns(4)
    with cols[0]:
        cfg["beta"] = st.slider(
            "β (Color)", 
            0.0, 2.0, 1.0, 0.1, format="%.1f",
            help="Controls the spectral slope of the noise. 0 = white, 1 = pink, 2 = brownian."
        )
    with cols[1]:
        cfg["mean"] = st.slider(
            "Mean", 
            -5.0, 5.0, 0.0, 0.1,
            help="Center value around which noise is distributed."
        )
    with cols[2]:
        cfg["std"] = st.slider(
            "Std Dev", 
            0.1, 5.0, 1.0, 0.1,
            help="Spread (volatility) of the noise values."
        )
    with cols[3]:
        cfg["drift"] = st.slider(
            "Drift", 
            -0.2, 0.2, 0.0, 0.01,
            help="Adds a constant upward or downward trend over time."
        )
    return cfg

def render_custom_series_controls() -> dict[str, any]:
    cfg = {}

    # Trend settings
    cols = st.columns(4)
    with cols[0]:
        trend_type = st.selectbox("Trend", options=[t.value for t in TrendType])
        cfg["trend_type"] = trend_type
    if trend_type == TrendType.LINEAR.value:
        with cols[1]:
            cfg["lin_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0)
        with cols[2]:
            cfg["lin_slope"] = st.slider("Slope", -1.0, 1.0, 0.05, step=0.05)
    elif trend_type == TrendType.QUADRATIC.value:
        with cols[1]:
            cfg["quad_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0)
        with cols[2]:
            cfg["quad_coef"] = st.slider("Quadratic Coef", -0.002, 0.002, 0.001, step=0.0001, format="%.4f")
    elif trend_type == TrendType.EXPONENTIAL.value:
        with cols[1]:
            cfg["exp_base"] = st.slider("Base", 1.000, 1.005, 1.001, step=0.001, format="%.3f")
        with cols[2]:
            cfg["exp_scale"] = st.slider("Scale", 0.0, 10.0, 1.0, step=0.1)

    # Seasonality settings
    cols = st.columns(4)
    with cols[0]:
        seas_type = st.selectbox("Seasonality", options=[s.value for s in SeasonalityType])
        cfg["seas_type"] = seas_type
    if seas_type != SeasonalityType.NONE.value:
        with cols[1]:
            cfg["seas_amp"] = st.slider("Amplitude", 0.1, 10.0, 1.0, step=0.1)
        with cols[2]:
            cfg["seas_period"] = st.slider("Period", 5, 200, 50, step=5)
        with cols[3]:
            if seas_type == SeasonalityType.SAWTOOTH.value:
                cfg["seas_width"] = st.slider("Wave Shape", 0.0, 1.0, 0.5, step=0.1)

    # Cycle settings
    cols = st.columns([.25, .1875, .1875, .1875, .1875])
    with cols[0]:
        cycle_enabled = st.checkbox("Cycle", value=False)
        cfg["cycle_enabled"] = cycle_enabled
    if cycle_enabled:
        with cols[1]:
            cfg["cyc_amp"] = st.slider("Amplitude", 0.0, 5.0, 1.0, 0.1, format="%.2f")
        with cols[2]:
            cfg["cyc_freq"] = st.slider("Base Freq", 0.000, 0.1, 0.03, 0.005, format="%.3f")
        with cols[3]:
            cfg["cyc_var"] = st.slider("Freq Var", 0.0, 0.1, 0.01, 0.005, format="%.3f")
        with cols[4]:
            cfg["cyc_decay"] = st.slider("Decay Rate", -0.01, 0.01, 0.0, 0.0005, format="%.3f")

    # Noise settings
    cols = st.columns(4)
    with cols[0]:
        noise_enabled = st.checkbox("Noise", value=False)
        cfg["noise_enabled"] = noise_enabled
    if noise_enabled:
        with cols[1]:
            cfg["noise_beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f")
        with cols[2]:
            cfg["noise_mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
        with cols[3]:
            cfg["noise_std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1)

    return cfg


def render_data_and_time_controls() -> dict[str, any]:
    with st.expander("Data & Time"):
        cols = st.columns([1, 1, 1, 2, 1, 1])
        with cols[0]:
            num_points = st.number_input(
                "Data Points", 
                10, 10000, 300, step=10,
                help="Total number of time steps to generate."
            )
        with cols[1]:
            rand_seed = st.number_input(
                "Rand Seed", 
                0, 100, 42, step=1,
                help="Random seed used for generating the time series (does not affect missing values)."
            )
        with cols[2]:
            allow_negative = st.checkbox(
                "Allow Neg", value=True,
                help="If unchecked, values will be shifted to be non-negative."
            )
        with cols[3]:
            start_time = st.text_input(
                "Starting Timestamp", value="2000-01-01 00:00:00",
                help="Start time for the generated series (format: 'YYYY-MM-DD HH:MM:SS')."
            )
        with cols[4]:
            time_interval = st.number_input(
                "Interval", min_value=1, value=60, step=1,
                help="Number of time units between each data point."
            )
        with cols[5]:
            interval_unit = st.selectbox(
                "Interval Unit", options=["ms", "s", "min", "h", "D"],
                format_func=lambda x: {
                    "ms": "ms", "s": "sec", "min": "min", 
                    "h": "hr", "D": "day"}.get(x, x),
                help="Time unit for the interval between steps."
            )

    return {
        "num_points": num_points,
        "rand_seed": rand_seed,
        "allow_negative": allow_negative,
        "start_time": start_time,
        "time_interval": time_interval,
        "interval_unit": interval_unit
    }

def render_missing_data_controls() -> dict[str, any]:
    with st.expander("Missing Values"):
        cols = st.columns([1, 1, 1])
        with cols[0]:
            missing_pct = st.slider(
                "Missing Data (%)", 0.0, 40.0, 0.0, step=0.5,
                help="Percentage of values to randomly remove from the series."
            )
        with cols[1]:
            missing_seed = st.number_input(
                "MV Rand Seed", 0, 100, 42, step=1,
                help="Random seed for missing values (does not affect time series generation)."
            )
        with cols[2]:
            missing_fill_method = st.selectbox(
                "Fill Method", options=[f.value for f in FillMethod],
                help="Choose how to fill in missing values. Forward fill: fill with last known value."
            )
    return {
        "missing_pct": missing_pct,
        "missing_seed": missing_seed,
        "missing_fill_method": missing_fill_method,
    }

def render_anomaly_controls(num_points) -> dict[str, any]:
    with st.expander("Anomalies"):
        cfg = {}

        cols = st.columns([0.8, 1, 0.5, 1])
        with cols[0]:
            anomaly_type = st.selectbox(
                "Anomaly Type",
                [a.value for a in AnomalyType]
            )
            cfg["anomaly_type"] = anomaly_type

            if anomaly_type == AnomalyType.VALUE_SPIKE.value:
                        
                with cols[1]:
                    idx_range = st.slider(
                        "Range (index)", 
                        0, num_points - 1, 
                        (int(num_points * 0.3), int(num_points * 0.4)),
                        help="Index range of the anomaly. Must span at least one point."
                    )
                with cols[2]:
                    anomaly_mode = st.selectbox(
                        "Mode", 
                        ["Mult", "Add"], 
                        index=0,
                        help="Multiplicative (scale values) or Additive (offset values) anomaly."
                    )
                with cols[3]:
                    magnitude = st.slider(
                        "Magnitude (% of original)", 
                        -300.0, 300.0, 100.0, step=10.0,
                        help="How strong the anomaly is. Negative values reduce the original signal."
                    )
                
                cfg["range"] = idx_range
                cfg["mode"] = anomaly_mode
                cfg["magnitude_pct"] = magnitude

        return cfg


# --- Styling ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Layout and Page Config ---
st.set_page_config(layout="wide")

# st.markdown('<div class="boxed-title">Visual Time Series Generator</div>', unsafe_allow_html=True)

left_col, spacer, right_col = st.columns([5, 0.5, 5])

with left_col:
    st.markdown('<div class="boxed-title">Visual Time Series Generator</div>', unsafe_allow_html=True)

    st.markdown(
        "##### Generate univariate time series data using a visual approach.\n\n"
        "Choose from:\n"
        "- **Noise**: Colored random noise with optional drift\n"
        "- **OU Process** – Mean-reverting stochastic process (Ornstein–Uhlenbeck)\n"
        "- **Custom**: Combine trend, seasonality, cycles, and noise components\n\n"
        "Optionally add missing values, apply simple fill methods, and inject labeled anomalies.\n\n"
        "Files saved as .csv will contain `timestamp, value, missing, anomaly`, "
        "where `missing` and `anomaly` are boolean flags."
    )

    with st.expander("Time Series", expanded=True):

        config = {"global": {}, "ou": {}, "custom": {}, "noise": {}}

        cols = st.columns([1, 3.0])

        with cols[0]:
            series_type = st.selectbox("Time Series Type", options=[s.value for s in SeriesType])
        config["global"]["series_type"] = series_type

        with cols[1]:

            if series_type == SeriesType.OU_PROCESS.value:
                config["ou"] = render_ou_controls()

            elif series_type == SeriesType.NOISE.value:
                config["noise"] = render_noise_controls()

            elif series_type == SeriesType.CUSTOM.value:
                config["custom"] = render_custom_series_controls()


    data_time_cfg = render_data_and_time_controls()
    missing_cfg = render_missing_data_controls()

    config["global"].update(data_time_cfg)
    config["global"].update(missing_cfg)

    anomaly_cfg = render_anomaly_controls(config["global"]["num_points"])
    config["global"].update(anomaly_cfg)


    # reusing rand_seed from global settings for the custom series.
    # do this here because config["global"]["rand_seed"] needs to be set first.
    if config["global"]["series_type"] == SeriesType.CUSTOM.value:
        config["custom"]["rand_seed"] = config["global"]["rand_seed"]
    

with right_col:

    df = generate_ts(config)

    fig = plot_series(df, series_type)

    st.plotly_chart(fig, use_container_width=True)

    summary_df = summarize_series(df["value"])

    # column_names = ["Mean", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"]
    colnames = summary_df.columns.tolist()
    column_config = {
        col: st.column_config.NumberColumn(col, width="small") for col in colnames
    }



    st.markdown("Series summary statistics:")
    st.dataframe(
    summary_df,
    use_container_width=False,
    hide_index=True,
    column_config=column_config,
    )
    

    csv = df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"time_series_{timestamp}.csv"
    st.download_button("Download CSV", data=csv, file_name=file_name, mime="text/csv")


