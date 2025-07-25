import streamlit as st
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy import signal
import colorednoise

from enums import SeriesType, FillMethod, TrendType, SeasonalityType, AnomalyType, MissingMode

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
            help="Average value of the noise."
        )
    with cols[2]:
        cfg["std"] = st.slider(
            "Std Dev", 
            0.1, 5.0, 1.0, 0.1,
            help="Standard deviation (volatility) of the noise."
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
        trend_type = st.selectbox(
            "Trend", 
            options=[t.value for t in TrendType],
            help="Adds long-term direction to the series."
        )
        cfg["trend_type"] = trend_type
    if trend_type == TrendType.LINEAR.value:
        with cols[1]:
            cfg["lin_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0,
                                             help="Starting value of the linear trend.")
        with cols[2]:
            cfg["lin_slope"] = st.slider("Slope", -1.0, 1.0, 0.05, step=0.01,
                                         help="Rate of increase or decrease in the linear trend.")
    elif trend_type == TrendType.QUADRATIC.value:
        with cols[1]:
            cfg["quad_intercept"] = st.slider("Intercept", -10.0, 10.0, 0.0, step=1.0,
                                              help="Starting value of the quadratic curve.")
        with cols[2]:
            cfg["quad_coef"] = st.slider("Quadratic Coef", -0.002, 0.002, 0.001, step=0.0001, format="%.4f",
                                         help="Controls the curvature of the quadratic trend.")
    elif trend_type == TrendType.EXPONENTIAL.value:
        with cols[1]:
            cfg["exp_base"] = st.slider("Base", 1.000, 1.005, 1.001, step=0.001, format="%.3f",
                                        help="Base of the exponential function (controls growth rate).")
        with cols[2]:
            cfg["exp_scale"] = st.slider("Scale", 0.0, 10.0, 1.0, step=0.1,
                                         help="Scale factor for exponential growth.")

    # Seasonality settings
    cols = st.columns(4)
    with cols[0]:
        seas_type = st.selectbox("Seasonality", options=[s.value for s in SeasonalityType],
                                 help="Seasonal pattern.")
        cfg["seas_type"] = seas_type
    if seas_type != SeasonalityType.NONE.value:
        with cols[1]:
            cfg["seas_amp"] = st.slider("Amplitude", 0.1, 10.0, 1.0, step=0.1,
                                        help="Height of the seasonal wave.")
        with cols[2]:
            cfg["seas_period"] = st.slider("Period", 5, 200, 50, step=5,
                                           help="Number of steps before the season repeats.")
        with cols[3]:
            if seas_type == SeasonalityType.SAWTOOTH.value:
                cfg["seas_width"] = st.slider("Wave Shape", 0.0, 1.0, 0.5, step=0.1,
                                              help="Skew of the sawtooth wave (0 = ramp up, 1 = ramp down).")

    # Cycle settings
    cols = st.columns([.25, .1875, .1875, .1875, .1875])
    with cols[0]:
        cycle_enabled = st.checkbox("Cycle", value=False,
                                    help="Simulates cycles (rises and falls without a fixed frequency).")
        cfg["cycle_enabled"] = cycle_enabled
    if cycle_enabled:
        with cols[1]:
            cfg["cyc_amp"] = st.slider("Amplitude", 0.0, 5.0, 1.0, 0.1, format="%.2f",
                                       help="Cycle height.")
        with cols[2]:
            cfg["cyc_freq"] = st.slider("Base Freq", 0.000, 0.1, 0.03, 0.005, format="%.3f",
                                        help="Base frequency of the cycle.")
        with cols[3]:
            cfg["cyc_var"] = st.slider("Freq Var", 0.0, 0.1, 0.01, 0.005, format="%.3f",
                                       help="Variation in frequency over time.")
        with cols[4]:
            cfg["cyc_decay"] = st.slider("Decay Rate", -0.01, 0.01, 0.0, 0.0005, format="%.3f",
                                         help="How quickly the cycle amplitude shrinks or grows.")

    # Noise settings
    cols = st.columns(4)
    with cols[0]:
        noise_enabled = st.checkbox("Noise", value=False,
                                    help="Adds random fluctuations to simulate randomness.")
        cfg["noise_enabled"] = noise_enabled
    if noise_enabled:
        with cols[1]:
            cfg["noise_beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f",
                                          help="Controls the spectral slope of the noise. 0 = white, 1 = pink, 2 = brownian.")
        with cols[2]:
            cfg["noise_mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1,
                                          help="Average value of the noise.")
        with cols[3]:
            cfg["noise_std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1,
                                         help="Standard deviation (volatility) of the noise.")

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
                help="Random seed used for generating the time series and anomaly noise. Missing values use a separate seed."
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
    cfg = {}

    with st.expander("Missing Values"):
        cols = st.columns([1, 0.3])
        with cols[0]:
            mode_cols = st.columns([1,1,1,0.7])
            with mode_cols[0]:
                missing_mode = st.selectbox(
                    "Mode", options=[m.value for m in MissingMode], index=0, 
                    help="Choose the type of missing data pattern to apply."
                )
            cfg["missing_mode"] = missing_mode

            if missing_mode == MissingMode.CLUSTER.value:
                with mode_cols[1]:
                    missing_pct = st.slider(
                        "Missing Data (%)", 0.0, 50.0, 0.0, step=0.5,
                        help="Percentage of values to randomly remove from the series."
                    )
                with mode_cols[2]:
                    clustering = st.slider(
                        "Gap Clustering", 0.0, 1.0, 0.0, step=0.01,
                        help="Controls how likely missing values are to appear in contiguous blocks.\n\n"
                            "0 = completely random, 1 = long stretches of missing values."
                    )
                with mode_cols[3]:
                    missing_seed = st.number_input(
                        "MV Seed", 0, 100, 42, step=1,
                        help="Random seed for missing values (does not affect time series generation)."
                    )
                cfg.update({
                    "missing_pct": missing_pct,
                    "gap_clustering": clustering,
                    "missing_seed": missing_seed,
                })

            elif missing_mode == MissingMode.NTH.value:
                with mode_cols[1]:
                    nth_value = st.number_input(
                        "Every Nth", 2, 100, 10, step=1,
                        help="Mask every Nth value in the time series."
                    )
                with mode_cols[2]:
                    span = st.number_input(
                        "Span", 1, nth_value, 1, step=1,
                        help="Number of consecutive values to mask at each step."
                    )
                with mode_cols[3]:
                    missing_seed = st.number_input(
                        "MV Rand Seed", 0, 1000, 42, step=1,
                        help="Random seed for missing values (does not affect time series generation)."
                    )
                cfg.update({
                    "nth_value": nth_value,
                    "nth_span": span,
                    "missing_seed": missing_seed,
                })

            elif missing_mode == MissingMode.CLIP.value:
                with mode_cols[1]:
                    threshold = st.number_input(
                        "Clip Threshold", 0.0, 1e6, 10.0, step=1.0,
                        help="Cutoff value for masking."
                    )
                with mode_cols[2]:
                    direction = st.selectbox(
                        "Direction", ["High", "Low", "Both"],
                        help="Choose to mask high, low values, or both."
                    )
                cfg.update({
                    "clip_threshold": threshold,
                    "clip_direction": direction,
                })

        # Fill method (shown regardless of mode)
        with cols[1]:
            missing_fill_method = st.selectbox(
                "Fill Method", options=[f.value for f in FillMethod],
                help="Choose how to fill in missing values. Forward fill: fill with last known value."
            )
        cfg["missing_fill_method"] = missing_fill_method

    return cfg


def render_anomaly_controls(num_points) -> dict[str, any]:
    with st.expander("Anomalies"):
        cfg = {}

        cols = st.columns([0.8, 0.9, 0.5, 0.6, 0.6])
        with cols[0]:
            anomaly_type = st.selectbox(
                "Anomaly Type",
                [a.value for a in AnomalyType]
            )
            cfg["anomaly_type"] = anomaly_type

            if anomaly_type == AnomalyType.CUSTOM.value:
                        
                with cols[1]:
                    idx_range = st.slider(
                        "Range (index)", 
                        0, num_points, 
                        (int(num_points * 0.3), int(num_points * 0.4)),
                        help="Index range of the anomaly. Must span at least one point. "
                             "To simulate a level shift or trend change, stretch the range toward the end."
                    )
                with cols[2]:
                    anomaly_mode = st.selectbox(
                        "Mode", 
                        ["Add", "Mult", "Slope"], 
                        index=0,
                        help="Choose how values are modified. "
                             "'Add' offsets values, 'Mult' scales them, 'Slope' applies a linear ramp."
                    )
                with cols[3]:
                    if anomaly_mode in ["Add", "Mult"]:
                        magnitude = st.slider(
                            "Magnitude (%)", 
                            -300.0, 300.0, 100.0, step=10.0,
                            help="Magnitude of the anomaly, as percentage of the original signal."
                        )
                        cfg["magnitude_pct"] = magnitude

                    elif anomaly_mode == "Slope":
                        slope = st.slider(
                            "Slope", 
                            -1.0, 1.0, 0.1, step=0.01,
                            help="Slope applied across the selected range. "
                                 "Positive = upward trend, negative = downward trend."
                        )
                        cfg["slope"] = slope

                # Add noise slider (always shown when anomaly is selected)
                with cols[4]:
                    noise_std = st.slider(
                        "Noise Std", 
                        0.0, 3.0, 0.0, step=0.1,
                        help="Amount of random noise to inject in the selected range.")
                    cfg["noise_std"] = noise_std

                cfg["range"] = idx_range
                cfg["mode"] = anomaly_mode

        return cfg


def render_plot_settings() -> dict:
    with st.expander("Plot Settings"):
        cols = st.columns([0.6, 1, 1, 0.8])
        with cols[0]:
            use_lines = st.checkbox(
                "Line plot",
                value=True,
                help="Connect data points with lines instead of showing individual markers."
            )
        with cols[1]:
            show_missing = st.checkbox(
                "Missing values overlay",
                value=True,
                help="Highlight regions where missing values were added."
            )
        with cols[2]:
            show_anomalies = st.checkbox(
                "Anomalies overlay",
                value=True,
                help="Highlight regions where anomalies were added."
            )

    return {
        "use_lines": use_lines,
        "show_anomalies": show_anomalies,
        "show_missing": show_missing,
    }

def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- Styling ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Layout and Page Config ---
st.set_page_config(
    layout="wide",
    page_title="Visual Time Series Generator",
    page_icon="images/favicon.png"
)
# st.set_page_config()

left_col, spacer, right_col = st.columns([5, 0.5, 5])

with left_col:
    st.markdown('<div class="boxed-title">Visual Time Series Generator</div>', unsafe_allow_html=True)

    st.markdown(
        "##### Generate univariate time series data using a visual approach."
    )

    with st.expander("About", expanded=False):
        tab_overview, tab_noise, tab_ou, tab_custom, tab_missing, tab_anomalies, tab_save = st.tabs([
            "Overview", "Noise", "OU Process", "Custom TS", "Missing Values", "Anomalies", "Save to File"
        ])

        with tab_overview:
            st.markdown(load_markdown("docs/overview.md"))
        with tab_noise:
            st.markdown(load_markdown("docs/noise.md"))
        with tab_ou:
            st.markdown(load_markdown("docs/ou_process.md"))
        with tab_custom:
            st.markdown(load_markdown("docs/custom_ts.md"))
        with tab_missing:
            st.markdown(load_markdown("docs/missing_values.md"))
        with tab_anomalies:
            st.markdown(load_markdown("docs/anomalies.md"))
        with tab_save:
            st.markdown(load_markdown("docs/save.md"))

    with st.expander("Time Series", expanded=True):

        config = {"global": {}, "ou": {}, "custom": {}, "noise": {}}

        cols = st.columns([0.8, 3.0])

        with cols[0]:
            series_type = st.selectbox("Type", options=[s.value for s in SeriesType])
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
    
    plot_settings = render_plot_settings()
    

with right_col:

    df = generate_ts(config)

    fig = plot_series(df, series_type, plot_settings)

    st.plotly_chart(fig, use_container_width=True)

    summary_df = summarize_series(df["value"])

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


