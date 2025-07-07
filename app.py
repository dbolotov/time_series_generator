import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy import signal
import colorednoise

from enums import SeriesType, FillMethod, TrendType, SeasonalityType

from functions import (
    generate_ts,
    generate_noise,
    generate_custom_series,
    generate_ou_process,
    generate_cycle_component,
    summarize_series
)

# --- Styling ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Layout and Page Config ---
st.set_page_config(layout="wide")

st.markdown('<div class="boxed-title">Visual Time Series Generator</div>', unsafe_allow_html=True)

left_col, spacer, right_col = st.columns([5, 0.5, 5])

with left_col:
    st.markdown("Generate univariate time series data. Optionally save in .csv format.")

    config = {"global": {}, "ou": {}, "custom": {}, "noise": {}}


    # st.subheader("Series Settings")
    st.markdown('<div class="section-header">Series Settings</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3.0])
    with col1:
        series_type = st.selectbox("Time Series Type", options=[s.value for s in SeriesType])
    config["global"]["series_type"] = series_type
    with col2:
        
        if series_type == SeriesType.OU_PROCESS.value:
            series_col1, series_col2, series_col3 = st.columns(3)
            with series_col1:
                config["ou"]["theta"] = st.slider("θ (Mean reversion)", 0.0, 1.0, 0.2, 0.001)
            with series_col2:
                config["ou"]["mu"] = st.slider("μ (Long-term mean)", -10.0, 10.0, 0.0, 0.1)
            with series_col3:
                config["ou"]["sigma"] = st.slider("σ (Volatility)", 0.01, 2.0, 0.3, 0.01)

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
            
            # Fourth row: Noise settings
            custom_noise_enabled = st.checkbox("Add Noise", value=False)
            config["custom"]["noise_enabled"] = custom_noise_enabled
            

            if custom_noise_enabled:
                noise_col1, noise_col2, noise_col3 = st.columns(3)
                with noise_col1:
                    config["custom"]["noise_beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f")
                with noise_col2:
                    config["custom"]["noise_mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
                with noise_col3:
                    config["custom"]["noise_std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1)

        elif series_type == SeriesType.NOISE.value:
            noise_col1, noise_col2, noise_col3, noise_col4 = st.columns(4)
            with noise_col1:
                config["noise"]["beta"] = st.slider("β (Color)", 0.0, 2.0, 1.0, 0.1, format="%.1f")
            with noise_col2:
                config["noise"]["mean"] = st.slider("Mean", -5.0, 5.0, 0.0, 0.1)
            with noise_col3:
                config["noise"]["std"] = st.slider("Std Dev", 0.1, 5.0, 1.0, 0.1)
            with noise_col4:
                config["noise"]["drift"] = st.slider("Drift", -0.2, 0.2, 0.0, 0.01)


    col_data, col_time = st.columns([3, 2])
    with col_data:
        st.markdown('<div class="section-header">Data Settings</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            num_points = st.slider("Number of Points", 100, 1000, 300, step=50)
        with col2:
            rand_seed = st.slider("Rand Seed", 0, 42, 100, step=1)
        with col3:
            allow_negative = st.checkbox("Allow Negatives", value=True)

    with col_time:
        st.markdown('<div class="section-header">Time Settings</div>', unsafe_allow_html=True)
        col4, col5 = st.columns([2, 1])
        with col4:
            start_time = st.text_input("Starting Timestamp", value="2000-01-01 00:00:00")
        with col5:
            time_interval = st.number_input("Interval (sec)", min_value=1, value=60, step=1)

    st.markdown('<div class="section-header">Missing Value Settings</div>', unsafe_allow_html=True)
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

    # reusing rand_seed from global settings for the custom series
    config["custom"]["rand_seed"] = config["global"]["rand_seed"]

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

    summary_df = summarize_series(df["value"])
    st.dataframe(summary_df, use_container_width=False)

    csv = df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"time_series_{timestamp}.csv"
    st.download_button("Download CSV", data=csv, file_name=file_name, mime="text/csv")
