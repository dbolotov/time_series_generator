import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import math
from scipy.stats import skew, kurtosis

# --- Functions ---
def ornstein_uhlenbeck_process(num_points, theta, mu, sigma):
    dt = 1
    ou = np.zeros(num_points)
    for t in range(1, num_points):
        ou[t] = ou[t-1] + theta * (mu - ou[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return ou

def generate_ts(
    series_type: str,
    num_points: int,
    rand_seed: int,
    start_time: str,
    time_interval: int,
    missing_pct: float = 0,
    missing_seed: int = 123,
    fill_method: str = "None",
    allow_negative: bool = True,
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 0.3,
    rw_drift: float = 0.0,
) -> pd.DataFrame:

    np.random.seed(rand_seed)

    if series_type == "White Noise":
        data = np.random.normal(0, 1, num_points)
    elif series_type == "Random Walk":
        steps = np.random.normal(0, 1, num_points) + rw_drift
        data = np.cumsum(steps)
    elif series_type == "Ornstein-Uhlenbeck":
        data = ornstein_uhlenbeck_process(num_points, theta, mu, sigma)

    if not allow_negative:
        current_min = np.min(data)
        if current_min < 0:
            data = data - current_min

    # Inject missing values
    if missing_pct > 0:
        np.random.seed(missing_seed)
        mask = np.random.rand(num_points) < (missing_pct / 100)
        data[mask] = np.nan

    # Apply fill method
    if fill_method == "Forward Fill":
        data = pd.Series(data).ffill().to_numpy()
    elif fill_method == "Fill with Zero":
        data = pd.Series(data).fillna(0).to_numpy()

    start = pd.to_datetime(start_time)
    timestamps = pd.date_range(start=start, periods=num_points, freq=pd.to_timedelta(time_interval, unit='s'))

    df = pd.DataFrame({"timestamp": timestamps, "value": data})
    return df

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
st.markdown(
    """
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
    </style>
""",
    unsafe_allow_html=True,
)



st.title("Time Series Data Generator")

# --- Column Layout ---
left_col, spacer, right_col = st.columns([5, 0.5, 5])

# --- Left Panel (Controls and Description) ---
with left_col:
    st.markdown("""
    Generate univariate time series data. Optionally save in .csv format.
    """)

    st.subheader("Series Settings")
    col1, col2 = st.columns([1, 2])
    with col1:
        series_type = st.selectbox("Time Series Type", ["White Noise", "Random Walk", "Ornstein-Uhlenbeck"])
    with col2:
        if series_type == "Ornstein-Uhlenbeck":
            ou_col1, ou_col2, ou_col3 = st.columns(3)
            with ou_col1:
                theta = st.slider("θ (mean reversion)", min_value=0.0, max_value=1.0, value=0.2, step=0.025, format="%.3f")
            with ou_col2:
                mu = st.slider("μ (long-term mean)", -10.0, 10.0, 0.0, step=0.1, format="%.1f")
            with ou_col3:
                sigma = st.slider("σ (volatility)", 0.01, 2.0, 0.3, step=0.01, format="%.2f")
            rw_drift = None
        elif series_type == "Random Walk":
            rw_col1, rw_col2, rw_col3 = st.columns(3)
            with rw_col1:
                rw_drift = st.slider("Drift", min_value=-0.2, max_value=0.2, value=0.0, step=0.01, format="%.2f")
            theta = mu = sigma = None
        else:
            theta = mu = sigma = rw_drift = None

    col_data, col_time = st.columns([3, 2])

    with col_data:
        with st.container():
            st.markdown("### Data Settings")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                num_points = st.slider("Number of Points", 100, 1000, 300, step=50, key="num_points_slider")
            with col2:
                # rand_seed = st.number_input("Rand Seed", value=42, step=1, key="rand_seed_input")
                rand_seed = st.slider("Rand Seed", 0, 42, 100, step=1, key="rand_seed_input")
            with col3:
                allow_negative = st.checkbox("Allow Negatives", value=True)

    with col_time:
        with st.container():
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
        # missing_seed = st.number_input("Missing Value Random Seed", value=123, step=1)
        missing_seed = st.slider("MV Rand Seed", 0, 42, 100, step=1, key="mv_rand_seed_input")
    with col8:
        fill_method = st.selectbox("Fill Method", ["None", "Forward Fill", "Fill with Zero"])


# --- Right Panel (Plot and Download) ---
with right_col:
    df = generate_ts(
        series_type=series_type, num_points=num_points, rand_seed=rand_seed,
        start_time=start_time, time_interval=time_interval,
        missing_pct=missing_pct, missing_seed=missing_seed,
        fill_method=fill_method, allow_negative=allow_negative,
        theta=theta, mu=mu, sigma=sigma, rw_drift=rw_drift
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=series_type,
            line=dict(width=1, color="#D2671A"),
        )
    )
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
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=file_name,
        mime="text/csv",
    )
