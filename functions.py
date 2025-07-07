import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import colorednoise
import plotly.graph_objects as go

from enums import SeriesType, TrendType, SeasonalityType, FillMethod

def generate_ou_process(num_points, theta, mu, sigma, rng):
    dt = 1
    ou = np.zeros(num_points)
    for t in range(1, num_points):
        ou[t] = ou[t-1] + theta * (mu - ou[t-1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return ou

def generate_cycle_component(num_points, amp=0.0, freq_base=0.00, freq_var=0.00, decay_rate=0.0):
    t = np.arange(num_points)

    # Slowly changing amplitude
    mod_amp = 1 + 0.5 * np.sin(2 * np.pi * t / (num_points * 0.8))

    # Slowly changing frequency
    cycle_freq = freq_base + freq_var * np.sin(2 * np.pi * t / (num_points * 0.6))
    cycle_phase = np.cumsum(cycle_freq)

    cycle = amp * mod_amp * np.sin(2 * np.pi * cycle_phase)

    # Decay
    decay = np.exp(-decay_rate * t)
    cycle *= decay

    return cycle

def generate_custom_series(num_points, cfg, rng):
    t = np.arange(num_points)

    trend_type = cfg["trend_type"]
    seas_type = cfg["seas_type"]
    lin_slope = cfg.get("lin_slope", 0.0)
    lin_intercept = cfg.get("lin_intercept", 0.0)
    seas_amp = cfg.get("seas_amp", 1.0)
    seas_period = cfg.get("seas_period", 50)
    seas_width = cfg.get("seas_width", 1.0)

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
    else:
        seasonal = 0

    y += seasonal

    if cfg.get("cycle_enabled"):
        cycle = generate_cycle_component(
            num_points,
            cfg.get("cyc_amp", 1.0),
            cfg.get("cyc_freq", 0.03),
            cfg.get("cyc_var", 0.01),
            cfg.get("cyc_decay", 0.0)
        )
        y += cycle

    if cfg.get("noise_enabled"):
        noise_cfg = {
            "noise": {
                "beta": cfg.get("noise_beta", 1.0),
                "mean": cfg.get("noise_mean", 0.0),
                "std": cfg.get("noise_std", 1.0),
                "drift": 0.0  # Drift is disabled for custom series
            }
        }
        noise = generate_noise(num_points, noise_cfg, rng)
        y += noise

    return y



def generate_noise(num_points, config, rng):
    cfg = config["noise"]
    beta = cfg.get("beta", 1.0)
    mean = cfg.get("mean", 0.0)
    std = cfg.get("std", 1.0)
    drift = cfg.get("drift", 0.0)

    # Use the passed-in rng directly
    noise = colorednoise.powerlaw_psd_gaussian(beta, num_points, random_state=rng)
    noise = noise * std + mean

    if drift != 0:
        noise += np.linspace(0, drift * num_points, num_points)

    return noise


def generate_ts(config):
    global_cfg = config["global"]
    rng = np.random.default_rng(global_cfg["rand_seed"])
    num_points = global_cfg["num_points"]

    series_type = global_cfg["series_type"]

    if series_type == SeriesType.OU_PROCESS.value:
        p = config["ou"]
        data = generate_ou_process(num_points, p["theta"], p["mu"], p["sigma"], rng)
    elif series_type == SeriesType.CUSTOM.value:
        data = generate_custom_series(num_points, config["custom"], rng)
    elif series_type == SeriesType.NOISE.value:
        data = generate_noise(num_points, config, rng)

    if not global_cfg["allow_negative"]:
        min_val = np.min(data)
        if min_val < 0:
            data = data - min_val

    if global_cfg["missing_pct"] > 0:
        # np.random.seed(global_cfg["missing_seed"])
        rng_missing = np.random.default_rng(global_cfg["missing_seed"])
        mask = rng_missing.random(num_points) < (global_cfg["missing_pct"] / 100)
        # mask = np.random.rand(num_points) < (global_cfg["missing_pct"] / 100)
        data[mask] = np.nan

    missing_fill_method = global_cfg["missing_fill_method"]
    if missing_fill_method == FillMethod.FORWARD.value:
        data = pd.Series(data).ffill().to_numpy()
    elif missing_fill_method == FillMethod.ZERO.value:
        data = pd.Series(data).fillna(0).to_numpy()

    start = pd.to_datetime(global_cfg["start_time"])
    interval = global_cfg["time_interval"]
    timestamps = pd.date_range(start=start, periods=num_points, freq=pd.to_timedelta(interval, unit='s'))

    return pd.DataFrame({"timestamp": timestamps, "value": data})

def summarize_series(series: pd.Series) -> pd.DataFrame:
    stats = {
        "Mean": series.mean(),
        "Std Dev": series.std(),
        "Min": series.min(),
        "Max": series.max(),
        "Skewness": skew(series, nan_policy='omit'),
        "Kurtosis": kurtosis(series, nan_policy='omit'),
    }
    return pd.DataFrame([stats]).round(3)


def plot_series(df: pd.DataFrame, series_type: str) -> go.Figure:
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
    return fig