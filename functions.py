import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import colorednoise
import plotly.graph_objects as go

from enums import SeriesType, TrendType, SeasonalityType, FillMethod, AnomalyType


def generate_noise(num_points, beta, mean, std, drift, rng):
    noise = colorednoise.powerlaw_psd_gaussian(beta, num_points, random_state=rng)
    noise = noise * std + mean
    drift_values = np.linspace(0, drift * (num_points - 1), num_points)
    return noise + drift_values


def generate_ou_process(num_points, theta, mu, sigma, rng):
    dt = 1
    ou = np.zeros(num_points)
    for t in range(1, num_points):
        ou[t] = (
            ou[t - 1]
            + theta * (mu - ou[t - 1]) * dt
            + sigma * np.sqrt(dt) * rng.normal()
        )
    return ou


def generate_cycle(num_points, amp=0.0, freq_base=0.00, freq_var=0.00, decay_rate=0.0):
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
        coef = cfg.get("quad_coef", 0.01)
        intercept = cfg.get("quad_intercept", 0.0)
        y = coef * t**2 + intercept
    elif trend_type == TrendType.EXPONENTIAL.value:
        base = cfg.get("exp_base", 1.01)
        scale = cfg.get("exp_scale", 1.0)
        y = scale * (base**t)
    else:
        y = np.zeros(num_points)

    if seas_type == SeasonalityType.SINE.value:
        seasonal = seas_amp * np.sin(2 * np.pi * t / seas_period)
    elif seas_type == SeasonalityType.SAWTOOTH.value:
        seasonal = seas_amp * signal.sawtooth(
            2 * np.pi * t / seas_period, width=seas_width
        )
    else:
        seasonal = 0

    y += seasonal

    if cfg.get("cycle_enabled"):
        cycle = generate_cycle(
            num_points,
            cfg.get("cyc_amp", 1.0),
            cfg.get("cyc_freq", 0.03),
            cfg.get("cyc_var", 0.01),
            cfg.get("cyc_decay", 0.0),
        )
        y += cycle

    if cfg.get("noise_enabled"):
        noise = generate_noise(
            num_points,
            cfg.get("noise_beta", 1.0),
            cfg.get("noise_mean", 0.0),
            cfg.get("noise_std", 1.0),
            0.0,  # drift is disabled in custom series
            rng,
        )

        y += noise

    return y


def generate_missing_mask(
    num_points: int, missing_pct: float, clustering: float, seed: int
) -> np.ndarray:
    """
    Returns a boolean mask with missing values placed to reflect gap clustering.
    """
    rng = np.random.default_rng(seed)
    n_missing = int(num_points * missing_pct / 100.0)
    mask = np.zeros(num_points, dtype=bool)

    if n_missing == 0:
        return mask

    # Parameters for clustering behavior
    avg_run_length = max(
        1, int(1 + clustering * 20)
    )  # Longer stretches for higher clustering
    placed = 0
    attempts = 0
    max_attempts = num_points * 2

    while placed < n_missing and attempts < max_attempts:
        start = rng.integers(0, num_points)
        run_length = rng.integers(1, avg_run_length + 1)
        end = min(start + run_length, num_points)

        for i in range(start, end):
            if not mask[i]:
                mask[i] = True
                placed += 1
                if placed >= n_missing:
                    break
        attempts += 1

    return mask



def apply_anomalies(data: np.ndarray, cfg: dict, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anomaly_type = cfg.get("anomaly_type", AnomalyType.NONE.value)

    if anomaly_type == AnomalyType.SPIKE_PLATEAU.value:
        start, end = cfg.get("range", (0, 0))
        pct = cfg.get("magnitude_pct", 100.0) / 100.0
        mode = cfg.get("mode", "Mult")

        if end > start:
            anomaly_indices = np.arange(start, end)
            labels[anomaly_indices] = 1

            if mode == "Mult":
                data[anomaly_indices] *= (1 + pct)
            elif mode == "Add":
                data[anomaly_indices] += pct * 10

    return data, labels


def generate_ts(config):
    global_cfg = config["global"]
    rng = np.random.default_rng(global_cfg["rand_seed"])
    num_points = global_cfg["num_points"]

    series_type = global_cfg["series_type"]

    if series_type == SeriesType.NOISE.value:
        p = config["noise"]
        data = generate_noise(
            num_points, p["beta"], p["mean"], p["std"], p["drift"], rng
        )
    elif series_type == SeriesType.OU_PROCESS.value:
        p = config["ou"]
        data = generate_ou_process(num_points, p["theta"], p["mu"], p["sigma"], rng)
    elif series_type == SeriesType.CUSTOM.value:
        data = generate_custom_series(num_points, config["custom"], rng)

    if not global_cfg["allow_negative"]:
        min_val = np.min(data)
        if min_val < 0:
            data = data - min_val

    # Store raw data
    value_raw = pd.Series(data.copy())

    # Add missing values
    was_missing = np.zeros(num_points, dtype=int)  # 0 = present, 1 = originally missing

    missing_pct = global_cfg.get("missing_pct", 0.0)
    clustering = global_cfg.get("gap_clustering", 0.0)

    if missing_pct > 0:
        is_missing = generate_missing_mask(
            num_points=num_points,
            missing_pct=missing_pct,
            clustering=clustering,
            seed=global_cfg["missing_seed"],
        )
        was_missing[is_missing] = 1
        data[is_missing] = np.nan

    # Fill missing values
    missing_fill_method = global_cfg["missing_fill_method"]
    if missing_fill_method == FillMethod.FORWARD.value:
        data = pd.Series(data).ffill().to_numpy()
    elif missing_fill_method == FillMethod.ZERO.value:
        data = pd.Series(data).fillna(0).to_numpy()

    start = pd.to_datetime(global_cfg["start_time"])
    interval = global_cfg["time_interval"]
    # timestamps = pd.date_range(start=start, periods=num_points, freq=pd.to_timedelta(interval, unit='s'))
    unit = global_cfg.get("interval_unit", "s")
    timestamps = pd.date_range(
        start=start, periods=num_points, freq=pd.to_timedelta(interval, unit=unit)
    )

    # labels = np.zeros(num_points, dtype=int)  # 0 = normal, 1 = anomaly

    # if global_cfg.get("anomaly_type") == AnomalyType.SPIKE_PLATEAU.value:
    #     start, end = global_cfg.get("range", (0, 0))
    #     pct = global_cfg.get("magnitude_pct", 100.0) / 100.0
    #     mode = global_cfg.get("mode", "Mult")

    #     if end > start:
    #         anomaly_indices = np.arange(start, end)
    #         labels[anomaly_indices] = 1

    #     if end > start:
    #         anomaly_indices = np.arange(start, end)
    #         labels[anomaly_indices] = 1

    #         if mode == "Mult":
    #             data[anomaly_indices] *= 1 + pct

    #         elif mode == "Add":
    #             data[anomaly_indices] += pct * 10

    labels = np.zeros(num_points, dtype=int)
    data, labels = apply_anomalies(data, global_cfg, labels)


    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": data,  # the filled values
            "value_raw": value_raw,  # the original version with NaNs
            "was_missing": was_missing,
            "anomaly": labels,
        }
    )

    return df


def summarize_series(series: pd.Series) -> pd.DataFrame:
    stats = {
        "Mean": series.mean(),
        "Std Dev": series.std(),
        "Min": series.min(),
        "Max": series.max(),
        "Range": series.max() - series.min(),
        "Skewness": skew(series, nan_policy="omit"),
        "Kurtosis": kurtosis(series, nan_policy="omit"),
        "ACF (lag 1)": series.autocorr(lag=1),
        "0 Crossings": ((series.shift(1) - series.mean()) * (series - series.mean()) < 0).sum()
    }
    return pd.DataFrame([stats]).round(3)


def _add_overlay_blocks(
    fig, df, column_name, timestamps_col, color, label, opacity=0.3
):
    """
    Add overlay blocks for missing values and anomalies during plots
    """
    mask = df[column_name] == 1
    indices = mask[mask].index

    if indices.empty:
        return

    # Find contiguous index runs
    runs = []
    start_idx = indices[0]

    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            runs.append((start_idx, indices[i - 1]))
            start_idx = indices[i]
    runs.append((start_idx, indices[-1]))

    for start, end in runs:
        x0 = (
            df[timestamps_col].iloc[start - 1]
            if start > 0
            else df[timestamps_col].iloc[start]
        )
        x1 = (
            df[timestamps_col].iloc[end + 1]
            if end + 1 < len(df)
            else df[timestamps_col].iloc[end]
        )

        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            line=dict(width=0),
            fillcolor=color,
            opacity=opacity,
            layer="below",
        )

    # Dummy trace for legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=6, color=color, symbol="square"),
            name=label,
        )
    )


def plot_series(df: pd.DataFrame, series_type: str, settings: dict) -> go.Figure:

    # colors
    main_color = "#D2671A"
    anomaly_color = "#D96A52"
    missing_color = "#5FBCD1"

    overlay_opacity = 0.3

    fig = go.Figure()

    # --- Main series line ---
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines" if settings["use_lines"] else "markers",
            name=series_type,
            line=dict(width=1, color=main_color),
            marker=dict(size=4) if not settings["use_lines"] else None,
        )
    )

    if (
        settings.get("show_anomalies")
        and "anomaly" in df.columns
        and df["anomaly"].any()
    ):
        _add_overlay_blocks(
            fig, df, "anomaly", "timestamp", anomaly_color, "Anomaly", overlay_opacity
        )

    if (
        settings.get("show_missing")
        and "was_missing" in df.columns
        and df["was_missing"].any()
    ):
        _add_overlay_blocks(
            fig,
            df,
            "was_missing",
            "timestamp",
            missing_color,
            "Missing",
            overlay_opacity,
        )

    # --- Layout ---
    fig.update_layout(
        title="Generated Series",
        height=500,
        xaxis_title="Time",
        yaxis_title="Value",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig
