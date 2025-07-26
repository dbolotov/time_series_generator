**Exporting Data**

Click **Download CSV** to save the generated time series. The file has the following columns:

- `timestamp`: Time index
- `value`: Final time series with all processing applied
- `value_raw`: Original series before missing values, fills, or anomalies
- `was_missing`: 1 if the value was artificially masked out, 0 otherwise
- `anomaly`: 1 if the value is part of a labeled anomaly, 0 otherwise

This format is useful for testing anomaly detection models or visualizing preprocessing effects.