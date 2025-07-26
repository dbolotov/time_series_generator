Simulate gaps in the time series data. Missing values are applied before anomalies and filled afterward.

**Mode**

- `None`: No missing values are added.
- `Clustered`:  Randomly masks a percentage of values, with optional clustering.
- `Every N`: Masks values in a repeating pattern (e.g. every 10th).
- `Clip`: Masks values based on their magnitude (e.g. all values above a threshold).

**Fill Method**

Choose how to fill in missing values after masking. Options: "Forward fill" (use last known value) or "Zero Fill".

**Details**

**Clustered Gaps**: Missing values are created by masking a target percentage of points. To simulate clustering (i.e. "gaps" of missing data), a run-based approach is used. For each gap, a random starting index and run length are chosen. When clustering is high, longer runs are more likely. This creates natural-looking bursts of missingness rather than scattered points. The process repeats until the desired number of missing values is reached.

**Every N**: Values are masked in a fixed repeating pattern (for example, every 10th point). Choose to mask one or multiple values in a row each time (e.g. mask 3 values every 10 steps). This is useful for simulating periodic outages or sampling limits.

**Clip**: Values are masked based on their magnitude. Choose to remove high values, low values, or both, based on a cutoff threshold. This simulates missingness that depends on the data itself (called "Missing Not At Random", or MNAR).
