Simulate gaps in the time series data. Missing values are applied to the data before anomalies.

**Parameters**
- `Missing Data (%)`: Percentage of values to mask out
- `Gap Clustering`: Controls how 'clustered' the gaps are (0 = random, 1 = more clustered)
- `Fill Method`: Choose forward fill or fill with zero

**Details**

Missing values are created by masking a target percentage of points in the time series. To simulate clustering (i.e. "gaps" of missing data), a run-based approach is used. For each gap, a random starting index and run length are chosen, with longer runs more likely when clustering is high. This creates natural-looking bursts of missingness rather than uniformly scattered points. The process repeats until the desired number of missing values is reached.