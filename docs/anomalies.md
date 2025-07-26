Add labeled anomalies to the time series using the Custom anomaly option. Define a range and apply a change using one of these modes:

- `Add`: Adds a constant offset
- `Mult`: Scales values based on a percentage
- `Slope`: Applies a linear ramp (trend shift)

`Noise Std`: Noise Standard Deviation; add Gaussian noise over the anomaly range to simulate instability

**Anomaly types**
- To create a level shift, stretch the range to either end of the data
- To create a spike, use a short range with high magnitude