Create a custom time series by combining multiple components: trend, seasonality, cycles, and noise.

**Components**
- `Trend`: Linear or non-linear increase/decrease
- `Seasonality`: Repeating seasonal pattern
- `Cycle`: Longer, slower oscillations (e.g., economic cycles)
- `Noise`: Random variation layered on top

**Details**

Each component can be turned on or off and configured independently. The components are combined additively, which allows for simulating a wide range of real-world behaviors.

This setup is useful for building datasets to test forecasting models, train anomaly detectors, or demonstrate how different signal types interact.