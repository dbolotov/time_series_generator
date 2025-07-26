
Create a sequence using the Ornstein–Uhlenbeck (OU) Process - a mean-reverting stochastic process. 

**Parameters**
- `θ`: Speed of reversion — how strongly the series is pulled toward its mean
- `μ`: Long-term mean the process reverts to
- `σ`: Volatility — controls the randomness around the mean

**Details**

The OU process models systems that exhibit random variation but remain anchored to a central tendency. Unlike pure noise, it incorporates memory: each value depends on both the previous value and a pull toward the mean. The reversion speed (θ) controls how quickly it corrects deviations, while the volatility (σ) controls how erratic it is.

Common use cases include modeling interest rates (e.g. Vasicek model), physical systems with friction (e.g. velocity of a particle in a fluid), or any process where random shocks are tempered by a stabilizing force.