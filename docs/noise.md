Create a sequence of random values using a colored noise process.  

**Parameters**
- `β`: Color of the noise. Set β to any value between 0 and 2 to explore a continuum between white, pink, and brownian noise.
- `Mean`, `Std Dev`: Mean and standard deviation of the noise
- `Drift`: Linear drift added to the noise over time

**Details**

The "color" of the noise refers to its spectral shape, which affects how values evolve over time. Lower-frequency noise (higher β) tends to be smoother and more correlated in the time domain.

- **White noise** (β = 0): Each value is independent and identically distributed (no memory). Use cases: Sensor noise, testing random baselines, simulation of measurement error.
- **Pink noise** (β = 1, also called 1/f noise): Has more power at low frequencies, creating smoother, more correlated behavior. Use cases: Heartbeat signals, audio noise, natural processes like rainfall or river flows.
- **Brownian noise** (β = 2, also called red noise): Very smooth and strongly correlated over time. Mimics the behavior of a random walk but is generated via spectral filtering to produce a 1/f² power spectrum. Use cases: Stock prices, climate records, cumulative effects over time.

