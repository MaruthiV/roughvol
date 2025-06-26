# Rough Volatility Research Library

A self-contained Python research library for calibrating and testing the rough Bergomi (rBergomi) stochastic-volatility model on live SPX option-chain data.

## Overview

The rough Bergomi model is a state-of-the-art stochastic volatility model that captures the "roughness" of volatility processes, which is a key feature of real market data. This library provides:

- **Live data fetching** from Yahoo Finance with intelligent caching
- **ATM-skew H seeding** for robust parameter initialization
- **Differential Evolution optimization** for model calibration
- **Monte Carlo pricing** with hybrid fBM scheme
- **Implied volatility calculation** using Brent root-finder
- **Exotic option pricing** (up-and-out calls)
- **Comprehensive output** including CSV files and 3D plots

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd roughvol

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Run calibration on SPX data (Jan-Jun 2024)
python -m roughvol --start 2024-01-01 --end 2024-06-01

# Quick run with fewer paths for testing
python -m roughvol --quick

# Custom parameters
python -m roughvol \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --maturities 30 60 90 \
  --paths 40000 \
  --output ./results
```


## Model Details

### Rough Bergomi Model

The rough Bergomi model is defined by:

```
dS_t = S_t * sqrt(V_t) * dW_t
V_t = V_0 * exp(η * W^H_t - 0.5 * η² * t^(2H))
```

where:
- `S_t` is the spot price
- `V_t` is the variance process
- `W_t` is a standard Brownian motion
- `W^H_t` is a fractional Brownian motion with Hurst parameter `H`
- `η` is the volatility of volatility
- `ρ` is the correlation between spot and volatility processes

### Parameter Bounds

- **H (Hurst)**: [0.05, 0.40] - Controls roughness (lower = rougher)
- **η (eta)**: [0.5, 4.0] - Volatility of volatility
- **ρ (rho)**: [-0.95, -0.1] - Spot-vol correlation (typically negative)

### Calibration Process

1. **H Estimation**: Fit power law to ATM skew: `σ(K) - σ(ATM) ∝ |K - S0|^H`
2. **Optimization**: Use Differential Evolution to minimize weighted RMSE
3. **Loss Function**: `Loss = Σ w_i (σ_market - σ_model)²` where `w_i = 1 / (ask_i - bid_i)`


## Command Line Options

```bash
python -m roughvol [OPTIONS]

Options:
  --start DATE           Start date (YYYY-MM-DD) [default: 2024-01-01]
  --end DATE             End date (YYYY-MM-DD) [default: 2024-06-01]
  --ticker SYMBOL        Stock ticker [default: SPX]
  --maturities DAYS      Target maturities in days [default: 30 60 90]
  --paths N              Number of Monte Carlo paths [default: 40000]
  --output DIR           Output directory [default: ./results]
  --config FILE          Configuration file [default: config.yaml]
  --quick                Quick run with fewer paths (5000)
  --maxiter N            Maximum optimization iterations [default: 100]
  --popsize N            Population size for DE [default: 15]
  --seed N               Random seed [default: 42]
  -h, --help             Show help message
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rbmc.py

# Run with verbose output
pytest -v tests/
```


## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing (optimization, root-finding)
- **statsmodels**: Statistical modeling
- **yfinance**: Yahoo Finance data
- **joblib**: Caching and parallel processing
- **matplotlib**: Plotting
- **pytest**: Testing framework
- **PyYAML**: Configuration file parsing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887-904.
2. Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*, 21(4), 931-965.
3. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949. # roughvol
