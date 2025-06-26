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

### Expected Output

```
Rough Volatility Research Library
========================================
Ticker: SPX
Period: 2024-01-01 to 2024-06-01
Maturities: [30, 60, 90] days
Monte Carlo paths: 40000
Output directory: ./results

Fetching market data...
Spot price: $5234.18
Risk-free rate: 0.052
Available strikes: 45
Available expiries: 3

Calibrating rough Bergomi model...
Estimated H from ATM skew: 0.127 (R² = 0.892)
Starting rough Bergomi calibration...
Calibration completed:
  H = 0.113
  η = 2.034
  ρ = -0.712
  RMSE = 0.423 vol pts

==================================================
CALIBRATION RESULTS
==================================================
Parameter   Value     
--------------------
H           0.113     
η           2.034     
ρ           -0.712    
RMSE        0.423 vol pts

==================================================
MATURITY-WISE PERFORMANCE
==================================================
Maturity    RMSE       Options   
--------------------------------
30          0.387      15        
60          0.445      15        
90          0.438      15        

Exotic Option Pricing Summary:
Up-and-Out Call (K=5234, B=5496, T=90d)
  rBergomi: $2.14
  Black-Scholes: $1.67
  Difference: $0.47 (+28.1%)

All results saved to: ./results
Files created:
  - params.json: Calibrated parameters
  - surface_market.csv: Market IV surface
  - surface_model.csv: Model IV surface
  - iv_surface.png: 3D IV surface plot
  - exotic_vs_bs.csv: Exotic option pricing
```

## Architecture

```
roughvol/
├── __init__.py          # Package initialization
├── rbmc.py             # Rough Bergomi Monte Carlo simulator
├── data.py             # Data fetcher with caching
├── price.py            # Monte Carlo pricer
├── calibrate.py        # Model calibrator
├── cli.py              # Command-line interface
├── config.yaml         # Configuration file
├── requirements.txt    # Dependencies
├── README.md           # This file
└── tests/              # Unit tests
    ├── __init__.py
    ├── test_rbmc.py    # Tests for Monte Carlo
    └── test_calibrate.py # Tests for calibration
```

### Key Components

#### 1. RoughBergomiMC (`rbmc.py`)
- Implements the rough Bergomi model using hybrid fBM scheme
- Uses Bennedsen-Lunde-Pakkanen FFT method for efficient fBM generation
- Supports both full path simulation and maturity-only pricing

#### 2. OptionDataFetcher (`data.py`)
- Fetches SPX spot and option data from Yahoo Finance
- Implements intelligent caching with joblib
- Filters data by target maturities
- Estimates risk-free rate from Treasury yields

#### 3. MonteCarloPricer (`price.py`)
- Monte Carlo pricing for vanilla and exotic options
- Implied volatility calculation using Brent root-finder
- Black-Scholes pricing for comparison
- Surface generation for visualization

#### 4. RoughBergomiCalibrator (`calibrate.py`)
- ATM-skew power-law fit for H parameter seeding
- Differential Evolution optimization
- Weighted loss function based on bid-ask spreads
- Maturity-wise performance evaluation

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

## Configuration

Edit `config.yaml` to customize default parameters:

```yaml
# Data settings
ticker: "SPX"
start_date: "2024-01-01"
end_date: "2024-06-01"
maturities: [30, 60, 90]

# Monte Carlo settings
default_paths: 40000
quick_paths: 5000

# Model parameters bounds
parameter_bounds:
  H: [0.05, 0.40]
  eta: [0.5, 4.0]
  rho: [-0.95, -0.1]

# Calibration settings
differential_evolution:
  maxiter: 100
  popsize: 15
  seed: 42
```

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

## Output Files

The library generates the following output files:

### `params.json`
```json
{
  "H": 0.113,
  "eta": 2.034,
  "rho": -0.712,
  "rmse": 0.423,
  "loss": 0.179,
  "success": true,
  "n_iterations": 67,
  "H_estimate": 0.127
}
```

### `surface_market.csv` / `surface_model.csv`
```csv
maturity_days,strike,moneyness,implied_vol
30,5000,0.955,0.187
30,5100,0.974,0.192
...
```

### `iv_surface.png`
3D scatter plot showing market vs model implied volatility surfaces.

### `exotic_vs_bs.csv`
```csv
option_type,strike,barrier,maturity_days,rbergomi_price,black_scholes_price,price_difference,price_ratio
Up-and-Out Call,5234,5496,90,2.14,1.67,0.47,1.281
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

## Performance

- **Typical runtime**: 2-5 minutes on a laptop
- **Memory usage**: ~2-4 GB for 40k paths
- **Target RMSE**: ≤ 0.5 volatility points for 30/60/90-day maturities
- **Caching**: 24-hour cache for Yahoo Finance data

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887-904.
2. Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*, 21(4), 931-965.
3. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949. # roughvol
