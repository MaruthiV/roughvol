"""
Command Line Interface for Rough Volatility Library

Provides command-line access to rough Bergomi model calibration and testing.
"""

import argparse
import sys
import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings

from .data import OptionDataFetcher
from .calibrate import RoughBergomiCalibrator
from .rbmc import RoughBergomiMC
from .price import MonteCarloPricer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def save_parameters(params: dict, output_dir: str) -> None:
    """Save calibrated parameters to JSON file."""
    output_path = os.path.join(output_dir, "params.json")
    
    # Convert numpy types to native Python types for JSON serialization
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            params_serializable[key] = float(value)
        else:
            params_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print(f"Parameters saved to {output_path}")


def save_surfaces(option_data, calibrated_params: dict, output_dir: str, 
                 n_paths: int = 40000) -> None:
    """Save market and model implied volatility surfaces to CSV."""
    # Create model and pricer
    model = RoughBergomiMC(
        calibrated_params['H'],
        calibrated_params['eta'],
        calibrated_params['rho']
    )
    pricer = MonteCarloPricer(model)
    
    S0 = option_data.spot_price
    r = option_data.risk_free_rate
    
    # Prepare data for CSV
    market_data = []
    model_data = []
    
    for expiry in option_data.expiry_dates:
        T = (expiry - datetime.now()).days / 365.0
        
        for strike in option_data.strikes:
            if (expiry, strike) in option_data.call_prices:
                market_price = option_data.call_prices[(expiry, strike)]
                
                # Market implied vol
                market_iv = pricer.implied_volatility_call(S0, strike, T, r, market_price)
                
                # Model implied vol
                model_price = pricer.price_vanilla_call(S0, strike, T, r, n_paths)
                model_iv = pricer.implied_volatility_call(S0, strike, T, r, model_price)
                
                if not (np.isnan(market_iv) or np.isnan(model_iv)):
                    market_data.append({
                        'maturity_days': int(T * 365),
                        'strike': strike,
                        'moneyness': strike / S0,
                        'implied_vol': market_iv
                    })
                    
                    model_data.append({
                        'maturity_days': int(T * 365),
                        'strike': strike,
                        'moneyness': strike / S0,
                        'implied_vol': model_iv
                    })
    
    # Save to CSV
    market_df = pd.DataFrame(market_data)
    model_df = pd.DataFrame(model_data)
    
    market_df.to_csv(os.path.join(output_dir, "surface_market.csv"), index=False)
    model_df.to_csv(os.path.join(output_dir, "surface_model.csv"), index=False)
    
    print(f"Surfaces saved to {output_dir}")


def create_iv_surface_plot(option_data, calibrated_params: dict, output_dir: str,
                          n_paths: int = 40000) -> None:
    """Create 3D implied volatility surface plot."""
    # Create model and pricer
    model = RoughBergomiMC(
        calibrated_params['H'],
        calibrated_params['eta'],
        calibrated_params['rho']
    )
    pricer = MonteCarloPricer(model)
    
    S0 = option_data.spot_price
    r = option_data.risk_free_rate
    
    # Collect data for plotting
    maturities = []
    strikes = []
    market_ivs = []
    model_ivs = []
    
    for expiry in option_data.expiry_dates:
        T = (expiry - datetime.now()).days / 365.0
        
        for strike in option_data.strikes:
            if (expiry, strike) in option_data.call_prices:
                market_price = option_data.call_prices[(expiry, strike)]
                
                # Market implied vol
                market_iv = pricer.implied_volatility_call(S0, strike, T, r, market_price)
                
                # Model implied vol
                model_price = pricer.price_vanilla_call(S0, strike, T, r, n_paths)
                model_iv = pricer.implied_volatility_call(S0, strike, T, r, model_price)
                
                if not (np.isnan(market_iv) or np.isnan(model_iv)):
                    maturities.append(T * 365)
                    strikes.append(strike / S0)  # Moneyness
                    market_ivs.append(market_iv)
                    model_ivs.append(model_iv)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Market surface (gray)
    scatter1 = ax.scatter(maturities, strikes, market_ivs, 
                         c='gray', alpha=0.6, s=20, label='Market')
    
    # Model surface (colored by IV level)
    scatter2 = ax.scatter(maturities, strikes, model_ivs, 
                         c=model_ivs, cmap=cm.viridis, s=30, label='Model')
    
    ax.set_xlabel('Maturity (days)')
    ax.set_ylabel('Moneyness (K/S0)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface: Market vs Model')
    
    # Add colorbar
    cbar = plt.colorbar(scatter2, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Model IV')
    
    ax.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iv_surface.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"IV surface plot saved to {output_dir}/iv_surface.png")


def price_exotic_vs_bs(option_data, calibrated_params: dict, output_dir: str,
                      n_paths: int = 40000) -> None:
    """Price exotic option and compare with Black-Scholes."""
    # Create model and pricer
    model = RoughBergomiMC(
        calibrated_params['H'],
        calibrated_params['eta'],
        calibrated_params['rho']
    )
    pricer = MonteCarloPricer(model)
    
    S0 = option_data.spot_price
    r = option_data.risk_free_rate
    
    # Price up-and-out call option
    K = S0  # ATM strike
    B = 1.05 * S0  # 5% above spot
    T = 90 / 365.0  # 90 days
    
    # Rough Bergomi price
    rbergomi_price = pricer.price_up_and_out_call(S0, K, B, T, r, n_paths)
    
    # Black-Scholes price (approximation)
    # Use ATM implied vol for BS
    atm_price = option_data.call_prices.get((option_data.expiry_dates[0], K), None)
    if atm_price is None:
        # Find closest ATM option
        atm_strike = min(option_data.strikes, key=lambda k: abs(k - S0))
        atm_price = option_data.call_prices.get((option_data.expiry_dates[0], atm_strike), 0.1)
    
    atm_iv = pricer.implied_volatility_call(S0, K, T, r, atm_price)
    if np.isnan(atm_iv):
        atm_iv = 0.2  # Default if IV calculation fails
    
    # Simple BS approximation for barrier option
    bs_price = pricer.black_scholes_call(S0, K, T, r, atm_iv) * 0.8  # Rough adjustment
    
    # Save results
    exotic_data = {
        'option_type': 'Up-and-Out Call',
        'strike': K,
        'barrier': B,
        'maturity_days': int(T * 365),
        'rbergomi_price': rbergomi_price,
        'black_scholes_price': bs_price,
        'price_difference': rbergomi_price - bs_price,
        'price_ratio': rbergomi_price / bs_price
    }
    
    exotic_df = pd.DataFrame([exotic_data])
    exotic_df.to_csv(os.path.join(output_dir, "exotic_vs_bs.csv"), index=False)
    
    print(f"Exotic pricing results saved to {output_dir}/exotic_vs_bs.csv")
    
    # Print summary
    print(f"\nExotic Option Pricing Summary:")
    print(f"Up-and-Out Call (K={K:.0f}, B={B:.0f}, T={T*365:.0f}d)")
    print(f"  rBergomi: ${rbergomi_price:.2f}")
    print(f"  Black-Scholes: ${bs_price:.2f}")
    print(f"  Difference: ${rbergomi_price - bs_price:.2f} ({((rbergomi_price/bs_price - 1)*100):+.1f}%)")


def print_results_table(params: dict, maturity_metrics: dict) -> None:
    """Print colored results table."""
    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    
    # Parameters table
    print(f"{'Parameter':<12} {'Value':<10}")
    print("-" * 22)
    print(f"{'H':<12} {params['H']:<10.3f}")
    print(f"{'η':<12} {params['eta']:<10.3f}")
    print(f"{'ρ':<12} {params['rho']:<10.3f}")
    print(f"{'RMSE':<12} {params['rmse']:<10.3f} vol pts")
    
    print("\n" + "="*50)
    print("MATURITY-WISE PERFORMANCE")
    print("="*50)
    
    # Maturity metrics table
    print(f"{'Maturity':<12} {'RMSE':<10} {'Options':<10}")
    print("-" * 32)
    
    for maturity, metrics in maturity_metrics.items():
        print(f"{maturity:<12} {metrics['rmse']:<10.3f} {metrics['n_options']:<10}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Rough Bergomi Model Calibration and Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m roughvol --start 2024-01-01 --end 2024-06-01
  python -m roughvol --maturities 30 60 90 --paths 5000 --output ./results
  python -m roughvol --quick  # Quick run with fewer paths
        """
    )
    
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-06-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--ticker', type=str, default='SPX',
                       help='Stock ticker symbol')
    parser.add_argument('--maturities', type=int, nargs='+', default=[30, 60, 90],
                       help='Target maturities in days')
    parser.add_argument('--paths', type=int, default=40000,
                       help='Number of Monte Carlo paths')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with fewer paths (5000)')
    parser.add_argument('--maxiter', type=int, default=100,
                       help='Maximum optimization iterations')
    parser.add_argument('--popsize', type=int, default=15,
                       help='Population size for differential evolution')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.quick:
        args.paths = 5000
    
    # Create output directory
    create_output_directory(args.output)
    
    try:
        print("Rough Volatility Research Library")
        print("=" * 40)
        print(f"Ticker: {args.ticker}")
        print(f"Period: {args.start} to {args.end}")
        print(f"Maturities: {args.maturities} days")
        print(f"Monte Carlo paths: {args.paths}")
        print(f"Output directory: {args.output}")
        print()
        
        # Fetch data
        print("Fetching market data...")
        fetcher = OptionDataFetcher()
        option_data = fetcher.fetch_data(
            args.ticker, args.start, args.end, args.maturities
        )
        
        print(f"Spot price: ${option_data.spot_price:.2f}")
        print(f"Risk-free rate: {option_data.risk_free_rate:.3f}")
        print(f"Available strikes: {len(option_data.strikes)}")
        print(f"Available expiries: {len(option_data.expiry_dates)}")
        print()
        
        # Calibrate model
        print("Calibrating rough Bergomi model...")
        calibrator = RoughBergomiCalibrator(option_data, args.paths)
        
        params = calibrator.calibrate(
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed
        )
        
        # Evaluate calibration
        print("Evaluating calibration quality...")
        maturity_metrics = calibrator.evaluate_calibration(params)
        
        # Save results
        print("Saving results...")
        save_parameters(params, args.output)
        save_surfaces(option_data, params, args.output, args.paths)
        create_iv_surface_plot(option_data, params, args.output, args.paths)
        price_exotic_vs_bs(option_data, params, args.output, args.paths)
        
        # Print results table
        print_results_table(params, maturity_metrics)
        
        print(f"\nAll results saved to: {args.output}")
        print("Files created:")
        print("  - params.json: Calibrated parameters")
        print("  - surface_market.csv: Market IV surface")
        print("  - surface_model.csv: Model IV surface")
        print("  - iv_surface.png: 3D IV surface plot")
        print("  - exotic_vs_bs.csv: Exotic option pricing")
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 