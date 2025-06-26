#!/usr/bin/env python3
"""
Example script demonstrating the roughvol library usage.

This script shows how to use the library programmatically
without the command-line interface.
"""

import numpy as np
from datetime import datetime, timedelta
from roughvol.data import OptionDataFetcher
from roughvol.calibrate import RoughBergomiCalibrator
from roughvol.rbmc import RoughBergomiMC
from roughvol.price import MonteCarloPricer


def main():
    """Main example function."""
    print("Rough Volatility Library - Example Usage")
    print("=" * 50)
    
    # 1. Fetch market data
    print("1. Fetching market data...")
    fetcher = OptionDataFetcher()
    
    # Use recent data for demonstration
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    option_data = fetcher.fetch_data(
        ticker="SPX",
        start_date=start_date,
        end_date=end_date,
        target_maturities=[30, 60, 90]
    )
    
    print(f"   Spot price: ${option_data.spot_price:.2f}")
    print(f"   Risk-free rate: {option_data.risk_free_rate:.3f}")
    print(f"   Available strikes: {len(option_data.strikes)}")
    print(f"   Available expiries: {len(option_data.expiry_dates)}")
    
    # 2. Calibrate model
    print("\n2. Calibrating rough Bergomi model...")
    calibrator = RoughBergomiCalibrator(option_data, n_paths=5000)  # Fewer paths for speed
    
    # Estimate H from ATM skew
    H_estimate = calibrator.estimate_h_from_atm_skew()
    print(f"   Estimated H from ATM skew: {H_estimate:.3f}")
    
    # Run calibration
    params = calibrator.calibrate(maxiter=20, popsize=10, seed=42)
    
    print(f"   Calibrated parameters:")
    print(f"     H = {params['H']:.3f}")
    print(f"     η = {params['eta']:.3f}")
    print(f"     ρ = {params['rho']:.3f}")
    print(f"     RMSE = {params['rmse']:.3f} vol pts")
    
    # 3. Create model and pricer
    print("\n3. Creating model and pricer...")
    model = RoughBergomiMC(params['H'], params['eta'], params['rho'])
    pricer = MonteCarloPricer(model)
    
    # 4. Price some options
    print("\n4. Pricing options...")
    S0 = option_data.spot_price
    r = option_data.risk_free_rate
    T = 0.25  # 3 months
    
    # ATM call option
    K_atm = S0
    call_price = pricer.price_vanilla_call(S0, K_atm, T, r, n_paths=5000)
    call_iv = pricer.implied_volatility_call(S0, K_atm, T, r, call_price)
    
    print(f"   ATM Call (K={K_atm:.0f}, T={T*365:.0f}d):")
    print(f"     Price: ${call_price:.2f}")
    print(f"     Implied Vol: {call_iv:.3f}")
    
    # OTM call option
    K_otm = S0 * 1.05
    otm_call_price = pricer.price_vanilla_call(S0, K_otm, T, r, n_paths=5000)
    otm_call_iv = pricer.implied_volatility_call(S0, K_otm, T, r, otm_call_price)
    
    print(f"   OTM Call (K={K_otm:.0f}, T={T*365:.0f}d):")
    print(f"     Price: ${otm_call_price:.2f}")
    print(f"     Implied Vol: {otm_call_iv:.3f}")
    
    # 5. Price exotic option
    print("\n5. Pricing exotic option...")
    B = S0 * 1.10  # 10% above spot
    exotic_price = pricer.price_up_and_out_call(S0, K_atm, B, T, r, n_paths=5000)
    
    # Compare with Black-Scholes
    bs_price = pricer.black_scholes_call(S0, K_atm, T, r, call_iv)
    
    print(f"   Up-and-Out Call (K={K_atm:.0f}, B={B:.0f}, T={T*365:.0f}d):")
    print(f"     rBergomi: ${exotic_price:.2f}")
    print(f"     Black-Scholes: ${bs_price:.2f}")
    print(f"     Difference: ${exotic_price - bs_price:.2f} ({((exotic_price/bs_price - 1)*100):+.1f}%)")
    
    # 6. Evaluate calibration quality
    print("\n6. Evaluating calibration quality...")
    maturity_metrics = calibrator.evaluate_calibration(params)
    
    print("   Maturity-wise performance:")
    for maturity, metrics in maturity_metrics.items():
        print(f"     {maturity}d: RMSE = {metrics['rmse']:.3f}, Options = {metrics['n_options']}")
    
    # 7. Generate implied volatility surface
    print("\n7. Generating implied volatility surface...")
    strikes = [S0 * 0.9, S0 * 0.95, S0, S0 * 1.05, S0 * 1.1]
    maturities = [30/365, 60/365, 90/365]
    
    iv_surface = pricer.implied_vol_surface(S0, strikes, maturities, r, n_paths=5000)
    
    print("   Implied Volatility Surface:")
    print("   Strike/Maturity", end="")
    for T in maturities:
        print(f"   {T*365:>6.0f}d", end="")
    print()
    
    for i, strike in enumerate(strikes):
        print(f"   {strike:>8.0f}", end="")
        for j, T in enumerate(maturities):
            iv = iv_surface[j, i]
            print(f"   {iv:>6.3f}", end="")
        print()
    
    print("\nExample completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main() 