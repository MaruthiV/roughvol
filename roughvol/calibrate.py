"""
Rough Bergomi Calibrator

Implements calibration of rough Bergomi model parameters using
ATM-skew power-law fit for H seeding and Differential Evolution optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
from .rbmc import RoughBergomiMC
from .price import MonteCarloPricer
from .data import OptionData


class RoughBergomiCalibrator:
    """
    Calibrator for rough Bergomi model parameters.
    
    Parameters
    ----------
    option_data : OptionData
        Market option data
    n_paths : int, optional
        Number of Monte Carlo paths for pricing (default: 40000)
    n_steps : int, optional
        Number of time steps for simulation (default: 252)
    """
    
    def __init__(self, option_data: OptionData, n_paths: int = 40000, n_steps: int = 252):
        self.option_data = option_data
        self.n_paths = n_paths
        self.n_steps = n_steps
        
        # Parameter bounds for optimization
        self.bounds = [
            (0.05, 0.40),  # H: Hurst parameter
            (0.5, 4.0),    # eta: Volatility of volatility
            (-0.95, -0.1)  # rho: Correlation
        ]
    
    def estimate_h_from_atm_skew(self) -> float:
        """
        Estimate Hurst parameter H from ATM skew using power-law fit.
        
        The ATM skew follows a power law: σ(K) - σ(ATM) ∝ |K - S0|^H
        
        Returns
        -------
        float
            Estimated Hurst parameter H
        """
        S0 = self.option_data.spot_price
        r = self.option_data.risk_free_rate
        
        # Collect ATM and near-ATM implied volatilities
        skew_data = []
        
        for expiry in self.option_data.expiry_dates:
            T = (expiry - pd.Timestamp.now()).days / 365.0
            
            # Find ATM strike (closest to spot)
            atm_strike = min(self.option_data.strikes, 
                           key=lambda k: abs(k - S0))
            
            # Get ATM implied vol
            if (expiry, atm_strike) in self.option_data.call_prices:
                atm_price = self.option_data.call_prices[(expiry, atm_strike)]
                
                # Calculate ATM implied vol using Black-Scholes
                atm_iv = self._price_to_iv_call(S0, atm_strike, T, r, atm_price)
                
                if not np.isnan(atm_iv):
                    # Get near-ATM strikes for skew calculation
                    for strike in self.option_data.strikes:
                        if 0.9 * S0 <= strike <= 1.1 * S0 and strike != atm_strike:
                            if (expiry, strike) in self.option_data.call_prices:
                                price = self.option_data.call_prices[(expiry, strike)]
                                iv = self._price_to_iv_call(S0, strike, T, r, price)
                                
                                if not np.isnan(iv):
                                    moneyness = np.log(strike / S0)
                                    skew = iv - atm_iv
                                    
                                    if abs(moneyness) > 0.01:  # Avoid very close strikes
                                        skew_data.append({
                                            'moneyness': abs(moneyness),
                                            'skew': abs(skew),
                                            'T': T
                                        })
        
        if len(skew_data) < 5:
            warnings.warn("Insufficient data for H estimation, using default H=0.1")
            return 0.1
        
        # Convert to arrays
        moneyness = np.array([d['moneyness'] for d in skew_data])
        skew = np.array([d['skew'] for d in skew_data])
        
        # Fit power law: skew ∝ moneyness^H
        # log(skew) = log(c) + H * log(moneyness)
        valid_mask = (moneyness > 0) & (skew > 0)
        
        if np.sum(valid_mask) < 3:
            warnings.warn("Insufficient valid data for H estimation, using default H=0.1")
            return 0.1
        
        log_moneyness = np.log(moneyness[valid_mask])
        log_skew = np.log(skew[valid_mask])
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_moneyness, log_skew
        )
        
        H_estimate = slope
        
        # Validate H estimate
        if not (0.05 <= H_estimate <= 0.40):
            warnings.warn(f"H estimate {H_estimate:.3f} outside valid range, using default H=0.1")
            H_estimate = 0.1
        
        print(f"Estimated H from ATM skew: {H_estimate:.3f} (R² = {r_value**2:.3f})")
        
        return H_estimate
    
    def _price_to_iv_call(self, S0: float, K: float, T: float, r: float, 
                          price: float) -> float:
        """Convert option price to implied volatility using Brent root-finder."""
        def objective(sigma):
            # Black-Scholes call price
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            bs_price = S0 * 0.5 * (1 + np.math.erf(d1 / np.sqrt(2))) - \
                      K * np.exp(-r * T) * 0.5 * (1 + np.math.erf(d2 / np.sqrt(2)))
            return bs_price - price
        
        try:
            from scipy.optimize import brentq
            sigma_min, sigma_max = 0.001, 5.0
            
            f_min = objective(sigma_min)
            f_max = objective(sigma_max)
            
            if f_min * f_max > 0:
                return np.nan
            
            sigma_implied = brentq(objective, sigma_min, sigma_max, 
                                 xtol=1e-8, rtol=1e-8)
            return sigma_implied
            
        except Exception:
            return np.nan
    
    def loss_function(self, params: Tuple[float, float, float]) -> float:
        """
        Loss function for calibration.
        
        Loss = Σ w_i (σ_market - σ_model)²
        where weights w_i = 1 / (ask_i - bid_i)
        
        Parameters
        ----------
        params : Tuple[float, float, float]
            Model parameters (H, eta, rho)
            
        Returns
        -------
        float
            Loss value
        """
        H, eta, rho = params
        
        try:
            # Create model and pricer
            model = RoughBergomiMC(H, eta, rho)
            pricer = MonteCarloPricer(model, self.n_steps)
            
            S0 = self.option_data.spot_price
            r = self.option_data.risk_free_rate
            
            total_loss = 0.0
            total_weight = 0.0
            
            # Calculate loss for each option
            for expiry in self.option_data.expiry_dates:
                T = (expiry - pd.Timestamp.now()).days / 365.0
                
                for strike in self.option_data.strikes:
                    # Check if we have call data
                    if (expiry, strike) in self.option_data.call_prices:
                        market_price = self.option_data.call_prices[(expiry, strike)]
                        bid = self.option_data.call_bids[(expiry, strike)]
                        ask = self.option_data.call_asks[(expiry, strike)]
                        
                        # Calculate weight (inverse of bid-ask spread)
                        spread = ask - bid
                        if spread > 0:
                            weight = 1.0 / spread
                        else:
                            weight = 1.0
                        
                        # Model price
                        model_price = pricer.price_vanilla_call(S0, strike, T, r, self.n_paths)
                        
                        # Market implied vol
                        market_iv = self._price_to_iv_call(S0, strike, T, r, market_price)
                        
                        # Model implied vol
                        model_iv = self._price_to_iv_call(S0, strike, T, r, model_price)
                        
                        if not (np.isnan(market_iv) or np.isnan(model_iv)):
                            iv_diff = market_iv - model_iv
                            total_loss += weight * iv_diff**2
                            total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                return total_loss / total_weight
            else:
                return float('inf')
                
        except Exception as e:
            warnings.warn(f"Error in loss function: {e}")
            return float('inf')
    
    def calibrate(self, maxiter: int = 100, popsize: int = 15, 
                  seed: int = 42) -> Dict:
        """
        Calibrate model parameters using Differential Evolution.
        
        Parameters
        ----------
        maxiter : int, optional
            Maximum iterations for optimization (default: 100)
        popsize : int, optional
            Population size for DE (default: 15)
        seed : int, optional
            Random seed for optimization (default: 42)
            
        Returns
        -------
        Dict
            Calibration results with parameters and metrics
        """
        print("Starting rough Bergomi calibration...")
        
        # Estimate H from ATM skew
        H_estimate = self.estimate_h_from_atm_skew()
        
        # Set random seed
        np.random.seed(seed)
        
        # Run Differential Evolution
        result = differential_evolution(
            self.loss_function,
            self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            disp=True
        )
        
        if not result.success:
            warnings.warn("Optimization did not converge successfully")
        
        # Extract results
        H_opt, eta_opt, rho_opt = result.x
        final_loss = result.fun
        
        # Calculate RMSE
        rmse = np.sqrt(final_loss)
        
        print(f"Calibration completed:")
        print(f"  H = {H_opt:.3f}")
        print(f"  η = {eta_opt:.3f}")
        print(f"  ρ = {rho_opt:.3f}")
        print(f"  RMSE = {rmse:.3f} vol pts")
        
        return {
            'H': H_opt,
            'eta': eta_opt,
            'rho': rho_opt,
            'rmse': rmse,
            'loss': final_loss,
            'success': result.success,
            'n_iterations': result.nit,
            'H_estimate': H_estimate
        }
    
    def evaluate_calibration(self, params: Dict) -> Dict:
        """
        Evaluate calibration quality on different maturities.
        
        Parameters
        ----------
        params : Dict
            Calibrated parameters
            
        Returns
        -------
        Dict
            Evaluation metrics by maturity
        """
        H, eta, rho = params['H'], params['eta'], params['rho']
        
        # Create model and pricer
        model = RoughBergomiMC(H, eta, rho)
        pricer = MonteCarloPricer(model, self.n_steps)
        
        S0 = self.option_data.spot_price
        r = self.option_data.risk_free_rate
        
        maturity_metrics = {}
        
        for expiry in self.option_data.expiry_dates:
            T = (expiry - pd.Timestamp.now()).days / 365.0
            maturity_days = int(T * 365)
            
            market_ivs = []
            model_ivs = []
            weights = []
            
            for strike in self.option_data.strikes:
                if (expiry, strike) in self.option_data.call_prices:
                    market_price = self.option_data.call_prices[(expiry, strike)]
                    bid = self.option_data.call_bids[(expiry, strike)]
                    ask = self.option_data.call_asks[(expiry, strike)]
                    
                    # Weight
                    spread = ask - bid
                    weight = 1.0 / spread if spread > 0 else 1.0
                    
                    # Market IV
                    market_iv = self._price_to_iv_call(S0, strike, T, r, market_price)
                    
                    # Model IV
                    model_price = pricer.price_vanilla_call(S0, strike, T, r, self.n_paths)
                    model_iv = self._price_to_iv_call(S0, strike, T, r, model_price)
                    
                    if not (np.isnan(market_iv) or np.isnan(model_iv)):
                        market_ivs.append(market_iv)
                        model_ivs.append(model_iv)
                        weights.append(weight)
            
            if market_ivs:
                # Calculate weighted RMSE
                iv_diffs = np.array(market_ivs) - np.array(model_ivs)
                weights = np.array(weights)
                
                weighted_rmse = np.sqrt(np.average(iv_diffs**2, weights=weights))
                
                maturity_metrics[maturity_days] = {
                    'rmse': weighted_rmse,
                    'n_options': len(market_ivs),
                    'mean_market_iv': np.mean(market_ivs),
                    'mean_model_iv': np.mean(model_ivs)
                }
        
        return maturity_metrics 