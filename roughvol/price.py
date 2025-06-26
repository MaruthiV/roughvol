"""
Monte Carlo Pricer

Implements Monte Carlo pricing for vanilla options and exotics
with implied volatility calculation using Brent root-finder.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, List, Tuple, Optional
import warnings
from .rbmc import RoughBergomiMC


class MonteCarloPricer:
    """
    Monte Carlo pricer for rough Bergomi model.
    
    Parameters
    ----------
    model : RoughBergomiMC
        Rough Bergomi model instance
    n_steps : int, optional
        Number of time steps for simulation (default: 252)
    """
    
    def __init__(self, model: RoughBergomiMC, n_steps: int = 252):
        self.model = model
        self.n_steps = n_steps
    
    def price_vanilla_call(self, S0: float, K: float, T: float, r: float,
                          n_paths: int) -> float:
        """
        Price vanilla call option using Monte Carlo.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        n_paths : int
            Number of Monte Carlo paths
            
        Returns
        -------
        float
            Option price
        """
        # Simulate final spot prices
        ST = self.model.simulate_at_maturity(S0, 0.04, T, self.n_steps, n_paths)
        
        # Calculate payoff: max(S_T - K, 0)
        payoffs = np.maximum(ST - K, 0)
        
        # Discount to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price
    
    def price_vanilla_put(self, S0: float, K: float, T: float, r: float,
                         n_paths: int) -> float:
        """
        Price vanilla put option using Monte Carlo.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        n_paths : int
            Number of Monte Carlo paths
            
        Returns
        -------
        float
            Option price
        """
        # Simulate final spot prices
        ST = self.model.simulate_at_maturity(S0, 0.04, T, self.n_steps, n_paths)
        
        # Calculate payoff: max(K - S_T, 0)
        payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price
    
    def price_up_and_out_call(self, S0: float, K: float, B: float, T: float,
                             r: float, n_paths: int) -> float:
        """
        Price up-and-out call option using Monte Carlo.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        B : float
            Barrier level
        T : float
            Time to maturity
        r : float
            Risk-free rate
        n_paths : int
            Number of Monte Carlo paths
            
        Returns
        -------
        float
            Option price
        """
        # Simulate full paths
        spot_paths, _ = self.model.simulate_paths(S0, 0.04, T, self.n_steps, n_paths)
        
        # Check if barrier is hit
        barrier_hit = np.any(spot_paths >= B, axis=1)
        
        # Calculate payoff: max(S_T - K, 0) if barrier not hit, 0 otherwise
        final_prices = spot_paths[:, -1]
        payoffs = np.where(barrier_hit, 0, np.maximum(final_prices - K, 0))
        
        # Discount to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price
    
    def black_scholes_call(self, S0: float, K: float, T: float, r: float,
                          sigma: float) -> float:
        """
        Black-Scholes call option price.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
            
        Returns
        -------
        float
            Option price
        """
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = S0 * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
        
        return price
    
    def black_scholes_put(self, S0: float, K: float, T: float, r: float,
                         sigma: float) -> float:
        """
        Black-Scholes put option price.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
            
        Returns
        -------
        float
            Option price
        """
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S0 * self._normal_cdf(-d1)
        
        return price
    
    def implied_volatility_call(self, S0: float, K: float, T: float, r: float,
                               market_price: float, sigma_guess: float = 0.2) -> float:
        """
        Calculate implied volatility for call option using Brent root-finder.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        market_price : float
            Market price of the option
        sigma_guess : float, optional
            Initial guess for volatility (default: 0.2)
            
        Returns
        -------
        float
            Implied volatility
        """
        def objective(sigma):
            return self.black_scholes_call(S0, K, T, r, sigma) - market_price
        
        # Find root using Brent's method
        try:
            # Bounds for volatility search
            sigma_min = 0.001
            sigma_max = 5.0
            
            # Check if solution exists
            f_min = objective(sigma_min)
            f_max = objective(sigma_max)
            
            if f_min * f_max > 0:
                # No solution in range, return NaN
                return np.nan
            
            sigma_implied = brentq(objective, sigma_min, sigma_max, 
                                 xtol=1e-8, rtol=1e-8)
            
            return sigma_implied
            
        except Exception as e:
            warnings.warn(f"Failed to find implied volatility: {e}")
            return np.nan
    
    def implied_volatility_put(self, S0: float, K: float, T: float, r: float,
                              market_price: float, sigma_guess: float = 0.2) -> float:
        """
        Calculate implied volatility for put option using Brent root-finder.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        market_price : float
            Market price of the option
        sigma_guess : float, optional
            Initial guess for volatility (default: 0.2)
            
        Returns
        -------
        float
            Implied volatility
        """
        def objective(sigma):
            return self.black_scholes_put(S0, K, T, r, sigma) - market_price
        
        # Find root using Brent's method
        try:
            # Bounds for volatility search
            sigma_min = 0.001
            sigma_max = 5.0
            
            # Check if solution exists
            f_min = objective(sigma_min)
            f_max = objective(sigma_max)
            
            if f_min * f_max > 0:
                # No solution in range, return NaN
                return np.nan
            
            sigma_implied = brentq(objective, sigma_min, sigma_max, 
                                 xtol=1e-8, rtol=1e-8)
            
            return sigma_implied
            
        except Exception as e:
            warnings.warn(f"Failed to find implied volatility: {e}")
            return np.nan
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    
    def price_surface(self, S0: float, strikes: List[float], 
                     maturities: List[float], r: float, n_paths: int,
                     option_type: str = 'call') -> np.ndarray:
        """
        Price option surface for given strikes and maturities.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        strikes : List[float]
            List of strike prices
        maturities : List[float]
            List of maturities in years
        r : float
            Risk-free rate
        n_paths : int
            Number of Monte Carlo paths
        option_type : str, optional
            Option type ('call' or 'put', default: 'call')
            
        Returns
        -------
        np.ndarray
            Price surface of shape (len(maturities), len(strikes))
        """
        surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                if option_type.lower() == 'call':
                    surface[i, j] = self.price_vanilla_call(S0, K, T, r, n_paths)
                else:
                    surface[i, j] = self.price_vanilla_put(S0, K, T, r, n_paths)
        
        return surface
    
    def implied_vol_surface(self, S0: float, strikes: List[float],
                          maturities: List[float], r: float, n_paths: int,
                          option_type: str = 'call') -> np.ndarray:
        """
        Calculate implied volatility surface.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        strikes : List[float]
            List of strike prices
        maturities : List[float]
            List of maturities in years
        r : float
            Risk-free rate
        n_paths : int
            Number of Monte Carlo paths
        option_type : str, optional
            Option type ('call' or 'put', default: 'call')
            
        Returns
        -------
        np.ndarray
            Implied volatility surface of shape (len(maturities), len(strikes))
        """
        # First get price surface
        price_surface = self.price_surface(S0, strikes, maturities, r, n_paths, option_type)
        
        # Calculate implied volatilities
        iv_surface = np.zeros_like(price_surface)
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                price = price_surface[i, j]
                
                if option_type.lower() == 'call':
                    iv = self.implied_volatility_call(S0, K, T, r, price)
                else:
                    iv = self.implied_volatility_put(S0, K, T, r, price)
                
                iv_surface[i, j] = iv
        
        return iv_surface 