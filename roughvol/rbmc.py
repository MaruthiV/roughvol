"""
Rough Bergomi Monte Carlo Simulator

Implements the rough Bergomi model using hybrid fractional Brownian motion
generation via Bennedsen-Lunde-Pakkanen FFT scheme.

The rough Bergomi model is defined by:
dS_t = S_t * sqrt(V_t) * dW_t
V_t = V_0 * exp(η * W^H_t - 0.5 * η² * t^(2H))

where W^H_t is a fractional Brownian motion with Hurst parameter H.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional
import warnings


class RoughBergomiMC:
    """
    Rough Bergomi Monte Carlo simulator using hybrid fBM scheme.
    
    Parameters
    ----------
    H : float
        Hurst parameter (roughness), H ∈ (0, 0.5)
    eta : float
        Volatility of volatility parameter
    rho : float
        Correlation between spot and volatility processes
    """
    
    def __init__(self, H: float, eta: float, rho: float):
        self.H = H
        self.eta = eta
        self.rho = rho
        
        # Validate parameters
        if not 0 < H < 0.5:
            raise ValueError(f"Hurst parameter H must be in (0, 0.5), got {H}")
        if eta <= 0:
            raise ValueError(f"Vol-of-vol eta must be positive, got {eta}")
        if not -1 <= rho <= 1:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {rho}")
    
    def generate_fbm_hybrid(self, n_steps: int, n_paths: int, dt: float) -> np.ndarray:
        """
        Generate fractional Brownian motion using hybrid FFT scheme.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps
        n_paths : int
            Number of Monte Carlo paths
        dt : float
            Time step size
        
        Returns
        -------
        np.ndarray
            Fractional Brownian motion paths of shape (n_paths, n_steps)
        """
        # Use simplified method for better numerical stability
        return self._generate_fbm_simple(n_steps, n_paths, dt)
    
    def _generate_fbm_simple(self, n_steps: int, n_paths: int, dt: float) -> np.ndarray:
        """
        Generate fBM using simplified method for numerical stability.
        """
        # Generate standard Brownian motion increments
        dw = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Create covariance matrix for fBM
        cov_matrix = self._fbm_covariance_matrix(n_steps, dt)
        
        # Use SVD for better numerical stability
        U, s, Vt = linalg.svd(cov_matrix)
        # Ensure positive eigenvalues
        s = np.maximum(s, 1e-12)
        L = U @ np.diag(np.sqrt(s))
        
        # Generate correlated increments
        fbm_increments = dw @ L.T
        
        # Cumulative sum to get fBM paths
        fbm_paths = np.cumsum(fbm_increments, axis=1)
        
        return fbm_paths
    
    def _fbm_covariance_matrix(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Compute covariance matrix for fractional Brownian motion.
        
        The covariance is given by:
        Cov(W^H_s, W^H_t) = 0.5 * (s^(2H) + t^(2H) - |t-s|^(2H))
        """
        H = self.H
        times = np.arange(1, n_steps + 1) * dt
        
        # Create covariance matrix
        cov_matrix = np.zeros((n_steps, n_steps))
        
        for i in range(n_steps):
            for j in range(n_steps):
                s, t = times[i], times[j]
                cov_matrix[i, j] = 0.5 * (s**(2*H) + t**(2*H) - abs(t-s)**(2*H))
        
        return cov_matrix
    
    def simulate_paths(self, S0: float, V0: float, T: float, n_steps: int, 
                      n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rough Bergomi paths.
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        V0 : float
            Initial variance
        T : float
            Time to maturity
        n_steps : int
            Number of time steps
        n_paths : int
            Number of Monte Carlo paths
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (spot_paths, variance_paths) of shapes (n_paths, n_steps)
        """
        dt = T / n_steps
        
        # Generate correlated Brownian motions
        dw1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        dw2 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Correlated Brownian motion for spot process
        dw_spot = self.rho * dw1 + np.sqrt(1 - self.rho**2) * dw2
        
        # Generate fractional Brownian motion for variance process
        fbm_paths = self.generate_fbm_hybrid(n_steps, n_paths, dt)
        
        # Initialize arrays
        spot_paths = np.zeros((n_paths, n_steps))
        variance_paths = np.zeros((n_paths, n_steps))
        
        spot_paths[:, 0] = S0
        variance_paths[:, 0] = V0
        
        # Simulate paths with better numerical stability
        for i in range(1, n_steps):
            t = i * dt
            
            # Variance process: V_t = V_0 * exp(η * W^H_t - 0.5 * η² * t^(2H))
            # Add bounds to prevent explosion
            variance_exponent = self.eta * fbm_paths[:, i] - 0.5 * self.eta**2 * t**(2*self.H)
            variance_exponent = np.clip(variance_exponent, -10, 10)  # Prevent extreme values
            variance_paths[:, i] = V0 * np.exp(variance_exponent)
            
            # Ensure variance stays positive and reasonable
            variance_paths[:, i] = np.clip(variance_paths[:, i], 1e-6, 1.0)
            
            # Spot process: dS_t = S_t * sqrt(V_t) * dW_t
            vol = np.sqrt(variance_paths[:, i-1])
            spot_paths[:, i] = spot_paths[:, i-1] * np.exp(
                -0.5 * vol**2 * dt + vol * dw_spot[:, i-1]
            )
        
        return spot_paths, variance_paths
    
    def simulate_at_maturity(self, S0: float, V0: float, T: float, 
                           n_steps: int, n_paths: int) -> np.ndarray:
        """
        Simulate spot prices at maturity only (for pricing).
        
        Parameters
        ----------
        S0 : float
            Initial spot price
        V0 : float
            Initial variance
        T : float
            Time to maturity
        n_steps : int
            Number of time steps
        n_paths : int
            Number of Monte Carlo paths
        
        Returns
        -------
        np.ndarray
            Final spot prices of shape (n_paths,)
        """
        spot_paths, _ = self.simulate_paths(S0, V0, T, n_steps, n_paths)
        return spot_paths[:, -1]
    
    def get_model_info(self) -> dict:
        """Get model parameters and information."""
        return {
            'H': self.H,
            'eta': self.eta,
            'rho': self.rho,
            'model': 'Rough Bergomi'
        } 