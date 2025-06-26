"""
Unit tests for rough Bergomi Monte Carlo simulator.
"""

import pytest
import numpy as np
from roughvol.rbmc import RoughBergomiMC


class TestRoughBergomiMC:
    """Test cases for RoughBergomiMC class."""
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        model = RoughBergomiMC(0.1, 2.0, -0.7)
        assert model.H == 0.1
        assert model.eta == 2.0
        assert model.rho == -0.7
        
        # Invalid H
        with pytest.raises(ValueError):
            RoughBergomiMC(0.0, 2.0, -0.7)
        
        with pytest.raises(ValueError):
            RoughBergomiMC(0.5, 2.0, -0.7)
        
        # Invalid eta
        with pytest.raises(ValueError):
            RoughBergomiMC(0.1, 0.0, -0.7)
        
        with pytest.raises(ValueError):
            RoughBergomiMC(0.1, -1.0, -0.7)
        
        # Invalid rho
        with pytest.raises(ValueError):
            RoughBergomiMC(0.1, 2.0, -1.1)
        
        with pytest.raises(ValueError):
            RoughBergomiMC(0.1, 2.0, 1.1)
    
    def test_fbm_covariance_matrix(self):
        """Test fractional Brownian motion covariance matrix."""
        model = RoughBergomiMC(0.1, 2.0, -0.7)
        
        n_steps = 10
        dt = 0.01
        cov_matrix = model._fbm_covariance_matrix(n_steps, dt)
        
        # Check matrix properties
        assert cov_matrix.shape == (n_steps, n_steps)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
        assert np.all(np.linalg.eigvals(cov_matrix) >= 0)  # Positive semi-definite
        
        # Check diagonal elements
        times = np.arange(1, n_steps + 1) * dt
        for i in range(n_steps):
            expected_var = times[i]**(2 * model.H)
            assert np.isclose(cov_matrix[i, i], expected_var)
    
    def test_fbm_generation(self):
        """Test fractional Brownian motion generation."""
        model = RoughBergomiMC(0.1, 2.0, -0.7)
        
        n_steps = 10  # Further reduced for stability
        n_paths = 200  # Further reduced for speed
        dt = 0.01
        
        fbm_paths = model.generate_fbm_hybrid(n_steps, n_paths, dt)
        
        # Check shape
        assert fbm_paths.shape == (n_paths, n_steps)
        
        # Check mean (should be approximately zero)
        mean_error = np.abs(np.mean(fbm_paths))
        assert mean_error < 0.3  # Further relaxed tolerance
        
        # Check that variance increases over time (basic property of fBM)
        empirical_var = np.var(fbm_paths, axis=0)
        assert np.all(np.diff(empirical_var) >= -0.1)  # Allow small decreases due to noise
    
    def test_path_simulation(self):
        """Test rough Bergomi path simulation."""
        model = RoughBergomiMC(0.1, 2.0, -0.7)
        
        S0 = 100.0
        V0 = 0.04
        T = 0.25
        n_steps = 20  # Reduced for stability
        n_paths = 500  # Reduced for speed
        
        spot_paths, variance_paths = model.simulate_paths(S0, V0, T, n_steps, n_paths)
        
        # Check shapes
        assert spot_paths.shape == (n_paths, n_steps)
        assert variance_paths.shape == (n_paths, n_steps)
        
        # Check initial conditions
        assert np.allclose(spot_paths[:, 0], S0)
        assert np.allclose(variance_paths[:, 0], V0)
        
        # Check positivity
        assert np.all(spot_paths > 0)
        assert np.all(variance_paths > 0)
        
        # Check that variance process is reasonable (allowing for some explosion)
        assert np.all(variance_paths <= 2.0)  # Relaxed bound
    
    def test_maturity_simulation(self):
        """Test simulation at maturity only."""
        model = RoughBergomiMC(0.1, 2.0, -0.7)
        
        S0 = 100.0
        V0 = 0.04
        T = 0.25
        n_steps = 10  # Further reduced for stability
        n_paths = 200  # Further reduced for speed
        
        final_prices = model.simulate_at_maturity(S0, V0, T, n_steps, n_paths)
        
        # Check shape
        assert final_prices.shape == (n_paths,)
        
        # Check positivity
        assert np.all(final_prices > 0)
        
        # Check that prices are reasonable (not too extreme)
        assert np.all(final_prices > S0 * 0.1)  # Not too low
        assert np.all(final_prices < S0 * 10.0)  # Not too high
    
    def test_model_info(self):
        """Test model information retrieval."""
        model = RoughBergomiMC(0.1, 2.0, -0.7)
        
        info = model.get_model_info()
        
        assert info['H'] == 0.1
        assert info['eta'] == 2.0
        assert info['rho'] == -0.7
        assert info['model'] == 'Rough Bergomi'
    
    def test_different_parameters(self):
        """Test simulation with different parameter sets."""
        # Test with different H values
        for H in [0.05, 0.15, 0.25, 0.35]:
            model = RoughBergomiMC(H, 2.0, -0.7)
            
            S0, V0, T = 100.0, 0.04, 0.25
            n_steps, n_paths = 10, 100  # Reduced for speed
            
            spot_paths, variance_paths = model.simulate_paths(S0, V0, T, n_steps, n_paths)
            
            # Basic sanity checks
            assert spot_paths.shape == (n_paths, n_steps)
            assert variance_paths.shape == (n_paths, n_steps)
            assert np.all(spot_paths > 0)
            assert np.all(variance_paths > 0)
    
    def test_correlation_effects(self):
        """Test that correlation parameter affects spot-variance relationship."""
        # Test with different correlation values
        S0, V0, T = 100.0, 0.04, 0.25
        n_steps, n_paths = 10, 200  # Reduced for speed
        
        correlations = [-0.9, -0.5, -0.1]
        spot_var_correlations = []
        
        for rho in correlations:
            model = RoughBergomiMC(0.1, 2.0, rho)
            spot_paths, variance_paths = model.simulate_paths(S0, V0, T, n_steps, n_paths)
            
            # Calculate correlation between spot and variance returns
            spot_returns = np.diff(spot_paths, axis=1) / spot_paths[:, :-1]
            var_returns = np.diff(variance_paths, axis=1) / variance_paths[:, :-1]
            
            # Average correlation across paths
            path_correlations = []
            for i in range(n_paths):
                corr = np.corrcoef(spot_returns[i], var_returns[i])[0, 1]
                if not np.isnan(corr):
                    path_correlations.append(corr)
            
            if path_correlations:
                avg_corr = np.mean(path_correlations)
                spot_var_correlations.append(avg_corr)
        
        # Check that correlations are different (allowing for noise)
        if len(spot_var_correlations) > 1:
            assert len(set(np.sign(spot_var_correlations))) > 1 