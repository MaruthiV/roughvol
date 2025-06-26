"""
Rough Volatility Research Library

A self-contained Python library for calibrating and testing the rough Bergomi (rBergomi) 
stochastic-volatility model on live SPX option-chain data.

Author: Quantitative Finance Engineer
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Engineer"

from .rbmc import RoughBergomiMC
from .data import OptionDataFetcher
from .price import MonteCarloPricer
from .calibrate import RoughBergomiCalibrator

__all__ = [
    "RoughBergomiMC",
    "OptionDataFetcher", 
    "MonteCarloPricer",
    "RoughBergomiCalibrator"
] 