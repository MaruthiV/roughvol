"""
Option Data Fetcher

Fetches SPX spot prices and option chain data from Yahoo Finance
with intelligent caching and data preprocessing.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import os
import warnings
from dataclasses import dataclass


@dataclass
class OptionData:
    """Container for option market data."""
    spot_price: float
    risk_free_rate: float
    expiry_dates: List[datetime]
    strikes: List[float]
    call_prices: Dict[Tuple[datetime, float], float]
    put_prices: Dict[Tuple[datetime, float], float]
    call_bids: Dict[Tuple[datetime, float], float]
    call_asks: Dict[Tuple[datetime, float], float]
    put_bids: Dict[Tuple[datetime, float], float]
    put_asks: Dict[Tuple[datetime, float], float]


class OptionDataFetcher:
    """
    Fetches and caches option data from Yahoo Finance.
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching data
    cache_timeout : int, optional
        Cache timeout in seconds (default: 24 hours)
    """
    
    def __init__(self, cache_dir: str = "/tmp/roughvol_cache", 
                 cache_timeout: int = 86400):
        self.cache_dir = cache_dir
        self.cache_timeout = cache_timeout
        
        # Setup caching
        os.makedirs(cache_dir, exist_ok=True)
        self.memory = joblib.Memory(cache_dir, verbose=0)
        
        # Cache the data fetching functions
        self._fetch_spot_data = self.memory.cache(
            self._fetch_spot_data_impl, 
            ignore=['self']
        )
        self._fetch_option_data = self.memory.cache(
            self._fetch_option_data_impl,
            ignore=['self']
        )
    
    def fetch_data(self, ticker: str, start_date: str, end_date: str,
                  target_maturities: List[int]) -> OptionData:
        """
        Fetch spot and option data for the given period.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        target_maturities : List[int]
            Target maturities in days
            
        Returns
        -------
        OptionData
            Container with all market data
        """
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Fetch spot data
        spot_data = self._fetch_spot_data(ticker, start_date, end_date)
        spot_price = spot_data['Close'].iloc[-1]
        
        # Estimate risk-free rate (using 3-month Treasury yield as proxy)
        risk_free_rate = self._estimate_risk_free_rate()
        
        # Fetch option data
        option_data = self._fetch_option_data(ticker, start_date, end_date)
        
        # Filter for target maturities
        filtered_data = self._filter_by_maturities(
            option_data, target_maturities, spot_price
        )
        
        return filtered_data
    
    def _fetch_spot_data_impl(self, ticker: str, start_date: str, 
                             end_date: str) -> pd.DataFrame:
        """Fetch spot price data."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch spot data for {ticker}: {e}")
    
    def _fetch_option_data_impl(self, ticker: str, start_date: str, 
                               end_date: str) -> Dict:
        """Fetch option chain data."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get all available expiration dates
            expirations = stock.options
            
            if not expirations:
                raise ValueError(f"No option data available for {ticker}")
            
            option_data = {}
            
            for expiry in expirations:
                try:
                    # Get option chain for this expiration
                    chain = stock.option_chain(expiry)
                    
                    # Store calls and puts
                    option_data[expiry] = {
                        'calls': chain.calls,
                        'puts': chain.puts
                    }
                    
                except Exception as e:
                    warnings.warn(f"Failed to fetch options for {expiry}: {e}")
                    continue
            
            return option_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch option data for {ticker}: {e}")
    
    def _estimate_risk_free_rate(self) -> float:
        """Estimate risk-free rate using 3-month Treasury yield."""
        try:
            # Use 3-month Treasury yield as proxy for risk-free rate
            treasury = yf.Ticker("^IRX")  # 13-week Treasury yield
            data = treasury.history(period="1mo")
            
            if not data.empty:
                # Convert from percentage to decimal
                rate = data['Close'].iloc[-1] / 100
                return max(rate, 0.001)  # Minimum 0.1%
            else:
                return 0.05  # Default 5% if data unavailable
                
        except Exception:
            return 0.05  # Default 5% if estimation fails
    
    def _filter_by_maturities(self, option_data: Dict, target_maturities: List[int],
                             spot_price: float) -> OptionData:
        """
        Filter option data to match target maturities.
        
        Parameters
        ----------
        option_data : Dict
            Raw option data from Yahoo Finance
        target_maturities : List[int]
            Target maturities in days
        spot_price : float
            Current spot price
            
        Returns
        -------
        OptionData
            Filtered option data
        """
        today = datetime.now()
        
        # Find closest expirations to target maturities
        selected_expiries = []
        for target_days in target_maturities:
            target_date = today + timedelta(days=target_days)
            
            # Find closest expiration
            closest_expiry = None
            min_diff = float('inf')
            
            for expiry_str in option_data.keys():
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                diff = abs((expiry_date - target_date).days)
                
                if diff < min_diff:
                    min_diff = diff
                    closest_expiry = expiry_str
            
            if closest_expiry and min_diff <= 30:  # Within 30 days
                selected_expiries.append(closest_expiry)
        
        # Extract data for selected expirations
        expiry_dates = []
        strikes = []
        call_prices = {}
        put_prices = {}
        call_bids = {}
        call_asks = {}
        put_bids = {}
        put_asks = {}
        
        for expiry in selected_expiries:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            expiry_dates.append(expiry_date)
            
            data = option_data[expiry]
            
            # Process calls
            for _, row in data['calls'].iterrows():
                strike = row['strike']
                mid_price = (row['bid'] + row['ask']) / 2
                
                strikes.append(strike)
                call_prices[(expiry_date, strike)] = mid_price
                call_bids[(expiry_date, strike)] = row['bid']
                call_asks[(expiry_date, strike)] = row['ask']
            
            # Process puts
            for _, row in data['puts'].iterrows():
                strike = row['strike']
                mid_price = (row['bid'] + row['ask']) / 2
                
                strikes.append(strike)
                put_prices[(expiry_date, strike)] = mid_price
                put_bids[(expiry_date, strike)] = row['bid']
                put_asks[(expiry_date, strike)] = row['ask']
        
        # Remove duplicates and sort
        strikes = sorted(list(set(strikes)))
        expiry_dates = sorted(expiry_dates)
        
        return OptionData(
            spot_price=spot_price,
            risk_free_rate=self._estimate_risk_free_rate(),
            expiry_dates=expiry_dates,
            strikes=strikes,
            call_prices=call_prices,
            put_prices=put_prices,
            call_bids=call_bids,
            call_asks=call_asks,
            put_bids=put_bids,
            put_asks=put_asks
        )
    
    def clear_cache(self):
        """Clear the data cache."""
        self.memory.clear()
    
    def get_cache_info(self) -> Dict:
        """Get cache information."""
        return {
            'cache_dir': self.cache_dir,
            'cache_timeout': self.cache_timeout,
            'cache_size': len(os.listdir(self.cache_dir)) if os.path.exists(self.cache_dir) else 0
        } 