"""
Technical Indicator Manager for algorithmic trading system.

This module contains the CIndicatorManager class which provides comprehensive
technical indicator calculations including moving averages, RSI, MACD, and more.
"""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
import warnings

from src.core.base import CBase, SystemProtocol

# Type aliases
NumericType = Union[int, float, np.number]
NumericList = Union[List[NumericType], np.ndarray]


class MAMethod(Enum):
    """Moving Average calculation methods."""
    SIMPLE = "Simple"
    EXPONENTIAL = "Exp" 
    WEIGHTED = "Weighted"
    HULL = "HullMA"
    TRIANGULAR = "Triangular"
    VARIABLE = "Variable"
    VOLUME = "Volume"
    WILDER = "Wilder"
    ZERO_LAG = "ZeroLag"
    TIME_SERIES = "TimeSeries"


@dataclass
class IndicatorConfig:
    """Configuration container for indicators."""
    
    # Common MA periods (Fibonacci sequence + common numbers)
    fibonacci_periods: List[int] = field(default_factory=lambda: [3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
    common_periods: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 45, 50, 100, 200, 500, 1000])
    
    # Supported MA methods
    ma_methods: List[str] = field(default_factory=lambda: [method.value for method in MAMethod])
    
    # Cache size for performance
    cache_size: int = 128


class CIndicatorManager(CBase):
    """
    Technical Indicator Manager providing comprehensive indicator calculations.
    
    Features:
    - Multiple Moving Average types (Simple, EMA, Hull, etc.)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Stochastic oscillators
    - Bulk calculation support
    - Performance optimization with caching
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize the indicator manager.
        
        Args:
            config: Optional configuration for indicators
        """
        super().__init__()
        self.config = config or IndicatorConfig()
        
        # Storage for calculated indicators
        self.ma_cache: Dict[str, np.ndarray] = {}
        self.rsi_cache: Dict[str, np.ndarray] = {}
        self.macd_cache: Dict[str, np.ndarray] = {}
        
        # Parameter tracking
        self.ma_params_list: List[str] = []
        self.ma_list: List[np.ndarray] = []
        self.ma_dictionary: Dict[str, np.ndarray] = {}
        
        self.rsi_params_list: List[str] = []
        self.rsi_list: List[np.ndarray] = []
        
        self.macd_params_list: List[str] = []
        self.macd_list: List[np.ndarray] = []
    
    def initialize(
        self,
        sistem: SystemProtocol,
        v: Any,
        open_data: NumericList,
        high_data: NumericList,
        low_data: NumericList,
        close_data: NumericList,
        volume_data: NumericList,
        lot_data: NumericList
    ) -> 'CIndicatorManager':
        """
        Initialize the indicator manager with market data.
        
        Args:
            sistem: Trading system reference
            v: Original data reference
            open_data: Opening prices
            high_data: High prices
            low_data: Low prices
            close_data: Closing prices
            volume_data: Volume data
            lot_data: Lot data
            
        Returns:
            Self for method chaining
        """
        self.set_data(sistem, v, open_data, high_data, low_data, close_data, volume_data, lot_data)
        return self
    
    def reset(self, sistem: SystemProtocol) -> 'CIndicatorManager':
        """
        Reset all cached indicators and parameters.
        
        Args:
            sistem: Trading system reference
            
        Returns:
            Self for method chaining
        """
        # Clear caches
        self.ma_cache.clear()
        self.rsi_cache.clear()
        self.macd_cache.clear()
        
        # Clear parameter lists
        self.ma_params_list.clear()
        self.ma_list.clear()
        self.ma_dictionary.clear()
        
        self.rsi_params_list.clear()
        self.rsi_list.clear()
        
        self.macd_params_list.clear()
        self.macd_list.clear()
        
        return self
    
    # ==================== Moving Average Calculations ====================
    
    def calculate_sma(self, source: NumericList, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average.
        
        Args:
            source: Price series
            period: Calculation period
            
        Returns:
            SMA values as numpy array
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        source_array = np.array(source, dtype=np.float64)
        if len(source_array) < period:
            return np.full(len(source_array), np.nan)
        
        # Use pandas rolling for efficient calculation
        series = pd.Series(source_array)
        sma = series.rolling(window=period, min_periods=period).mean()
        
        return sma.values
    
    def calculate_ema(self, source: NumericList, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            source: Price series
            period: Calculation period
            
        Returns:
            EMA values as numpy array
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        source_array = np.array(source, dtype=np.float64)
        if len(source_array) == 0:
            return np.array([])
        
        # Calculate EMA using pandas for efficiency
        series = pd.Series(source_array)
        ema = series.ewm(span=period, adjust=False).mean()
        
        return ema.values
    
    def calculate_wma(self, source: NumericList, period: int) -> np.ndarray:
        """
        Calculate Weighted Moving Average.
        
        Args:
            source: Price series
            period: Calculation period
            
        Returns:
            WMA values as numpy array
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        source_array = np.array(source, dtype=np.float64)
        if len(source_array) < period:
            return np.full(len(source_array), np.nan)
        
        # Create weights (1, 2, 3, ..., period)
        weights = np.arange(1, period + 1, dtype=np.float64)
        weights = weights / weights.sum()
        
        # Calculate WMA using convolution
        wma = np.convolve(source_array, weights[::-1], mode='valid')
        
        # Pad with NaN for the initial period
        result = np.full(len(source_array), np.nan)
        result[period-1:] = wma
        
        return result
    
    def calculate_hull_ma(self, source: NumericList, period: int) -> np.ndarray:
        """
        Calculate Hull Moving Average.
        
        Args:
            source: Price series
            period: Calculation period
            
        Returns:
            Hull MA values as numpy array
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        source_array = np.array(source, dtype=np.float64)
        
        # Hull MA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))
        
        wma_half = self.calculate_wma(source_array, half_period)
        wma_full = self.calculate_wma(source_array, period)
        
        # Calculate 2*WMA(n/2) - WMA(n)
        hull_source = 2 * wma_half - wma_full
        
        # Apply WMA with sqrt(n) period
        hull_ma = self.calculate_wma(hull_source, sqrt_period)
        
        return hull_ma
    
    def calculate_ma(
        self,
        sistem: SystemProtocol,
        source: NumericList,
        method: str = "Simple",
        period: int = 14
    ) -> np.ndarray:
        """
        Calculate Moving Average using specified method.
        
        Args:
            sistem: Trading system reference
            source: Price series
            method: MA calculation method
            period: Calculation period
            
        Returns:
            MA values as numpy array
        """
        cache_key = f"{method}_{period}_{hash(tuple(source) if isinstance(source, list) else source.tobytes())}"
        
        if cache_key in self.ma_cache:
            return self.ma_cache[cache_key]
        
        method_upper = method.upper()
        
        if method_upper in ["SIMPLE", "SMA"]:
            result = self.calculate_sma(source, period)
        elif method_upper in ["EXPONENTIAL", "EXP", "EMA"]:
            result = self.calculate_ema(source, period)
        elif method_upper in ["WEIGHTED", "WMA"]:
            result = self.calculate_wma(source, period)
        elif method_upper in ["HULL", "HULLMA"]:
            result = self.calculate_hull_ma(source, period)
        elif method_upper == "TRIANGULAR":
            # Triangular MA is SMA of SMA
            sma_period = (period + 1) // 2
            sma1 = self.calculate_sma(source, sma_period)
            result = self.calculate_sma(sma1, sma_period)
        elif method_upper == "WILDER":
            # Wilder's smoothing (similar to EMA with different alpha)
            alpha = 1.0 / period
            result = self._calculate_wilder_ma(source, alpha)
        else:
            # Default to Simple MA for unknown methods
            warnings.warn(f"Unknown MA method '{method}', using Simple MA")
            result = self.calculate_sma(source, period)
        
        # Cache result if cache is not full
        if len(self.ma_cache) < self.config.cache_size:
            self.ma_cache[cache_key] = result
        
        return result
    
    def _calculate_wilder_ma(self, source: NumericList, alpha: float) -> np.ndarray:
        """Calculate Wilder's smoothing (internal helper)."""
        source_array = np.array(source, dtype=np.float64)
        if len(source_array) == 0:
            return np.array([])
        
        # For RSI, start with simple average of first period
        period = int(1.0 / alpha)
        result = np.zeros_like(source_array)
        
        # Calculate initial average
        if len(source_array) >= period:
            result[period-1] = np.mean(source_array[:period])
            
            # Apply Wilder smoothing from period onwards
            for i in range(period, len(source_array)):
                result[i] = alpha * source_array[i] + (1 - alpha) * result[i-1]
        else:
            # Not enough data, just use simple average where possible
            for i in range(len(source_array)):
                if i == 0:
                    result[i] = source_array[i]
                else:
                    result[i] = alpha * source_array[i] + (1 - alpha) * result[i-1]
        
        return result
    
    # ==================== RSI Calculation ====================
    
    def calculate_rsi(self, sistem: SystemProtocol, source: NumericList, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index.
        
        Args:
            sistem: Trading system reference
            source: Price series
            period: Calculation period
            
        Returns:
            RSI values as numpy array
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        cache_key = f"RSI_{period}_{hash(tuple(source) if isinstance(source, list) else source.tobytes())}"
        
        if cache_key in self.rsi_cache:
            return self.rsi_cache[cache_key]
        
        source_array = np.array(source, dtype=np.float64)
        if len(source_array) < period + 1:
            return np.full(len(source_array), np.nan)
        
        # Calculate price differences
        diff = np.diff(source_array)
        
        # Separate gains and losses
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        
        # Initialize result array
        result = np.full(len(source_array), np.nan)
        
        # Calculate initial averages (SMA for first period)
        if len(gains) >= period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate RSI for the period+1 index (period gains/losses + 1 for original index)
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                result[period] = 100 - (100 / (1 + rs))
            else:
                result[period] = 100.0 if avg_gain > 0 else 50.0
            
            # Continue with Wilder's smoothing
            alpha = 1.0 / period
            for i in range(period + 1, len(source_array)):
                gain_idx = i - 1  # diff index is one less than source index
                loss_idx = i - 1
                
                # Wilder's smoothing
                avg_gain = alpha * gains[gain_idx] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[loss_idx] + (1 - alpha) * avg_loss
                
                # Calculate RSI
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    result[i] = 100 - (100 / (1 + rs))
                else:
                    result[i] = 100.0 if avg_gain > 0 else 50.0
        
        # Cache result
        if len(self.rsi_cache) < self.config.cache_size:
            self.rsi_cache[cache_key] = result
        
        return result
    
    # ==================== MACD Calculation ====================
    
    def calculate_macd(
        self,
        sistem: SystemProtocol,
        source: NumericList,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            sistem: Trading system reference
            source: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        cache_key = f"MACD_{fast_period}_{slow_period}_{signal_period}_{hash(tuple(source) if isinstance(source, list) else source.tobytes())}"
        
        if cache_key in self.macd_cache:
            return self.macd_cache[cache_key]
        
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(source, fast_period)
        slow_ema = self.calculate_ema(source, slow_period)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Signal line = EMA of MACD line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Histogram = MACD line - Signal line
        histogram = macd_line - signal_line
        
        result = (macd_line, signal_line, histogram)
        
        # Cache result
        if len(self.macd_cache) < self.config.cache_size:
            self.macd_cache[cache_key] = result
        
        return result
    
    # ==================== Bulk Operations ====================
    
    def fill_ma_list(
        self,
        sistem: SystemProtocol,
        source: NumericList,
        methods: Union[str, List[str]],
        periods: List[int]
    ) -> List[np.ndarray]:
        """
        Calculate multiple Moving Averages in bulk.
        
        Args:
            sistem: Trading system reference
            source: Price series
            methods: MA method(s) to calculate
            periods: List of periods to calculate
            
        Returns:
            List of MA arrays
        """
        if isinstance(methods, str):
            methods = [methods]
        
        self.ma_list.clear()
        self.ma_params_list.clear()
        
        for method in methods:
            for period in periods:
                param = f"{method}_{period}"
                self.ma_params_list.append(param)
                
                ma = self.calculate_ma(sistem, source, method, period)
                self.ma_list.append(ma)
                self.ma_dictionary[str(period)] = ma
        
        return self.ma_list.copy()
    
    def fill_rsi_list(
        self,
        sistem: SystemProtocol,
        source: NumericList,
        periods: List[int]
    ) -> List[np.ndarray]:
        """
        Calculate multiple RSI values in bulk.
        
        Args:
            sistem: Trading system reference
            source: Price series
            periods: List of periods to calculate
            
        Returns:
            List of RSI arrays
        """
        self.rsi_list.clear()
        self.rsi_params_list.clear()
        
        for period in periods:
            param = str(period)
            self.rsi_params_list.append(param)
            
            rsi = self.calculate_rsi(sistem, source, period)
            self.rsi_list.append(rsi)
        
        return self.rsi_list.copy()
    
    def fill_macd_list(
        self,
        sistem: SystemProtocol,
        source: NumericList,
        fast_periods: List[int],
        slow_periods: List[int],
        signal_period: int = 9
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate multiple MACD values in bulk.
        
        Args:
            sistem: Trading system reference
            source: Price series
            fast_periods: List of fast EMA periods
            slow_periods: List of slow EMA periods
            signal_period: Signal line period
            
        Returns:
            List of MACD tuples (line, signal, histogram)
        """
        self.macd_list.clear()
        self.macd_params_list.clear()
        
        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow:
                    param = f"{fast}_{slow}_{signal_period}"
                    self.macd_params_list.append(param)
                    
                    macd = self.calculate_macd(sistem, source, fast, slow, signal_period)
                    self.macd_list.append(macd)
        
        return self.macd_list.copy()
    
    # ==================== Utility Methods ====================
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            'ma_cache_size': len(self.ma_cache),
            'rsi_cache_size': len(self.rsi_cache),
            'macd_cache_size': len(self.macd_cache),
            'max_cache_size': self.config.cache_size
        }
    
    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self.ma_cache.clear()
        self.rsi_cache.clear()
        self.macd_cache.clear()
    
    def __repr__(self) -> str:
        """String representation of the indicator manager."""
        return f"CIndicatorManager(bars={self.bar_count}, cache_stats={self.get_cache_stats()})"