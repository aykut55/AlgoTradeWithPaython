"""
Unit tests for the CIndicatorManager class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.indicators.indicator_manager import CIndicatorManager, IndicatorConfig, MAMethod
from src.core.base import SystemProtocol


class MockSystem:
    """Mock system class for testing."""
    
    def __init__(self):
        self.messages = []
    
    def mesaj(self, message: str) -> None:
        """Mock message handler."""
        self.messages.append(message)


class TestIndicatorConfig:
    """Test cases for IndicatorConfig."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = IndicatorConfig()
        assert len(config.fibonacci_periods) > 0
        assert len(config.common_periods) > 0
        assert len(config.ma_methods) > 0
        assert config.cache_size == 128


class TestCIndicatorManager:
    """Test cases for CIndicatorManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.indicator_manager = CIndicatorManager()
        self.mock_system = MockSystem()
        
        # Create sample market data
        self.sample_data = np.array([
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            110.0, 109.0, 108.0, 107.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0
        ], dtype=np.float64)
    
    # ==================== Initialization Tests ====================
    
    def test_indicator_manager_creation(self):
        """Test CIndicatorManager instantiation."""
        manager = CIndicatorManager()
        assert isinstance(manager, CIndicatorManager)
        assert manager.bar_count == 0
        assert len(manager.ma_cache) == 0
    
    def test_initialize_and_reset(self):
        """Test initialize and reset methods."""
        # Initialize with data
        result = self.indicator_manager.initialize(
            self.mock_system, None,
            self.sample_data, self.sample_data, self.sample_data,
            self.sample_data, self.sample_data, [1.0] * len(self.sample_data)
        )
        
        assert result == self.indicator_manager
        assert self.indicator_manager.bar_count == len(self.sample_data)
        
        # Reset should clear everything
        result_reset = self.indicator_manager.reset(self.mock_system)
        assert result_reset == self.indicator_manager
        assert len(self.indicator_manager.ma_cache) == 0
    
    # ==================== Simple Moving Average Tests ====================
    
    def test_calculate_sma_basic(self):
        """Test basic SMA calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        period = 3
        
        sma = self.indicator_manager.calculate_sma(data, period)
        
        # First 2 values should be NaN, then [2, 3, 4, 5, 6, 7, 8, 9]
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        assert sma[2] == 2.0  # (1+2+3)/3
        assert sma[3] == 3.0  # (2+3+4)/3
        assert sma[-1] == 9.0  # (8+9+10)/3
    
    def test_calculate_sma_edge_cases(self):
        """Test SMA edge cases."""
        # Empty data
        sma = self.indicator_manager.calculate_sma([], 5)
        assert len(sma) == 0
        
        # Period larger than data
        sma = self.indicator_manager.calculate_sma([1, 2], 5)
        assert all(np.isnan(sma))
        
        # Invalid period
        with pytest.raises(ValueError):
            self.indicator_manager.calculate_sma([1, 2, 3], 0)
    
    # ==================== Exponential Moving Average Tests ====================
    
    def test_calculate_ema_basic(self):
        """Test basic EMA calculation."""
        data = [1, 2, 3, 4, 5]
        period = 3
        
        ema = self.indicator_manager.calculate_ema(data, period)
        
        # EMA should start with first value and converge
        assert len(ema) == len(data)
        assert ema[0] == 1.0  # First value
        assert ema[-1] > ema[0]  # Should increase with increasing data
        
        # EMA should be smoother than SMA (less lag)
        sma = self.indicator_manager.calculate_sma(data, period)
        # Last values: EMA should react more quickly
        assert not np.isnan(ema[-1])
    
    def test_calculate_ema_edge_cases(self):
        """Test EMA edge cases."""
        # Empty data
        ema = self.indicator_manager.calculate_ema([], 5)
        assert len(ema) == 0
        
        # Single value
        ema = self.indicator_manager.calculate_ema([100], 3)
        assert len(ema) == 1
        assert ema[0] == 100
    
    # ==================== Weighted Moving Average Tests ====================
    
    def test_calculate_wma_basic(self):
        """Test basic WMA calculation."""
        data = [1, 2, 3, 4, 5]
        period = 3
        
        wma = self.indicator_manager.calculate_wma(data, period)
        
        # First 2 values should be NaN
        assert np.isnan(wma[0])
        assert np.isnan(wma[1])
        
        # Third value: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 2.33...
        expected_third = (1*1 + 2*2 + 3*3) / (1+2+3)
        assert abs(wma[2] - expected_third) < 1e-10
    
    def test_calculate_wma_edge_cases(self):
        """Test WMA edge cases."""
        # Period larger than data
        wma = self.indicator_manager.calculate_wma([1, 2], 5)
        assert all(np.isnan(wma))
    
    # ==================== Hull Moving Average Tests ====================
    
    def test_calculate_hull_ma_basic(self):
        """Test basic Hull MA calculation."""
        data = np.arange(1, 21, dtype=float)  # 1, 2, 3, ..., 20
        period = 10
        
        hull_ma = self.indicator_manager.calculate_hull_ma(data, period)
        
        assert len(hull_ma) == len(data)
        # Hull MA should have some valid values (not all NaN)
        assert not all(np.isnan(hull_ma))
    
    # ==================== Generic MA Tests ====================
    
    def test_calculate_ma_different_methods(self):
        """Test MA calculation with different methods."""
        data = self.sample_data
        period = 5
        
        # Test different methods
        sma = self.indicator_manager.calculate_ma(self.mock_system, data, "Simple", period)
        ema = self.indicator_manager.calculate_ma(self.mock_system, data, "Exp", period)
        wma = self.indicator_manager.calculate_ma(self.mock_system, data, "Weighted", period)
        hull = self.indicator_manager.calculate_ma(self.mock_system, data, "Hull", period)
        
        # All should have same length
        assert len(sma) == len(ema) == len(wma) == len(hull) == len(data)
        
        # Different methods should produce different results
        assert not np.array_equal(sma, ema)
        assert not np.array_equal(sma, wma)
    
    def test_calculate_ma_caching(self):
        """Test MA calculation caching."""
        data = self.sample_data
        period = 5
        
        # First calculation
        ma1 = self.indicator_manager.calculate_ma(self.mock_system, data, "Simple", period)
        cache_size_1 = len(self.indicator_manager.ma_cache)
        
        # Second calculation with same parameters should use cache
        ma2 = self.indicator_manager.calculate_ma(self.mock_system, data, "Simple", period)
        cache_size_2 = len(self.indicator_manager.ma_cache)
        
        # Results should be identical and cache size shouldn't increase
        np.testing.assert_array_equal(ma1, ma2)
        assert cache_size_1 == cache_size_2
    
    def test_calculate_ma_unknown_method(self):
        """Test MA calculation with unknown method."""
        data = self.sample_data
        
        with pytest.warns(UserWarning, match="Unknown MA method"):
            ma = self.indicator_manager.calculate_ma(self.mock_system, data, "UnknownMethod", 5)
            # Should default to Simple MA
            expected = self.indicator_manager.calculate_sma(data, 5)
            np.testing.assert_array_equal(ma, expected)
    
    # ==================== RSI Tests ====================
    
    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create data with clear trend
        uptrend_data = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
        downtrend_data = np.array([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100], dtype=float)
        
        rsi_up = self.indicator_manager.calculate_rsi(self.mock_system, uptrend_data, 5)
        rsi_down = self.indicator_manager.calculate_rsi(self.mock_system, downtrend_data, 5)
        
        # RSI should be between 0 and 100
        valid_rsi_up = rsi_up[~np.isnan(rsi_up)]
        valid_rsi_down = rsi_down[~np.isnan(rsi_down)]
        
        assert all(0 <= val <= 100 for val in valid_rsi_up)
        assert all(0 <= val <= 100 for val in valid_rsi_down)
        
        # Uptrend should have higher RSI than downtrend
        if len(valid_rsi_up) > 0 and len(valid_rsi_down) > 0:
            assert np.mean(valid_rsi_up) > np.mean(valid_rsi_down)
    
    def test_calculate_rsi_edge_cases(self):
        """Test RSI edge cases."""
        # Insufficient data
        rsi = self.indicator_manager.calculate_rsi(self.mock_system, [100, 101], 14)
        assert all(np.isnan(rsi))
        
        # Invalid period
        with pytest.raises(ValueError):
            self.indicator_manager.calculate_rsi(self.mock_system, self.sample_data, 0)
    
    def test_calculate_rsi_caching(self):
        """Test RSI caching."""
        data = self.sample_data
        period = 14
        
        # First calculation
        rsi1 = self.indicator_manager.calculate_rsi(self.mock_system, data, period)
        cache_size_1 = len(self.indicator_manager.rsi_cache)
        
        # Second calculation should use cache
        rsi2 = self.indicator_manager.calculate_rsi(self.mock_system, data, period)
        cache_size_2 = len(self.indicator_manager.rsi_cache)
        
        np.testing.assert_array_equal(rsi1, rsi2)
        assert cache_size_1 == cache_size_2
    
    # ==================== MACD Tests ====================
    
    def test_calculate_macd_basic(self):
        """Test basic MACD calculation."""
        data = self.sample_data
        
        macd_line, signal_line, histogram = self.indicator_manager.calculate_macd(
            self.mock_system, data, 5, 10, 3
        )
        
        # All arrays should have same length as input data
        assert len(macd_line) == len(signal_line) == len(histogram) == len(data)
        
        # Histogram should be MACD line - Signal line
        np.testing.assert_array_almost_equal(histogram, macd_line - signal_line)
        
        # MACD values should be finite (not NaN or inf) for later periods
        valid_indices = ~(np.isnan(macd_line) | np.isnan(signal_line))
        if np.any(valid_indices):
            assert all(np.isfinite(macd_line[valid_indices]))
            assert all(np.isfinite(signal_line[valid_indices]))
    
    def test_calculate_macd_parameters(self):
        """Test MACD with different parameters."""
        data = self.sample_data
        
        # Standard MACD (12, 26, 9)
        macd1, signal1, hist1 = self.indicator_manager.calculate_macd(self.mock_system, data)
        
        # Faster MACD (5, 10, 3)
        macd2, signal2, hist2 = self.indicator_manager.calculate_macd(self.mock_system, data, 5, 10, 3)
        
        # Results should be different
        assert not np.allclose(macd1, macd2, equal_nan=True)
    
    def test_calculate_macd_edge_cases(self):
        """Test MACD edge cases."""
        # Invalid parameters
        with pytest.raises(ValueError):
            self.indicator_manager.calculate_macd(self.mock_system, self.sample_data, 10, 5, 3)  # fast >= slow
    
    # ==================== Bulk Operations Tests ====================
    
    def test_fill_ma_list_single_method(self):
        """Test bulk MA calculation with single method."""
        data = self.sample_data
        periods = [5, 10, 20]
        
        ma_list = self.indicator_manager.fill_ma_list(
            self.mock_system, data, "Simple", periods
        )
        
        assert len(ma_list) == len(periods)
        assert len(self.indicator_manager.ma_params_list) == len(periods)
        
        # Check parameter names
        for i, period in enumerate(periods):
            assert self.indicator_manager.ma_params_list[i] == f"Simple_{period}"
    
    def test_fill_ma_list_multiple_methods(self):
        """Test bulk MA calculation with multiple methods."""
        data = self.sample_data
        methods = ["Simple", "Exp"]
        periods = [5, 10]
        
        ma_list = self.indicator_manager.fill_ma_list(
            self.mock_system, data, methods, periods
        )
        
        expected_count = len(methods) * len(periods)
        assert len(ma_list) == expected_count
        assert len(self.indicator_manager.ma_params_list) == expected_count
    
    def test_fill_rsi_list(self):
        """Test bulk RSI calculation."""
        data = self.sample_data
        periods = [14, 21, 28]
        
        rsi_list = self.indicator_manager.fill_rsi_list(
            self.mock_system, data, periods
        )
        
        assert len(rsi_list) == len(periods)
        assert len(self.indicator_manager.rsi_params_list) == len(periods)
        
        # Check that each RSI has correct length
        for rsi in rsi_list:
            assert len(rsi) == len(data)
    
    def test_fill_macd_list(self):
        """Test bulk MACD calculation."""
        data = self.sample_data
        fast_periods = [5, 12]
        slow_periods = [10, 26]
        
        macd_list = self.indicator_manager.fill_macd_list(
            self.mock_system, data, fast_periods, slow_periods, 9
        )
        
        # Should only include valid combinations (fast < slow)
        expected_count = 3  # (5,10), (5,26), (12,26)
        assert len(macd_list) == expected_count
        assert len(self.indicator_manager.macd_params_list) == expected_count
        
        # Each MACD should be a tuple of 3 arrays
        for macd_tuple in macd_list:
            assert len(macd_tuple) == 3
            for array in macd_tuple:
                assert len(array) == len(data)
    
    # ==================== Utility Methods Tests ====================
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Calculate some indicators to populate cache
        self.indicator_manager.calculate_ma(self.mock_system, self.sample_data, "Simple", 10)
        self.indicator_manager.calculate_rsi(self.mock_system, self.sample_data, 14)
        
        stats = self.indicator_manager.get_cache_stats()
        
        assert 'ma_cache_size' in stats
        assert 'rsi_cache_size' in stats
        assert 'macd_cache_size' in stats
        assert 'max_cache_size' in stats
        
        assert stats['ma_cache_size'] >= 1
        assert stats['rsi_cache_size'] >= 1
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Populate caches
        self.indicator_manager.calculate_ma(self.mock_system, self.sample_data, "Simple", 10)
        self.indicator_manager.calculate_rsi(self.mock_system, self.sample_data, 14)
        
        # Verify caches have content
        assert len(self.indicator_manager.ma_cache) > 0
        assert len(self.indicator_manager.rsi_cache) > 0
        
        # Clear caches
        self.indicator_manager.clear_cache()
        
        # Verify caches are empty
        assert len(self.indicator_manager.ma_cache) == 0
        assert len(self.indicator_manager.rsi_cache) == 0
        assert len(self.indicator_manager.macd_cache) == 0
    
    def test_repr(self):
        """Test string representation."""
        # Initialize with some data
        self.indicator_manager.initialize(
            self.mock_system, None,
            self.sample_data, self.sample_data, self.sample_data,
            self.sample_data, self.sample_data, [1.0] * len(self.sample_data)
        )
        
        repr_str = repr(self.indicator_manager)
        assert "CIndicatorManager" in repr_str
        assert f"bars={len(self.sample_data)}" in repr_str
        assert "cache_stats" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])