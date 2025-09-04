"""
Unit tests for the CUtils class.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.utils.utils import CUtils
from src.core.base import SystemProtocol


class MockSystem:
    """Mock system class for testing."""
    
    def __init__(self):
        self.messages = []
    
    def mesaj(self, message: str) -> None:
        """Mock message handler."""
        self.messages.append(message)


class TestCUtils:
    """Test cases for CUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utils = CUtils()
        self.mock_system = MockSystem()
    
    # ==================== Initialization Tests ====================
    
    def test_utils_creation(self):
        """Test CUtils instantiation."""
        utils = CUtils()
        assert isinstance(utils, CUtils)
    
    def test_initialize_and_reset(self):
        """Test initialize and reset methods."""
        result_init = self.utils.initialize(self.mock_system)
        assert result_init == self.utils  # Should return self
        
        result_reset = self.utils.reset(self.mock_system)
        assert result_reset == self.utils  # Should return self
    
    # ==================== Data Type Conversion Tests ====================
    
    def test_get_bool_various_inputs(self):
        """Test boolean conversion with various inputs."""
        # String comparisons
        assert self.utils.get_bool("1", "1") is True
        assert self.utils.get_bool("0", "1") is False
        assert self.utils.get_bool("yes", "yes") is True
        
        # Integer comparisons
        assert self.utils.get_bool(1, 1) is True
        assert self.utils.get_bool(0, 1) is False
        assert self.utils.get_bool(5, 5) is True
        
        # Default boolean conversion
        assert self.utils.get_bool(True) is True
        assert self.utils.get_bool(False) is False
        assert self.utils.get_bool(1) is True
        assert self.utils.get_bool(0) is False
    
    def test_integer_conversions(self):
        """Test integer conversion functions."""
        # Valid conversions
        assert self.utils.get_integer16("123") == 123
        assert self.utils.get_integer32("456") == 456
        assert self.utils.get_integer64("789") == 789
        assert self.utils.get_integer("999") == 999
        
        # Invalid conversions should return 0
        assert self.utils.get_integer16("invalid") == 0
        assert self.utils.get_integer32(None) == 0
        
        # 16-bit range clamping
        assert self.utils.get_integer16("40000") == 32767  # Clamped to max
        assert self.utils.get_integer16("-40000") == -32768  # Clamped to min
    
    def test_float_conversions(self):
        """Test float conversion functions."""
        # Valid conversions
        assert self.utils.get_float("123.45") == 123.45
        assert self.utils.get_double("678.90") == 678.90
        
        # Invalid conversions should return 0.0
        assert self.utils.get_float("invalid") == 0.0
        assert self.utils.get_double(None) == 0.0
    
    # ==================== Mathematical Utilities Tests ====================
    
    def test_get_max_min(self):
        """Test max and min functions."""
        assert self.utils.get_max(5, 10) == 10
        assert self.utils.get_max(10, 5) == 10
        assert self.utils.get_max(-5, -10) == -5
        
        assert self.utils.get_min(5, 10) == 5
        assert self.utils.get_min(10, 5) == 5
        assert self.utils.get_min(-5, -10) == -10
    
    # ==================== Crossover Detection Tests ====================
    
    def test_yukari_kesti_series_crossover(self):
        """Test upward crossover between two series."""
        # Test data: series crossing upward at index 3
        series1 = [1.0, 2.0, 3.0, 5.0, 6.0]  # Crosses above series2
        series2 = [4.0, 4.0, 4.0, 4.0, 4.0]  # Constant line at 4.0
        
        # Should detect crossover at index 3
        assert self.utils.yukari_kesti(self.mock_system, 3, series1, series2) is True
        
        # No crossover at other indices
        assert self.utils.yukari_kesti(self.mock_system, 1, series1, series2) is False
        assert self.utils.yukari_kesti(self.mock_system, 2, series1, series2) is False
        assert self.utils.yukari_kesti(self.mock_system, 4, series1, series2) is False
    
    def test_yukari_kesti_level_crossover(self):
        """Test upward crossover with static level."""
        series = [1.0, 2.0, 3.0, 5.0, 6.0]
        level = 4.0
        
        # Should detect crossover at index 3
        assert self.utils.yukari_kesti(self.mock_system, 3, series, level=level) is True
        
        # No crossover at other indices
        assert self.utils.yukari_kesti(self.mock_system, 1, series, level=level) is False
        assert self.utils.yukari_kesti(self.mock_system, 2, series, level=level) is False
    
    def test_asagi_kesti_series_crossover(self):
        """Test downward crossover between two series."""
        # Test data: series crossing downward at index 3
        series1 = [6.0, 5.0, 4.5, 3.0, 2.0]  # Crosses below series2
        series2 = [4.0, 4.0, 4.0, 4.0, 4.0]  # Constant line at 4.0
        
        # Should detect crossover at index 3
        assert self.utils.asagi_kesti(self.mock_system, 3, series1, series2) is True
        
        # No crossover at other indices
        assert self.utils.asagi_kesti(self.mock_system, 1, series1, series2) is False
        assert self.utils.asagi_kesti(self.mock_system, 2, series1, series2) is False
    
    def test_asagi_kesti_level_crossover(self):
        """Test downward crossover with static level."""
        series = [6.0, 5.0, 4.5, 3.0, 2.0]
        level = 4.0
        
        # Should detect crossover at index 3
        assert self.utils.asagi_kesti(self.mock_system, 3, series, level=level) is True
        
        # No crossover at other indices
        assert self.utils.asagi_kesti(self.mock_system, 1, series, level=level) is False
    
    def test_crossover_equality_handling(self):
        """Test crossover detection with equality conditions."""
        # Test case: series1 crosses series2 exactly at equality
        series1 = [3.0, 3.0, 4.0, 4.0, 5.0]
        series2 = [4.0, 4.0, 4.0, 4.0, 4.0]
        
        # With equality included (default) - crosses at 4.0 = 4.0
        assert self.utils.yukari_kesti(self.mock_system, 2, series1, series2, esitlik_dahil=True) is True
        
        # Without equality - exact equality doesn't count as crossover
        assert self.utils.yukari_kesti(self.mock_system, 2, series1, series2, esitlik_dahil=False) is False
        
        # Test case with clear crossover: 3.5 -> 4.5 crosses 4.0
        series3 = [2.0, 3.0, 3.5, 4.5, 5.0]
        series4 = [4.0, 4.0, 4.0, 4.0, 4.0]
        
        # This should work with both settings since 3.5 < 4.0 and 4.5 > 4.0
        assert self.utils.yukari_kesti(self.mock_system, 3, series3, series4, esitlik_dahil=False) is True
        assert self.utils.yukari_kesti(self.mock_system, 3, series3, series4, esitlik_dahil=True) is True
    
    def test_crossover_edge_cases(self):
        """Test crossover detection edge cases."""
        series = [1.0, 2.0, 3.0]
        
        # Index 0 should return False (need previous bar)
        assert self.utils.yukari_kesti(self.mock_system, 0, series, level=1.5) is False
        
        # Missing parameters should raise ValueError
        with pytest.raises(ValueError):
            self.utils.yukari_kesti(self.mock_system, 1, series)  # No list_y or level
        
        # Out of bounds should return False
        assert self.utils.yukari_kesti(self.mock_system, 10, series, level=1.5) is False
    
    def test_crossover_with_numpy_arrays(self):
        """Test crossover detection with numpy arrays."""
        series1 = np.array([1.0, 2.0, 3.0, 5.0])
        series2 = np.array([4.0, 4.0, 4.0, 4.0])
        
        assert self.utils.yukari_kesti(self.mock_system, 3, series1, series2) is True
    
    # ==================== Level Generation Tests ====================
    
    def test_create_levels_basic(self):
        """Test basic level creation."""
        levels = self.utils.create_levels(self.mock_system, 10, 0, 2)
        
        # Should have levels 0, 2, 4, 6, 8, 10
        expected_keys = [0, 2, 4, 6, 8, 10]
        assert list(levels.keys()) == expected_keys
        
        # Each level should be an array of 100 values (default bar_count)
        for key in expected_keys:
            assert len(levels[key]) == 100
            assert all(val == key for val in levels[key])
    
    def test_create_levels_custom_bar_count(self):
        """Test level creation with custom bar count."""
        levels = self.utils.create_levels(self.mock_system, 5, 0, 1, bar_count=50)
        
        # Should have levels 0, 1, 2, 3, 4, 5
        assert len(levels) == 6
        
        # Each level should have 50 bars
        for level_array in levels.values():
            assert len(level_array) == 50
    
    def test_create_levels_from_list(self):
        """Test level creation from specific values."""
        level_values = [10, 20, 30, 50, 70]
        levels = self.utils.create_levels_from_list(self.mock_system, level_values, bar_count=25)
        
        assert len(levels) == 5
        assert set(levels.keys()) == {10.0, 20.0, 30.0, 50.0, 70.0}
        
        # Each level should have 25 bars
        for level_array in levels.values():
            assert len(level_array) == 25
    
    # ==================== Additional Utility Tests ====================
    
    def test_is_rising_falling(self):
        """Test trend detection functions."""
        rising_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        falling_data = [5.0, 4.0, 3.0, 2.0, 1.0]
        flat_data = [3.0, 3.0, 3.0, 3.0, 3.0]
        
        assert self.utils.is_rising(rising_data) is True
        assert self.utils.is_rising(falling_data) is False
        assert self.utils.is_rising(flat_data) is False
        
        assert self.utils.is_falling(falling_data) is True
        assert self.utils.is_falling(rising_data) is False
        assert self.utils.is_falling(flat_data) is False
        
        # Test with different periods
        assert self.utils.is_rising(rising_data, period=2) is True
        assert self.utils.is_rising([1.0, 5.0, 2.0], period=2) is True  # 2.0 > 1.0
    
    def test_percentage_change(self):
        """Test percentage change calculation."""
        assert self.utils.percentage_change(110, 100) == 10.0
        assert self.utils.percentage_change(90, 100) == -10.0
        assert self.utils.percentage_change(100, 100) == 0.0
        
        # Division by zero should return 0
        assert self.utils.percentage_change(50, 0) == 0.0
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert self.utils.safe_divide(10, 2) == 5.0
        assert self.utils.safe_divide(7, 3) == pytest.approx(2.333, rel=1e-3)
        
        # Division by zero should return 0
        assert self.utils.safe_divide(10, 0) == 0.0
    
    def test_clamp(self):
        """Test value clamping function."""
        assert self.utils.clamp(5, 0, 10) == 5  # Within bounds
        assert self.utils.clamp(-5, 0, 10) == 0  # Below min
        assert self.utils.clamp(15, 0, 10) == 10  # Above max
        assert self.utils.clamp(7.5, 2.5, 12.5) == 7.5  # Float values
    
    def test_repr(self):
        """Test string representation."""
        assert repr(self.utils) == "CUtils()"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])