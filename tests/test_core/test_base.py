"""
Unit tests for the CBase class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.core.base import CBase, MarketData, SystemProtocol


class MockSystem:
    """Mock system class for testing."""
    
    def __init__(self):
        self.messages = []
    
    def mesaj(self, message: str) -> None:
        """Mock message handler."""
        self.messages.append(message)


class TestMarketData:
    """Test cases for MarketData class."""
    
    def test_market_data_creation(self):
        """Test MarketData creation and properties."""
        data = MarketData()
        assert data.bar_count == 0
        assert data.last_bar_index == 0
    
    def test_market_data_with_data(self):
        """Test MarketData with actual data."""
        data = MarketData(
            open=[1.0, 2.0, 3.0],
            high=[1.5, 2.5, 3.5],
            low=[0.5, 1.5, 2.5],
            close=[1.2, 2.2, 3.2],
            volume=[100, 200, 300],
            lot=[1, 1, 1]
        )
        assert data.bar_count == 3
        assert data.last_bar_index == 2
    
    def test_market_data_validation_success(self):
        """Test successful market data validation."""
        data = MarketData(
            open=[1.0, 2.0],
            high=[1.5, 2.5],
            low=[0.5, 1.5],
            close=[1.2, 2.2],
            volume=[100, 200],
            lot=[1, 1]
        )
        assert data.validate() is True
    
    def test_market_data_validation_failure(self):
        """Test failed market data validation."""
        data = MarketData(
            open=[1.0, 2.0, 3.0],  # 3 elements
            high=[1.5, 2.5],       # 2 elements - inconsistent
            low=[0.5, 1.5],
            close=[1.2, 2.2],
            volume=[100, 200],
            lot=[1, 1]
        )
        assert data.validate() is False
    
    def test_market_data_to_dataframe(self):
        """Test conversion to DataFrame."""
        data = MarketData(
            open=[1.0, 2.0],
            high=[1.5, 2.5],
            low=[0.5, 1.5],
            close=[1.2, 2.2],
            volume=[100, 200],
            lot=[1, 1]
        )
        df = data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume', 'lot']


class TestCBase:
    """Test cases for CBase class."""
    
    def test_cbase_creation(self):
        """Test CBase instantiation."""
        base = TestableBase(id_value=123)
        assert base.id == 123
        assert base.bar_count == 0
        assert base.last_bar_index == 0
    
    def test_show_message(self):
        """Test message sending functionality."""
        base = TestableBase()
        mock_system = MockSystem()
        
        base.show_message(mock_system, "Test message")
        assert len(mock_system.messages) == 1
        assert mock_system.messages[0] == "Test message"
    
    def test_set_data(self):
        """Test setting market data."""
        base = TestableBase()
        mock_system = MockSystem()
        
        open_data = [1.0, 2.0, 3.0]
        high_data = [1.5, 2.5, 3.5]
        low_data = [0.5, 1.5, 2.5]
        close_data = [1.2, 2.2, 3.2]
        volume_data = [100.0, 200.0, 300.0]
        lot_data = [1.0, 1.0, 1.0]
        
        base.set_data(
            mock_system, None, open_data, high_data, low_data,
            close_data, volume_data, lot_data
        )
        
        assert base.bar_count == 3
        assert base.last_bar_index == 2
        assert base.open == open_data
        assert base.high == high_data
        assert base.low == low_data
        assert base.close == close_data
        assert base.volume == volume_data
        assert base.lot == lot_data
    
    def test_set_data_validation_error(self):
        """Test validation error during data setting."""
        base = TestableBase()
        mock_system = MockSystem()
        
        with pytest.raises(ValueError, match="Market data validation failed"):
            base.set_data(
                mock_system, None,
                [1.0, 2.0, 3.0],  # 3 elements
                [1.5, 2.5],       # 2 elements - inconsistent
                [0.5, 1.5],
                [1.2, 2.2],
                [100.0, 200.0],
                [1.0, 1.0]
            )
    
    def test_set_data_from_dataframe(self):
        """Test setting data from DataFrame."""
        base = TestableBase()
        
        df = pd.DataFrame({
            'open': [1.0, 2.0, 3.0],
            'high': [1.5, 2.5, 3.5],
            'low': [0.5, 1.5, 2.5],
            'close': [1.2, 2.2, 3.2],
            'volume': [100, 200, 300],
            'lot': [1, 1, 1]
        })
        
        base.set_data_from_dataframe(df)
        
        assert base.bar_count == 3
        assert base.open == [1.0, 2.0, 3.0]
        assert base.close == [1.2, 2.2, 3.2]
    
    def test_set_data_from_dataframe_missing_columns(self):
        """Test error when DataFrame is missing required columns."""
        base = TestableBase()
        
        df = pd.DataFrame({
            'open': [1.0, 2.0],
            'high': [1.5, 2.5]
            # Missing required columns
        })
        
        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            base.set_data_from_dataframe(df)
    
    def test_get_ohlcv_dataframe(self):
        """Test getting DataFrame from market data."""
        base = TestableBase()
        base.set_data_from_dataframe(pd.DataFrame({
            'open': [1.0, 2.0],
            'high': [1.5, 2.5],
            'low': [0.5, 1.5],
            'close': [1.2, 2.2],
            'volume': [100, 200],
            'lot': [1, 1]
        }))
        
        df = base.get_ohlcv_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_get_price_series(self):
        """Test getting specific price series."""
        base = TestableBase()
        base.set_data_from_dataframe(pd.DataFrame({
            'open': [1.0, 2.0, 3.0],
            'high': [1.5, 2.5, 3.5],
            'low': [0.5, 1.5, 2.5],
            'close': [1.2, 2.2, 3.2],
            'volume': [100, 200, 300]
        }))
        
        close_series = base.get_price_series('close')
        assert isinstance(close_series, np.ndarray)
        np.testing.assert_array_equal(close_series, [1.2, 2.2, 3.2])
        
        open_series = base.get_price_series('open')
        np.testing.assert_array_equal(open_series, [1.0, 2.0, 3.0])
    
    def test_get_price_series_invalid_type(self):
        """Test error with invalid price type."""
        base = TestableBase()
        
        with pytest.raises(ValueError, match="Invalid price_type"):
            base.get_price_series('invalid')
    
    def test_properties(self):
        """Test property access."""
        base = TestableBase()
        
        # Test setting via properties
        base.open = [1.0, 2.0]
        base.close = [1.1, 2.1]
        
        assert base.open == [1.0, 2.0]
        assert base.close == [1.1, 2.1]
        assert base.bar_count == 2
        assert base.last_bar_index == 1
    
    def test_repr_and_len(self):
        """Test string representation and length."""
        base = TestableBase(id_value=42)
        base.open = [1.0, 2.0, 3.0]
        base.close = [1.1, 2.1, 3.1]
        
        assert "CBase(id=42, bars=3)" in repr(base)
        assert len(base) == 3


class TestableBase(CBase):
    """Concrete implementation of CBase for testing."""
    pass


if __name__ == "__main__":
    pytest.main([__file__])