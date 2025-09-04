"""
Base class for the algorithmic trading system.

This module contains the CBase class which serves as the foundation
for all trading system components, providing common functionality
for market data handling and system communication.
"""

from typing import List, Optional, Any, Protocol
from abc import ABC
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


class SystemProtocol(Protocol):
    """Protocol defining the interface for the trading system."""
    
    def mesaj(self, message: str) -> None:
        """Send a message to the system."""
        ...
    
    def grafik_fiyat_oku(self, data: Any, field_name: str) -> List[float]:
        """Read price data from graphics data."""
        ...


@dataclass
class MarketData:
    """Market data container with OHLCV information."""
    
    open: List[float] = field(default_factory=list)
    high: List[float] = field(default_factory=list)
    low: List[float] = field(default_factory=list)
    close: List[float] = field(default_factory=list)
    volume: List[float] = field(default_factory=list)
    lot: List[float] = field(default_factory=list)
    
    @property
    def bar_count(self) -> int:
        """Get the number of bars in the dataset."""
        return len(self.close) if len(self.close) > 0 else 0
    
    @property
    def last_bar_index(self) -> int:
        """Get the index of the last bar."""
        return max(0, self.bar_count - 1)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        return pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'lot': self.lot
        })
    
    def validate(self) -> bool:
        """Validate that all data series have the same length."""
        lengths = [
            len(self.open), len(self.high), len(self.low),
            len(self.close), len(self.volume), len(self.lot)
        ]
        return len(set(lengths)) == 1 and all(length > 0 for length in lengths)


class CBase(ABC):
    """
    Base class for all trading system components.
    
    Provides common functionality for:
    - Market data storage and management
    - System communication
    - Data validation and parsing
    """
    
    def __init__(self, id_value: int = 0):
        """
        Initialize the base class.
        
        Args:
            id_value: Unique identifier for this instance
        """
        self.id: int = id_value
        self.market_data: MarketData = MarketData()
        self.v: Optional[Any] = None  # Original data reference
        
        # Legacy property access for backward compatibility
        self._open: List[float] = []
        self._high: List[float] = []
        self._low: List[float] = []
        self._close: List[float] = []
        self._volume: List[float] = []
        self._lot: List[float] = []
    
    @property
    def open(self) -> List[float]:
        """Get open prices."""
        return self.market_data.open
    
    @open.setter
    def open(self, value: List[float]) -> None:
        """Set open prices."""
        self.market_data.open = value
    
    @property
    def high(self) -> List[float]:
        """Get high prices."""
        return self.market_data.high
    
    @high.setter
    def high(self, value: List[float]) -> None:
        """Set high prices."""
        self.market_data.high = value
    
    @property
    def low(self) -> List[float]:
        """Get low prices."""
        return self.market_data.low
    
    @low.setter
    def low(self, value: List[float]) -> None:
        """Set low prices."""
        self.market_data.low = value
    
    @property
    def close(self) -> List[float]:
        """Get close prices."""
        return self.market_data.close
    
    @close.setter
    def close(self, value: List[float]) -> None:
        """Set close prices."""
        self.market_data.close = value
    
    @property
    def volume(self) -> List[float]:
        """Get volume data."""
        return self.market_data.volume
    
    @volume.setter
    def volume(self, value: List[float]) -> None:
        """Set volume data."""
        self.market_data.volume = value
    
    @property
    def lot(self) -> List[float]:
        """Get lot data."""
        return self.market_data.lot
    
    @lot.setter
    def lot(self, value: List[float]) -> None:
        """Set lot data."""
        self.market_data.lot = value
    
    @property
    def bar_count(self) -> int:
        """Get the number of bars in the dataset."""
        return self.market_data.bar_count
    
    @property
    def last_bar_index(self) -> int:
        """Get the index of the last bar."""
        return self.market_data.last_bar_index
    
    def show_message(self, sistem: SystemProtocol, message: str) -> None:
        """
        Send a message to the system.
        
        Args:
            sistem: System object that can receive messages
            message: Message to send
        """
        sistem.mesaj(message)
    
    def set_data(
        self,
        sistem: SystemProtocol,
        v: Any,
        open_data: List[float],
        high_data: List[float],
        low_data: List[float],
        close_data: List[float],
        volume_data: List[float],
        lot_data: List[float]
    ) -> None:
        """
        Set market data for this instance.
        
        Args:
            sistem: System object
            v: Original data reference
            open_data: Opening prices
            high_data: High prices
            low_data: Low prices
            close_data: Closing prices
            volume_data: Volume data
            lot_data: Lot data
        """
        self.v = v
        self.market_data.open = open_data.copy()
        self.market_data.high = high_data.copy()
        self.market_data.low = low_data.copy()
        self.market_data.close = close_data.copy()
        self.market_data.volume = volume_data.copy()
        self.market_data.lot = lot_data.copy()
        
        # Validate data consistency
        if not self.market_data.validate():
            raise ValueError("Market data validation failed: inconsistent data lengths")
    
    def set_data_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set market data from a pandas DataFrame.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, lot
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        self.market_data.open = df['open'].tolist()
        self.market_data.high = df['high'].tolist()
        self.market_data.low = df['low'].tolist()
        self.market_data.close = df['close'].tolist()
        self.market_data.volume = df['volume'].tolist()
        lot_data = df.get('lot', [1.0] * len(df))
        self.market_data.lot = lot_data.tolist() if hasattr(lot_data, 'tolist') else list(lot_data)
    
    def get_ohlcv_dataframe(self) -> pd.DataFrame:
        """
        Get market data as a pandas DataFrame.
        
        Returns:
            DataFrame with OHLCV data
        """
        return self.market_data.to_dataframe()
    
    def get_price_series(self, price_type: str = "close") -> np.ndarray:
        """
        Get a specific price series as numpy array.
        
        Args:
            price_type: Type of price ('open', 'high', 'low', 'close')
            
        Returns:
            Price series as numpy array
            
        Raises:
            ValueError: If price_type is invalid
        """
        price_map = {
            'open': self.market_data.open,
            'high': self.market_data.high,
            'low': self.market_data.low,
            'close': self.market_data.close,
            'volume': self.market_data.volume,
            'lot': self.market_data.lot
        }
        
        if price_type.lower() not in price_map:
            raise ValueError(f"Invalid price_type: {price_type}")
        
        return np.array(price_map[price_type.lower()])
    
    def __repr__(self) -> str:
        """String representation of the base class."""
        return f"CBase(id={self.id}, bars={self.bar_count})"
    
    def __len__(self) -> int:
        """Return the number of bars."""
        return self.bar_count