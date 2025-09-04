# CBase Class Documentation

## Overview

`CBase` is the fundamental base class for all trading system components in the algorithmic trading system. It provides essential functionality for market data management, system communication, and serves as the foundation for all other trading classes.

## Class Hierarchy

```
CBase (Abstract Base Class)
├── MarketData (Data Container)
├── SystemProtocol (Interface)
└── [All other trading classes inherit from CBase]
```

## Key Features

### 1. Market Data Management
- **OHLCV Data Storage**: Handles Open, High, Low, Close, Volume, and Lot data
- **Data Validation**: Ensures consistency across all data series
- **Flexible Data Input**: Accepts both list and pandas DataFrame formats
- **Data Conversion**: Provides seamless conversion between formats

### 2. System Integration
- **Protocol-Based Design**: Uses `SystemProtocol` for loose coupling
- **Message Communication**: Standardized messaging with trading system
- **Backward Compatibility**: Maintains C# interface compatibility

### 3. Type Safety
- **Type Hints**: Full type annotation support
- **Protocol Validation**: Runtime type checking where needed
- **Data Structure Validation**: Automatic validation of market data integrity

## API Reference

### Core Properties

```python
@property
def bar_count(self) -> int:
    """Get the number of bars in the dataset."""

@property  
def last_bar_index(self) -> int:
    """Get the index of the last bar."""

# Price data properties
@property
def open(self) -> List[float]:
@property  
def high(self) -> List[float]:
@property
def low(self) -> List[float]:
@property
def close(self) -> List[float]:
@property
def volume(self) -> List[float]:
@property
def lot(self) -> List[float]:
```

### Core Methods

```python
def set_data(self, sistem, v, open_data, high_data, low_data, 
             close_data, volume_data, lot_data) -> None:
    """Set market data from individual lists."""

def set_data_from_dataframe(self, df: pd.DataFrame) -> None:
    """Set market data from pandas DataFrame."""

def get_ohlcv_dataframe(self) -> pd.DataFrame:
    """Get market data as pandas DataFrame."""

def get_price_series(self, price_type: str = "close") -> np.ndarray:
    """Get specific price series as numpy array."""

def show_message(self, sistem: SystemProtocol, message: str) -> None:
    """Send message to the trading system."""
```

## Usage Examples

### Basic Usage

```python
from src.core.base import CBase
import pandas as pd

class MyTradingClass(CBase):
    def __init__(self):
        super().__init__(id_value=1)

# Create instance
trader = MyTradingClass()

# Set data from lists
trader.set_data(
    sistem=mock_system,
    v=None,
    open_data=[1.0, 2.0, 3.0],
    high_data=[1.5, 2.5, 3.5],
    low_data=[0.5, 1.5, 2.5], 
    close_data=[1.2, 2.2, 3.2],
    volume_data=[100, 200, 300],
    lot_data=[1, 1, 1]
)

print(f"Bars: {trader.bar_count}")  # Output: Bars: 3
```

### DataFrame Integration

```python
# Create from DataFrame
df = pd.DataFrame({
    'open': [1.0, 2.0, 3.0],
    'high': [1.5, 2.5, 3.5],
    'low': [0.5, 1.5, 2.5],
    'close': [1.2, 2.2, 3.2],
    'volume': [100, 200, 300]
})

trader.set_data_from_dataframe(df)

# Get back as DataFrame
result_df = trader.get_ohlcv_dataframe()
```

### Price Series Access

```python
# Get specific price series
import numpy as np

close_prices = trader.get_price_series('close')
open_prices = trader.get_price_series('open')

# Use with numpy operations
sma_20 = np.convolve(close_prices, np.ones(20)/20, mode='valid')
```

## Integration with C# System

The Python `CBase` maintains compatibility with the original C# interface:

### C# Original:
```csharp
public void SetData(dynamic Sistem, dynamic V, dynamic Open, 
                   dynamic High, dynamic Low, dynamic Close, 
                   dynamic Volume, dynamic Lot)

public void ShowMessage(dynamic Sistem, string Message)
```

### Python Equivalent:
```python  
def set_data(self, sistem: SystemProtocol, v: Any, 
            open_data: List[float], high_data: List[float], ...):

def show_message(self, sistem: SystemProtocol, message: str):
```

## Testing

Comprehensive test coverage includes:
- Market data validation
- DataFrame integration  
- Property access
- Error handling
- Type checking

Run tests:
```bash
pytest tests/test_core/test_base.py -v
```

## Best Practices

1. **Always validate data** before processing
2. **Use type hints** for better IDE support
3. **Handle exceptions** appropriately in derived classes
4. **Prefer DataFrame** for complex data operations
5. **Use numpy arrays** for mathematical operations

## Migration from C#

Key differences when porting C# code:

| C# | Python | Notes |
|---|---|---|
| `dynamic` | `Any` | Use protocols for better typing |
| `List<float>` | `List[float]` | Same interface |
| Properties | `@property` | Python property decorators |
| Exception handling | Same pattern | Python exception handling |
| Memory management | Automatic | Python GC handles cleanup |

## Performance Considerations

- **Data Copying**: `set_data()` creates copies for data integrity
- **NumPy Integration**: Use `get_price_series()` for mathematical operations
- **DataFrame Operations**: Leverage pandas for complex data manipulation
- **Memory Usage**: Monitor memory usage with large datasets

## Future Enhancements

- [ ] Add support for tick-level data
- [ ] Implement data streaming capabilities  
- [ ] Add more data validation rules
- [ ] Support for multiple timeframes
- [ ] Integration with real-time data feeds