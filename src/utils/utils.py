"""
Utility functions for technical analysis and general trading operations.

This module contains the CUtils class which provides essential utility functions
including crossover detection, data type conversions, mathematical operations,
and level generation functions.
"""

from typing import List, Union, TypeVar, Generic, Any, Optional, Dict
import numpy as np
from src.core.base import SystemProtocol

T = TypeVar('T')
NumericType = Union[int, float, np.number]
NumericList = Union[List[NumericType], np.ndarray]


class CUtils:
    """
    Utility class providing technical analysis and general trading utilities.
    
    This class contains essential functions for:
    - Crossover detection (YukariKesti/AsagiKesti)
    - Data type conversions
    - Mathematical operations
    - Level generation
    """
    
    def __init__(self):
        """Initialize the utils class."""
        pass
    
    def initialize(self, sistem: SystemProtocol) -> 'CUtils':
        """
        Initialize the utils with system reference.
        
        Args:
            sistem: Trading system reference
            
        Returns:
            Self for method chaining
        """
        return self
    
    def reset(self, sistem: SystemProtocol) -> 'CUtils':
        """
        Reset the utils state.
        
        Args:
            sistem: Trading system reference
            
        Returns:
            Self for method chaining
        """
        return self
    
    # ==================== Data Type Conversions ====================
    
    def get_bool(self, value: Any, true_value: Union[str, int] = 1) -> bool:
        """
        Convert value to boolean.
        
        Args:
            value: Value to convert
            true_value: Value that represents True
            
        Returns:
            Boolean result
        """
        if isinstance(value, (str, int)) and isinstance(true_value, type(value)):
            return value == true_value
        
        try:
            return bool(value)
        except (ValueError, TypeError):
            return False
    
    def get_integer16(self, value: Any) -> int:
        """Convert value to 16-bit integer."""
        try:
            result = int(value)
            return max(-32768, min(32767, result))  # 16-bit range
        except (ValueError, TypeError):
            return 0
    
    def get_integer32(self, value: Any) -> int:
        """Convert value to 32-bit integer."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def get_integer64(self, value: Any) -> int:
        """Convert value to 64-bit integer."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def get_integer(self, value: Any) -> int:
        """Convert value to integer (alias for get_integer32)."""
        return self.get_integer32(value)
    
    def get_float(self, value: Any) -> float:
        """Convert value to float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_double(self, value: Any) -> float:
        """Convert value to double (float in Python)."""
        return self.get_float(value)
    
    # ==================== Mathematical Utilities ====================
    
    def get_max(self, lhs: T, rhs: T) -> T:
        """
        Get maximum of two values.
        
        Args:
            lhs: Left hand side value
            rhs: Right hand side value
            
        Returns:
            Maximum value
        """
        return max(lhs, rhs)
    
    def get_min(self, lhs: T, rhs: T) -> T:
        """
        Get minimum of two values.
        
        Args:
            lhs: Left hand side value
            rhs: Right hand side value
            
        Returns:
            Minimum value
        """
        return min(lhs, rhs)
    
    # ==================== Crossover Detection ====================
    
    def yukari_kesti(
        self, 
        sistem: SystemProtocol, 
        i: int, 
        list_x: NumericList, 
        list_y: Optional[NumericList] = None,
        level: Optional[NumericType] = None,
        esitlik_dahil: bool = True
    ) -> bool:
        """
        Detect upward crossover.
        
        Detects when list_x crosses above list_y (or level) at position i.
        
        Args:
            sistem: Trading system reference
            i: Current bar index
            list_x: Primary data series
            list_y: Secondary data series (optional if level is provided)
            level: Static level for crossover (optional if list_y is provided)
            esitlik_dahil: Include equality in comparison
            
        Returns:
            True if upward crossover detected
            
        Raises:
            ValueError: If neither list_y nor level is provided, or if i < 1
        """
        if i < 1:
            return False
        
        if list_y is None and level is None:
            raise ValueError("Either list_y or level must be provided")
        
        if len(list_x) <= i:
            return False
        
        # Convert to numpy arrays for consistent indexing
        x_array = np.array(list_x) if not isinstance(list_x, np.ndarray) else list_x
        
        # Case 1: Crossover with another series
        if list_y is not None:
            if len(list_y) <= i:
                return False
            
            y_array = np.array(list_y) if not isinstance(list_y, np.ndarray) else list_y
            
            prev_x, curr_x = float(x_array[i-1]), float(x_array[i])
            prev_y, curr_y = float(y_array[i-1]), float(y_array[i])
            
            if esitlik_dahil:
                return prev_x < prev_y and curr_x >= curr_y
            else:
                return prev_x < prev_y and curr_x > curr_y
        
        # Case 2: Crossover with static level  
        else:
            prev_x, curr_x = float(x_array[i-1]), float(x_array[i])
            level_val = float(level)
            
            if esitlik_dahil:
                return prev_x < level_val and curr_x >= level_val
            else:
                return prev_x < level_val and curr_x > level_val
    
    def asagi_kesti(
        self,
        sistem: SystemProtocol,
        i: int,
        list_x: NumericList,
        list_y: Optional[NumericList] = None,
        level: Optional[NumericType] = None,
        esitlik_dahil: bool = True
    ) -> bool:
        """
        Detect downward crossover.
        
        Detects when list_x crosses below list_y (or level) at position i.
        
        Args:
            sistem: Trading system reference
            i: Current bar index
            list_x: Primary data series
            list_y: Secondary data series (optional if level is provided)
            level: Static level for crossover (optional if list_y is provided)
            esitlik_dahil: Include equality in comparison
            
        Returns:
            True if downward crossover detected
            
        Raises:
            ValueError: If neither list_y nor level is provided, or if i < 1
        """
        if i < 1:
            return False
        
        if list_y is None and level is None:
            raise ValueError("Either list_y or level must be provided")
        
        if len(list_x) <= i:
            return False
        
        # Convert to numpy arrays for consistent indexing
        x_array = np.array(list_x) if not isinstance(list_x, np.ndarray) else list_x
        
        # Case 1: Crossover with another series
        if list_y is not None:
            if len(list_y) <= i:
                return False
            
            y_array = np.array(list_y) if not isinstance(list_y, np.ndarray) else list_y
            
            prev_x, curr_x = float(x_array[i-1]), float(x_array[i])
            prev_y, curr_y = float(y_array[i-1]), float(y_array[i])
            
            if esitlik_dahil:
                return prev_x > prev_y and curr_x <= curr_y
            else:
                return prev_x > prev_y and curr_x < curr_y
        
        # Case 2: Crossover with static level
        else:
            prev_x, curr_x = float(x_array[i-1]), float(x_array[i])
            level_val = float(level)
            
            if esitlik_dahil:
                return prev_x > level_val and curr_x <= level_val
            else:
                return prev_x > level_val and curr_x < level_val
    
    # ==================== Level Generation ====================
    
    def create_levels(
        self, 
        sistem: SystemProtocol, 
        max_value: NumericType, 
        min_value: NumericType = 0,
        step: NumericType = 1,
        bar_count: Optional[int] = None
    ) -> Dict[NumericType, List[float]]:
        """
        Create horizontal levels for chart analysis.
        
        Args:
            sistem: Trading system reference
            max_value: Maximum level value
            min_value: Minimum level value (default: 0)
            step: Step between levels (default: 1)
            bar_count: Number of bars to extend levels (if None, uses 100)
            
        Returns:
            Dictionary mapping level values to arrays of that level
        """
        if bar_count is None:
            bar_count = 100
        
        levels = {}
        current = float(min_value)
        max_val = float(max_value)
        step_val = float(step)
        
        while current <= max_val:
            levels[current] = [current] * bar_count
            current += step_val
        
        return levels
    
    def create_levels_from_list(
        self,
        sistem: SystemProtocol,
        level_values: List[NumericType],
        bar_count: Optional[int] = None
    ) -> Dict[NumericType, List[float]]:
        """
        Create levels from a list of specific values.
        
        Args:
            sistem: Trading system reference
            level_values: List of level values to create
            bar_count: Number of bars to extend levels
            
        Returns:
            Dictionary mapping level values to arrays
        """
        if bar_count is None:
            bar_count = 100
        
        levels = {}
        for level in level_values:
            level_val = float(level)
            levels[level_val] = [level_val] * bar_count
        
        return levels
    
    # ==================== Additional Utility Functions ====================
    
    def is_rising(self, data: NumericList, period: int = 1) -> bool:
        """
        Check if the data series is rising over the specified period.
        
        Args:
            data: Data series to check
            period: Number of periods to look back
            
        Returns:
            True if data is rising
        """
        if len(data) < period + 1:
            return False
        
        data_array = np.array(data) if not isinstance(data, np.ndarray) else data
        return float(data_array[-1]) > float(data_array[-1 - period])
    
    def is_falling(self, data: NumericList, period: int = 1) -> bool:
        """
        Check if the data series is falling over the specified period.
        
        Args:
            data: Data series to check
            period: Number of periods to look back
            
        Returns:
            True if data is falling
        """
        if len(data) < period + 1:
            return False
        
        data_array = np.array(data) if not isinstance(data, np.ndarray) else data
        return float(data_array[-1]) < float(data_array[-1 - period])
    
    def percentage_change(
        self, 
        current_value: NumericType, 
        previous_value: NumericType
    ) -> float:
        """
        Calculate percentage change between two values.
        
        Args:
            current_value: Current value
            previous_value: Previous value
            
        Returns:
            Percentage change
        """
        if previous_value == 0:
            return 0.0
        
        return ((float(current_value) - float(previous_value)) / float(previous_value)) * 100.0
    
    def safe_divide(self, numerator: NumericType, denominator: NumericType) -> float:
        """
        Perform safe division avoiding division by zero.
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            
        Returns:
            Division result or 0.0 if denominator is zero
        """
        if denominator == 0:
            return 0.0
        
        return float(numerator) / float(denominator)
    
    def clamp(self, value: NumericType, min_val: NumericType, max_val: NumericType) -> NumericType:
        """
        Clamp value between min and max bounds.
        
        Args:
            value: Value to clamp
            min_val: Minimum bound
            max_val: Maximum bound
            
        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, value))
    
    def __repr__(self) -> str:
        """String representation of CUtils."""
        return "CUtils()"