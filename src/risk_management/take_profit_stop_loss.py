"""
Advanced take profit and stop loss management system.

This module implements sophisticated risk management strategies including:
- Percentage-based take profit/stop loss
- Trailing stops with adaptive algorithms
- Level-based risk management
- Multi-level exit strategies
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum, IntEnum
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol
from src.trading.signals import Direction
from src.utils.utils import CUtils


class TPSLType(Enum):
    """Take Profit/Stop Loss calculation types."""
    PERCENTAGE_BASIC = "percentage_basic"
    PERCENTAGE_TRAILING = "percentage_trailing"  
    PERCENTAGE_LAST_PRICE = "percentage_last_price"
    PERCENTAGE_MULTI_LEVEL = "percentage_multi_level"
    ABSOLUTE_LEVEL = "absolute_level"
    ABSOLUTE_MULTI_LEVEL = "absolute_multi_level"


class TPSLResult(IntEnum):
    """Take Profit/Stop Loss trigger results."""
    NO_ACTION = 0
    LONG_TRIGGERED = 1
    SHORT_TRIGGERED = -1


@dataclass
class TPSLConfiguration:
    """Configuration for take profit and stop loss parameters."""
    
    # Basic settings
    take_profit_enabled: bool = True
    stop_loss_enabled: bool = True
    trailing_stop_enabled: bool = False
    
    # Percentage-based settings
    take_profit_percentage: float = 2.0
    stop_loss_percentage: float = 1.0
    trailing_stop_percentage: float = 1.0
    
    # Level-based settings  
    take_profit_level: float = 100.0
    stop_loss_level: float = -50.0
    
    # Multi-level settings
    level_start: int = 2
    level_end: int = 10
    level_multiplier: float = 0.01
    
    # Advanced settings
    use_crossover_detection: bool = True
    recalculate_on_new_position: bool = True


class CKarAlZararKes(CBase):
    """
    Comprehensive Take Profit and Stop Loss management system.
    
    Provides multiple strategies for risk management:
    - Basic percentage-based TP/SL
    - Trailing stops that follow profitable positions
    - Multi-level exit strategies
    - Absolute price level management
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Core components
        self.trader = None
        self.utils = CUtils()
        self.config = TPSLConfiguration()
        
        # State tracking
        self.take_profit_levels: List[float] = []
        self.stop_loss_levels: List[float] = []
        self.trailing_stop_levels: List[float] = []
        
        # Calculation flags
        self._take_profit_enabled: bool = True
        self._stop_loss_enabled: bool = True
        self._trailing_stop_enabled: bool = False
        
        # Last calculated values
        self._last_take_profit_price: float = 0.0
        self._last_stop_loss_price: float = 0.0
        self._last_entry_price: float = 0.0
        self._last_direction: Optional[Direction] = None
        
        # Performance tracking
        self.triggers_count: Dict[str, int] = {
            "take_profit": 0,
            "stop_loss": 0,
            "trailing_stop": 0
        }
    
    def initialize(self, system: SystemProtocol, trader) -> 'CKarAlZararKes':
        """Initialize the TP/SL manager with trader reference."""
        self.trader = trader
        return self
    
    def reset(self, system: SystemProtocol) -> 'CKarAlZararKes':
        """Reset all TP/SL calculations and state."""
        self.take_profit_levels.clear()
        self.stop_loss_levels.clear() 
        self.trailing_stop_levels.clear()
        
        self._last_take_profit_price = 0.0
        self._last_stop_loss_price = 0.0
        self._last_entry_price = 0.0
        self._last_direction = None
        
        self.triggers_count = {"take_profit": 0, "stop_loss": 0, "trailing_stop": 0}
        return self
    
    def configure(self, **kwargs) -> 'CKarAlZararKes':
        """Configure TP/SL parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    
    # Type 1: Basic Percentage-based TP/SL
    def calculate_take_profit_percentage(
        self, 
        system: SystemProtocol, 
        bar_index: int,
        take_profit_percentage: float = 2.0,
        reference_prices: Optional[List[float]] = None
    ) -> TPSLResult:
        """
        Calculate take profit based on percentage from entry price.
        
        Args:
            system: Trading system reference
            bar_index: Current bar index
            take_profit_percentage: TP percentage (e.g., 2.0 for 2%)
            reference_prices: Price array to check against (defaults to close prices)
            
        Returns:
            TPSLResult indicating if TP was triggered
        """
        if not self.config.take_profit_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        current_price = reference_prices[bar_index] if reference_prices else system.market_data.close[bar_index]
        
        # Get trader's last position info
        if not hasattr(self.trader, 'get_last_position_info'):
            return TPSLResult.NO_ACTION
            
        position_info = self.trader.get_last_position_info()
        if not position_info:
            return TPSLResult.NO_ACTION
        
        entry_price = position_info.get('entry_price', 0.0)
        direction = position_info.get('direction')
        
        if entry_price == 0.0 or not direction:
            return TPSLResult.NO_ACTION
        
        # Calculate TP level
        if direction == Direction.LONG:
            tp_level = entry_price * (1.0 + take_profit_percentage * 0.01)
            if current_price >= tp_level:
                self.triggers_count["take_profit"] += 1
                self._last_take_profit_price = tp_level
                return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            tp_level = entry_price * (1.0 - take_profit_percentage * 0.01)
            if current_price <= tp_level:
                self.triggers_count["take_profit"] += 1
                self._last_take_profit_price = tp_level
                return TPSLResult.SHORT_TRIGGERED
        
        # Store current TP level
        if len(self.take_profit_levels) <= bar_index:
            self.take_profit_levels.extend([0.0] * (bar_index - len(self.take_profit_levels) + 1))
        self.take_profit_levels[bar_index] = tp_level if 'tp_level' in locals() else current_price
        
        return TPSLResult.NO_ACTION
    
    def calculate_stop_loss_percentage(
        self,
        system: SystemProtocol,
        bar_index: int,
        stop_loss_percentage: float = 1.0,
        reference_prices: Optional[List[float]] = None
    ) -> TPSLResult:
        """Calculate stop loss based on percentage from entry price."""
        if not self.config.stop_loss_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        current_price = reference_prices[bar_index] if reference_prices else system.market_data.close[bar_index]
        
        position_info = self.trader.get_last_position_info() if hasattr(self.trader, 'get_last_position_info') else {}
        entry_price = position_info.get('entry_price', 0.0)
        direction = position_info.get('direction')
        
        if entry_price == 0.0 or not direction:
            return TPSLResult.NO_ACTION
        
        # Calculate SL level
        if direction == Direction.LONG:
            sl_level = entry_price * (1.0 - abs(stop_loss_percentage) * 0.01)
            if current_price <= sl_level:
                self.triggers_count["stop_loss"] += 1
                self._last_stop_loss_price = sl_level
                return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            sl_level = entry_price * (1.0 + abs(stop_loss_percentage) * 0.01)
            if current_price >= sl_level:
                self.triggers_count["stop_loss"] += 1
                self._last_stop_loss_price = sl_level
                return TPSLResult.SHORT_TRIGGERED
        
        # Store current SL level
        if len(self.stop_loss_levels) <= bar_index:
            self.stop_loss_levels.extend([0.0] * (bar_index - len(self.stop_loss_levels) + 1))
        self.stop_loss_levels[bar_index] = sl_level if 'sl_level' in locals() else current_price
        
        return TPSLResult.NO_ACTION
    
    # Type 2: Trailing Stop Implementation
    def calculate_trailing_stop_percentage(
        self,
        system: SystemProtocol,
        bar_index: int,
        trailing_percentage: float = 1.0,
        reference_prices: Optional[List[float]] = None
    ) -> TPSLResult:
        """
        Calculate trailing stop that follows favorable price movement.
        """
        if not self.config.trailing_stop_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        current_price = reference_prices[bar_index] if reference_prices else system.market_data.close[bar_index]
        
        position_info = self.trader.get_last_position_info() if hasattr(self.trader, 'get_last_position_info') else {}
        direction = position_info.get('direction')
        
        if not direction:
            return TPSLResult.NO_ACTION
        
        # Initialize or extend trailing stop levels
        if len(self.trailing_stop_levels) <= bar_index:
            self.trailing_stop_levels.extend([0.0] * (bar_index - len(self.trailing_stop_levels) + 1))
        
        # Calculate new trailing stop level
        if direction == Direction.LONG:
            # For long positions, trailing stop moves up with price
            new_trailing_level = current_price * (1.0 - abs(trailing_percentage) * 0.01)
            
            # Only update if new level is higher (more favorable)
            if self.trailing_stop_levels[bar_index] == 0.0 or new_trailing_level > self.trailing_stop_levels[bar_index]:
                self.trailing_stop_levels[bar_index] = new_trailing_level
            
            # Check for trigger
            if current_price <= self.trailing_stop_levels[bar_index]:
                self.triggers_count["trailing_stop"] += 1
                return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            # For short positions, trailing stop moves down with price
            new_trailing_level = current_price * (1.0 + abs(trailing_percentage) * 0.01)
            
            # Only update if new level is lower (more favorable)
            if self.trailing_stop_levels[bar_index] == 0.0 or new_trailing_level < self.trailing_stop_levels[bar_index]:
                self.trailing_stop_levels[bar_index] = new_trailing_level
            
            # Check for trigger
            if current_price >= self.trailing_stop_levels[bar_index]:
                self.triggers_count["trailing_stop"] += 1
                return TPSLResult.SHORT_TRIGGERED
        
        return TPSLResult.NO_ACTION
    
    # Type 3: Multi-level TP/SL using crossover detection
    def calculate_take_profit_multi_level(
        self,
        system: SystemProtocol,
        bar_index: int,
        level_start: int = 2,
        level_end: int = 10,
        multiplier: float = 0.01
    ) -> TPSLResult:
        """
        Calculate take profit using multiple levels with crossover detection.
        """
        if not self.config.take_profit_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        position_info = self.trader.get_last_position_info() if hasattr(self.trader, 'get_last_position_info') else {}
        entry_price = position_info.get('entry_price', 0.0)
        direction = position_info.get('direction')
        
        if entry_price == 0.0 or not direction:
            return TPSLResult.NO_ACTION
        
        current_price = system.market_data.close[bar_index]
        
        if direction == Direction.LONG:
            # Check multiple TP levels for long position
            for level in range(level_start, level_end + 1):
                tp_level = entry_price * (1.0 + level * multiplier)
                
                if self.config.use_crossover_detection:
                    # Use crossover detection
                    if self.utils.asagi_kesti(system, bar_index, [current_price], tp_level):
                        self.triggers_count["take_profit"] += 1
                        self._last_take_profit_price = tp_level
                        return TPSLResult.LONG_TRIGGERED
                else:
                    # Simple level check
                    if current_price >= tp_level:
                        self.triggers_count["take_profit"] += 1
                        self._last_take_profit_price = tp_level
                        return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            # Check multiple TP levels for short position
            for level in range(level_start, level_end + 1):
                tp_level = entry_price * (1.0 - level * multiplier)
                
                if self.config.use_crossover_detection:
                    if self.utils.yukari_kesti(system, bar_index, [current_price], tp_level):
                        self.triggers_count["take_profit"] += 1
                        self._last_take_profit_price = tp_level
                        return TPSLResult.SHORT_TRIGGERED
                else:
                    if current_price <= tp_level:
                        self.triggers_count["take_profit"] += 1
                        self._last_take_profit_price = tp_level
                        return TPSLResult.SHORT_TRIGGERED
        
        return TPSLResult.NO_ACTION
    
    def calculate_stop_loss_multi_level(
        self,
        system: SystemProtocol,
        bar_index: int,
        level_start: int = -2,
        level_end: int = -10,
        multiplier: float = 0.01
    ) -> TPSLResult:
        """Calculate stop loss using multiple levels with crossover detection."""
        if not self.config.stop_loss_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        position_info = self.trader.get_last_position_info() if hasattr(self.trader, 'get_last_position_info') else {}
        entry_price = position_info.get('entry_price', 0.0)
        direction = position_info.get('direction')
        
        if entry_price == 0.0 or not direction:
            return TPSLResult.NO_ACTION
        
        current_price = system.market_data.close[bar_index]
        
        # Convert negative levels to positive for calculation
        pos_level_start = abs(level_start)
        pos_level_end = abs(level_end)
        
        if direction == Direction.LONG:
            # Check multiple SL levels for long position
            for level in range(pos_level_start, pos_level_end + 1):
                sl_level = entry_price * (1.0 - level * multiplier)
                
                if self.config.use_crossover_detection:
                    if self.utils.asagi_kesti(system, bar_index, [current_price], sl_level):
                        self.triggers_count["stop_loss"] += 1
                        self._last_stop_loss_price = sl_level
                        return TPSLResult.LONG_TRIGGERED
                else:
                    if current_price <= sl_level:
                        self.triggers_count["stop_loss"] += 1
                        self._last_stop_loss_price = sl_level
                        return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            # Check multiple SL levels for short position
            for level in range(pos_level_start, pos_level_end + 1):
                sl_level = entry_price * (1.0 + level * multiplier)
                
                if self.config.use_crossover_detection:
                    if self.utils.yukari_kesti(system, bar_index, [current_price], sl_level):
                        self.triggers_count["stop_loss"] += 1
                        self._last_stop_loss_price = sl_level
                        return TPSLResult.SHORT_TRIGGERED
                else:
                    if current_price >= sl_level:
                        self.triggers_count["stop_loss"] += 1
                        self._last_stop_loss_price = sl_level
                        return TPSLResult.SHORT_TRIGGERED
        
        return TPSLResult.NO_ACTION
    
    # Type 4: Absolute Level-based TP/SL
    def calculate_take_profit_absolute(
        self,
        system: SystemProtocol,
        bar_index: int,
        take_profit_level: float = 100.0
    ) -> TPSLResult:
        """Calculate take profit based on absolute price level."""
        if not self.config.take_profit_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        position_info = self.trader.get_last_position_info() if hasattr(self.trader, 'get_last_position_info') else {}
        entry_price = position_info.get('entry_price', 0.0)
        direction = position_info.get('direction')
        
        if entry_price == 0.0 or not direction:
            return TPSLResult.NO_ACTION
        
        current_price = system.market_data.close[bar_index]
        
        if direction == Direction.LONG:
            tp_level = entry_price + abs(take_profit_level)
            if current_price >= tp_level:
                self.triggers_count["take_profit"] += 1
                self._last_take_profit_price = tp_level
                return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            tp_level = entry_price - abs(take_profit_level)
            if current_price <= tp_level:
                self.triggers_count["take_profit"] += 1
                self._last_take_profit_price = tp_level
                return TPSLResult.SHORT_TRIGGERED
        
        return TPSLResult.NO_ACTION
    
    def calculate_stop_loss_absolute(
        self,
        system: SystemProtocol,
        bar_index: int,
        stop_loss_level: float = -50.0
    ) -> TPSLResult:
        """Calculate stop loss based on absolute price level."""
        if not self.config.stop_loss_enabled or not self.trader:
            return TPSLResult.NO_ACTION
        
        position_info = self.trader.get_last_position_info() if hasattr(self.trader, 'get_last_position_info') else {}
        entry_price = position_info.get('entry_price', 0.0)
        direction = position_info.get('direction')
        
        if entry_price == 0.0 or not direction:
            return TPSLResult.NO_ACTION
        
        current_price = system.market_data.close[bar_index]
        sl_distance = abs(stop_loss_level)
        
        if direction == Direction.LONG:
            sl_level = entry_price - sl_distance
            if current_price <= sl_level:
                self.triggers_count["stop_loss"] += 1
                self._last_stop_loss_price = sl_level
                return TPSLResult.LONG_TRIGGERED
        
        elif direction == Direction.SHORT:
            sl_level = entry_price + sl_distance
            if current_price >= sl_level:
                self.triggers_count["stop_loss"] += 1
                self._last_stop_loss_price = sl_level
                return TPSLResult.SHORT_TRIGGERED
        
        return TPSLResult.NO_ACTION
    
    # Convenience methods with default parameters
    def check_take_profit(self, system: SystemProtocol, bar_index: int) -> TPSLResult:
        """Check take profit using default configuration."""
        return self.calculate_take_profit_percentage(
            system, bar_index, self.config.take_profit_percentage
        )
    
    def check_stop_loss(self, system: SystemProtocol, bar_index: int) -> TPSLResult:
        """Check stop loss using default configuration."""
        return self.calculate_stop_loss_percentage(
            system, bar_index, self.config.stop_loss_percentage
        )
    
    def check_trailing_stop(self, system: SystemProtocol, bar_index: int) -> TPSLResult:
        """Check trailing stop using default configuration."""
        return self.calculate_trailing_stop_percentage(
            system, bar_index, self.config.trailing_stop_percentage
        )
    
    # Status and information methods
    def get_current_levels(self, bar_index: int) -> Dict[str, float]:
        """Get current TP/SL levels for the given bar."""
        return {
            "take_profit": self.take_profit_levels[bar_index] if bar_index < len(self.take_profit_levels) else 0.0,
            "stop_loss": self.stop_loss_levels[bar_index] if bar_index < len(self.stop_loss_levels) else 0.0,
            "trailing_stop": self.trailing_stop_levels[bar_index] if bar_index < len(self.trailing_stop_levels) else 0.0
        }
    
    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get statistics about TP/SL triggers."""
        total_triggers = sum(self.triggers_count.values())
        return {
            "total_triggers": total_triggers,
            "take_profit_triggers": self.triggers_count["take_profit"],
            "stop_loss_triggers": self.triggers_count["stop_loss"],
            "trailing_stop_triggers": self.triggers_count["trailing_stop"],
            "take_profit_percentage": (self.triggers_count["take_profit"] / max(1, total_triggers)) * 100,
            "stop_loss_percentage": (self.triggers_count["stop_loss"] / max(1, total_triggers)) * 100,
            "trailing_stop_percentage": (self.triggers_count["trailing_stop"] / max(1, total_triggers)) * 100
        }
    
    def get_last_triggered_prices(self) -> Dict[str, float]:
        """Get the last triggered TP/SL prices."""
        return {
            "last_take_profit_price": self._last_take_profit_price,
            "last_stop_loss_price": self._last_stop_loss_price
        }
    
    def is_enabled(self, tp_sl_type: str) -> bool:
        """Check if a specific TP/SL type is enabled."""
        if tp_sl_type == "take_profit":
            return self.config.take_profit_enabled
        elif tp_sl_type == "stop_loss":
            return self.config.stop_loss_enabled
        elif tp_sl_type == "trailing_stop":
            return self.config.trailing_stop_enabled
        return False
    
    def enable_feature(self, feature: str, enabled: bool = True) -> 'CKarAlZararKes':
        """Enable or disable specific features."""
        if feature == "take_profit":
            self.config.take_profit_enabled = enabled
        elif feature == "stop_loss":
            self.config.stop_loss_enabled = enabled
        elif feature == "trailing_stop":
            self.config.trailing_stop_enabled = enabled
        return self
    
    def get_configuration(self) -> TPSLConfiguration:
        """Get current configuration."""
        return self.config
    
    def __str__(self) -> str:
        """String representation of TP/SL manager."""
        stats = self.get_trigger_statistics()
        return (
            f"CKarAlZararKes(tp_enabled={self.config.take_profit_enabled}, "
            f"sl_enabled={self.config.stop_loss_enabled}, "
            f"trailing_enabled={self.config.trailing_stop_enabled}, "
            f"total_triggers={stats['total_triggers']})"
        )