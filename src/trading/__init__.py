"""
Trading module for algorithmic trading system.

This module provides comprehensive trading functionality including:
- Signal generation and processing
- Position and order management
- Risk management
- Trade execution and tracking
- Performance monitoring
"""

from .signals import (
    Direction,
    SignalType, 
    SignalInfo,
    PositionInfo,
    CSignals
)

from .trader import (
    OrderType,
    OrderStatus,
    OrderInfo,
    TradeInfo,
    RiskSettings,
    CTrader
)

__all__ = [
    # Signals
    "Direction",
    "SignalType",
    "SignalInfo", 
    "PositionInfo",
    "CSignals",
    
    # Trading
    "OrderType",
    "OrderStatus", 
    "OrderInfo",
    "TradeInfo",
    "RiskSettings",
    "CTrader"
]