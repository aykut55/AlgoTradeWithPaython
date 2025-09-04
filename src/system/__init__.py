"""
System orchestration module for algorithmic trading system.

This module provides the main system wrapper and orchestration components
that coordinate all trading system elements including traders, indicators,
utilities, and strategy execution.
"""

from .system_wrapper import (
    ExecutionMode,
    ReportingLevel,
    SystemConfiguration,
    StrategySignals,
    ExecutionStatistics,
    SystemWrapper
)

from .backtest_engine import (
    BacktestMode,
    BacktestConfiguration,
    BacktestMetrics,
    BacktestResults,
    CBacktestEngine
)

__all__ = [
    # System Wrapper
    "ExecutionMode",
    "ReportingLevel", 
    "SystemConfiguration",
    "StrategySignals",
    "ExecutionStatistics",
    "SystemWrapper",
    
    # Backtest Engine
    "BacktestMode",
    "BacktestConfiguration", 
    "BacktestMetrics",
    "BacktestResults",
    "CBacktestEngine"
]