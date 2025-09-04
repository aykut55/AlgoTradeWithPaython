"""
Statistics module for algorithmic trading system.

This module provides comprehensive trading performance analysis, including
profit/loss metrics, drawdown analysis, win/loss ratios, and risk-adjusted returns.
"""

from .statistics_manager import (
    TradingStatistics,
    PerformanceMetrics,
    DrawdownAnalysis,
    CStatistics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_profit_factor
)

__all__ = [
    "TradingStatistics",
    "PerformanceMetrics", 
    "DrawdownAnalysis",
    "CStatistics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio", 
    "calculate_profit_factor"
]