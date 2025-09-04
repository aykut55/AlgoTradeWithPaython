"""
Core module for the algorithmic trading system.

This module provides fundamental base classes and protocols
that form the foundation of the trading system architecture.
"""

from .base import CBase, MarketData, SystemProtocol

__all__ = [
    'CBase',
    'MarketData', 
    'SystemProtocol'
]