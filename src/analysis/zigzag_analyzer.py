"""
ZigZag pattern analysis system for algorithmic trading.

This module contains the CZigZagAnalyzer class which identifies
ZigZag patterns in price data, analyzes market structure,
and provides advanced pattern-based trading signals.
"""

import sys
import os

# Add the src directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque

# Import from core module
from core.base import SystemProtocol


class SignalType(Enum):
    """Simple signal type enumeration for ZigZag analysis."""
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"


@dataclass
class MarketDataPoint:
    """Single market data point for ZigZag analysis."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class ZigZagType(Enum):
    """ZigZag calculation types."""
    PERCENTAGE = "PERCENTAGE"       # Percentage-based ZigZag
    ABSOLUTE = "ABSOLUTE"           # Absolute value-based ZigZag
    ATR_BASED = "ATR_BASED"         # ATR (Average True Range) based


class TrendDirection(Enum):
    """Trend direction enumeration."""
    UP = "UP"                       # Uptrend
    DOWN = "DOWN"                   # Downtrend
    SIDEWAYS = "SIDEWAYS"           # Sideways/consolidation


class ZigZagLevel(Enum):
    """ZigZag significance levels."""
    MINOR = "MINOR"                 # Minor swings (short-term)
    INTERMEDIATE = "INTERMEDIATE"   # Intermediate swings (medium-term)
    MAJOR = "MAJOR"                 # Major swings (long-term)


@dataclass
class ZigZagPoint:
    """ZigZag turning point."""
    
    index: int                      # Index in the data series
    timestamp: datetime             # Timestamp of the point
    price: float                    # Price at turning point
    is_high: bool                   # True for peaks, False for troughs
    level: ZigZagLevel = ZigZagLevel.MINOR
    
    # Additional analysis
    volume: float = 0.0             # Volume at turning point
    strength: float = 0.0           # Strength of the turning point
    retracement_pct: float = 0.0    # Retracement percentage
    
    def __str__(self) -> str:
        direction = "HIGH" if self.is_high else "LOW"
        return f"ZigZag{direction}({self.price:.4f} at {self.timestamp})"


@dataclass
class ZigZagSwing:
    """ZigZag swing between two points."""
    
    start_point: ZigZagPoint        # Starting point
    end_point: ZigZagPoint          # Ending point
    
    # Swing characteristics
    swing_size: float = 0.0         # Price difference
    swing_percentage: float = 0.0   # Percentage move
    duration: timedelta = timedelta() # Time duration
    avg_volume: float = 0.0         # Average volume during swing
    
    def __post_init__(self):
        """Calculate swing characteristics."""
        self.swing_size = abs(self.end_point.price - self.start_point.price)
        if self.start_point.price > 0:
            self.swing_percentage = (self.swing_size / self.start_point.price) * 100
        self.duration = self.end_point.timestamp - self.start_point.timestamp
    
    @property
    def is_up_swing(self) -> bool:
        """Check if this is an up swing."""
        return self.end_point.price > self.start_point.price
    
    @property
    def is_down_swing(self) -> bool:
        """Check if this is a down swing."""
        return self.end_point.price < self.start_point.price


@dataclass
class ZigZagPattern:
    """Identified ZigZag pattern."""
    
    pattern_type: str               # Pattern type (e.g., "DOUBLE_TOP", "HEAD_SHOULDERS")
    points: List[ZigZagPoint]       # Pattern points
    confidence: float = 0.0         # Pattern confidence (0-1)
    target_price: float = 0.0       # Target price if pattern completes
    stop_loss: float = 0.0          # Stop loss level
    
    # Pattern characteristics
    symmetry_score: float = 0.0     # Symmetry score (0-1)
    volume_confirmation: bool = False # Volume confirms the pattern
    time_symmetry: float = 0.0      # Time symmetry score


class CZigZagAnalyzer:
    """
    Advanced ZigZag pattern analyzer.
    
    Features:
    - Multiple ZigZag calculation methods
    - Multi-timeframe analysis
    - Pattern recognition (double tops/bottoms, head & shoulders, etc.)
    - Trend analysis and market structure
    - Support/resistance level identification
    - Fibonacci retracement analysis
    - Volume confirmation
    - Real-time pattern alerts
    """
    
    def __init__(self):
        """Initialize ZigZag analyzer."""
        self.is_initialized = False
        
        # Configuration
        self.zigzag_type = ZigZagType.PERCENTAGE
        self.threshold_percentage = 5.0  # 5% threshold for percentage-based
        self.threshold_absolute = 10.0   # Absolute threshold
        self.atr_multiplier = 2.0        # ATR multiplier for ATR-based
        
        # Data storage
        self.price_data: List[MarketDataPoint] = []
        self.zigzag_points: List[ZigZagPoint] = []
        self.swings: List[ZigZagSwing] = []
        self.patterns: List[ZigZagPattern] = []
        
        # Analysis parameters
        self.max_data_points = 5000      # Maximum data points to keep
        self.min_swing_size = 2.0        # Minimum swing size for significance
        self.pattern_lookback = 100      # Lookback period for pattern recognition
        
        # Market structure analysis
        self.current_trend = TrendDirection.SIDEWAYS
        self.trend_strength = 0.0
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        
        # Fibonacci levels
        self.fibonacci_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        self.current_retracements: Dict[str, float] = {}
    
    def initialize(self, system: SystemProtocol) -> 'CZigZagAnalyzer':
        """
        Initialize ZigZag analyzer.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CZigZagAnalyzer':
        """
        Reset ZigZag analyzer.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.price_data.clear()
        self.zigzag_points.clear()
        self.swings.clear()
        self.patterns.clear()
        self.support_levels.clear()
        self.resistance_levels.clear()
        self.current_retracements.clear()
        
        self.current_trend = TrendDirection.SIDEWAYS
        self.trend_strength = 0.0
        
        return self
    
    # ========== Configuration Methods ==========
    
    def set_zigzag_type(self, zigzag_type: ZigZagType) -> 'CZigZagAnalyzer':
        """Set ZigZag calculation type."""
        self.zigzag_type = zigzag_type
        return self
    
    def set_percentage_threshold(self, percentage: float) -> 'CZigZagAnalyzer':
        """Set percentage threshold for ZigZag calculation."""
        self.threshold_percentage = percentage
        return self
    
    def set_absolute_threshold(self, threshold: float) -> 'CZigZagAnalyzer':
        """Set absolute threshold for ZigZag calculation."""
        self.threshold_absolute = threshold
        return self
    
    def set_atr_multiplier(self, multiplier: float) -> 'CZigZagAnalyzer':
        """Set ATR multiplier for ATR-based ZigZag."""
        self.atr_multiplier = multiplier
        return self
    
    # ========== Data Management ==========
    
    def add_market_data(self, market_data: MarketDataPoint) -> None:
        """
        Add new market data point.
        
        Args:
            market_data: Market data to add
        """
        self.price_data.append(market_data)
        
        # Maintain data size limit
        if len(self.price_data) > self.max_data_points:
            self.price_data.pop(0)
        
        # Update ZigZag analysis
        self._update_zigzag_analysis()
    
    def add_price_data_bulk(self, data: List[MarketDataPoint]) -> None:
        """
        Add bulk price data for historical analysis.
        
        Args:
            data: List of market data
        """
        self.price_data.extend(data)
        
        # Maintain data size limit
        if len(self.price_data) > self.max_data_points:
            excess = len(self.price_data) - self.max_data_points
            self.price_data = self.price_data[excess:]
        
        # Recalculate entire ZigZag
        self._recalculate_zigzag()
    
    # ========== ZigZag Calculation ==========
    
    def _update_zigzag_analysis(self) -> None:
        """Update ZigZag analysis with new data point."""
        if len(self.price_data) < 3:
            return
        
        # Check if new ZigZag point should be added
        self._check_for_new_zigzag_point()
        
        # Update current analysis
        self._update_market_structure()
        self._update_patterns()
    
    def _recalculate_zigzag(self) -> None:
        """Recalculate entire ZigZag from scratch."""
        self.zigzag_points.clear()
        self.swings.clear()
        
        if len(self.price_data) < 3:
            return
        
        # Calculate ZigZag points
        if self.zigzag_type == ZigZagType.PERCENTAGE:
            self._calculate_percentage_zigzag()
        elif self.zigzag_type == ZigZagType.ABSOLUTE:
            self._calculate_absolute_zigzag()
        elif self.zigzag_type == ZigZagType.ATR_BASED:
            self._calculate_atr_zigzag()
        
        # Create swings
        self._create_swings()
        
        # Analyze market structure
        self._analyze_market_structure()
        
        # Identify patterns
        self._identify_patterns()
    
    def _calculate_percentage_zigzag(self) -> None:
        """Calculate percentage-based ZigZag."""
        if not self.price_data:
            return
        
        # Start with first high/low
        current_extreme = self.price_data[0]
        current_extreme_idx = 0
        looking_for_high = True  # Start looking for high
        
        for i, data in enumerate(self.price_data[1:], 1):
            if looking_for_high:
                # Looking for a high
                if data.high > current_extreme.high:
                    current_extreme = data
                    current_extreme_idx = i
                else:
                    # Check if we've fallen enough to confirm the high
                    decline_pct = ((current_extreme.high - data.low) / current_extreme.high) * 100
                    if decline_pct >= self.threshold_percentage:
                        # Confirm the high
                        self._add_zigzag_point(
                            index=current_extreme_idx,
                            data=current_extreme,
                            is_high=True
                        )
                        
                        # Start looking for low
                        current_extreme = data
                        current_extreme_idx = i
                        looking_for_high = False
            else:
                # Looking for a low
                if data.low < current_extreme.low:
                    current_extreme = data
                    current_extreme_idx = i
                else:
                    # Check if we've risen enough to confirm the low
                    rise_pct = ((data.high - current_extreme.low) / current_extreme.low) * 100
                    if rise_pct >= self.threshold_percentage:
                        # Confirm the low
                        self._add_zigzag_point(
                            index=current_extreme_idx,
                            data=current_extreme,
                            is_high=False
                        )
                        
                        # Start looking for high
                        current_extreme = data
                        current_extreme_idx = i
                        looking_for_high = True
    
    def _calculate_absolute_zigzag(self) -> None:
        """Calculate absolute value-based ZigZag."""
        if not self.price_data:
            return
        
        current_extreme = self.price_data[0]
        current_extreme_idx = 0
        looking_for_high = True
        
        for i, data in enumerate(self.price_data[1:], 1):
            if looking_for_high:
                if data.high > current_extreme.high:
                    current_extreme = data
                    current_extreme_idx = i
                else:
                    decline = current_extreme.high - data.low
                    if decline >= self.threshold_absolute:
                        self._add_zigzag_point(
                            index=current_extreme_idx,
                            data=current_extreme,
                            is_high=True
                        )
                        current_extreme = data
                        current_extreme_idx = i
                        looking_for_high = False
            else:
                if data.low < current_extreme.low:
                    current_extreme = data
                    current_extreme_idx = i
                else:
                    rise = data.high - current_extreme.low
                    if rise >= self.threshold_absolute:
                        self._add_zigzag_point(
                            index=current_extreme_idx,
                            data=current_extreme,
                            is_high=False
                        )
                        current_extreme = data
                        current_extreme_idx = i
                        looking_for_high = True
    
    def _calculate_atr_zigzag(self) -> None:
        """Calculate ATR-based ZigZag."""
        if len(self.price_data) < 14:  # Need at least 14 periods for ATR
            return
        
        # Calculate ATR
        atr_values = self._calculate_atr(period=14)
        if not atr_values:
            return
        
        current_extreme = self.price_data[14]  # Start after ATR calculation
        current_extreme_idx = 14
        looking_for_high = True
        
        for i, data in enumerate(self.price_data[15:], 15):
            if i >= len(atr_values):
                break
                
            threshold = atr_values[i] * self.atr_multiplier
            
            if looking_for_high:
                if data.high > current_extreme.high:
                    current_extreme = data
                    current_extreme_idx = i
                else:
                    decline = current_extreme.high - data.low
                    if decline >= threshold:
                        self._add_zigzag_point(
                            index=current_extreme_idx,
                            data=current_extreme,
                            is_high=True
                        )
                        current_extreme = data
                        current_extreme_idx = i
                        looking_for_high = False
            else:
                if data.low < current_extreme.low:
                    current_extreme = data
                    current_extreme_idx = i
                else:
                    rise = data.high - current_extreme.low
                    if rise >= threshold:
                        self._add_zigzag_point(
                            index=current_extreme_idx,
                            data=current_extreme,
                            is_high=False
                        )
                        current_extreme = data
                        current_extreme_idx = i
                        looking_for_high = True
    
    def _calculate_atr(self, period: int = 14) -> List[float]:
        """Calculate Average True Range."""
        if len(self.price_data) < period + 1:
            return []
        
        true_ranges = []
        for i in range(1, len(self.price_data)):
            current = self.price_data[i]
            previous = self.price_data[i-1]
            
            tr1 = current.high - current.low
            tr2 = abs(current.high - previous.close)
            tr3 = abs(current.low - previous.close)
            
            true_ranges.append(max(tr1, tr2, tr3))
        
        # Calculate ATR using exponential moving average
        atr_values = []
        if len(true_ranges) >= period:
            # First ATR is simple average
            first_atr = sum(true_ranges[:period]) / period
            atr_values.append(first_atr)
            
            # Subsequent ATRs use exponential smoothing
            for i in range(period, len(true_ranges)):
                atr = ((atr_values[-1] * (period - 1)) + true_ranges[i]) / period
                atr_values.append(atr)
        
        return atr_values
    
    def _add_zigzag_point(self, index: int, data: MarketDataPoint, is_high: bool) -> None:
        """Add a ZigZag point."""
        price = data.high if is_high else data.low
        
        point = ZigZagPoint(
            index=index,
            timestamp=data.timestamp,
            price=price,
            is_high=is_high,
            volume=data.volume
        )
        
        self.zigzag_points.append(point)
    
    def _check_for_new_zigzag_point(self) -> None:
        """Check if latest data creates a new ZigZag point."""
        if len(self.zigzag_points) < 1 or len(self.price_data) < 2:
            return
        
        latest_data = self.price_data[-1]
        latest_index = len(self.price_data) - 1
        last_point = self.zigzag_points[-1]
        
        # Determine what we're looking for next
        looking_for_high = not last_point.is_high
        
        # Check if threshold is met
        if self.zigzag_type == ZigZagType.PERCENTAGE:
            if looking_for_high:
                rise_pct = ((latest_data.high - last_point.price) / last_point.price) * 100
                if rise_pct >= self.threshold_percentage:
                    self._add_zigzag_point(latest_index, latest_data, True)
            else:
                decline_pct = ((last_point.price - latest_data.low) / last_point.price) * 100
                if decline_pct >= self.threshold_percentage:
                    self._add_zigzag_point(latest_index, latest_data, False)
    
    # ========== Swing Analysis ==========
    
    def _create_swings(self) -> None:
        """Create swing objects from ZigZag points."""
        self.swings.clear()
        
        for i in range(len(self.zigzag_points) - 1):
            swing = ZigZagSwing(
                start_point=self.zigzag_points[i],
                end_point=self.zigzag_points[i + 1]
            )
            
            # Calculate average volume during swing
            start_idx = swing.start_point.index
            end_idx = swing.end_point.index
            if end_idx > start_idx:
                volumes = [data.volume for data in self.price_data[start_idx:end_idx+1]]
                swing.avg_volume = sum(volumes) / len(volumes) if volumes else 0
            
            self.swings.append(swing)
    
    def get_recent_swings(self, count: int = 5) -> List[ZigZagSwing]:
        """Get most recent swings."""
        return self.swings[-count:] if len(self.swings) >= count else self.swings
    
    def get_swing_statistics(self) -> Dict[str, Any]:
        """Get swing statistics."""
        if not self.swings:
            return {}
        
        up_swings = [s for s in self.swings if s.is_up_swing]
        down_swings = [s for s in self.swings if s.is_down_swing]
        
        return {
            'total_swings': len(self.swings),
            'up_swings': len(up_swings),
            'down_swings': len(down_swings),
            'avg_up_swing_size': sum(s.swing_size for s in up_swings) / len(up_swings) if up_swings else 0,
            'avg_down_swing_size': sum(s.swing_size for s in down_swings) / len(down_swings) if down_swings else 0,
            'avg_up_swing_percentage': sum(s.swing_percentage for s in up_swings) / len(up_swings) if up_swings else 0,
            'avg_down_swing_percentage': sum(s.swing_percentage for s in down_swings) / len(down_swings) if down_swings else 0
        }
    
    # ========== Market Structure Analysis ==========
    
    def _analyze_market_structure(self) -> None:
        """Analyze overall market structure from ZigZag."""
        if len(self.zigzag_points) < 4:
            return
        
        recent_points = self.zigzag_points[-4:]  # Last 4 points
        
        # Analyze trend based on highs and lows
        highs = [p for p in recent_points if p.is_high]
        lows = [p for p in recent_points if not p.is_high]
        
        if len(highs) >= 2 and len(lows) >= 2:
            # Check for higher highs and higher lows (uptrend)
            if (highs[-1].price > highs[-2].price and 
                lows[-1].price > lows[-2].price):
                self.current_trend = TrendDirection.UP
                self.trend_strength = min(1.0, (highs[-1].price - highs[-2].price) / highs[-2].price * 10)
            
            # Check for lower highs and lower lows (downtrend)
            elif (highs[-1].price < highs[-2].price and 
                  lows[-1].price < lows[-2].price):
                self.current_trend = TrendDirection.DOWN
                self.trend_strength = min(1.0, (highs[-2].price - highs[-1].price) / highs[-2].price * 10)
            
            else:
                self.current_trend = TrendDirection.SIDEWAYS
                self.trend_strength = 0.0
        
        # Update support and resistance levels
        self._update_support_resistance()
    
    def _update_market_structure(self) -> None:
        """Update market structure with new data."""
        self._analyze_market_structure()
    
    def _update_support_resistance(self) -> None:
        """Update support and resistance levels."""
        if len(self.zigzag_points) < 3:
            return
        
        self.support_levels.clear()
        self.resistance_levels.clear()
        
        # Collect recent levels
        recent_points = self.zigzag_points[-10:] if len(self.zigzag_points) >= 10 else self.zigzag_points
        
        # Group similar levels
        tolerance = 0.01  # 1% tolerance for level grouping
        
        for point in recent_points:
            if point.is_high:
                # Check if this resistance level already exists
                exists = False
                for level in self.resistance_levels:
                    if abs(point.price - level) / level <= tolerance:
                        exists = True
                        break
                if not exists:
                    self.resistance_levels.append(point.price)
            else:
                # Check if this support level already exists
                exists = False
                for level in self.support_levels:
                    if abs(point.price - level) / level <= tolerance:
                        exists = True
                        break
                if not exists:
                    self.support_levels.append(point.price)
        
        # Sort levels
        self.support_levels.sort(reverse=True)  # Highest support first
        self.resistance_levels.sort()           # Lowest resistance first
    
    # ========== Pattern Recognition ==========
    
    def _identify_patterns(self) -> None:
        """Identify chart patterns from ZigZag points."""
        self.patterns.clear()
        
        if len(self.zigzag_points) < 5:
            return
        
        # Look for various patterns
        self._find_double_tops_bottoms()
        self._find_head_and_shoulders()
        self._find_triangles()
        self._find_flags_pennants()
    
    def _update_patterns(self) -> None:
        """Update patterns with new data."""
        # Only look for new patterns if we have enough new data
        if len(self.zigzag_points) >= 5:
            recent_patterns = []
            
            # Check for patterns in recent data
            recent_points = self.zigzag_points[-10:]
            if len(recent_points) >= 5:
                # Simplified pattern check - could be expanded
                pass
    
    def _find_double_tops_bottoms(self) -> None:
        """Find double top and double bottom patterns."""
        for i in range(len(self.zigzag_points) - 4):
            points = self.zigzag_points[i:i+5]
            
            # Double top: Low-High-Low-High-Low with similar highs
            if (not points[0].is_high and points[1].is_high and not points[2].is_high and 
                points[3].is_high and not points[4].is_high):
                
                high1, high2 = points[1].price, points[3].price
                tolerance = 0.02  # 2% tolerance
                
                if abs(high1 - high2) / max(high1, high2) <= tolerance:
                    pattern = ZigZagPattern(
                        pattern_type="DOUBLE_TOP",
                        points=points,
                        confidence=0.7,
                        target_price=points[2].price,  # Target at middle low
                        stop_loss=max(high1, high2) * 1.02
                    )
                    self.patterns.append(pattern)
            
            # Double bottom: High-Low-High-Low-High with similar lows
            elif (points[0].is_high and not points[1].is_high and points[2].is_high and 
                  not points[3].is_high and points[4].is_high):
                
                low1, low2 = points[1].price, points[3].price
                tolerance = 0.02
                
                if abs(low1 - low2) / max(low1, low2) <= tolerance:
                    pattern = ZigZagPattern(
                        pattern_type="DOUBLE_BOTTOM",
                        points=points,
                        confidence=0.7,
                        target_price=points[2].price,  # Target at middle high
                        stop_loss=min(low1, low2) * 0.98
                    )
                    self.patterns.append(pattern)
    
    def _find_head_and_shoulders(self) -> None:
        """Find head and shoulders patterns."""
        for i in range(len(self.zigzag_points) - 6):
            points = self.zigzag_points[i:i+7]
            
            # Head and shoulders: Low-High-Low-High-Low-High-Low
            # with middle high being the highest (head)
            if len([p for p in points if p.is_high]) == 3 and len([p for p in points if not p.is_high]) == 4:
                highs = [p for p in points if p.is_high]
                lows = [p for p in points if not p.is_high]
                
                # Check if middle high is the head (highest)
                if len(highs) == 3 and highs[1].price > highs[0].price and highs[1].price > highs[2].price:
                    # Check shoulder symmetry
                    shoulder_diff = abs(highs[0].price - highs[2].price)
                    head_height = highs[1].price - max(lows[1].price, lows[2].price)
                    
                    if shoulder_diff / head_height <= 0.1:  # Shoulders should be similar
                        pattern = ZigZagPattern(
                            pattern_type="HEAD_AND_SHOULDERS",
                            points=points,
                            confidence=0.8,
                            target_price=min(lows[1].price, lows[2].price),
                            stop_loss=highs[1].price
                        )
                        self.patterns.append(pattern)
    
    def _find_triangles(self) -> None:
        """Find triangle patterns (ascending, descending, symmetrical)."""
        if len(self.zigzag_points) < 6:
            return
        
        # Look for converging trend lines
        recent_points = self.zigzag_points[-8:]
        highs = [p for p in recent_points if p.is_high]
        lows = [p for p in recent_points if not p.is_high]
        
        if len(highs) >= 3 and len(lows) >= 3:
            # Check for ascending triangle (flat resistance, rising support)
            high_trend = self._calculate_trend_slope([p.price for p in highs[-3:]])
            low_trend = self._calculate_trend_slope([p.price for p in lows[-3:]])
            
            if abs(high_trend) < 0.001 and low_trend > 0.001:  # Flat highs, rising lows
                pattern = ZigZagPattern(
                    pattern_type="ASCENDING_TRIANGLE",
                    points=recent_points,
                    confidence=0.6
                )
                self.patterns.append(pattern)
            elif abs(low_trend) < 0.001 and high_trend < -0.001:  # Flat lows, falling highs
                pattern = ZigZagPattern(
                    pattern_type="DESCENDING_TRIANGLE",
                    points=recent_points,
                    confidence=0.6
                )
                self.patterns.append(pattern)
    
    def _find_flags_pennants(self) -> None:
        """Find flag and pennant patterns."""
        if len(self.zigzag_points) < 5:
            return
        
        # Flags and pennants are continuation patterns
        # Look for consolidation after strong moves
        recent_swings = self.get_recent_swings(3)
        
        if len(recent_swings) >= 2:
            # Check for strong initial move followed by consolidation
            initial_swing = recent_swings[0]
            consolidation_swings = recent_swings[1:]
            
            # Strong initial move
            if initial_swing.swing_percentage > 5.0:
                # Small consolidation moves
                consolidation_sizes = [s.swing_percentage for s in consolidation_swings]
                avg_consolidation = sum(consolidation_sizes) / len(consolidation_sizes)
                
                if avg_consolidation < initial_swing.swing_percentage * 0.3:  # Consolidation < 30% of initial move
                    pattern_type = "FLAG" if initial_swing.is_up_swing else "BEAR_FLAG"
                    pattern = ZigZagPattern(
                        pattern_type=pattern_type,
                        points=self.zigzag_points[-5:],
                        confidence=0.5
                    )
                    self.patterns.append(pattern)
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope from values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    # ========== Fibonacci Analysis ==========
    
    def calculate_fibonacci_retracements(self, swing: ZigZagSwing) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels for a swing."""
        retracements = {}
        swing_size = swing.swing_size
        
        if swing.is_up_swing:
            # For upswings, retracements are below the high
            base_price = swing.end_point.price
            for level in self.fibonacci_levels:
                retracements[level] = base_price - (swing_size * level)
        else:
            # For downswings, retracements are above the low
            base_price = swing.end_point.price
            for level in self.fibonacci_levels:
                retracements[level] = base_price + (swing_size * level)
        
        return retracements
    
    def get_current_retracement_level(self, current_price: float) -> Optional[Tuple[float, float]]:
        """Get current Fibonacci retracement level."""
        if len(self.swings) < 1:
            return None
        
        last_swing = self.swings[-1]
        retracements = self.calculate_fibonacci_retracements(last_swing)
        
        # Find closest retracement level
        closest_level = None
        min_distance = float('inf')
        
        for level, price in retracements.items():
            distance = abs(current_price - price)
            if distance < min_distance:
                min_distance = distance
                closest_level = (level, price)
        
        return closest_level
    
    # ========== Trading Signals ==========
    
    def get_trading_signal(self, current_price: float) -> SignalType:
        """
        Get trading signal based on ZigZag analysis.
        
        Args:
            current_price: Current market price
            
        Returns:
            Trading signal
        """
        if not self.is_initialized or len(self.zigzag_points) < 3:
            return SignalType.FLAT
        
        signal = SignalType.FLAT
        
        # Check pattern signals
        for pattern in self.patterns:
            if pattern.confidence > 0.7:
                if pattern.pattern_type in ["DOUBLE_BOTTOM", "HEAD_AND_SHOULDERS_INVERSE"]:
                    signal = SignalType.BUY
                elif pattern.pattern_type in ["DOUBLE_TOP", "HEAD_AND_SHOULDERS"]:
                    signal = SignalType.SELL
        
        # Check trend signals
        if self.current_trend == TrendDirection.UP and self.trend_strength > 0.5:
            if signal != SignalType.SELL:  # Don't conflict with pattern signals
                signal = SignalType.BUY
        elif self.current_trend == TrendDirection.DOWN and self.trend_strength > 0.5:
            if signal != SignalType.BUY:
                signal = SignalType.SELL
        
        # Check support/resistance signals
        for support in self.support_levels[:2]:  # Check top 2 support levels
            if abs(current_price - support) / support < 0.005:  # Within 0.5%
                signal = SignalType.BUY
                break
        
        for resistance in self.resistance_levels[:2]:  # Check top 2 resistance levels
            if abs(current_price - resistance) / resistance < 0.005:  # Within 0.5%
                signal = SignalType.SELL
                break
        
        return signal
    
    # ========== Analysis Results ==========
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive ZigZag analysis summary."""
        return {
            'zigzag_points_count': len(self.zigzag_points),
            'swings_count': len(self.swings),
            'patterns_count': len(self.patterns),
            'current_trend': self.current_trend.value,
            'trend_strength': self.trend_strength,
            'support_levels': self.support_levels[:5],  # Top 5
            'resistance_levels': self.resistance_levels[:5],  # Top 5
            'recent_patterns': [p.pattern_type for p in self.patterns[-3:]],
            'zigzag_type': self.zigzag_type.value,
            'threshold_percentage': self.threshold_percentage,
            'data_points': len(self.price_data)
        }
    
    def get_pattern_alerts(self) -> List[Dict[str, Any]]:
        """Get pattern-based alerts."""
        alerts = []
        
        for pattern in self.patterns[-5:]:  # Recent patterns
            if pattern.confidence > 0.6:
                alerts.append({
                    'pattern_type': pattern.pattern_type,
                    'confidence': pattern.confidence,
                    'target_price': pattern.target_price,
                    'stop_loss': pattern.stop_loss,
                    'points_count': len(pattern.points),
                    'timestamp': pattern.points[-1].timestamp if pattern.points else datetime.now()
                })
        
        return alerts
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CZigZagAnalyzer(points={len(self.zigzag_points)}, swings={len(self.swings)}, "
                f"patterns={len(self.patterns)}, trend={self.current_trend.value})")