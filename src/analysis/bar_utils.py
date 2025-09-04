"""
Advanced bar and candlestick analysis utilities for trading systems.

This module provides comprehensive bar analysis including:
- Candlestick pattern recognition
- Bar statistics and analysis
- Price action analysis  
- Volume analysis
- Gap detection and analysis
- Trend analysis from bars
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class BarAnalysisType(Enum):
    """Types of bar analysis."""
    CANDLESTICK_PATTERNS = "candlestick_patterns"
    PRICE_ACTION = "price_action"
    VOLUME_ANALYSIS = "volume_analysis"
    GAP_ANALYSIS = "gap_analysis"
    TREND_ANALYSIS = "trend_analysis"
    VOLATILITY = "volatility"


class BarPatternType(Enum):
    """Candlestick pattern types."""
    # Single bar patterns
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    INVERTED_HAMMER = "inverted_hammer"
    SPINNING_TOP = "spinning_top"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"
    
    # Multi-bar patterns
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"


class GapType(Enum):
    """Gap types in price action."""
    NO_GAP = "no_gap"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    BREAKAWAY_GAP = "breakaway_gap"
    MEASURING_GAP = "measuring_gap"
    EXHAUSTION_GAP = "exhaustion_gap"


@dataclass
class CandlestickPattern:
    """Information about detected candlestick pattern."""
    
    pattern_type: BarPatternType
    confidence: float  # 0.0 to 1.0
    bar_index: int
    signal_strength: str  # "weak", "moderate", "strong"
    bullish: bool
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"{self.pattern_type.value.replace('_', ' ').title()}"


@dataclass 
class BarStatistics:
    """Comprehensive bar statistics."""
    
    # Basic OHLC stats
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: float = 0.0
    
    # Calculated values
    body_size: float = 0.0
    upper_shadow: float = 0.0
    lower_shadow: float = 0.0
    total_range: float = 0.0
    
    # Percentages
    body_percentage: float = 0.0
    upper_shadow_percentage: float = 0.0
    lower_shadow_percentage: float = 0.0
    
    # Classification
    is_bullish: bool = False
    is_bearish: bool = False
    is_doji: bool = False
    is_long_body: bool = False
    is_long_upper_shadow: bool = False
    is_long_lower_shadow: bool = False
    
    # Gap information
    gap_type: GapType = GapType.NO_GAP
    gap_size: float = 0.0
    gap_percentage: float = 0.0


@dataclass
class BarAnalysisResult:
    """Results of comprehensive bar analysis."""
    
    bar_index: int
    analysis_types: List[BarAnalysisType]
    bar_stats: BarStatistics
    patterns: List[CandlestickPattern] = None
    
    # Analysis results
    trend_direction: str = "neutral"  # "up", "down", "neutral"
    volatility_level: str = "normal"   # "low", "normal", "high"
    volume_analysis: str = "normal"    # "low", "normal", "high", "climax"
    
    # Signals
    buy_signal_strength: float = 0.0   # 0.0 to 1.0
    sell_signal_strength: float = 0.0  # 0.0 to 1.0
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []


class CBarUtils(CBase):
    """
    Comprehensive bar and candlestick analysis utilities.
    
    Provides advanced analysis of price bars including:
    - Candlestick pattern recognition
    - Price action analysis
    - Volume analysis
    - Gap detection
    - Trend identification
    - Volatility measurement
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Analysis parameters
        self.doji_threshold = 0.1        # Body size threshold for doji (as % of range)
        self.long_body_threshold = 0.6   # Long body threshold (as % of range)
        self.long_shadow_threshold = 0.3 # Long shadow threshold (as % of range)
        self.gap_threshold = 0.002       # Minimum gap size (as % of price)
        
        # Pattern confidence thresholds
        self.min_confidence = 0.5
        self.high_confidence = 0.8
        
        # Cached analysis results
        self._cached_patterns: Dict[int, List[CandlestickPattern]] = {}
        self._cached_statistics: Dict[int, BarStatistics] = {}
        
        # Volume analysis settings
        self.volume_sma_period = 20
        self.high_volume_multiplier = 1.5
        self.low_volume_multiplier = 0.7
        
    def initialize(
        self, 
        system: SystemProtocol,
        market_data: Optional[any] = None
    ) -> 'CBarUtils':
        """Initialize bar utils with market data."""
        if market_data is not None:
            self.set_market_data(market_data)
        
        self.reset(system)
        return self
    
    def reset(self, system: SystemProtocol) -> 'CBarUtils':
        """Reset all cached analysis results."""
        self._cached_patterns.clear()
        self._cached_statistics.clear()
        return self
    
    # Core bar statistics calculation
    def calculate_bar_statistics(self, system: SystemProtocol, bar_index: int) -> BarStatistics:
        """Calculate comprehensive statistics for a single bar."""
        if bar_index in self._cached_statistics:
            return self._cached_statistics[bar_index]
        
        if not self._validate_bar_index(system, bar_index):
            return BarStatistics()
        
        # Get OHLC data
        md = system.market_data
        open_price = md.opens[bar_index]
        high_price = md.highs[bar_index]  
        low_price = md.lows[bar_index]
        close_price = md.closes[bar_index]
        volume = md.volumes[bar_index] if hasattr(md, 'volumes') and md.volumes else 0.0
        
        # Calculate basic metrics
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        # Calculate percentages (avoid division by zero)
        if total_range > 0:
            body_percentage = (body_size / total_range) * 100
            upper_shadow_percentage = (upper_shadow / total_range) * 100
            lower_shadow_percentage = (lower_shadow / total_range) * 100
        else:
            body_percentage = upper_shadow_percentage = lower_shadow_percentage = 0.0
        
        # Classifications
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        is_doji = body_percentage <= (self.doji_threshold * 100)
        is_long_body = body_percentage >= (self.long_body_threshold * 100)
        is_long_upper_shadow = upper_shadow_percentage >= (self.long_shadow_threshold * 100)
        is_long_lower_shadow = lower_shadow_percentage >= (self.long_shadow_threshold * 100)
        
        # Gap analysis
        gap_type, gap_size, gap_percentage = self._analyze_gap(system, bar_index)
        
        stats = BarStatistics(
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            body_size=body_size,
            upper_shadow=upper_shadow,
            lower_shadow=lower_shadow,
            total_range=total_range,
            body_percentage=body_percentage,
            upper_shadow_percentage=upper_shadow_percentage,
            lower_shadow_percentage=lower_shadow_percentage,
            is_bullish=is_bullish,
            is_bearish=is_bearish,
            is_doji=is_doji,
            is_long_body=is_long_body,
            is_long_upper_shadow=is_long_upper_shadow,
            is_long_lower_shadow=is_long_lower_shadow,
            gap_type=gap_type,
            gap_size=gap_size,
            gap_percentage=gap_percentage
        )
        
        self._cached_statistics[bar_index] = stats
        return stats
    
    def _analyze_gap(self, system: SystemProtocol, bar_index: int) -> Tuple[GapType, float, float]:
        """Analyze gap between current and previous bar."""
        if bar_index <= 0 or not self._validate_bar_index(system, bar_index - 1):
            return GapType.NO_GAP, 0.0, 0.0
        
        md = system.market_data
        prev_close = md.closes[bar_index - 1]
        current_open = md.opens[bar_index]
        
        gap_size = current_open - prev_close
        gap_percentage = abs(gap_size) / prev_close * 100 if prev_close > 0 else 0.0
        
        # Determine gap type
        if gap_percentage < (self.gap_threshold * 100):
            return GapType.NO_GAP, gap_size, gap_percentage
        
        if gap_size > 0:
            return GapType.GAP_UP, gap_size, gap_percentage
        else:
            return GapType.GAP_DOWN, gap_size, gap_percentage
    
    # Candlestick pattern recognition
    def detect_candlestick_patterns(
        self, 
        system: SystemProtocol, 
        bar_index: int,
        lookback_bars: int = 3
    ) -> List[CandlestickPattern]:
        """Detect candlestick patterns at specified bar."""
        if bar_index in self._cached_patterns:
            return self._cached_patterns[bar_index]
        
        patterns = []
        
        # Single bar patterns
        patterns.extend(self._detect_single_bar_patterns(system, bar_index))
        
        # Multi-bar patterns (if enough history)
        if bar_index >= lookback_bars:
            patterns.extend(self._detect_multi_bar_patterns(system, bar_index, lookback_bars))
        
        # Filter by minimum confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        
        self._cached_patterns[bar_index] = patterns
        return patterns
    
    def _detect_single_bar_patterns(self, system: SystemProtocol, bar_index: int) -> List[CandlestickPattern]:
        """Detect single bar candlestick patterns."""
        patterns = []
        stats = self.calculate_bar_statistics(system, bar_index)
        
        # Doji
        if stats.is_doji:
            confidence = 1.0 - (stats.body_percentage / 10.0)  # Smaller body = higher confidence
            patterns.append(CandlestickPattern(
                pattern_type=BarPatternType.DOJI,
                confidence=max(0.5, confidence),
                bar_index=bar_index,
                signal_strength="moderate",
                bullish=False,  # Neutral pattern
                description="Indecision candlestick with very small body"
            ))
        
        # Hammer (bullish at support)
        if (stats.lower_shadow_percentage >= 50 and 
            stats.upper_shadow_percentage <= 15 and 
            stats.body_percentage <= 30):
            
            confidence = (stats.lower_shadow_percentage + (30 - stats.body_percentage)) / 100
            patterns.append(CandlestickPattern(
                pattern_type=BarPatternType.HAMMER,
                confidence=min(1.0, confidence),
                bar_index=bar_index,
                signal_strength="strong" if confidence > self.high_confidence else "moderate",
                bullish=True,
                description="Potential reversal signal with long lower shadow"
            ))
        
        # Shooting Star (bearish at resistance)
        if (stats.upper_shadow_percentage >= 50 and 
            stats.lower_shadow_percentage <= 15 and 
            stats.body_percentage <= 30):
            
            confidence = (stats.upper_shadow_percentage + (30 - stats.body_percentage)) / 100
            patterns.append(CandlestickPattern(
                pattern_type=BarPatternType.SHOOTING_STAR,
                confidence=min(1.0, confidence),
                bar_index=bar_index,
                signal_strength="strong" if confidence > self.high_confidence else "moderate",
                bullish=False,
                description="Potential reversal signal with long upper shadow"
            ))
        
        # Marubozu patterns (very long bodies, minimal shadows)
        if stats.body_percentage >= 85:
            if stats.is_bullish:
                patterns.append(CandlestickPattern(
                    pattern_type=BarPatternType.MARUBOZU_BULLISH,
                    confidence=stats.body_percentage / 100,
                    bar_index=bar_index,
                    signal_strength="strong",
                    bullish=True,
                    description="Strong bullish momentum with minimal shadows"
                ))
            else:
                patterns.append(CandlestickPattern(
                    pattern_type=BarPatternType.MARUBOZU_BEARISH,
                    confidence=stats.body_percentage / 100,
                    bar_index=bar_index,
                    signal_strength="strong",
                    bullish=False,
                    description="Strong bearish momentum with minimal shadows"
                ))
        
        return patterns
    
    def _detect_multi_bar_patterns(
        self, 
        system: SystemProtocol, 
        bar_index: int, 
        lookback: int
    ) -> List[CandlestickPattern]:
        """Detect multi-bar candlestick patterns."""
        patterns = []
        
        if bar_index < 1:
            return patterns
        
        current_stats = self.calculate_bar_statistics(system, bar_index)
        prev_stats = self.calculate_bar_statistics(system, bar_index - 1)
        
        # Engulfing patterns
        if self._is_engulfing_pattern(prev_stats, current_stats):
            if current_stats.is_bullish:
                patterns.append(CandlestickPattern(
                    pattern_type=BarPatternType.ENGULFING_BULLISH,
                    confidence=0.8,
                    bar_index=bar_index,
                    signal_strength="strong",
                    bullish=True,
                    description="Bullish engulfing pattern - potential reversal"
                ))
            else:
                patterns.append(CandlestickPattern(
                    pattern_type=BarPatternType.ENGULFING_BEARISH,
                    confidence=0.8,
                    bar_index=bar_index,
                    signal_strength="strong",
                    bullish=False,
                    description="Bearish engulfing pattern - potential reversal"
                ))
        
        # Three-bar patterns
        if bar_index >= 2:
            three_bar_patterns = self._detect_three_bar_patterns(system, bar_index)
            patterns.extend(three_bar_patterns)
        
        return patterns
    
    def _is_engulfing_pattern(self, prev_stats: BarStatistics, current_stats: BarStatistics) -> bool:
        """Check if current bar engulfs the previous bar."""
        if prev_stats.is_bullish == current_stats.is_bullish:
            return False
        
        # Current bar must completely engulf previous bar
        current_body_top = max(current_stats.open_price, current_stats.close_price)
        current_body_bottom = min(current_stats.open_price, current_stats.close_price)
        prev_body_top = max(prev_stats.open_price, prev_stats.close_price)
        prev_body_bottom = min(prev_stats.open_price, prev_stats.close_price)
        
        return (current_body_top > prev_body_top and 
                current_body_bottom < prev_body_bottom and
                current_stats.body_size > prev_stats.body_size * 1.1)  # At least 10% larger
    
    def _detect_three_bar_patterns(self, system: SystemProtocol, bar_index: int) -> List[CandlestickPattern]:
        """Detect three-bar patterns like Morning/Evening Star."""
        patterns = []
        
        stats_2 = self.calculate_bar_statistics(system, bar_index - 2)
        stats_1 = self.calculate_bar_statistics(system, bar_index - 1)  
        stats_0 = self.calculate_bar_statistics(system, bar_index)
        
        # Morning Star (bullish reversal)
        if (stats_2.is_bearish and stats_2.is_long_body and
            stats_1.is_doji and
            stats_0.is_bullish and stats_0.is_long_body):
            
            patterns.append(CandlestickPattern(
                pattern_type=BarPatternType.MORNING_STAR,
                confidence=0.85,
                bar_index=bar_index,
                signal_strength="strong",
                bullish=True,
                description="Morning Star - strong bullish reversal pattern"
            ))
        
        # Evening Star (bearish reversal)  
        if (stats_2.is_bullish and stats_2.is_long_body and
            stats_1.is_doji and
            stats_0.is_bearish and stats_0.is_long_body):
            
            patterns.append(CandlestickPattern(
                pattern_type=BarPatternType.EVENING_STAR,
                confidence=0.85,
                bar_index=bar_index,
                signal_strength="strong", 
                bullish=False,
                description="Evening Star - strong bearish reversal pattern"
            ))
        
        return patterns
    
    # Volume analysis
    def analyze_volume(self, system: SystemProtocol, bar_index: int) -> str:
        """Analyze volume characteristics at specified bar."""
        if not hasattr(system.market_data, 'volumes') or not system.market_data.volumes:
            return "normal"  # No volume data
        
        if bar_index < self.volume_sma_period:
            return "normal"  # Not enough data for analysis
        
        md = system.market_data
        current_volume = md.volumes[bar_index]
        
        # Calculate average volume over the period
        start_idx = bar_index - self.volume_sma_period + 1
        avg_volume = np.mean(md.volumes[start_idx:bar_index + 1])
        
        if current_volume >= avg_volume * self.high_volume_multiplier:
            return "high"
        elif current_volume <= avg_volume * self.low_volume_multiplier:
            return "low"
        else:
            return "normal"
    
    # Trend analysis
    def analyze_trend(self, system: SystemProtocol, bar_index: int, lookback: int = 5) -> str:
        """Analyze trend direction based on recent bars."""
        if bar_index < lookback:
            return "neutral"
        
        md = system.market_data
        
        # Calculate simple trend based on close prices
        closes = md.closes[bar_index - lookback + 1:bar_index + 1]
        
        # Linear regression slope to determine trend
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        # Normalize slope relative to price
        price_normalized_slope = slope / closes[-1] * 100
        
        if price_normalized_slope > 0.1:
            return "up"
        elif price_normalized_slope < -0.1:
            return "down"
        else:
            return "neutral"
    
    # Volatility analysis
    def calculate_volatility(self, system: SystemProtocol, bar_index: int, period: int = 14) -> float:
        """Calculate volatility using True Range method."""
        if bar_index < period:
            return 0.0
        
        md = system.market_data
        true_ranges = []
        
        for i in range(bar_index - period + 1, bar_index + 1):
            if i > 0:
                # True Range = max(High-Low, High-PrevClose, PrevClose-Low)
                high_low = md.highs[i] - md.lows[i]
                high_prev_close = abs(md.highs[i] - md.closes[i-1])
                prev_close_low = abs(md.closes[i-1] - md.lows[i])
                
                tr = max(high_low, high_prev_close, prev_close_low)
                true_ranges.append(tr)
            else:
                true_ranges.append(md.highs[i] - md.lows[i])
        
        return np.mean(true_ranges) if true_ranges else 0.0
    
    def analyze_volatility_level(self, system: SystemProtocol, bar_index: int) -> str:
        """Classify volatility level as low, normal, or high."""
        if bar_index < 20:
            return "normal"
        
        current_volatility = self.calculate_volatility(system, bar_index, 14)
        
        # Calculate historical average volatility
        volatilities = []
        for i in range(max(20, bar_index - 100), bar_index):
            vol = self.calculate_volatility(system, i, 14)
            if vol > 0:
                volatilities.append(vol)
        
        if not volatilities:
            return "normal"
        
        avg_volatility = np.mean(volatilities)
        std_volatility = np.std(volatilities)
        
        if current_volatility > avg_volatility + std_volatility:
            return "high"
        elif current_volatility < avg_volatility - std_volatility:
            return "low"
        else:
            return "normal"
    
    # Comprehensive analysis
    def analyze_bar(
        self, 
        system: SystemProtocol, 
        bar_index: int,
        analysis_types: List[BarAnalysisType] = None
    ) -> BarAnalysisResult:
        """Perform comprehensive analysis of a single bar."""
        if analysis_types is None:
            analysis_types = list(BarAnalysisType)
        
        # Calculate basic statistics
        bar_stats = self.calculate_bar_statistics(system, bar_index)
        
        result = BarAnalysisResult(
            bar_index=bar_index,
            analysis_types=analysis_types,
            bar_stats=bar_stats
        )
        
        # Perform requested analyses
        if BarAnalysisType.CANDLESTICK_PATTERNS in analysis_types:
            result.patterns = self.detect_candlestick_patterns(system, bar_index)
        
        if BarAnalysisType.TREND_ANALYSIS in analysis_types:
            result.trend_direction = self.analyze_trend(system, bar_index)
        
        if BarAnalysisType.VOLATILITY in analysis_types:
            result.volatility_level = self.analyze_volatility_level(system, bar_index)
        
        if BarAnalysisType.VOLUME_ANALYSIS in analysis_types:
            result.volume_analysis = self.analyze_volume(system, bar_index)
        
        # Calculate signal strengths
        result.buy_signal_strength = self._calculate_buy_signal_strength(result)
        result.sell_signal_strength = self._calculate_sell_signal_strength(result)
        
        return result
    
    def _calculate_buy_signal_strength(self, result: BarAnalysisResult) -> float:
        """Calculate overall buy signal strength from analysis."""
        strength = 0.0
        
        # Bullish patterns contribute to buy signal
        if result.patterns:
            bullish_patterns = [p for p in result.patterns if p.bullish]
            if bullish_patterns:
                avg_confidence = np.mean([p.confidence for p in bullish_patterns])
                strength += avg_confidence * 0.5
        
        # Uptrend contributes to buy signal
        if result.trend_direction == "up":
            strength += 0.3
        
        # High volume confirms signals
        if result.volume_analysis == "high":
            strength += 0.2
        
        return min(1.0, strength)
    
    def _calculate_sell_signal_strength(self, result: BarAnalysisResult) -> float:
        """Calculate overall sell signal strength from analysis."""
        strength = 0.0
        
        # Bearish patterns contribute to sell signal
        if result.patterns:
            bearish_patterns = [p for p in result.patterns if not p.bullish]
            if bearish_patterns:
                avg_confidence = np.mean([p.confidence for p in bearish_patterns])
                strength += avg_confidence * 0.5
        
        # Downtrend contributes to sell signal
        if result.trend_direction == "down":
            strength += 0.3
        
        # High volume confirms signals
        if result.volume_analysis == "high":
            strength += 0.2
        
        return min(1.0, strength)
    
    # Utility methods
    def _validate_bar_index(self, system: SystemProtocol, bar_index: int) -> bool:
        """Validate that bar index is within valid range."""
        if not hasattr(system.market_data, 'closes'):
            return False
        
        return 0 <= bar_index < len(system.market_data.closes)
    
    def get_analysis_summary(self, system: SystemProtocol, bar_index: int) -> Dict[str, any]:
        """Get summary of all analysis for a bar."""
        if not self._validate_bar_index(system, bar_index):
            return {}
        
        analysis = self.analyze_bar(system, bar_index)
        
        summary = {
            "bar_index": bar_index,
            "basic_info": {
                "open": analysis.bar_stats.open_price,
                "high": analysis.bar_stats.high_price, 
                "low": analysis.bar_stats.low_price,
                "close": analysis.bar_stats.close_price,
                "volume": analysis.bar_stats.volume,
                "is_bullish": analysis.bar_stats.is_bullish
            },
            "technical_analysis": {
                "body_percentage": round(analysis.bar_stats.body_percentage, 2),
                "upper_shadow_percentage": round(analysis.bar_stats.upper_shadow_percentage, 2),
                "lower_shadow_percentage": round(analysis.bar_stats.lower_shadow_percentage, 2),
                "is_doji": analysis.bar_stats.is_doji,
                "gap_type": analysis.bar_stats.gap_type.value
            },
            "market_analysis": {
                "trend_direction": analysis.trend_direction,
                "volatility_level": analysis.volatility_level,
                "volume_analysis": analysis.volume_analysis
            },
            "signals": {
                "buy_strength": round(analysis.buy_signal_strength, 3),
                "sell_strength": round(analysis.sell_signal_strength, 3),
                "patterns_detected": len(analysis.patterns) if analysis.patterns else 0
            }
        }
        
        if analysis.patterns:
            summary["candlestick_patterns"] = [
                {
                    "type": p.pattern_type.value,
                    "confidence": round(p.confidence, 3),
                    "bullish": p.bullish,
                    "strength": p.signal_strength
                }
                for p in analysis.patterns
            ]
        
        return summary
    
    def clear_cache(self) -> 'CBarUtils':
        """Clear all cached analysis results."""
        self._cached_patterns.clear()
        self._cached_statistics.clear()
        return self
    
    def __str__(self) -> str:
        """String representation of bar utils."""
        return (
            f"CBarUtils(cached_patterns={len(self._cached_patterns)}, "
            f"cached_stats={len(self._cached_statistics)}, "
            f"doji_threshold={self.doji_threshold})"
        )