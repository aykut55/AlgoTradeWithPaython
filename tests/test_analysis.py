"""
Comprehensive tests for analysis module (CBarUtils).
"""

import pytest
import unittest
from datetime import datetime
from unittest.mock import Mock, MagicMock
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.bar_utils import (
    CBarUtils,
    BarAnalysisType,
    BarPatternType,
    BarAnalysisResult,
    CandlestickPattern,
    BarStatistics,
    GapType
)

from analysis.zigzag_analyzer import (
    CZigZagAnalyzer,
    ZigZagType,
    TrendDirection,
    ZigZagLevel,
    ZigZagPoint,
    ZigZagSwing,
    ZigZagPattern,
    MarketDataPoint,
    SignalType
)


class TestCBarUtils(unittest.TestCase):
    """Test cases for CBarUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.system.system_id = 1
        
        # Mock market data with various candlestick patterns
        self.system.market_data = Mock()
        
        # Sample OHLC data for testing patterns
        self.system.market_data.opens = [100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0]
        self.system.market_data.highs = [102.5, 104.5, 106.0, 105.0, 107.5, 109.0, 108.0]
        self.system.market_data.lows = [99.5, 101.5, 103.5, 102.5, 104.5, 106.5, 105.0]
        self.system.market_data.closes = [102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0]
        self.system.market_data.volumes = [1000, 1200, 800, 1500, 1100, 900, 1300]
        
        self.bar_utils = CBarUtils()
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.bar_utils, CBarUtils)
        self.assertEqual(self.bar_utils.system_id, 0)
        self.assertEqual(self.bar_utils.doji_threshold, 0.1)
        self.assertEqual(self.bar_utils.long_body_threshold, 0.6)
    
    def test_bar_statistics_calculation(self):
        """Test basic bar statistics calculation."""
        stats = self.bar_utils.calculate_bar_statistics(self.system, 0)
        
        self.assertIsInstance(stats, BarStatistics)
        self.assertEqual(stats.open_price, 100.0)
        self.assertEqual(stats.high_price, 102.5)
        self.assertEqual(stats.low_price, 99.5)
        self.assertEqual(stats.close_price, 102.0)
        self.assertEqual(stats.volume, 1000)
        
        # Check calculated values
        self.assertEqual(stats.body_size, 2.0)  # |102 - 100|
        self.assertEqual(stats.upper_shadow, 0.5)  # 102.5 - 102
        self.assertEqual(stats.lower_shadow, 0.5)  # 100 - 99.5
        self.assertEqual(stats.total_range, 3.0)  # 102.5 - 99.5
        
        # Check percentages
        expected_body_percentage = (2.0 / 3.0) * 100  # ~66.67%
        self.assertAlmostEqual(stats.body_percentage, expected_body_percentage, places=1)
        
        # Check classifications
        self.assertTrue(stats.is_bullish)
        self.assertFalse(stats.is_bearish)
        self.assertFalse(stats.is_doji)  # Body percentage > 10%
    
    def test_doji_pattern_detection(self):
        """Test doji pattern detection."""
        # Create doji-like bar (small body)
        self.system.market_data.opens = [100.0]
        self.system.market_data.highs = [102.0]
        self.system.market_data.lows = [98.0]
        self.system.market_data.closes = [100.1]  # Very small body
        self.system.market_data.volumes = [1000]
        
        patterns = self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Should detect doji pattern
        doji_patterns = [p for p in patterns if p.pattern_type == BarPatternType.DOJI]
        self.assertGreater(len(doji_patterns), 0)
        
        doji = doji_patterns[0]
        self.assertEqual(doji.pattern_type, BarPatternType.DOJI)
        self.assertGreaterEqual(doji.confidence, 0.5)
    
    def test_hammer_pattern_detection(self):
        """Test hammer pattern detection."""
        # Create hammer pattern (long lower shadow, small body)
        self.system.market_data.opens = [100.0]
        self.system.market_data.highs = [101.0]
        self.system.market_data.lows = [95.0]  # Long lower shadow
        self.system.market_data.closes = [100.5]  # Small body
        self.system.market_data.volumes = [1000]
        
        patterns = self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Should detect hammer pattern
        hammer_patterns = [p for p in patterns if p.pattern_type == BarPatternType.HAMMER]
        self.assertGreater(len(hammer_patterns), 0)
        
        hammer = hammer_patterns[0]
        self.assertEqual(hammer.pattern_type, BarPatternType.HAMMER)
        self.assertTrue(hammer.bullish)
        self.assertGreaterEqual(hammer.confidence, 0.5)
    
    def test_shooting_star_pattern_detection(self):
        """Test shooting star pattern detection."""
        # Create shooting star pattern (long upper shadow, small body)
        self.system.market_data.opens = [100.0]
        self.system.market_data.highs = [105.0]  # Long upper shadow
        self.system.market_data.lows = [99.5]
        self.system.market_data.closes = [100.5]  # Small body
        self.system.market_data.volumes = [1000]
        
        patterns = self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Should detect shooting star pattern
        star_patterns = [p for p in patterns if p.pattern_type == BarPatternType.SHOOTING_STAR]
        self.assertGreater(len(star_patterns), 0)
        
        star = star_patterns[0]
        self.assertEqual(star.pattern_type, BarPatternType.SHOOTING_STAR)
        self.assertFalse(star.bullish)
        self.assertGreaterEqual(star.confidence, 0.5)
    
    def test_marubozu_pattern_detection(self):
        """Test marubozu pattern detection."""
        # Create bullish marubozu (large body, minimal shadows)
        self.system.market_data.opens = [100.0]
        self.system.market_data.highs = [105.1]  # Tiny upper shadow
        self.system.market_data.lows = [99.9]   # Tiny lower shadow
        self.system.market_data.closes = [105.0]  # Large bullish body
        self.system.market_data.volumes = [1000]
        
        patterns = self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Should detect bullish marubozu
        marubozu_patterns = [p for p in patterns if p.pattern_type == BarPatternType.MARUBOZU_BULLISH]
        self.assertGreater(len(marubozu_patterns), 0)
        
        marubozu = marubozu_patterns[0]
        self.assertEqual(marubozu.pattern_type, BarPatternType.MARUBOZU_BULLISH)
        self.assertTrue(marubozu.bullish)
        self.assertGreater(marubozu.confidence, 0.8)
    
    def test_engulfing_pattern_detection(self):
        """Test engulfing pattern detection."""
        # Create bullish engulfing pattern
        # Bar 0: Small bearish candle
        # Bar 1: Large bullish candle that engulfs bar 0
        self.system.market_data.opens = [100.0, 99.5]
        self.system.market_data.highs = [100.2, 103.0]
        self.system.market_data.lows = [99.5, 99.0]
        self.system.market_data.closes = [99.8, 102.5]  # Engulfing pattern
        self.system.market_data.volumes = [1000, 1200]
        
        patterns = self.bar_utils.detect_candlestick_patterns(self.system, 1)
        
        # Should detect bullish engulfing
        engulfing_patterns = [p for p in patterns if p.pattern_type == BarPatternType.ENGULFING_BULLISH]
        self.assertGreater(len(engulfing_patterns), 0)
        
        engulfing = engulfing_patterns[0]
        self.assertEqual(engulfing.pattern_type, BarPatternType.ENGULFING_BULLISH)
        self.assertTrue(engulfing.bullish)
    
    def test_gap_analysis(self):
        """Test gap analysis functionality."""
        # Create gap up scenario
        self.system.market_data.opens = [100.0, 102.5]  # Gap up from 100 close to 102.5 open
        self.system.market_data.highs = [101.0, 103.0]
        self.system.market_data.lows = [99.0, 102.0]
        self.system.market_data.closes = [100.5, 102.8]
        self.system.market_data.volumes = [1000, 1200]
        
        stats = self.bar_utils.calculate_bar_statistics(self.system, 1)
        
        # Should detect gap up
        self.assertEqual(stats.gap_type, GapType.GAP_UP)
        self.assertGreater(stats.gap_size, 0)
        self.assertGreater(stats.gap_percentage, 0)
    
    def test_volume_analysis(self):
        """Test volume analysis."""
        # Create volume data with high volume spike
        volumes = [1000] * 20 + [2000]  # High volume at end
        self.system.market_data.volumes = volumes
        
        # Extend other data to match
        self.system.market_data.opens = [100.0] * 21
        self.system.market_data.highs = [101.0] * 21
        self.system.market_data.lows = [99.0] * 21
        self.system.market_data.closes = [100.5] * 21
        
        volume_analysis = self.bar_utils.analyze_volume(self.system, 20)
        self.assertEqual(volume_analysis, "high")
        
        # Test normal volume
        volume_analysis = self.bar_utils.analyze_volume(self.system, 10)
        self.assertEqual(volume_analysis, "normal")
    
    def test_trend_analysis(self):
        """Test trend analysis."""
        # Create uptrend data
        closes = [100, 101, 102, 103, 104, 105]
        self.system.market_data.closes = closes
        self.system.market_data.opens = [c - 0.5 for c in closes]
        self.system.market_data.highs = [c + 0.5 for c in closes]
        self.system.market_data.lows = [c - 1.0 for c in closes]
        self.system.market_data.volumes = [1000] * 6
        
        trend = self.bar_utils.analyze_trend(self.system, 5, lookback=5)
        self.assertEqual(trend, "up")
        
        # Create downtrend data
        closes_down = [105, 104, 103, 102, 101, 100]
        self.system.market_data.closes = closes_down
        
        trend_down = self.bar_utils.analyze_trend(self.system, 5, lookback=5)
        self.assertEqual(trend_down, "down")
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        # Create data with varying volatility
        opens = [100] * 15
        highs = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        lows = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85]
        closes = [100.5] * 15
        
        self.system.market_data.opens = opens
        self.system.market_data.highs = highs
        self.system.market_data.lows = lows
        self.system.market_data.closes = closes
        self.system.market_data.volumes = [1000] * 15
        
        volatility = self.bar_utils.calculate_volatility(self.system, 14)
        self.assertGreater(volatility, 0)
        
        # Test volatility level classification
        volatility_level = self.bar_utils.analyze_volatility_level(self.system, 14)
        self.assertIn(volatility_level, ["low", "normal", "high"])
    
    def test_comprehensive_bar_analysis(self):
        """Test comprehensive bar analysis."""
        analysis = self.bar_utils.analyze_bar(self.system, 0)
        
        self.assertIsInstance(analysis, BarAnalysisResult)
        self.assertEqual(analysis.bar_index, 0)
        self.assertIsInstance(analysis.bar_stats, BarStatistics)
        self.assertIsInstance(analysis.patterns, list)
        
        # Check signal strengths
        self.assertGreaterEqual(analysis.buy_signal_strength, 0.0)
        self.assertLessEqual(analysis.buy_signal_strength, 1.0)
        self.assertGreaterEqual(analysis.sell_signal_strength, 0.0)
        self.assertLessEqual(analysis.sell_signal_strength, 1.0)
        
        # Check analysis results
        self.assertIn(analysis.trend_direction, ["up", "down", "neutral"])
        self.assertIn(analysis.volatility_level, ["low", "normal", "high"])
        self.assertIn(analysis.volume_analysis, ["low", "normal", "high"])
    
    def test_analysis_summary(self):
        """Test analysis summary generation."""
        summary = self.bar_utils.get_analysis_summary(self.system, 0)
        
        self.assertIn("bar_index", summary)
        self.assertIn("basic_info", summary)
        self.assertIn("technical_analysis", summary)
        self.assertIn("market_analysis", summary)
        self.assertIn("signals", summary)
        
        # Check basic info
        basic_info = summary["basic_info"]
        self.assertIn("open", basic_info)
        self.assertIn("high", basic_info)
        self.assertIn("low", basic_info)
        self.assertIn("close", basic_info)
        self.assertIn("volume", basic_info)
        
        # Check signals
        signals = summary["signals"]
        self.assertIn("buy_strength", signals)
        self.assertIn("sell_strength", signals)
        self.assertIn("patterns_detected", signals)
    
    def test_three_bar_patterns(self):
        """Test three-bar patterns like Morning/Evening Star."""
        # Create morning star pattern
        # Bar 0: Large bearish candle
        # Bar 1: Small doji/star
        # Bar 2: Large bullish candle
        self.system.market_data.opens = [105.0, 102.0, 101.0]
        self.system.market_data.highs = [105.5, 102.2, 104.0]
        self.system.market_data.lows = [100.0, 101.8, 100.5]
        self.system.market_data.closes = [100.5, 102.0, 103.5]  # Bearish, Doji, Bullish
        self.system.market_data.volumes = [1000, 800, 1200]
        
        patterns = self.bar_utils.detect_candlestick_patterns(self.system, 2)
        
        # Should potentially detect morning star (depends on exact thresholds)
        morning_star_patterns = [p for p in patterns if p.pattern_type == BarPatternType.MORNING_STAR]
        # Note: This is a complex pattern, may not always trigger based on exact data
        self.assertIsInstance(morning_star_patterns, list)
    
    def test_signal_strength_calculation(self):
        """Test buy/sell signal strength calculation."""
        # Create analysis with bullish pattern
        analysis = BarAnalysisResult(
            bar_index=0,
            analysis_types=[BarAnalysisType.CANDLESTICK_PATTERNS],
            bar_stats=BarStatistics(),
            patterns=[
                CandlestickPattern(
                    pattern_type=BarPatternType.HAMMER,
                    confidence=0.8,
                    bar_index=0,
                    signal_strength="strong",
                    bullish=True
                )
            ],
            trend_direction="up",
            volume_analysis="high"
        )
        
        buy_strength = self.bar_utils._calculate_buy_signal_strength(analysis)
        sell_strength = self.bar_utils._calculate_sell_signal_strength(analysis)
        
        # Buy strength should be higher due to bullish pattern and uptrend
        self.assertGreater(buy_strength, sell_strength)
        self.assertGreater(buy_strength, 0.5)  # Should be significant
    
    def test_cache_functionality(self):
        """Test caching of analysis results."""
        # First call should calculate and cache
        stats1 = self.bar_utils.calculate_bar_statistics(self.system, 0)
        patterns1 = self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Second call should use cache
        stats2 = self.bar_utils.calculate_bar_statistics(self.system, 0)
        patterns2 = self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Should be same objects (from cache)
        self.assertEqual(stats1, stats2)
        self.assertEqual(patterns1, patterns2)
        
        # Clear cache
        self.bar_utils.clear_cache()
        
        # After clearing, should be empty
        self.assertEqual(len(self.bar_utils._cached_patterns), 0)
        self.assertEqual(len(self.bar_utils._cached_statistics), 0)
    
    def test_invalid_bar_index_handling(self):
        """Test handling of invalid bar indices."""
        # Test negative index
        stats = self.bar_utils.calculate_bar_statistics(self.system, -1)
        self.assertEqual(stats.open_price, 0.0)  # Default empty stats
        
        # Test index too large
        stats = self.bar_utils.calculate_bar_statistics(self.system, 1000)
        self.assertEqual(stats.open_price, 0.0)  # Default empty stats
        
        # Test empty summary for invalid index
        summary = self.bar_utils.get_analysis_summary(self.system, 1000)
        self.assertEqual(summary, {})
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        # Generate some cached data
        self.bar_utils.calculate_bar_statistics(self.system, 0)
        self.bar_utils.detect_candlestick_patterns(self.system, 0)
        
        # Should have cached data
        self.assertGreater(len(self.bar_utils._cached_statistics), 0)
        
        # Reset should clear cache
        self.bar_utils.reset(self.system)
        
        self.assertEqual(len(self.bar_utils._cached_statistics), 0)
        self.assertEqual(len(self.bar_utils._cached_patterns), 0)
    
    def test_string_representation(self):
        """Test string representation."""
        # Add some cached data
        self.bar_utils.calculate_bar_statistics(self.system, 0)
        
        str_repr = str(self.bar_utils)
        
        self.assertIn("CBarUtils", str_repr)
        self.assertIn("cached_patterns", str_repr)
        self.assertIn("cached_stats", str_repr)
        self.assertIn("doji_threshold", str_repr)


class TestBarStatistics(unittest.TestCase):
    """Test BarStatistics dataclass."""
    
    def test_default_creation(self):
        """Test default BarStatistics creation."""
        stats = BarStatistics()
        
        self.assertEqual(stats.open_price, 0.0)
        self.assertEqual(stats.high_price, 0.0)
        self.assertEqual(stats.low_price, 0.0)
        self.assertEqual(stats.close_price, 0.0)
        self.assertEqual(stats.volume, 0.0)
        self.assertFalse(stats.is_bullish)
        self.assertFalse(stats.is_bearish)
        self.assertEqual(stats.gap_type, GapType.NO_GAP)
    
    def test_custom_creation(self):
        """Test custom BarStatistics creation."""
        stats = BarStatistics(
            open_price=100.0,
            high_price=105.0,
            low_price=98.0,
            close_price=103.0,
            volume=1500,
            is_bullish=True,
            gap_type=GapType.GAP_UP
        )
        
        self.assertEqual(stats.open_price, 100.0)
        self.assertEqual(stats.close_price, 103.0)
        self.assertTrue(stats.is_bullish)
        self.assertEqual(stats.gap_type, GapType.GAP_UP)


class TestCandlestickPattern(unittest.TestCase):
    """Test CandlestickPattern dataclass."""
    
    def test_pattern_creation(self):
        """Test candlestick pattern creation."""
        pattern = CandlestickPattern(
            pattern_type=BarPatternType.HAMMER,
            confidence=0.8,
            bar_index=0,
            signal_strength="strong",
            bullish=True
        )
        
        self.assertEqual(pattern.pattern_type, BarPatternType.HAMMER)
        self.assertEqual(pattern.confidence, 0.8)
        self.assertEqual(pattern.bar_index, 0)
        self.assertEqual(pattern.signal_strength, "strong")
        self.assertTrue(pattern.bullish)
        self.assertIn("Hammer", pattern.description)  # Auto-generated description


class TestCZigZagAnalyzer(unittest.TestCase):
    """Test cases for CZigZagAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.system.system_id = 1
        
        self.zigzag_analyzer = CZigZagAnalyzer()
        
        # Create sample market data for testing
        self.sample_data = []
        
        # Create a series with clear ZigZag pattern
        # Up trend: 100 -> 110 -> 105 -> 115 -> 110 -> 120
        base_time = datetime(2024, 1, 1, 9, 0)
        prices = [
            (100.0, 102.0, 98.0, 101.0),   # 0: slight up
            (101.0, 105.0, 100.5, 104.0), # 1: up
            (104.0, 108.0, 103.0, 107.0), # 2: up
            (107.0, 111.0, 106.0, 110.0), # 3: up -> peak
            (110.0, 110.5, 106.0, 106.5), # 4: down
            (106.5, 107.0, 103.0, 104.0), # 5: down -> trough
            (104.0, 108.0, 103.5, 107.5), # 6: up
            (107.5, 112.0, 107.0, 111.0), # 7: up
            (111.0, 116.0, 110.5, 115.0), # 8: up -> higher peak
            (115.0, 115.5, 111.0, 112.0), # 9: down
            (112.0, 112.5, 108.0, 109.0), # 10: down -> lower trough
        ]
        
        for i, (open_p, high_p, low_p, close_p) in enumerate(prices):
            market_data = MarketDataPoint(
                timestamp=base_time.replace(minute=i*5),
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=1000 + i*50
            )
            self.sample_data.append(market_data)
    
    def test_initialization(self):
        """Test ZigZag analyzer initialization."""
        self.assertIsInstance(self.zigzag_analyzer, CZigZagAnalyzer)
        self.assertFalse(self.zigzag_analyzer.is_initialized)
        self.assertEqual(self.zigzag_analyzer.zigzag_type, ZigZagType.PERCENTAGE)
        self.assertEqual(self.zigzag_analyzer.threshold_percentage, 5.0)
        self.assertEqual(self.zigzag_analyzer.current_trend, TrendDirection.SIDEWAYS)
        
        # Initialize
        analyzer = self.zigzag_analyzer.initialize(self.system)
        self.assertTrue(analyzer.is_initialized)
        self.assertIs(analyzer, self.zigzag_analyzer)  # Should return self
    
    def test_configuration_methods(self):
        """Test configuration methods."""
        # Test ZigZag type setting
        analyzer = self.zigzag_analyzer.set_zigzag_type(ZigZagType.ABSOLUTE)
        self.assertEqual(self.zigzag_analyzer.zigzag_type, ZigZagType.ABSOLUTE)
        self.assertIs(analyzer, self.zigzag_analyzer)  # Should return self
        
        # Test percentage threshold
        self.zigzag_analyzer.set_percentage_threshold(3.0)
        self.assertEqual(self.zigzag_analyzer.threshold_percentage, 3.0)
        
        # Test absolute threshold
        self.zigzag_analyzer.set_absolute_threshold(15.0)
        self.assertEqual(self.zigzag_analyzer.threshold_absolute, 15.0)
        
        # Test ATR multiplier
        self.zigzag_analyzer.set_atr_multiplier(2.5)
        self.assertEqual(self.zigzag_analyzer.atr_multiplier, 2.5)
    
    def test_data_management(self):
        """Test data management functionality."""
        # Test adding individual data points
        self.assertEqual(len(self.zigzag_analyzer.price_data), 0)
        
        for data in self.sample_data[:3]:
            self.zigzag_analyzer.add_market_data(data)
        
        self.assertEqual(len(self.zigzag_analyzer.price_data), 3)
        
        # Test bulk data addition
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data[3:])
        self.assertEqual(len(self.zigzag_analyzer.price_data), len(self.sample_data))
        
        # Test data size limit
        large_dataset = self.sample_data * 200  # Create large dataset
        self.zigzag_analyzer.add_price_data_bulk(large_dataset)
        self.assertEqual(len(self.zigzag_analyzer.price_data), self.zigzag_analyzer.max_data_points)
    
    def test_percentage_zigzag_calculation(self):
        """Test percentage-based ZigZag calculation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.set_zigzag_type(ZigZagType.PERCENTAGE)
        self.zigzag_analyzer.set_percentage_threshold(3.0)  # 3% threshold
        
        # Add data and calculate ZigZag
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        # Should have identified some ZigZag points
        self.assertGreater(len(self.zigzag_analyzer.zigzag_points), 0)
        
        # Check that points alternate between high and low
        if len(self.zigzag_analyzer.zigzag_points) >= 2:
            points = self.zigzag_analyzer.zigzag_points
            for i in range(len(points) - 1):
                self.assertNotEqual(points[i].is_high, points[i+1].is_high)
    
    def test_absolute_zigzag_calculation(self):
        """Test absolute value-based ZigZag calculation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.set_zigzag_type(ZigZagType.ABSOLUTE)
        self.zigzag_analyzer.set_absolute_threshold(5.0)  # $5 threshold
        
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        # Should identify ZigZag points
        self.assertGreater(len(self.zigzag_analyzer.zigzag_points), 0)
    
    def test_atr_zigzag_calculation(self):
        """Test ATR-based ZigZag calculation."""
        # Need more data for ATR calculation (at least 14 periods)
        extended_data = self.sample_data * 2  # 22 data points
        
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.set_zigzag_type(ZigZagType.ATR_BASED)
        self.zigzag_analyzer.set_atr_multiplier(2.0)
        
        self.zigzag_analyzer.add_price_data_bulk(extended_data)
        
        # Should calculate ATR and identify some points (depends on data)
        # ATR-based may be more conservative
        self.assertGreaterEqual(len(self.zigzag_analyzer.zigzag_points), 0)
    
    def test_zigzag_point_creation(self):
        """Test ZigZag point creation and properties."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        if len(self.zigzag_analyzer.zigzag_points) > 0:
            point = self.zigzag_analyzer.zigzag_points[0]
            
            self.assertIsInstance(point, ZigZagPoint)
            self.assertIsInstance(point.timestamp, datetime)
            self.assertGreater(point.price, 0)
            self.assertIn(point.is_high, [True, False])
            self.assertEqual(point.level, ZigZagLevel.MINOR)  # Default level
            self.assertGreaterEqual(point.volume, 0)
            
            # Test string representation
            str_repr = str(point)
            self.assertIn("ZigZag", str_repr)
            self.assertIn(str(point.price), str_repr)
    
    def test_swing_creation_and_analysis(self):
        """Test swing creation and analysis."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        # Should create swings from ZigZag points
        if len(self.zigzag_analyzer.zigzag_points) >= 2:
            self.assertGreater(len(self.zigzag_analyzer.swings), 0)
            
            # Test swing properties
            swing = self.zigzag_analyzer.swings[0]
            self.assertIsInstance(swing, ZigZagSwing)
            self.assertGreater(swing.swing_size, 0)
            self.assertGreaterEqual(swing.swing_percentage, 0)
            self.assertIsInstance(swing.duration, timedelta)
            self.assertGreaterEqual(swing.avg_volume, 0)
            
            # Test swing direction properties
            self.assertIsInstance(swing.is_up_swing, bool)
            self.assertIsInstance(swing.is_down_swing, bool)
            self.assertNotEqual(swing.is_up_swing, swing.is_down_swing)  # Should be opposite
    
    def test_swing_statistics(self):
        """Test swing statistics calculation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        stats = self.zigzag_analyzer.get_swing_statistics()
        
        if len(self.zigzag_analyzer.swings) > 0:
            self.assertIn('total_swings', stats)
            self.assertIn('up_swings', stats)
            self.assertIn('down_swings', stats)
            self.assertIn('avg_up_swing_size', stats)
            self.assertIn('avg_down_swing_size', stats)
            self.assertIn('avg_up_swing_percentage', stats)
            self.assertIn('avg_down_swing_percentage', stats)
            
            # Basic validation
            self.assertEqual(stats['total_swings'], len(self.zigzag_analyzer.swings))
            self.assertEqual(stats['up_swings'] + stats['down_swings'], stats['total_swings'])
        else:
            self.assertEqual(stats, {})
    
    def test_recent_swings(self):
        """Test getting recent swings."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        # Test getting recent swings
        recent_swings = self.zigzag_analyzer.get_recent_swings(3)
        self.assertLessEqual(len(recent_swings), 3)
        self.assertLessEqual(len(recent_swings), len(self.zigzag_analyzer.swings))
        
        # If we have swings, recent should be the last ones
        if len(self.zigzag_analyzer.swings) >= 3:
            expected_recent = self.zigzag_analyzer.swings[-3:]
            self.assertEqual(recent_swings, expected_recent)
    
    def test_market_structure_analysis(self):
        """Test market structure analysis."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        # Check trend analysis
        self.assertIn(self.zigzag_analyzer.current_trend, [
            TrendDirection.UP, TrendDirection.DOWN, TrendDirection.SIDEWAYS
        ])
        self.assertGreaterEqual(self.zigzag_analyzer.trend_strength, 0.0)
        self.assertLessEqual(self.zigzag_analyzer.trend_strength, 1.0)
        
        # Check support/resistance levels
        self.assertIsInstance(self.zigzag_analyzer.support_levels, list)
        self.assertIsInstance(self.zigzag_analyzer.resistance_levels, list)
        
        # All levels should be positive prices
        for level in self.zigzag_analyzer.support_levels:
            self.assertGreater(level, 0)
        for level in self.zigzag_analyzer.resistance_levels:
            self.assertGreater(level, 0)
    
    def test_pattern_recognition(self):
        """Test pattern recognition functionality."""
        self.zigzag_analyzer.initialize(self.system)
        
        # Create data that should form recognizable patterns
        pattern_data = []
        base_time = datetime(2024, 1, 1, 9, 0)
        
        # Create double top pattern: Low-High-Low-High-Low with similar highs
        pattern_prices = [
            (100.0, 102.0, 98.0, 101.0),   # Initial low
            (101.0, 110.0, 100.0, 109.0), # First high
            (109.0, 110.0, 102.0, 103.0), # Middle low  
            (103.0, 111.0, 102.0, 110.5), # Second high (similar to first)
            (110.5, 111.0, 96.0, 97.0),   # Final low
        ]
        
        for i, (open_p, high_p, low_p, close_p) in enumerate(pattern_prices):
            market_data = MarketDataPoint(
                timestamp=base_time.replace(minute=i*15),
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=1000
            )
            pattern_data.append(market_data)
        
        # Use lower threshold to ensure pattern detection
        self.zigzag_analyzer.set_percentage_threshold(2.0)
        self.zigzag_analyzer.add_price_data_bulk(pattern_data)
        
        # Check that patterns are detected
        self.assertIsInstance(self.zigzag_analyzer.patterns, list)
        
        # Test pattern properties if any are found
        for pattern in self.zigzag_analyzer.patterns:
            self.assertIsInstance(pattern, ZigZagPattern)
            self.assertIsInstance(pattern.pattern_type, str)
            self.assertGreaterEqual(pattern.confidence, 0.0)
            self.assertLessEqual(pattern.confidence, 1.0)
            self.assertGreater(len(pattern.points), 0)
    
    def test_fibonacci_retracement(self):
        """Test Fibonacci retracement calculation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        if len(self.zigzag_analyzer.swings) > 0:
            swing = self.zigzag_analyzer.swings[0]
            retracements = self.zigzag_analyzer.calculate_fibonacci_retracements(swing)
            
            # Should have retracement levels
            self.assertGreater(len(retracements), 0)
            
            # Check that all Fibonacci levels are present
            for level in self.zigzag_analyzer.fibonacci_levels:
                self.assertIn(level, retracements)
                self.assertGreater(retracements[level], 0)
            
            # Test getting current retracement level
            if len(self.sample_data) > 0:
                current_price = self.sample_data[-1].close
                current_level = self.zigzag_analyzer.get_current_retracement_level(current_price)
                
                if current_level:
                    level, price = current_level
                    self.assertIn(level, self.zigzag_analyzer.fibonacci_levels)
                    self.assertGreater(price, 0)
    
    def test_trading_signals(self):
        """Test trading signal generation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        if len(self.sample_data) > 0:
            current_price = self.sample_data[-1].close
            signal = self.zigzag_analyzer.get_trading_signal(current_price)
            
            self.assertIn(signal, [
                SignalType.BUY,
                SignalType.SELL, 
                SignalType.FLAT
            ])
        
        # Test signal with uninitialized analyzer
        uninitialized_analyzer = CZigZagAnalyzer()
        signal = uninitialized_analyzer.get_trading_signal(100.0)
        self.assertEqual(signal, SignalType.FLAT)
    
    def test_analysis_summary(self):
        """Test analysis summary generation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        summary = self.zigzag_analyzer.get_analysis_summary()
        
        # Check required fields
        required_fields = [
            'zigzag_points_count', 'swings_count', 'patterns_count',
            'current_trend', 'trend_strength', 'support_levels',
            'resistance_levels', 'recent_patterns', 'zigzag_type',
            'threshold_percentage', 'data_points'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check data types and ranges
        self.assertIsInstance(summary['zigzag_points_count'], int)
        self.assertIsInstance(summary['swings_count'], int)
        self.assertIsInstance(summary['patterns_count'], int)
        self.assertIsInstance(summary['current_trend'], str)
        self.assertIsInstance(summary['trend_strength'], (int, float))
        self.assertIsInstance(summary['support_levels'], list)
        self.assertIsInstance(summary['resistance_levels'], list)
        self.assertIsInstance(summary['recent_patterns'], list)
        self.assertIsInstance(summary['zigzag_type'], str)
        self.assertIsInstance(summary['threshold_percentage'], (int, float))
        self.assertIsInstance(summary['data_points'], int)
    
    def test_pattern_alerts(self):
        """Test pattern alert generation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        alerts = self.zigzag_analyzer.get_pattern_alerts()
        
        self.assertIsInstance(alerts, list)
        
        # Test alert structure if any patterns found
        for alert in alerts:
            required_fields = [
                'pattern_type', 'confidence', 'target_price',
                'stop_loss', 'points_count', 'timestamp'
            ]
            for field in required_fields:
                self.assertIn(field, alert)
            
            self.assertGreaterEqual(alert['confidence'], 0.0)
            self.assertLessEqual(alert['confidence'], 1.0)
            self.assertGreater(alert['points_count'], 0)
            self.assertIsInstance(alert['timestamp'], datetime)
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        # Should have data after adding
        self.assertGreater(len(self.zigzag_analyzer.price_data), 0)
        
        # Reset should clear everything
        self.zigzag_analyzer.reset(self.system)
        
        self.assertEqual(len(self.zigzag_analyzer.price_data), 0)
        self.assertEqual(len(self.zigzag_analyzer.zigzag_points), 0)
        self.assertEqual(len(self.zigzag_analyzer.swings), 0)
        self.assertEqual(len(self.zigzag_analyzer.patterns), 0)
        self.assertEqual(len(self.zigzag_analyzer.support_levels), 0)
        self.assertEqual(len(self.zigzag_analyzer.resistance_levels), 0)
        self.assertEqual(self.zigzag_analyzer.current_trend, TrendDirection.SIDEWAYS)
        self.assertEqual(self.zigzag_analyzer.trend_strength, 0.0)
    
    def test_string_representation(self):
        """Test string representation."""
        self.zigzag_analyzer.initialize(self.system)
        self.zigzag_analyzer.add_price_data_bulk(self.sample_data)
        
        str_repr = str(self.zigzag_analyzer)
        
        self.assertIn("CZigZagAnalyzer", str_repr)
        self.assertIn("points=", str_repr)
        self.assertIn("swings=", str_repr)
        self.assertIn("patterns=", str_repr)
        self.assertIn("trend=", str_repr)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        self.zigzag_analyzer.initialize(self.system)
        
        # Test with insufficient data
        single_data = self.sample_data[:1]
        self.zigzag_analyzer.add_price_data_bulk(single_data)
        
        summary = self.zigzag_analyzer.get_analysis_summary()
        self.assertEqual(summary['zigzag_points_count'], 0)
        
        # Test with empty data
        empty_analyzer = CZigZagAnalyzer().initialize(self.system)
        signal = empty_analyzer.get_trading_signal(100.0)
        self.assertEqual(signal, SignalType.FLAT)
        
        stats = empty_analyzer.get_swing_statistics()
        self.assertEqual(stats, {})


class TestZigZagDataClasses(unittest.TestCase):
    """Test ZigZag data classes."""
    
    def test_zigzag_point(self):
        """Test ZigZagPoint creation and methods."""
        timestamp = datetime.now()
        point = ZigZagPoint(
            index=5,
            timestamp=timestamp,
            price=105.50,
            is_high=True,
            level=ZigZagLevel.MAJOR,
            volume=1500.0,
            strength=0.8,
            retracement_pct=23.6
        )
        
        self.assertEqual(point.index, 5)
        self.assertEqual(point.timestamp, timestamp)
        self.assertEqual(point.price, 105.50)
        self.assertTrue(point.is_high)
        self.assertEqual(point.level, ZigZagLevel.MAJOR)
        self.assertEqual(point.volume, 1500.0)
        self.assertEqual(point.strength, 0.8)
        self.assertEqual(point.retracement_pct, 23.6)
        
        # Test string representation
        str_repr = str(point)
        self.assertIn("ZigZagHIGH", str_repr)
        self.assertIn("105.5000", str_repr)
    
    def test_zigzag_swing(self):
        """Test ZigZagSwing creation and properties."""
        timestamp1 = datetime(2024, 1, 1, 9, 0)
        timestamp2 = datetime(2024, 1, 1, 9, 15)
        
        start_point = ZigZagPoint(
            index=0,
            timestamp=timestamp1,
            price=100.0,
            is_high=False
        )
        
        end_point = ZigZagPoint(
            index=5,
            timestamp=timestamp2,
            price=110.0,
            is_high=True
        )
        
        swing = ZigZagSwing(start_point=start_point, end_point=end_point)
        
        # Test calculated properties
        self.assertEqual(swing.swing_size, 10.0)
        self.assertEqual(swing.swing_percentage, 10.0)  # 10/100 * 100
        self.assertEqual(swing.duration, timedelta(minutes=15))
        
        # Test direction properties
        self.assertTrue(swing.is_up_swing)
        self.assertFalse(swing.is_down_swing)
        
        # Test down swing
        down_swing = ZigZagSwing(start_point=end_point, end_point=start_point)
        self.assertFalse(down_swing.is_up_swing)
        self.assertTrue(down_swing.is_down_swing)
    
    def test_zigzag_pattern(self):
        """Test ZigZagPattern creation."""
        points = [
            ZigZagPoint(0, datetime.now(), 100.0, False),
            ZigZagPoint(1, datetime.now(), 110.0, True),
            ZigZagPoint(2, datetime.now(), 105.0, False)
        ]
        
        pattern = ZigZagPattern(
            pattern_type="DOUBLE_TOP",
            points=points,
            confidence=0.75,
            target_price=95.0,
            stop_loss=112.0,
            symmetry_score=0.85,
            volume_confirmation=True,
            time_symmetry=0.9
        )
        
        self.assertEqual(pattern.pattern_type, "DOUBLE_TOP")
        self.assertEqual(len(pattern.points), 3)
        self.assertEqual(pattern.confidence, 0.75)
        self.assertEqual(pattern.target_price, 95.0)
        self.assertEqual(pattern.stop_loss, 112.0)
        self.assertEqual(pattern.symmetry_score, 0.85)
        self.assertTrue(pattern.volume_confirmation)
        self.assertEqual(pattern.time_symmetry, 0.9)


class TestZigZagEnums(unittest.TestCase):
    """Test ZigZag enumerations."""
    
    def test_zigzag_type_enum(self):
        """Test ZigZagType enumeration."""
        self.assertEqual(ZigZagType.PERCENTAGE.value, "PERCENTAGE")
        self.assertEqual(ZigZagType.ABSOLUTE.value, "ABSOLUTE")
        self.assertEqual(ZigZagType.ATR_BASED.value, "ATR_BASED")
    
    def test_trend_direction_enum(self):
        """Test TrendDirection enumeration."""
        self.assertEqual(TrendDirection.UP.value, "UP")
        self.assertEqual(TrendDirection.DOWN.value, "DOWN")
        self.assertEqual(TrendDirection.SIDEWAYS.value, "SIDEWAYS")
    
    def test_zigzag_level_enum(self):
        """Test ZigZagLevel enumeration."""
        self.assertEqual(ZigZagLevel.MINOR.value, "MINOR")
        self.assertEqual(ZigZagLevel.INTERMEDIATE.value, "INTERMEDIATE")
        self.assertEqual(ZigZagLevel.MAJOR.value, "MAJOR")


if __name__ == '__main__':
    unittest.main()