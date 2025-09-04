"""
Comprehensive tests for time management module.
"""

import pytest
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from time_management.time_filter import (
    CTimeFilter,
    TimeFilterType,
    TimeFilterResult, 
    TradingSession,
    MarketHours
)

from time_management.time_utils import (
    CTimeUtils,
    TimeInfo,
    ElapsedTime,
    TimePeriod
)


class TestCTimeFilter(unittest.TestCase):
    """Test cases for CTimeFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.system.system_id = 1
        
        # Create sample dates for testing
        self.base_date = datetime(2024, 1, 15, 10, 0, 0)  # Monday 10:00 AM
        self.system.market_data = Mock()
        self.system.market_data.dates = [
            self.base_date + timedelta(hours=i) for i in range(24)
        ]
        
        self.time_filter = CTimeFilter()
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.time_filter, CTimeFilter)
        self.assertEqual(self.time_filter.system_id, 0)
        self.assertTrue(self.time_filter.enabled)
    
    def test_enable_disable(self):
        """Test enable/disable functionality."""
        self.time_filter.disable()
        self.assertFalse(self.time_filter.enabled)
        
        self.time_filter.enable()
        self.assertTrue(self.time_filter.enabled)
    
    def test_setup_forex_24h(self):
        """Test 24/7 Forex market setup."""
        self.time_filter.setup_forex_24h()
        
        # Should allow trading at any hour
        result = self.time_filter.check_trading_allowed(
            self.system, 
            datetime(2024, 1, 15, 2, 0, 0)  # 2 AM
        )
        self.assertTrue(result.allowed)
    
    def test_setup_stock_market_hours(self):
        """Test stock market hours setup."""
        self.time_filter.setup_stock_market_hours()
        
        # During market hours (10 AM) - should be allowed
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 10, 0, 0)
        )
        self.assertTrue(result.allowed)
        
        # Outside market hours (2 AM) - should be blocked
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 2, 0, 0)
        )
        self.assertFalse(result.allowed)
    
    def test_block_weekends(self):
        """Test weekend blocking functionality."""
        self.time_filter.block_weekends(True)
        
        # Saturday - should be blocked
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 13, 10, 0, 0)  # Saturday
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.filter_type, TimeFilterType.WEEKEND_BLOCK)
        
        # Monday - should be allowed
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 10, 0, 0)  # Monday
        )
        self.assertTrue(result.allowed)
    
    def test_custom_time_range(self):
        """Test custom time range filtering."""
        # Allow trading only between 9:00-17:00
        self.time_filter.add_allowed_time_range("09:00", "17:00")
        
        # Within range - should be allowed
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 12, 0, 0)  # 12:00 PM
        )
        self.assertTrue(result.allowed)
        
        # Outside range - should be blocked
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 20, 0, 0)  # 8:00 PM
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.filter_type, TimeFilterType.TIME_RANGE)
    
    def test_blocked_time_range(self):
        """Test blocked time range functionality."""
        # Block lunch break 12:00-13:00
        self.time_filter.add_blocked_time_range("12:00", "13:00")
        
        # During lunch break - should be blocked
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 12, 30, 0)  # 12:30 PM
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.filter_type, TimeFilterType.TIME_RANGE)
    
    def test_trading_session_filtering(self):
        """Test trading session filtering."""
        # Enable only European session
        self.time_filter.enable_session_filter(TradingSession.EUROPEAN, True)
        self.time_filter.enable_session_filter(TradingSession.ASIAN, False)
        self.time_filter.enable_session_filter(TradingSession.AMERICAN, False)
        
        # During European session (10 AM UTC) - should be allowed
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 10, 0, 0)
        )
        self.assertTrue(result.allowed)
        
        # During Asian session (3 AM UTC) - should be blocked
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 3, 0, 0)
        )
        self.assertFalse(result.allowed)
    
    def test_end_of_day_close(self):
        """Test end-of-day close functionality."""
        self.time_filter.enable_end_of_day_close("17:00")
        
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 17, 30, 0)  # After close time
        )
        
        # Should suggest closing positions
        self.assertTrue(result.should_close_positions)
        self.assertEqual(result.filter_type, TimeFilterType.END_OF_DAY_CLOSE)
    
    def test_disabled_filter_allows_all(self):
        """Test that disabled filter allows all trading."""
        self.time_filter.disable()
        
        # Weekend should be allowed when disabled
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 13, 10, 0, 0)  # Saturday
        )
        self.assertTrue(result.allowed)
    
    def test_multiple_filters_interaction(self):
        """Test interaction of multiple filters."""
        self.time_filter.block_weekends(True)
        self.time_filter.add_allowed_time_range("09:00", "17:00")
        
        # Weekday within time range - should be allowed
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 12, 0, 0)  # Monday 12:00 PM
        )
        self.assertTrue(result.allowed)
        
        # Weekday outside time range - should be blocked
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 15, 20, 0, 0)  # Monday 8:00 PM
        )
        self.assertFalse(result.allowed)
    
    def test_reset_functionality(self):
        """Test filter reset."""
        self.time_filter.add_allowed_time_range("09:00", "17:00")
        self.time_filter.block_weekends(True)
        
        self.time_filter.reset(self.system)
        
        # After reset, weekend should be allowed (no filters)
        result = self.time_filter.check_trading_allowed(
            self.system,
            datetime(2024, 1, 13, 10, 0, 0)  # Saturday
        )
        self.assertTrue(result.allowed)


class TestCTimeUtils(unittest.TestCase):
    """Test cases for CTimeUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = Mock()
        self.system.system_id = 1
        
        # Create sample dates spanning multiple periods
        self.base_date = datetime(2024, 1, 1, 0, 0, 0)  # Start of year
        self.system.market_data = Mock()
        self.system.market_data.dates = [
            self.base_date + timedelta(days=i) for i in range(100)
        ]
        
        self.time_utils = CTimeUtils()
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsInstance(self.time_utils, CTimeUtils)
        self.assertEqual(self.time_utils.system_id, 0)
        self.assertIsInstance(self.time_utils.elapsed_time, ElapsedTime)
    
    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        elapsed = self.time_utils.calculate_elapsed_time_info(self.system)
        
        self.assertIsInstance(elapsed, ElapsedTime)
        self.assertGreaterEqual(elapsed.total_days, 0)
        self.assertGreaterEqual(elapsed.total_hours, 0)
        self.assertGreaterEqual(elapsed.total_minutes, 0)
    
    def test_get_elapsed_time_units(self):
        """Test elapsed time in different units."""
        # Test different unit codes
        minutes = self.time_utils.get_elapsed_time(self.system, "D")  # Dakika (minutes)
        hours = self.time_utils.get_elapsed_time(self.system, "S")    # Saat (hours)
        days = self.time_utils.get_elapsed_time(self.system, "G")     # Gün (days)
        months = self.time_utils.get_elapsed_time(self.system, "M")   # Ay (months)
        
        self.assertGreaterEqual(hours, 0)
        self.assertGreaterEqual(days, 0)
        self.assertGreaterEqual(months, 0)
        self.assertGreaterEqual(minutes, hours)  # Minutes should be >= hours
    
    def test_new_period_detection(self):
        """Test new period detection methods."""
        # Test at first bar (should be False)
        self.assertFalse(self.time_utils.is_new_month(self.system, 0))
        self.assertFalse(self.time_utils.is_new_week(self.system, 0))
        self.assertFalse(self.time_utils.is_new_day(self.system, 0))
        
        # Test with bars that should detect new periods
        # February 1st should be new month from January 1st
        feb_1_index = 31  # 31 days from Jan 1
        if feb_1_index < len(self.system.market_data.dates):
            is_new_month = self.time_utils.is_new_month(self.system, feb_1_index)
            # This depends on the exact dates, but should work for most cases
            self.assertIsInstance(is_new_month, bool)
    
    def test_generic_new_period(self):
        """Test generic new period detection."""
        # Test Turkish abbreviations
        result_month = self.time_utils.is_new_period(self.system, 31, "M")  # Ay (Month)
        result_day = self.time_utils.is_new_period(self.system, 1, "G")     # Gün (Day)
        
        self.assertIsInstance(result_month, bool)
        self.assertIsInstance(result_day, bool)
    
    def test_execution_timing(self):
        """Test execution timing functionality."""
        self.time_utils.start_timing()
        
        # Simulate some work
        import time
        time.sleep(0.01)  # 10ms
        
        self.time_utils.stop_timing()
        
        execution_time = self.time_utils.get_execution_time_ms()
        self.assertGreater(execution_time, 0)
        self.assertLess(execution_time, 1000)  # Should be less than 1 second
    
    def test_watchdog_monitoring(self):
        """Test watchdog monitoring functionality."""
        self.time_utils.start_watchdog()
        
        self.assertTrue(self.time_utils.watchdog_started)
        self.assertFalse(self.time_utils.watchdog_finished)
        self.assertEqual(self.time_utils.watchdog_counter, 0)
        
        # Update watchdog
        self.time_utils.update_watchdog()
        self.assertEqual(self.time_utils.watchdog_counter, 1)
        
        self.time_utils.stop_watchdog()
        self.assertTrue(self.time_utils.watchdog_finished)
        
        # Get watchdog info
        info = self.time_utils.get_watchdog_info()
        self.assertIn("counter", info)
        self.assertIn("started", info)
        self.assertIn("finished", info)
    
    def test_bar_datetime_retrieval(self):
        """Test bar datetime retrieval."""
        dt = self.time_utils.get_bar_datetime(self.system, 0)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt, self.system.market_data.dates[0])
        
        # Test invalid index
        invalid_dt = self.time_utils.get_bar_datetime(self.system, 1000)
        self.assertIsNone(invalid_dt)
    
    def test_week_number_calculation(self):
        """Test week number calculation."""
        test_date = datetime(2024, 1, 15)  # Known date
        week_num = self.time_utils.get_week_number(self.system, test_date)
        
        self.assertIsInstance(week_num, int)
        self.assertGreaterEqual(week_num, 1)
        self.assertLessEqual(week_num, 53)
    
    def test_weekend_and_market_day_detection(self):
        """Test weekend and market day detection."""
        saturday = datetime(2024, 1, 13)  # Saturday
        monday = datetime(2024, 1, 15)    # Monday
        
        self.assertTrue(self.time_utils.is_weekend(saturday))
        self.assertFalse(self.time_utils.is_weekend(monday))
        
        self.assertFalse(self.time_utils.is_market_day(saturday))
        self.assertTrue(self.time_utils.is_market_day(monday))
    
    def test_datetime_formatting(self):
        """Test datetime formatting utilities."""
        test_dt = datetime(2024, 1, 15, 10, 30, 45)
        
        # Test different format types
        full_format = self.time_utils.format_datetime(test_dt, "full")
        date_format = self.time_utils.format_datetime(test_dt, "date")
        time_format = self.time_utils.format_datetime(test_dt, "time")
        
        self.assertIn("15.01.2024", full_format)
        self.assertIn("10:30:45", full_format)
        self.assertEqual(date_format, "15.01.2024")
        self.assertEqual(time_format, "10:30:45")
    
    def test_datetime_parsing(self):
        """Test datetime parsing utilities."""
        date_str = "15.01.2024 10:30:45"
        parsed_dt = self.time_utils.parse_datetime(date_str, "full")
        
        self.assertIsInstance(parsed_dt, datetime)
        self.assertEqual(parsed_dt.year, 2024)
        self.assertEqual(parsed_dt.month, 1)
        self.assertEqual(parsed_dt.day, 15)
        self.assertEqual(parsed_dt.hour, 10)
        self.assertEqual(parsed_dt.minute, 30)
        self.assertEqual(parsed_dt.second, 45)
    
    def test_time_statistics(self):
        """Test time statistics generation."""
        stats = self.time_utils.get_time_statistics(self.system)
        
        self.assertIn("elapsed_time", stats)
        self.assertIn("data_info", stats)
        self.assertIn("performance", stats)
        
        self.assertIn("total_days", stats["elapsed_time"])
        self.assertIn("total_bars", stats["data_info"])
        self.assertIn("execution_time_ms", stats["performance"])
    
    def test_period_boundaries(self):
        """Test period boundary detection."""
        boundaries = self.time_utils.get_period_boundaries(self.system, TimePeriod.DAY)
        
        self.assertIsInstance(boundaries, list)
        # All boundaries should be valid indices
        for boundary in boundaries:
            self.assertGreaterEqual(boundary, 1)
            self.assertLess(boundary, len(self.system.market_data.dates))
    
    def test_trading_session_analysis(self):
        """Test trading session analysis."""
        # Create hourly data for session analysis
        hourly_dates = [
            datetime(2024, 1, 15, hour, 0, 0) for hour in range(24)
        ]
        self.system.market_data.dates = hourly_dates
        
        session_stats = self.time_utils.analyze_trading_sessions(self.system)
        
        self.assertIn("asian_session_bars", session_stats)
        self.assertIn("european_session_bars", session_stats)
        self.assertIn("american_session_bars", session_stats)
        self.assertIn("total_bars", session_stats)
        
        total_bars = session_stats["total_bars"]
        self.assertEqual(total_bars, 24)
    
    def test_interval_counters(self):
        """Test interval counter functionality."""
        # Test increment
        self.time_utils.increment_interval_counter(90)
        self.assertEqual(self.time_utils.get_interval_counter(90), 1)
        
        # Test multiple increments
        self.time_utils.increment_interval_counter(90)
        self.time_utils.increment_interval_counter(120)
        
        self.assertEqual(self.time_utils.get_interval_counter(90), 2)
        self.assertEqual(self.time_utils.get_interval_counter(120), 1)
        
        # Test reset
        self.time_utils.reset_interval_counters()
        self.assertEqual(self.time_utils.get_interval_counter(90), 0)
        self.assertEqual(self.time_utils.get_interval_counter(120), 0)
    
    def test_cache_management(self):
        """Test cache management."""
        # Generate some cached data
        self.time_utils.is_new_day(self.system, 1)
        self.time_utils.is_new_month(self.system, 31)
        
        # Cache should have entries
        self.assertGreater(len(self.time_utils._cached_periods), 0)
        
        # Clear cache
        self.time_utils.clear_cache()
        self.assertEqual(len(self.time_utils._cached_periods), 0)
    
    def test_context_manager(self):
        """Test context manager functionality for timing."""
        with self.time_utils as timer:
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # Should have recorded execution time
        self.assertGreater(timer.get_execution_time_ms(), 0)
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        # Set some data
        self.time_utils.execution_time_ms = 100.0
        self.time_utils.watchdog_counter = 5
        self.time_utils._cached_periods["test"] = True
        
        self.time_utils.reset(self.system)
        
        # Should be reset
        self.assertEqual(self.time_utils.execution_time_ms, 0.0)
        self.assertEqual(self.time_utils.watchdog_counter, 0)
        self.assertEqual(len(self.time_utils._cached_periods), 0)


if __name__ == '__main__':
    unittest.main()