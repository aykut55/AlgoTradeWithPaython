"""
Advanced time utilities for trading systems.

This module provides comprehensive time analysis and management functions:
- Elapsed time calculations
- New period detection (month, week, day, hour)
- Performance timing and benchmarks
- Execution time tracking
- Time-based bar analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class TimePeriod(Enum):
    """Time period types for new period detection."""
    MINUTE = "minute"
    HOUR = "hour" 
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class TimeInfo:
    """Information about execution and elapsed times."""
    
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    current_time_ms: float = 0.0
    elapsed_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.end_time is None:
            self.end_time = datetime.now()


@dataclass
class ElapsedTime:
    """Elapsed time breakdown in different units."""
    
    total_minutes: float = 0.0
    total_hours: float = 0.0  
    total_days: float = 0.0
    total_months: float = 0.0
    total_years: float = 0.0
    
    # Formatted representations
    minutes_str: str = "0"
    hours_str: str = "0"
    days_str: str = "0"
    months_str: str = "0.0"
    years_str: str = "0.0"


class CTimeUtils(CBase):
    """
    Comprehensive time utilities for trading systems.
    
    Provides advanced time analysis including:
    - Elapsed time calculations from data start
    - New time period detection (month, week, day, hour)
    - High-precision execution timing
    - Performance benchmarking
    - Time-based analysis tools
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Elapsed time tracking
        self.elapsed_time = ElapsedTime()
        
        # Execution timing
        self.start_time_tick: int = 0
        self.stop_time_tick: int = 0
        self.current_time_tick: int = 0
        self.execution_time_ms: float = 0.0
        
        # Performance counters
        self.watchdog_counter: int = 0
        self.watchdog_started: bool = False
        self.watchdog_finished: bool = False
        
        # Internal counters for various intervals
        self._interval_counters: Dict[int, int] = {
            90: 0, 120: 0, 150: 0, 180: 0,
            240: 0, 300: 0, 480: 0
        }
        
        # Time format constants
        self.DATETIME_FORMAT = "%d.%m.%Y %H:%M:%S"
        self.DATE_FORMAT = "%d.%m.%Y" 
        self.TIME_FORMAT = "%H:%M:%S"
        
        # Cached values for performance
        self._last_calculated_bar: int = -1
        self._cached_periods: Dict[str, bool] = {}
    
    def initialize(
        self, 
        system: SystemProtocol,
        market_data: Optional[any] = None
    ) -> 'CTimeUtils':
        """Initialize time utils with market data."""
        if market_data is not None:
            self.set_market_data(market_data)
        
        self.reset(system)
        return self
    
    def reset(self, system: SystemProtocol) -> 'CTimeUtils':
        """Reset all time calculations and counters."""
        self.elapsed_time = ElapsedTime()
        
        self.start_time_tick = 0
        self.stop_time_tick = 0
        self.current_time_tick = 0
        self.execution_time_ms = 0.0
        
        self.watchdog_counter = 0
        self.watchdog_started = False
        self.watchdog_finished = False
        
        self._interval_counters = {k: 0 for k in self._interval_counters.keys()}
        self._last_calculated_bar = -1
        self._cached_periods.clear()
        
        return self
    
    # Elapsed time calculations
    def calculate_elapsed_time_info(self, system: SystemProtocol) -> ElapsedTime:
        """
        Calculate comprehensive elapsed time information from data start.
        
        Calculates time elapsed from first bar to current time in various units.
        """
        if not hasattr(system.market_data, 'dates') or not system.market_data.dates:
            return self.elapsed_time
        
        start_date = system.market_data.dates[0]
        current_date = datetime.now()
        
        time_diff = current_date - start_date
        
        # Calculate in different units
        total_minutes = time_diff.total_seconds() / 60
        total_hours = total_minutes / 60
        total_days = time_diff.days + (time_diff.seconds / 86400)
        total_months = total_days / 30.4  # Average month length
        total_years = total_days / 365.25  # Average year length
        
        # Update elapsed time object
        self.elapsed_time.total_minutes = total_minutes
        self.elapsed_time.total_hours = total_hours
        self.elapsed_time.total_days = total_days
        self.elapsed_time.total_months = total_months
        self.elapsed_time.total_years = total_years
        
        # Format strings
        self.elapsed_time.minutes_str = f"{int(total_minutes)}"
        self.elapsed_time.hours_str = f"{int(total_hours)}"
        self.elapsed_time.days_str = f"{int(total_days)}"
        self.elapsed_time.months_str = f"{total_months:.1f}"
        self.elapsed_time.years_str = f"{total_years:.2f}"
        
        return self.elapsed_time
    
    def get_elapsed_time(self, system: SystemProtocol, unit: str = "M") -> float:
        """
        Get elapsed time in specified unit.
        
        Args:
            system: Trading system reference
            unit: Time unit - "D" (minutes), "S" (hours), "G" (days), "M" (months), "Y" (years)
        """
        self.calculate_elapsed_time_info(system)
        
        unit_map = {
            "D": self.elapsed_time.total_minutes,  # "Dakika" 
            "S": self.elapsed_time.total_hours,    # "Saat"
            "G": self.elapsed_time.total_days,     # "Gün"
            "M": self.elapsed_time.total_months,   # "Ay" (Month)
            "Y": self.elapsed_time.total_years     # "Yıl" (Year)
        }
        
        return unit_map.get(unit, 0.0)
    
    # New period detection methods
    def is_new_month(self, system: SystemProtocol, bar_index: int) -> bool:
        """Check if current bar represents start of a new month."""
        return self._is_new_period(system, bar_index, TimePeriod.MONTH)
    
    def is_new_week(self, system: SystemProtocol, bar_index: int) -> bool:
        """Check if current bar represents start of a new week."""
        return self._is_new_period(system, bar_index, TimePeriod.WEEK)
    
    def is_new_day(self, system: SystemProtocol, bar_index: int) -> bool:
        """Check if current bar represents start of a new day."""
        return self._is_new_period(system, bar_index, TimePeriod.DAY)
    
    def is_new_hour(self, system: SystemProtocol, bar_index: int) -> bool:
        """Check if current bar represents start of a new hour."""
        return self._is_new_period(system, bar_index, TimePeriod.HOUR)
    
    def is_new_period(self, system: SystemProtocol, bar_index: int, period: str) -> bool:
        """
        Generic new period detection.
        
        Args:
            system: Trading system reference
            bar_index: Current bar index
            period: Period type - "M" (month), "H" (week), "G" (day), "S" (hour)
        """
        period_map = {
            "M": TimePeriod.MONTH,
            "H": TimePeriod.WEEK,  # "Hafta"
            "G": TimePeriod.DAY,   # "Gün"
            "S": TimePeriod.HOUR   # "Saat"
        }
        
        period_enum = period_map.get(period)
        if period_enum:
            return self._is_new_period(system, bar_index, period_enum)
        
        return False
    
    def _is_new_period(self, system: SystemProtocol, bar_index: int, period: TimePeriod) -> bool:
        """Internal method for new period detection with caching."""
        if bar_index <= 0:
            return False
        
        # Check cache
        cache_key = f"{period.value}_{bar_index}"
        if cache_key in self._cached_periods:
            return self._cached_periods[cache_key]
        
        if not hasattr(system.market_data, 'dates') or bar_index >= len(system.market_data.dates):
            return False
        
        current_date = system.market_data.dates[bar_index]
        previous_date = system.market_data.dates[bar_index - 1]
        
        result = False
        
        if period == TimePeriod.MONTH:
            result = current_date.month != previous_date.month or current_date.year != previous_date.year
        
        elif period == TimePeriod.WEEK:
            # Check if it's a new day first, then check if it's a different week
            is_new_day = current_date.date() != previous_date.date()
            if is_new_day:
                current_week = current_date.isocalendar().week
                previous_week = previous_date.isocalendar().week
                result = current_week != previous_week or current_date.year != previous_date.year
        
        elif period == TimePeriod.DAY:
            result = current_date.date() != previous_date.date()
        
        elif period == TimePeriod.HOUR:
            result = current_date.hour != previous_date.hour
        
        elif period == TimePeriod.MINUTE:
            result = current_date.minute != previous_date.minute
        
        # Cache result
        self._cached_periods[cache_key] = result
        return result
    
    # Execution timing methods
    def start_timing(self) -> 'CTimeUtils':
        """Start execution timing."""
        self.start_time_tick = int(time.time() * 1000)  # Milliseconds
        return self
    
    def StartTimer(self, sistem=None) -> 'CTimeUtils':
        """Start timer - Python equivalent of C# StartTimer method."""
        return self.start_timing()
    
    def StopTimer(self, sistem=None) -> 'CTimeUtils':
        """Stop timer - Python equivalent of C# StopTimer method."""
        return self.stop_timing()
    
    def stop_timing(self) -> 'CTimeUtils':
        """Stop execution timing and calculate elapsed time."""
        self.stop_time_tick = int(time.time() * 1000)
        self.execution_time_ms = self.stop_time_tick - self.start_time_tick
        return self
    
    def get_current_time_ms(self) -> int:
        """Get current time in milliseconds."""
        self.current_time_tick = int(time.time() * 1000)
        return self.current_time_tick
    
    def get_execution_time_ms(self) -> float:
        """Get last recorded execution time in milliseconds."""
        return self.execution_time_ms
    
    def get_last_execution_time(self, system: SystemProtocol) -> str:
        """Get formatted string of last execution time."""
        return datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    
    # Watchdog and performance monitoring
    def start_watchdog(self) -> 'CTimeUtils':
        """Start watchdog timer for performance monitoring."""
        self.watchdog_counter = 0
        self.watchdog_started = True
        self.watchdog_finished = False
        self.start_timing()
        return self
    
    def update_watchdog(self) -> 'CTimeUtils':
        """Update watchdog counter."""
        if self.watchdog_started and not self.watchdog_finished:
            self.watchdog_counter += 1
        return self
    
    def stop_watchdog(self) -> 'CTimeUtils':
        """Stop watchdog and record final timing."""
        self.watchdog_finished = True
        self.stop_timing()
        return self
    
    def get_watchdog_info(self) -> Dict[str, any]:
        """Get watchdog monitoring information."""
        return {
            "counter": self.watchdog_counter,
            "started": self.watchdog_started,
            "finished": self.watchdog_finished,
            "execution_time_ms": self.execution_time_ms
        }
    
    # Bar and datetime utilities
    def get_bar_datetime(self, system: SystemProtocol, bar_index: int) -> Optional[datetime]:
        """Get datetime for specific bar."""
        if (hasattr(system.market_data, 'dates') and 
            bar_index < len(system.market_data.dates) and 
            bar_index >= 0):
            return system.market_data.dates[bar_index]
        return None
    
    def get_week_number(self, system: SystemProtocol, target_date: datetime) -> int:
        """Get ISO week number for given date."""
        return target_date.isocalendar().week
    
    def is_weekend(self, target_date: datetime) -> bool:
        """Check if given date is weekend."""
        return target_date.weekday() >= 5  # Saturday=5, Sunday=6
    
    def is_market_day(self, target_date: datetime) -> bool:
        """Check if given date is a typical market day (Monday-Friday)."""
        return target_date.weekday() < 5
    
    # Time formatting utilities
    def format_datetime(self, dt: datetime, format_type: str = "full") -> str:
        """Format datetime with predefined formats."""
        formats = {
            "full": self.DATETIME_FORMAT,
            "date": self.DATE_FORMAT,
            "time": self.TIME_FORMAT,
            "iso": "%Y-%m-%d %H:%M:%S",
            "compact": "%Y%m%d_%H%M%S"
        }
        
        format_str = formats.get(format_type, self.DATETIME_FORMAT)
        return dt.strftime(format_str)
    
    def parse_datetime(self, date_str: str, format_type: str = "full") -> Optional[datetime]:
        """Parse datetime string with predefined formats."""
        formats = {
            "full": self.DATETIME_FORMAT,
            "date": self.DATE_FORMAT,
            "time": self.TIME_FORMAT,
            "iso": "%Y-%m-%d %H:%M:%S",
            "compact": "%Y%m%d_%H%M%S"
        }
        
        format_str = formats.get(format_type, self.DATETIME_FORMAT)
        try:
            return datetime.strptime(date_str, format_str)
        except ValueError:
            return None
    
    # Analysis and statistics methods
    def get_time_statistics(self, system: SystemProtocol) -> Dict[str, any]:
        """Get comprehensive time analysis statistics."""
        elapsed = self.calculate_elapsed_time_info(system)
        
        total_bars = len(system.market_data.dates) if hasattr(system.market_data, 'dates') else 0
        
        stats = {
            "elapsed_time": {
                "total_days": elapsed.total_days,
                "total_hours": elapsed.total_hours,
                "total_minutes": elapsed.total_minutes,
                "total_months": elapsed.total_months,
                "formatted": {
                    "days": elapsed.days_str,
                    "hours": elapsed.hours_str,
                    "months": elapsed.months_str
                }
            },
            "data_info": {
                "total_bars": total_bars,
                "start_date": self.format_datetime(system.market_data.dates[0], "full") if total_bars > 0 else None,
                "end_date": self.format_datetime(system.market_data.dates[-1], "full") if total_bars > 0 else None
            },
            "performance": {
                "execution_time_ms": self.execution_time_ms,
                "watchdog_counter": self.watchdog_counter,
                "cache_size": len(self._cached_periods)
            }
        }
        
        return stats
    
    def get_period_boundaries(self, system: SystemProtocol, period: TimePeriod) -> List[int]:
        """
        Get list of bar indices where new periods start.
        
        Args:
            system: Trading system reference  
            period: Period type to find boundaries for
            
        Returns:
            List of bar indices where new periods begin
        """
        boundaries = []
        
        if not hasattr(system.market_data, 'dates'):
            return boundaries
        
        total_bars = len(system.market_data.dates)
        
        for i in range(1, total_bars):
            if self._is_new_period(system, i, period):
                boundaries.append(i)
        
        return boundaries
    
    def analyze_trading_sessions(self, system: SystemProtocol) -> Dict[str, any]:
        """Analyze trading sessions within the data."""
        if not hasattr(system.market_data, 'dates'):
            return {}
        
        session_stats = {
            "asian_session_bars": 0,
            "european_session_bars": 0, 
            "american_session_bars": 0,
            "overlap_bars": 0,
            "total_bars": len(system.market_data.dates)
        }
        
        for date in system.market_data.dates:
            hour = date.hour
            
            # Asian session: 00:00-09:00 UTC
            if 0 <= hour < 9:
                session_stats["asian_session_bars"] += 1
            
            # European session: 07:00-16:00 UTC  
            if 7 <= hour < 16:
                session_stats["european_session_bars"] += 1
            
            # American session: 13:00-22:00 UTC
            if 13 <= hour < 22:
                session_stats["american_session_bars"] += 1
            
            # Overlap periods
            if (7 <= hour < 9) or (13 <= hour < 16):  # Asian-European or European-American
                session_stats["overlap_bars"] += 1
        
        return session_stats
    
    # Utility methods for intervals and counters
    def increment_interval_counter(self, interval_minutes: int) -> 'CTimeUtils':
        """Increment counter for specific time interval."""
        if interval_minutes in self._interval_counters:
            self._interval_counters[interval_minutes] += 1
        return self
    
    def get_interval_counter(self, interval_minutes: int) -> int:
        """Get current value of interval counter."""
        return self._interval_counters.get(interval_minutes, 0)
    
    def reset_interval_counters(self) -> 'CTimeUtils':
        """Reset all interval counters."""
        self._interval_counters = {k: 0 for k in self._interval_counters.keys()}
        return self
    
    def clear_cache(self) -> 'CTimeUtils':
        """Clear all cached calculations."""
        self._cached_periods.clear()
        self._last_calculated_bar = -1
        return self
    
    # Context manager support for timing
    def __enter__(self):
        """Start timing when entering context."""
        self.start_timing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context."""
        self.stop_timing()
    
    def __str__(self) -> str:
        """String representation of time utils."""
        return (
            f"CTimeUtils(execution_time={self.execution_time_ms}ms, "
            f"watchdog_counter={self.watchdog_counter}, "
            f"cache_size={len(self._cached_periods)})"
        )