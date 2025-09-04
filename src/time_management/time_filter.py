"""
Time-based trading filters and session management.

This module provides sophisticated time filtering capabilities for trading systems:
- Market session filtering (Asian, European, American sessions)
- Trading hours restrictions
- Holiday and weekend filters
- Custom time-based rules
- Position management based on time
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from datetime import datetime, time, timedelta, date
import calendar

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.base import CBase, SystemProtocol


class TimeFilterType(Enum):
    """Types of time filters available."""
    MARKET_HOURS = "market_hours"
    TRADING_SESSION = "trading_session" 
    WEEKDAY_FILTER = "weekday_filter"
    HOLIDAY_FILTER = "holiday_filter"
    CUSTOM_TIME_RANGE = "custom_time_range"
    END_OF_DAY_CLOSE = "end_of_day_close"


class TimeFilterResult(Enum):
    """Results of time filter checks."""
    ALLOW_TRADING = "allow_trading"
    BLOCK_TRADING = "block_trading"
    FORCE_CLOSE_POSITIONS = "force_close_positions"
    END_OF_SESSION = "end_of_session"


class TradingSession(Enum):
    """Major trading sessions."""
    ASIAN = "asian"
    EUROPEAN = "european"
    AMERICAN = "american"
    OVERLAP_ASIAN_EUROPEAN = "overlap_asian_european"
    OVERLAP_EUROPEAN_AMERICAN = "overlap_european_american"


@dataclass
class MarketHours:
    """Market hours configuration for different markets."""
    
    # Forex major sessions (UTC times)
    forex_sessions: Dict[TradingSession, Tuple[time, time]] = None
    
    # Stock market hours
    nyse_hours: Tuple[time, time] = None
    lse_hours: Tuple[time, time] = None  
    tse_hours: Tuple[time, time] = None
    
    # Custom market hours
    custom_hours: Dict[str, Tuple[time, time]] = None
    
    def __post_init__(self):
        if self.forex_sessions is None:
            self.forex_sessions = {
                TradingSession.ASIAN: (time(0, 0), time(9, 0)),      # UTC: 00:00-09:00
                TradingSession.EUROPEAN: (time(7, 0), time(16, 0)),   # UTC: 07:00-16:00 
                TradingSession.AMERICAN: (time(13, 0), time(22, 0)),  # UTC: 13:00-22:00
            }
        
        if self.nyse_hours is None:
            self.nyse_hours = (time(14, 30), time(21, 0))  # UTC: 14:30-21:00 (NYSE: 9:30-16:00 EST)
            
        if self.lse_hours is None:
            self.lse_hours = (time(8, 0), time(16, 30))    # UTC: 08:00-16:30
            
        if self.tse_hours is None:
            self.tse_hours = (time(0, 0), time(6, 0))      # UTC: 00:00-06:00 (JST: 9:00-15:00)
            
        if self.custom_hours is None:
            self.custom_hours = {}


class CTimeFilter(CBase):
    """
    Comprehensive time-based trading filter system.
    
    Features:
    - Market session filtering
    - Trading hours enforcement  
    - Weekend/holiday blocking
    - End-of-day position management
    - Custom time-based rules
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        
        # Core components
        self.trader = None
        self.market_hours = MarketHours()
        
        # Filter configuration
        self.enabled_filters: Set[TimeFilterType] = set()
        self.allowed_weekdays: Set[int] = {0, 1, 2, 3, 4}  # Monday-Friday
        self.holidays: Set[date] = set()
        
        # Session settings
        self.allowed_sessions: Set[TradingSession] = {
            TradingSession.EUROPEAN, 
            TradingSession.AMERICAN
        }
        
        # Time range settings
        self.custom_trading_start: Optional[time] = None
        self.custom_trading_end: Optional[time] = None
        
        # End-of-day settings
        self.end_of_day_close_enabled: bool = False
        self.end_of_day_close_time: time = time(21, 50)  # 10 minutes before market close
        self.end_of_day_warning_time: time = time(21, 45)  # 15 minutes warning
        
        # State tracking
        self.current_session: Optional[TradingSession] = None
        self.last_check_time: Optional[datetime] = None
        self.positions_closed_eod: bool = False
        
        # Statistics
        self.filter_triggers: Dict[TimeFilterType, int] = {
            filter_type: 0 for filter_type in TimeFilterType
        }
    
    def initialize(self, system: SystemProtocol, trader) -> 'CTimeFilter':
        """Initialize the time filter with trader reference."""
        self.trader = trader
        return self
    
    def reset(self, system: SystemProtocol) -> 'CTimeFilter':
        """Reset all time filter state."""
        self.current_session = None
        self.last_check_time = None
        self.positions_closed_eod = False
        self.filter_triggers = {filter_type: 0 for filter_type in TimeFilterType}
        return self
    
    # Configuration methods
    def enable_filter(self, filter_type: TimeFilterType) -> 'CTimeFilter':
        """Enable a specific time filter."""
        self.enabled_filters.add(filter_type)
        return self
    
    def disable_filter(self, filter_type: TimeFilterType) -> 'CTimeFilter':
        """Disable a specific time filter."""
        self.enabled_filters.discard(filter_type)
        return self
    
    def set_allowed_weekdays(self, weekdays: List[int]) -> 'CTimeFilter':
        """Set allowed weekdays (0=Monday, 6=Sunday)."""
        self.allowed_weekdays = set(weekdays)
        return self
    
    def set_allowed_sessions(self, sessions: List[TradingSession]) -> 'CTimeFilter':
        """Set allowed trading sessions."""
        self.allowed_sessions = set(sessions)
        return self
    
    def add_holiday(self, holiday_date: date) -> 'CTimeFilter':
        """Add a holiday to the filter."""
        self.holidays.add(holiday_date)
        return self
    
    def add_holidays(self, holidays: List[date]) -> 'CTimeFilter':
        """Add multiple holidays."""
        self.holidays.update(holidays)
        return self
    
    def set_custom_trading_hours(self, start_time: time, end_time: time) -> 'CTimeFilter':
        """Set custom trading hours."""
        self.custom_trading_start = start_time
        self.custom_trading_end = end_time
        return self
    
    def enable_end_of_day_close(
        self, 
        close_time: time = time(21, 50),
        warning_time: time = time(21, 45)
    ) -> 'CTimeFilter':
        """Enable end-of-day position closing."""
        self.end_of_day_close_enabled = True
        self.end_of_day_close_time = close_time
        self.end_of_day_warning_time = warning_time
        self.enable_filter(TimeFilterType.END_OF_DAY_CLOSE)
        return self
    
    # Core filtering methods
    def check_trading_allowed(
        self, 
        system: SystemProtocol, 
        bar_index: int,
        current_time: Optional[datetime] = None
    ) -> TimeFilterResult:
        """
        Check if trading is allowed at the current time.
        
        Args:
            system: Trading system reference
            bar_index: Current bar index
            current_time: Current time (uses system time if None)
            
        Returns:
            TimeFilterResult indicating the action to take
        """
        if current_time is None:
            if hasattr(system.market_data, 'dates') and bar_index < len(system.market_data.dates):
                current_time = system.market_data.dates[bar_index]
            else:
                current_time = datetime.now()
        
        self.last_check_time = current_time
        
        # Check each enabled filter
        for filter_type in self.enabled_filters:
            result = self._apply_filter(filter_type, current_time, system, bar_index)
            
            if result != TimeFilterResult.ALLOW_TRADING:
                self.filter_triggers[filter_type] += 1
                return result
        
        return TimeFilterResult.ALLOW_TRADING
    
    def _apply_filter(
        self, 
        filter_type: TimeFilterType, 
        current_time: datetime,
        system: SystemProtocol,
        bar_index: int
    ) -> TimeFilterResult:
        """Apply a specific filter type."""
        
        if filter_type == TimeFilterType.WEEKDAY_FILTER:
            return self._check_weekday_filter(current_time)
        
        elif filter_type == TimeFilterType.HOLIDAY_FILTER:
            return self._check_holiday_filter(current_time)
        
        elif filter_type == TimeFilterType.MARKET_HOURS:
            return self._check_market_hours_filter(current_time)
        
        elif filter_type == TimeFilterType.TRADING_SESSION:
            return self._check_trading_session_filter(current_time)
        
        elif filter_type == TimeFilterType.CUSTOM_TIME_RANGE:
            return self._check_custom_time_range_filter(current_time)
        
        elif filter_type == TimeFilterType.END_OF_DAY_CLOSE:
            return self._check_end_of_day_filter(current_time)
        
        return TimeFilterResult.ALLOW_TRADING
    
    def _check_weekday_filter(self, current_time: datetime) -> TimeFilterResult:
        """Check if current weekday is allowed for trading."""
        weekday = current_time.weekday()
        return TimeFilterResult.ALLOW_TRADING if weekday in self.allowed_weekdays else TimeFilterResult.BLOCK_TRADING
    
    def _check_holiday_filter(self, current_time: datetime) -> TimeFilterResult:
        """Check if current date is a holiday."""
        current_date = current_time.date()
        return TimeFilterResult.BLOCK_TRADING if current_date in self.holidays else TimeFilterResult.ALLOW_TRADING
    
    def _check_market_hours_filter(self, current_time: datetime) -> TimeFilterResult:
        """Check if current time is within market hours."""
        current_time_only = current_time.time()
        
        # Check NYSE hours as default
        start_time, end_time = self.market_hours.nyse_hours
        
        if self._is_time_in_range(current_time_only, start_time, end_time):
            return TimeFilterResult.ALLOW_TRADING
        else:
            return TimeFilterResult.BLOCK_TRADING
    
    def _check_trading_session_filter(self, current_time: datetime) -> TimeFilterResult:
        """Check if current time is within allowed trading sessions."""
        current_session = self._get_current_session(current_time)
        self.current_session = current_session
        
        if current_session in self.allowed_sessions:
            return TimeFilterResult.ALLOW_TRADING
        
        # Check for session overlaps
        if TradingSession.OVERLAP_ASIAN_EUROPEAN in self.allowed_sessions:
            if self._is_in_overlap_period(current_time, TradingSession.ASIAN, TradingSession.EUROPEAN):
                return TimeFilterResult.ALLOW_TRADING
        
        if TradingSession.OVERLAP_EUROPEAN_AMERICAN in self.allowed_sessions:
            if self._is_in_overlap_period(current_time, TradingSession.EUROPEAN, TradingSession.AMERICAN):
                return TimeFilterResult.ALLOW_TRADING
        
        return TimeFilterResult.BLOCK_TRADING
    
    def _check_custom_time_range_filter(self, current_time: datetime) -> TimeFilterResult:
        """Check custom time range filter."""
        if self.custom_trading_start is None or self.custom_trading_end is None:
            return TimeFilterResult.ALLOW_TRADING
        
        current_time_only = current_time.time()
        
        if self._is_time_in_range(current_time_only, self.custom_trading_start, self.custom_trading_end):
            return TimeFilterResult.ALLOW_TRADING
        else:
            return TimeFilterResult.BLOCK_TRADING
    
    def _check_end_of_day_filter(self, current_time: datetime) -> TimeFilterResult:
        """Check end-of-day position closing filter."""
        if not self.end_of_day_close_enabled:
            return TimeFilterResult.ALLOW_TRADING
        
        current_time_only = current_time.time()
        
        # Force close positions at EOD time
        if current_time_only >= self.end_of_day_close_time:
            return TimeFilterResult.FORCE_CLOSE_POSITIONS
        
        # Warning period before EOD
        if current_time_only >= self.end_of_day_warning_time:
            return TimeFilterResult.END_OF_SESSION
        
        return TimeFilterResult.ALLOW_TRADING
    
    # Helper methods
    def _get_current_session(self, current_time: datetime) -> Optional[TradingSession]:
        """Determine current trading session."""
        current_time_only = current_time.time()
        
        for session, (start, end) in self.market_hours.forex_sessions.items():
            if self._is_time_in_range(current_time_only, start, end):
                return session
        
        return None
    
    def _is_time_in_range(self, current_time: time, start_time: time, end_time: time) -> bool:
        """Check if current time is within the given range."""
        if start_time <= end_time:
            # Normal range (e.g., 09:00-17:00)
            return start_time <= current_time <= end_time
        else:
            # Overnight range (e.g., 22:00-06:00)
            return current_time >= start_time or current_time <= end_time
    
    def _is_in_overlap_period(
        self, 
        current_time: datetime, 
        session1: TradingSession, 
        session2: TradingSession
    ) -> bool:
        """Check if current time is in overlap period between two sessions."""
        current_time_only = current_time.time()
        
        session1_hours = self.market_hours.forex_sessions.get(session1)
        session2_hours = self.market_hours.forex_sessions.get(session2)
        
        if not session1_hours or not session2_hours:
            return False
        
        # Find overlap period
        overlap_start = max(session1_hours[0], session2_hours[0])
        overlap_end = min(session1_hours[1], session2_hours[1])
        
        if overlap_start < overlap_end:
            return self._is_time_in_range(current_time_only, overlap_start, overlap_end)
        
        return False
    
    # Information and status methods
    def get_current_session_info(self) -> Dict[str, any]:
        """Get information about current trading session."""
        return {
            "current_session": self.current_session.value if self.current_session else None,
            "last_check_time": self.last_check_time,
            "positions_closed_eod": self.positions_closed_eod,
            "enabled_filters": [f.value for f in self.enabled_filters],
            "allowed_sessions": [s.value for s in self.allowed_sessions]
        }
    
    def get_filter_statistics(self) -> Dict[str, int]:
        """Get statistics about filter triggers."""
        return dict(self.filter_triggers)
    
    def is_trading_time(self, current_time: Optional[datetime] = None) -> bool:
        """Simple check if it's currently trading time."""
        if current_time is None:
            current_time = datetime.now()
        
        # Quick check using basic filters
        weekday = current_time.weekday()
        if weekday not in self.allowed_weekdays:
            return False
        
        if current_time.date() in self.holidays:
            return False
        
        return True
    
    def get_next_trading_session(self, current_time: Optional[datetime] = None) -> Optional[datetime]:
        """Get the start time of the next trading session."""
        if current_time is None:
            current_time = datetime.now()
        
        # Simple implementation - find next allowed session
        current_time_only = current_time.time()
        
        for session in self.allowed_sessions:
            if session in self.market_hours.forex_sessions:
                start_time, _ = self.market_hours.forex_sessions[session]
                if current_time_only < start_time:
                    return datetime.combine(current_time.date(), start_time)
        
        # If no session today, try tomorrow
        next_day = current_time.date() + timedelta(days=1)
        if next_day.weekday() in self.allowed_weekdays and next_day not in self.holidays:
            if self.allowed_sessions:
                first_session = next(iter(self.allowed_sessions))
                if first_session in self.market_hours.forex_sessions:
                    start_time, _ = self.market_hours.forex_sessions[first_session]
                    return datetime.combine(next_day, start_time)
        
        return None
    
    def get_market_hours_info(self, market: str = "forex") -> Dict[str, any]:
        """Get market hours information."""
        if market == "forex":
            return {session.value: (start.strftime("%H:%M"), end.strftime("%H:%M")) 
                   for session, (start, end) in self.market_hours.forex_sessions.items()}
        elif market == "nyse":
            start, end = self.market_hours.nyse_hours
            return {"nyse": (start.strftime("%H:%M"), end.strftime("%H:%M"))}
        elif market == "lse":
            start, end = self.market_hours.lse_hours  
            return {"lse": (start.strftime("%H:%M"), end.strftime("%H:%M"))}
        else:
            return {}
    
    # Utility methods for common scenarios
    def setup_forex_24h(self) -> 'CTimeFilter':
        """Set up for 24-hour forex trading (Monday-Friday)."""
        self.set_allowed_weekdays([0, 1, 2, 3, 4])  # Monday-Friday
        self.set_allowed_sessions([
            TradingSession.ASIAN,
            TradingSession.EUROPEAN, 
            TradingSession.AMERICAN
        ])
        self.enable_filter(TimeFilterType.WEEKDAY_FILTER)
        self.enable_filter(TimeFilterType.TRADING_SESSION)
        return self
    
    def setup_stock_market_hours(self, market: str = "nyse") -> 'CTimeFilter':
        """Set up for stock market hours."""
        if market == "nyse":
            start, end = self.market_hours.nyse_hours
        elif market == "lse":
            start, end = self.market_hours.lse_hours
        else:
            start, end = time(9, 0), time(17, 0)  # Default hours
        
        self.set_custom_trading_hours(start, end)
        self.set_allowed_weekdays([0, 1, 2, 3, 4])  # Monday-Friday
        self.enable_filter(TimeFilterType.WEEKDAY_FILTER)
        self.enable_filter(TimeFilterType.CUSTOM_TIME_RANGE)
        return self
    
    def setup_end_of_day_management(self) -> 'CTimeFilter':
        """Set up end-of-day position management."""
        self.enable_end_of_day_close()
        return self
    
    def add_common_holidays_us(self, year: int = None) -> 'CTimeFilter':
        """Add common US holidays for the given year."""
        if year is None:
            year = datetime.now().year
        
        holidays = [
            date(year, 1, 1),   # New Year's Day
            date(year, 7, 4),   # Independence Day
            date(year, 12, 25), # Christmas
        ]
        
        # Add Martin Luther King Jr. Day (3rd Monday in January)
        jan_first = date(year, 1, 1)
        days_to_add = (7 - jan_first.weekday()) % 7 + 14  # 3rd Monday
        mlk_day = jan_first + timedelta(days=days_to_add)
        holidays.append(mlk_day)
        
        self.add_holidays(holidays)
        self.enable_filter(TimeFilterType.HOLIDAY_FILTER)
        return self
    
    def __str__(self) -> str:
        """String representation of time filter."""
        enabled_count = len(self.enabled_filters)
        total_triggers = sum(self.filter_triggers.values())
        
        return (
            f"CTimeFilter(enabled_filters={enabled_count}, "
            f"allowed_weekdays={sorted(self.allowed_weekdays)}, "
            f"allowed_sessions={len(self.allowed_sessions)}, "
            f"total_triggers={total_triggers})"
        )