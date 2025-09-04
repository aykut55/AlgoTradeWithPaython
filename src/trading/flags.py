"""
Trading Flags Management for algorithmic trading.

This module contains the CFlags class which handles comprehensive
trading flag management, signal coordination, system states,
and conditional logic for trading operations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

from ..core.base import SystemProtocol


class TradingFlagType(Enum):
    """Trading flag types."""
    SYSTEM = "SYSTEM"                   # System-level flags
    STRATEGY = "STRATEGY"               # Strategy-specific flags
    RISK = "RISK"                       # Risk management flags
    MARKET = "MARKET"                   # Market condition flags
    POSITION = "POSITION"               # Position management flags
    ORDER = "ORDER"                     # Order management flags
    SIGNAL = "SIGNAL"                   # Trading signal flags
    CONDITION = "CONDITION"             # Conditional flags
    USER = "USER"                       # User-defined flags


class FlagState(Enum):
    """Flag state enumeration."""
    INACTIVE = "INACTIVE"               # Flag is inactive/false
    ACTIVE = "ACTIVE"                   # Flag is active/true
    PENDING = "PENDING"                 # Flag is pending activation
    EXPIRED = "EXPIRED"                 # Flag has expired
    DISABLED = "DISABLED"               # Flag is disabled


class FlagPriority(Enum):
    """Flag priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class SystemFlags(Flag):
    """System-level trading flags."""
    NONE = 0
    TRADING_ENABLED = auto()            # Trading is enabled
    MARKET_OPEN = auto()                # Market is open
    DATA_FEED_OK = auto()               # Data feed is working
    RISK_CHECK_OK = auto()              # Risk checks passed
    EMERGENCY_STOP = auto()             # Emergency stop activated
    MAINTENANCE_MODE = auto()           # System in maintenance
    BACKTEST_MODE = auto()              # Running in backtest mode
    PAPER_TRADING = auto()              # Paper trading mode
    LIVE_TRADING = auto()               # Live trading mode


class StrategyFlags(Flag):
    """Strategy-level trading flags."""
    NONE = 0
    STRATEGY_ACTIVE = auto()            # Strategy is active
    SIGNAL_GENERATED = auto()           # Trading signal generated
    ENTRY_CONDITIONS_MET = auto()       # Entry conditions satisfied
    EXIT_CONDITIONS_MET = auto()        # Exit conditions satisfied
    FILTER_PASSED = auto()              # Strategy filters passed
    CONFIRMATION_REQUIRED = auto()      # Manual confirmation needed
    TREND_BULLISH = auto()              # Bullish trend detected
    TREND_BEARISH = auto()              # Bearish trend detected


class RiskFlags(Flag):
    """Risk management flags."""
    NONE = 0
    RISK_OK = auto()                    # Risk levels acceptable
    POSITION_LIMIT_OK = auto()          # Position limits OK
    DRAWDOWN_OK = auto()                # Drawdown within limits
    VOLATILITY_OK = auto()              # Volatility acceptable
    CORRELATION_OK = auto()             # Correlation limits OK
    MARGIN_OK = auto()                  # Margin requirements OK
    EXPOSURE_WARNING = auto()           # Exposure warning
    STOP_LOSS_HIT = auto()              # Stop loss triggered


@dataclass
class TradingFlag:
    """Trading flag definition."""
    
    name: str
    flag_type: TradingFlagType
    state: FlagState = FlagState.INACTIVE
    priority: FlagPriority = FlagPriority.NORMAL
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    # Conditions
    condition_callback: Optional[Callable[[], bool]] = None
    auto_expire_seconds: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    
    # Values
    value: Any = None
    threshold: Optional[float] = None
    
    # History
    activation_count: int = 0
    last_activated: Optional[datetime] = None
    last_deactivated: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if flag is currently active."""
        return self.state == FlagState.ACTIVE
    
    def is_expired(self) -> bool:
        """Check if flag has expired."""
        if self.state == FlagState.EXPIRED:
            return True
        
        if (self.auto_expire_seconds and self.last_activated and 
            datetime.now() - self.last_activated > timedelta(seconds=self.auto_expire_seconds)):
            return True
        
        return False
    
    def activate(self) -> bool:
        """Activate the flag."""
        if self.state == FlagState.DISABLED:
            return False
        
        if self.state != FlagState.ACTIVE:
            self.state = FlagState.ACTIVE
            self.updated_at = datetime.now()
            self.last_activated = self.updated_at
            self.activation_count += 1
            return True
        
        return False
    
    def deactivate(self) -> bool:
        """Deactivate the flag."""
        if self.state == FlagState.ACTIVE:
            self.state = FlagState.INACTIVE
            self.updated_at = datetime.now()
            self.last_deactivated = self.updated_at
            return True
        
        return False


@dataclass
class FlagEvent:
    """Flag state change event."""
    
    timestamp: datetime
    flag_name: str
    old_state: FlagState
    new_state: FlagState
    trigger: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class CFlags:
    """
    Comprehensive trading flags management system.
    
    Features:
    - Multiple flag types (system, strategy, risk, etc.)
    - Flag dependencies and conflicts
    - Conditional flag evaluation
    - Auto-expiring flags
    - Flag priority system
    - Event-driven flag changes
    - Flag history and analytics
    - Thread-safe operations
    - Custom flag definitions
    """
    
    def __init__(self):
        """Initialize flags manager."""
        self.is_initialized = False
        
        # Flag storage
        self.flags: Dict[str, TradingFlag] = {}
        self.flag_groups: Dict[TradingFlagType, Set[str]] = defaultdict(set)
        
        # Built-in flag sets
        self.system_flags = SystemFlags.NONE
        self.strategy_flags = StrategyFlags.NONE
        self.risk_flags = RiskFlags.NONE
        
        # Event system
        self.flag_events: deque = deque(maxlen=1000)
        self.event_subscribers: Dict[str, List[Callable[[FlagEvent], None]]] = defaultdict(list)
        self.global_subscribers: List[Callable[[FlagEvent], None]] = []
        
        # Conditional evaluation
        self.condition_checkers: Dict[str, Callable[[], bool]] = {}
        self.evaluation_interval = 1.0  # seconds
        
        # Threading
        self.flags_lock = threading.RLock()
        self.evaluation_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_flags': 0,
            'active_flags': 0,
            'flag_activations': 0,
            'flag_deactivations': 0,
            'condition_evaluations': 0,
            'event_notifications': 0
        }
        
        # Initialize built-in flags
        self._initialize_builtin_flags()
    
    def initialize(self, system: SystemProtocol) -> 'CFlags':
        """
        Initialize flags manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.flags_lock:
            # Start condition evaluation thread
            self._start_evaluation_thread()
            
            self.is_initialized = True
        
        return self
    
    def reset(self, system: SystemProtocol) -> 'CFlags':
        """
        Reset flags manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.flags_lock:
            # Stop evaluation thread
            self.running = False
            if self.evaluation_thread:
                self.evaluation_thread.join(timeout=5.0)
            
            # Reset all flags
            for flag in self.flags.values():
                flag.state = FlagState.INACTIVE
            
            # Clear events and subscribers
            self.flag_events.clear()
            self.event_subscribers.clear()
            self.global_subscribers.clear()
            
            # Reset built-in flags
            self.system_flags = SystemFlags.NONE
            self.strategy_flags = StrategyFlags.NONE
            self.risk_flags = RiskFlags.NONE
        
        return self
    
    def _initialize_builtin_flags(self) -> None:
        """Initialize built-in trading flags."""
        # System flags
        system_flag_definitions = [
            ("trading_enabled", "Trading operations are enabled"),
            ("market_open", "Market is currently open"),
            ("data_feed_ok", "Data feed is functioning properly"),
            ("risk_check_ok", "Risk management checks are passing"),
            ("emergency_stop", "Emergency stop is activated"),
            ("maintenance_mode", "System is in maintenance mode"),
            ("backtest_mode", "Running in backtest mode"),
            ("paper_trading", "Paper trading mode active"),
            ("live_trading", "Live trading mode active")
        ]
        
        for name, desc in system_flag_definitions:
            self.define_flag(
                name=name,
                flag_type=TradingFlagType.SYSTEM,
                description=desc,
                priority=FlagPriority.HIGH
            )
        
        # Strategy flags
        strategy_flag_definitions = [
            ("strategy_active", "Strategy is currently active"),
            ("signal_generated", "Trading signal has been generated"),
            ("entry_conditions_met", "Entry conditions are satisfied"),
            ("exit_conditions_met", "Exit conditions are satisfied"),
            ("filter_passed", "Strategy filters have passed"),
            ("confirmation_required", "Manual confirmation is required"),
            ("trend_bullish", "Bullish trend detected"),
            ("trend_bearish", "Bearish trend detected")
        ]
        
        for name, desc in strategy_flag_definitions:
            self.define_flag(
                name=name,
                flag_type=TradingFlagType.STRATEGY,
                description=desc
            )
        
        # Risk flags
        risk_flag_definitions = [
            ("risk_ok", "Risk levels are acceptable"),
            ("position_limit_ok", "Position limits are within bounds"),
            ("drawdown_ok", "Drawdown is within acceptable limits"),
            ("volatility_ok", "Market volatility is acceptable"),
            ("margin_ok", "Margin requirements are satisfied"),
            ("exposure_warning", "Portfolio exposure warning"),
            ("stop_loss_hit", "Stop loss has been triggered")
        ]
        
        for name, desc in risk_flag_definitions:
            self.define_flag(
                name=name,
                flag_type=TradingFlagType.RISK,
                description=desc,
                priority=FlagPriority.HIGH
            )
    
    def _start_evaluation_thread(self) -> None:
        """Start conditional evaluation thread."""
        self.running = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_worker, daemon=True)
        self.evaluation_thread.start()
    
    def _evaluation_worker(self) -> None:
        """Background worker for flag condition evaluation."""
        import time
        
        while self.running:
            try:
                self._evaluate_conditions()
                time.sleep(self.evaluation_interval)
            except Exception:
                time.sleep(self.evaluation_interval * 2)  # Back off on error
    
    def _evaluate_conditions(self) -> None:
        """Evaluate all conditional flags."""
        with self.flags_lock:
            for flag_name, flag in self.flags.items():
                if flag.condition_callback and flag.state != FlagState.DISABLED:
                    try:
                        should_be_active = flag.condition_callback()
                        current_active = flag.is_active()
                        
                        if should_be_active and not current_active:
                            self._activate_flag(flag_name, "condition_met")
                        elif not should_be_active and current_active:
                            self._deactivate_flag(flag_name, "condition_not_met")
                        
                        self.stats['condition_evaluations'] += 1
                        
                    except Exception:
                        pass  # Ignore condition evaluation errors
                
                # Check for expiration
                if flag.is_expired() and flag.state == FlagState.ACTIVE:
                    self._expire_flag(flag_name)
    
    # ========== Flag Definition and Management ==========
    
    def define_flag(self, name: str, flag_type: TradingFlagType,
                   description: str = "", priority: FlagPriority = FlagPriority.NORMAL,
                   condition_callback: Optional[Callable[[], bool]] = None,
                   auto_expire_seconds: Optional[int] = None,
                   depends_on: Optional[List[str]] = None,
                   conflicts_with: Optional[List[str]] = None) -> bool:
        """
        Define a new trading flag.
        
        Args:
            name: Flag name (must be unique)
            flag_type: Type of flag
            description: Flag description
            priority: Flag priority level
            condition_callback: Optional condition callback
            auto_expire_seconds: Auto-expiration time in seconds
            depends_on: List of flags this flag depends on
            conflicts_with: List of flags this flag conflicts with
            
        Returns:
            True if defined successfully
        """
        with self.flags_lock:
            if name in self.flags:
                return False
            
            flag = TradingFlag(
                name=name,
                flag_type=flag_type,
                description=description,
                priority=priority,
                condition_callback=condition_callback,
                auto_expire_seconds=auto_expire_seconds,
                depends_on=depends_on or [],
                conflicts_with=conflicts_with or []
            )
            
            self.flags[name] = flag
            self.flag_groups[flag_type].add(name)
            self.stats['total_flags'] += 1
            
            return True
    
    def remove_flag(self, name: str) -> bool:
        """Remove a flag definition."""
        with self.flags_lock:
            if name not in self.flags:
                return False
            
            flag = self.flags[name]
            
            # Remove from group
            self.flag_groups[flag.flag_type].discard(name)
            
            # Remove the flag
            del self.flags[name]
            self.stats['total_flags'] -= 1
            
            return True
    
    def get_flag(self, name: str) -> Optional[TradingFlag]:
        """Get flag by name."""
        return self.flags.get(name)
    
    def flag_exists(self, name: str) -> bool:
        """Check if flag exists."""
        return name in self.flags
    
    # ========== Flag State Management ==========
    
    def set_flag(self, name: str, active: bool = True, trigger: str = "manual") -> bool:
        """Set flag state."""
        if active:
            return self.activate_flag(name, trigger)
        else:
            return self.deactivate_flag(name, trigger)
    
    def activate_flag(self, name: str, trigger: str = "manual") -> bool:
        """Activate a flag."""
        return self._activate_flag(name, trigger)
    
    def deactivate_flag(self, name: str, trigger: str = "manual") -> bool:
        """Deactivate a flag."""
        return self._deactivate_flag(name, trigger)
    
    def _activate_flag(self, name: str, trigger: str) -> bool:
        """Internal flag activation with dependency checking."""
        with self.flags_lock:
            if name not in self.flags:
                return False
            
            flag = self.flags[name]
            
            # Check if flag is disabled
            if flag.state == FlagState.DISABLED:
                return False
            
            # Check dependencies
            if not self._check_dependencies(name):
                return False
            
            # Check conflicts
            if not self._resolve_conflicts(name):
                return False
            
            # Activate the flag
            old_state = flag.state
            if flag.activate():
                self.stats['active_flags'] += 1
                self.stats['flag_activations'] += 1
                
                # Create and publish event
                event = FlagEvent(
                    timestamp=datetime.now(),
                    flag_name=name,
                    old_state=old_state,
                    new_state=flag.state,
                    trigger=trigger
                )
                
                self._publish_event(event)
                return True
        
        return False
    
    def _deactivate_flag(self, name: str, trigger: str) -> bool:
        """Internal flag deactivation."""
        with self.flags_lock:
            if name not in self.flags:
                return False
            
            flag = self.flags[name]
            old_state = flag.state
            
            if flag.deactivate():
                self.stats['active_flags'] -= 1
                self.stats['flag_deactivations'] += 1
                
                # Create and publish event
                event = FlagEvent(
                    timestamp=datetime.now(),
                    flag_name=name,
                    old_state=old_state,
                    new_state=flag.state,
                    trigger=trigger
                )
                
                self._publish_event(event)
                return True
        
        return False
    
    def _expire_flag(self, name: str) -> bool:
        """Expire a flag."""
        with self.flags_lock:
            if name not in self.flags:
                return False
            
            flag = self.flags[name]
            old_state = flag.state
            flag.state = FlagState.EXPIRED
            flag.updated_at = datetime.now()
            
            if old_state == FlagState.ACTIVE:
                self.stats['active_flags'] -= 1
            
            # Create and publish event
            event = FlagEvent(
                timestamp=datetime.now(),
                flag_name=name,
                old_state=old_state,
                new_state=flag.state,
                trigger="expired"
            )
            
            self._publish_event(event)
            return True
    
    def _check_dependencies(self, name: str) -> bool:
        """Check if flag dependencies are satisfied."""
        flag = self.flags[name]
        
        for dep_name in flag.depends_on:
            if dep_name in self.flags:
                dep_flag = self.flags[dep_name]
                if not dep_flag.is_active():
                    return False
            else:
                return False  # Dependency doesn't exist
        
        return True
    
    def _resolve_conflicts(self, name: str) -> bool:
        """Resolve flag conflicts."""
        flag = self.flags[name]
        
        for conflict_name in flag.conflicts_with:
            if conflict_name in self.flags:
                conflict_flag = self.flags[conflict_name]
                if conflict_flag.is_active():
                    # Check priorities
                    if flag.priority.value <= conflict_flag.priority.value:
                        return False  # Cannot activate due to higher priority conflict
                    else:
                        # Deactivate conflicting flag
                        self._deactivate_flag(conflict_name, f"conflict_with_{name}")
        
        return True
    
    # ========== Flag Queries ==========
    
    def is_flag_active(self, name: str) -> bool:
        """Check if a flag is active."""
        flag = self.flags.get(name)
        return flag.is_active() if flag else False
    
    def get_active_flags(self, flag_type: Optional[TradingFlagType] = None) -> List[str]:
        """Get list of active flags, optionally filtered by type."""
        active_flags = []
        
        with self.flags_lock:
            for name, flag in self.flags.items():
                if flag.is_active():
                    if flag_type is None or flag.flag_type == flag_type:
                        active_flags.append(name)
        
        return active_flags
    
    def get_flags_by_type(self, flag_type: TradingFlagType) -> List[str]:
        """Get all flags of a specific type."""
        return list(self.flag_groups.get(flag_type, set()))
    
    def get_flag_value(self, name: str, default: Any = None) -> Any:
        """Get flag value."""
        flag = self.flags.get(name)
        return flag.value if flag else default
    
    def set_flag_value(self, name: str, value: Any) -> bool:
        """Set flag value."""
        with self.flags_lock:
            if name in self.flags:
                self.flags[name].value = value
                self.flags[name].updated_at = datetime.now()
                return True
        return False
    
    # ========== Compound Flag Operations ==========
    
    def all_flags_active(self, flag_names: List[str]) -> bool:
        """Check if all specified flags are active."""
        return all(self.is_flag_active(name) for name in flag_names)
    
    def any_flags_active(self, flag_names: List[str]) -> bool:
        """Check if any of the specified flags are active."""
        return any(self.is_flag_active(name) for name in flag_names)
    
    def activate_multiple(self, flag_names: List[str], trigger: str = "batch") -> Dict[str, bool]:
        """Activate multiple flags."""
        results = {}
        for name in flag_names:
            results[name] = self.activate_flag(name, trigger)
        return results
    
    def deactivate_multiple(self, flag_names: List[str], trigger: str = "batch") -> Dict[str, bool]:
        """Deactivate multiple flags."""
        results = {}
        for name in flag_names:
            results[name] = self.deactivate_flag(name, trigger)
        return results
    
    # ========== Event System ==========
    
    def _publish_event(self, event: FlagEvent) -> None:
        """Publish flag event to subscribers."""
        self.flag_events.append(event)
        self.stats['event_notifications'] += 1
        
        # Notify flag-specific subscribers
        for subscriber in self.event_subscribers.get(event.flag_name, []):
            try:
                subscriber(event)
            except Exception:
                pass
        
        # Notify global subscribers
        for subscriber in self.global_subscribers:
            try:
                subscriber(event)
            except Exception:
                pass
    
    def subscribe_to_flag(self, flag_name: str, callback: Callable[[FlagEvent], None]) -> bool:
        """Subscribe to events for a specific flag."""
        if flag_name in self.flags:
            self.event_subscribers[flag_name].append(callback)
            return True
        return False
    
    def subscribe_to_all_flags(self, callback: Callable[[FlagEvent], None]) -> bool:
        """Subscribe to all flag events."""
        if callback not in self.global_subscribers:
            self.global_subscribers.append(callback)
            return True
        return False
    
    def get_flag_events(self, flag_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get flag events history."""
        events = list(self.flag_events)
        
        if flag_name:
            events = [e for e in events if e.flag_name == flag_name]
        
        # Sort by timestamp (newest first) and limit
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                'timestamp': event.timestamp.isoformat(),
                'flag_name': event.flag_name,
                'old_state': event.old_state.value,
                'new_state': event.new_state.value,
                'trigger': event.trigger,
                'details': event.details
            }
            for event in events
        ]
    
    # ========== Information Methods ==========
    
    def get_flags_summary(self) -> Dict[str, Any]:
        """Get comprehensive flags summary."""
        with self.flags_lock:
            active_by_type = defaultdict(int)
            total_by_type = defaultdict(int)
            
            for flag in self.flags.values():
                total_by_type[flag.flag_type.value] += 1
                if flag.is_active():
                    active_by_type[flag.flag_type.value] += 1
            
            return {
                'total_flags': len(self.flags),
                'active_flags': self.stats['active_flags'],
                'by_type': {
                    flag_type.value: {
                        'total': total_by_type[flag_type.value],
                        'active': active_by_type[flag_type.value]
                    }
                    for flag_type in TradingFlagType
                },
                'statistics': self.stats.copy(),
                'recent_events': len(self.flag_events)
            }
    
    def get_flag_details(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific flag."""
        flag = self.flags.get(name)
        if not flag:
            return None
        
        return {
            'name': flag.name,
            'type': flag.flag_type.value,
            'state': flag.state.value,
            'priority': flag.priority.value,
            'description': flag.description,
            'is_active': flag.is_active(),
            'is_expired': flag.is_expired(),
            'created_at': flag.created_at.isoformat(),
            'updated_at': flag.updated_at.isoformat() if flag.updated_at else None,
            'activation_count': flag.activation_count,
            'last_activated': flag.last_activated.isoformat() if flag.last_activated else None,
            'last_deactivated': flag.last_deactivated.isoformat() if flag.last_deactivated else None,
            'value': flag.value,
            'threshold': flag.threshold,
            'auto_expire_seconds': flag.auto_expire_seconds,
            'depends_on': flag.depends_on,
            'conflicts_with': flag.conflicts_with,
            'has_condition': flag.condition_callback is not None
        }
    
    def export_flags_state(self) -> Dict[str, Any]:
        """Export current state of all flags."""
        return {
            name: {
                'state': flag.state.value,
                'value': flag.value,
                'last_activated': flag.last_activated.isoformat() if flag.last_activated else None,
                'activation_count': flag.activation_count
            }
            for name, flag in self.flags.items()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        active_count = len([f for f in self.flags.values() if f.is_active()])
        return f"CFlags(total={len(self.flags)}, active={active_count})"