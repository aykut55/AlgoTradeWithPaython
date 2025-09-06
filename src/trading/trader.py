"""
Trading logic implementation for the algorithmic trading system.

This module contains the CTrader class which manages the trading operations,
position management, order execution, and integration with signals and indicators.
"""

# Check if running as script
if __name__ == "__main__":
    print("Error: This module should be imported, not run directly.")
    print("Use: from src.trading.trader import CTrader")
    print("Or run: python main.py")
    exit(1)

from typing import Optional, Dict, Any, List, Protocol, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

try:
    from ..core.base import CBase, SystemProtocol
    from ..utils.utils import CUtils
    from ..indicators.indicator_manager import CIndicatorManager
    from .signals import CSignals, Direction, SignalType, SignalInfo
except ImportError as e:
    # This happens when running the file directly
    if __name__ == "__main__":
        pass  # Already handled above
    else:
        raise e


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"


@dataclass
class OrderInfo:
    """Information about a trading order."""
    order_id: str
    order_type: OrderType
    direction: Direction
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    bar_number: int = 0
    timestamp: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_pending(self) -> bool:
        return self.status == OrderStatus.PENDING


@dataclass
class TradeInfo:
    """Information about an executed trade."""
    trade_id: str
    direction: Direction
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 1.0
    entry_bar: int = 0
    exit_bar: Optional[int] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    
    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None
    
    @property
    def unrealized_pnl(self) -> float:
        if self.is_closed:
            return self.pnl
        return 0.0


@dataclass
class RiskSettings:
    """Risk management settings."""
    max_position_size: float = 1.0
    take_profit_points: float = 0.0
    stop_loss_points: float = 0.0
    max_daily_loss: float = 0.0
    max_open_positions: int = 1
    position_sizing_method: str = "FIXED"  # FIXED, PERCENT_RISK, KELLY
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    def is_valid(self) -> bool:
        """Validate risk settings."""
        return (self.max_position_size > 0 and 
                self.take_profit_points >= 0 and 
                self.stop_loss_points >= 0)


class CTrader(CBase):
    """
    Main trading class that manages all trading operations.
    
    Integrates with:
    - CSignals for trading signals
    - CIndicatorManager for technical analysis
    - CUtils for utility functions
    """
    
    def __init__(self, id_value: int = 0, name: str = "CTrader"):
        """
        Initialize trading system.
        
        Args:
            id_value: Unique identifier
            name: Trader name
        """
        super().__init__(id_value)
        self.name = name
        
        # Core components
        self.signals = CSignals()
        self.utils = CUtils()
        self.indicators: Optional[CIndicatorManager] = None
        
        # Import kar_al_zarar_kes after avoiding circular import
        from .kar_al_zarar_kes import CKarAlZararKes
        self.kar_al_zarar_kes = CKarAlZararKes()
        self.kar_al_zarar_kes.initialize(None, self)
        
        # Trading state
        self.is_initialized: bool = False
        self.current_bar: int = 0
        self.current_price: float = 0.0
        
        # Risk management
        self.risk_settings = RiskSettings()
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.commission_per_trade: float = 0.0
        
        # Orders and trades
        self.pending_orders: Dict[str, OrderInfo] = {}
        self.filled_orders: Dict[str, OrderInfo] = {}
        self.open_trades: Dict[str, TradeInfo] = {}
        self.closed_trades: List[TradeInfo] = []
        
        # Trading statistics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.max_drawdown: float = 0.0
        self.max_runup: float = 0.0
        
        # Position tracking
        self.current_position_size: float = 0.0
        self.average_entry_price: float = 0.0
        
        # Position checking methods (compatibility with main.py)
        self.position = "FLAT"  # Simple position tracking for compatibility
        
        # Event callbacks
        self.on_signal_generated: Optional[Callable[[SignalInfo], None]] = None
        self.on_order_filled: Optional[Callable[[OrderInfo], None]] = None
        self.on_trade_closed: Optional[Callable[[TradeInfo], None]] = None
    
    def initialize(self, system: SystemProtocol, v: Any) -> None:
        """
        Initialize the trading system with market data.
        
        Args:
            system: System interface
            v: Original data reference
        """
        if not self.market_data.validate():
            raise ValueError("Invalid market data for trader initialization")
        
        # Initialize components
        self.utils.initialize(system)
        
        # Create indicator manager if data is available
        if self.bar_count > 0:
            self.indicators = CIndicatorManager()
            self.indicators.initialize(
                system, v,
                self.open, self.high, self.low, self.close,
                self.volume, self.lot
            )
        
        self.is_initialized = True
        self.show_message(system, f"CTrader '{self.name}' initialized with {self.bar_count} bars")
    
    def Start(self, sistem=None) -> None:
        """
        Start trading system - Python equivalent of C# Start method.
        
        Args:
            sistem: System interface (for compatibility with C# version)
        """
        if not self.is_initialized:
            self.show_message(sistem, f"Warning: CTrader '{self.name}' not initialized before Start()")
        
        # Reset trading state for new session
        self.current_bar = 0
        self.current_price = 0.0
        self.daily_pnl = 0.0
        
        # Clear pending orders from previous session
        self.pending_orders.clear()
        
        # Initialize kar_al_zarar_kes if needed
        if hasattr(self, 'kar_al_zarar_kes') and self.kar_al_zarar_kes:
            self.kar_al_zarar_kes.reset()
        
        self.show_message(sistem, f"CTrader '{self.name}' started successfully")
    
    def update_bar(self, system: SystemProtocol, bar_number: int) -> None:
        """
        Update current bar and process trading logic.
        
        Args:
            system: System interface
            bar_number: Current bar number
        """
        if not self.is_initialized:
            raise RuntimeError("Trader not initialized")
        
        if bar_number < 0 or bar_number >= self.bar_count:
            raise ValueError(f"Invalid bar number: {bar_number}")
        
        self.current_bar = bar_number
        self.current_price = self.close[bar_number]
        
        # Update signals with current data
        self.signals.update_current_data(bar_number, self.current_price)
        
        # Process pending orders
        self._process_pending_orders(system, bar_number)
        
        # Update open positions
        self._update_open_positions(system, bar_number)
        
        # Check risk management
        self._check_risk_management(system, bar_number)
    
    def generate_buy_signal(self, system: SystemProtocol, bar_number: int, 
                           price: Optional[float] = None) -> Optional[SignalInfo]:
        """
        Generate a buy signal.
        
        Args:
            system: System interface
            bar_number: Bar number for signal
            price: Optional price override
            
        Returns:
            SignalInfo if signal generated, None otherwise
        """
        if not self._can_open_position(Direction.LONG):
            return None
        
        signal_price = price or self.close[bar_number]
        
        # Set buy signal
        self.signals.al = True
        
        # Process the signal
        signal_info = self.signals.process_signals(bar_number, signal_price)
        
        if signal_info:
            # Create trade record when opening position
            self._create_trade_record(system, signal_info)
            
            if self.on_signal_generated:
                self.on_signal_generated(signal_info)
        
        # Reset signal after processing
        self.signals.reset_signals()
        
        return signal_info
    
    def generate_sell_signal(self, system: SystemProtocol, bar_number: int,
                            price: Optional[float] = None) -> Optional[SignalInfo]:
        """
        Generate a sell signal.
        
        Args:
            system: System interface
            bar_number: Bar number for signal
            price: Optional price override
            
        Returns:
            SignalInfo if signal generated, None otherwise
        """
        if not self._can_open_position(Direction.SHORT):
            return None
        
        signal_price = price or self.close[bar_number]
        
        # Set sell signal
        self.signals.sat = True
        
        # Process the signal
        signal_info = self.signals.process_signals(bar_number, signal_price)
        
        if signal_info:
            # Create trade record when opening position
            self._create_trade_record(system, signal_info)
            
            if self.on_signal_generated:
                self.on_signal_generated(signal_info)
        
        # Reset signal after processing
        self.signals.reset_signals()
        
        return signal_info
    
    def close_position(self, system: SystemProtocol, bar_number: int,
                      reason: str = "Manual", price: Optional[float] = None) -> Optional[SignalInfo]:
        """
        Close current position.
        
        Args:
            system: System interface
            bar_number: Bar number for close
            reason: Reason for closing
            price: Optional price override
            
        Returns:
            SignalInfo if position closed, None otherwise
        """
        if self.signals.position.is_flat:
            return None
        
        signal_price = price or self.close[bar_number]
        
        # Set flatten signal
        if reason == "TakeProfit":
            self.signals.kar_al = True
        elif reason == "StopLoss":
            self.signals.zarar_kes = True
        else:
            self.signals.flat_ol = True
        
        # Process the signal
        signal_info = self.signals.process_signals(bar_number, signal_price)
        
        if signal_info:
            # Create trade record when closing position
            self._close_trade_record(system, signal_info, reason)
            
            if self.on_signal_generated:
                self.on_signal_generated(signal_info)
        
        # Reset signal after processing
        self.signals.reset_signals()
        
        return signal_info
    
    def set_risk_settings(self, settings: RiskSettings) -> None:
        """Set risk management settings."""
        if not settings.is_valid():
            raise ValueError("Invalid risk settings")
        self.risk_settings = settings
    
    def get_position_info(self) -> Dict[str, Any]:
        """Get current position information."""
        position = self.signals.position
        
        return {
            "direction": position.direction.value,
            "size": position.quantity,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "unrealized_pnl": position.unrealized_pnl,
            "entry_bar": position.entry_bar,
            "current_bar": position.current_bar,
            "bars_in_trade": position.current_bar - position.entry_bar if not position.is_flat else 0
        }
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_win = np.mean([t.pnl for t in self.closed_trades if t.pnl > 0]) if self.winning_trades > 0 else 0.0
        avg_loss = np.mean([t.pnl for t in self.closed_trades if t.pnl < 0]) if self.losing_trades > 0 else 0.0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": abs(avg_win * self.winning_trades / max(1, avg_loss * self.losing_trades)),
            "max_drawdown": self.max_drawdown,
            "max_runup": self.max_runup,
            "open_positions": len(self.open_trades),
            "pending_orders": len(self.pending_orders),
            "current_position": self.get_position_info()
        }
    
    def get_signal_history(self, count: int = 10) -> List[SignalInfo]:
        """Get recent signal history."""
        return self.signals.get_last_signals(count)
    
    def _can_open_position(self, direction: Direction) -> bool:
        """Check if we can open a position in the given direction."""
        # Check if we're already in the same direction
        if direction == Direction.LONG and self.signals.position.is_long:
            return False
        if direction == Direction.SHORT and self.signals.position.is_short:
            return False
        
        # Check daily loss limit
        if (self.risk_settings.max_daily_loss > 0 and 
            abs(self.daily_pnl) >= self.risk_settings.max_daily_loss):
            return False
        
        # Allow position reversal - we can open opposite direction even if already in position
        # The signal processing will handle closing the existing position first
        
        return True
    
    def _process_pending_orders(self, system: SystemProtocol, bar_number: int) -> None:
        """Process pending orders against current market data."""
        current_high = self.high[bar_number]
        current_low = self.low[bar_number]
        current_close = self.close[bar_number]
        
        orders_to_fill = []
        
        for order_id, order in self.pending_orders.items():
            filled = False
            fill_price = 0.0
            
            if order.order_type == OrderType.MARKET:
                filled = True
                fill_price = current_close
            elif order.order_type == OrderType.LIMIT:
                if order.direction == Direction.LONG and order.price and current_low <= order.price:
                    filled = True
                    fill_price = order.price
                elif order.direction == Direction.SHORT and order.price and current_high >= order.price:
                    filled = True
                    fill_price = order.price
            elif order.order_type == OrderType.STOP:
                if order.direction == Direction.LONG and order.stop_price and current_high >= order.stop_price:
                    filled = True
                    fill_price = max(order.stop_price, current_close)
                elif order.direction == Direction.SHORT and order.stop_price and current_low <= order.stop_price:
                    filled = True
                    fill_price = min(order.stop_price, current_close)
            
            if filled:
                orders_to_fill.append((order_id, fill_price))
        
        # Fill orders
        for order_id, fill_price in orders_to_fill:
            self._fill_order(system, order_id, fill_price, bar_number)
    
    def _fill_order(self, system: SystemProtocol, order_id: str, 
                   fill_price: float, bar_number: int) -> None:
        """Fill a pending order."""
        if order_id not in self.pending_orders:
            return
        
        order = self.pending_orders[order_id]
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        
        # Move to filled orders
        self.filled_orders[order_id] = order
        del self.pending_orders[order_id]
        
        # Create trade record
        trade_id = f"TRADE_{len(self.closed_trades) + len(self.open_trades) + 1}"
        trade = TradeInfo(
            trade_id=trade_id,
            direction=order.direction,
            entry_price=fill_price,
            quantity=order.quantity,
            entry_bar=bar_number,
            entry_time=datetime.now()
        )
        
        self.open_trades[trade_id] = trade
        
        if self.on_order_filled:
            self.on_order_filled(order)
    
    def _update_open_positions(self, system: SystemProtocol, bar_number: int) -> None:
        """Update unrealized P&L for open positions."""
        current_price = self.close[bar_number]
        
        for trade_id, trade in self.open_trades.items():
            if trade.direction == Direction.LONG:
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:  # SHORT
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
            
            trade.pnl = unrealized_pnl - self.commission_per_trade
    
    def _close_trade_record(self, system: SystemProtocol, signal_info: SignalInfo, reason: str) -> None:
        """Close trade record and update statistics."""
        if not self.open_trades:
            return
        
        # Find the open trade to close
        trade_id = list(self.open_trades.keys())[0]  # Close first open trade
        trade = self.open_trades[trade_id]
        
        # Update trade info
        trade.exit_price = signal_info.price
        trade.exit_bar = signal_info.bar_number
        trade.exit_time = datetime.now()
        
        # Calculate final P&L
        if trade.direction == Direction.LONG:
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
        
        trade.pnl -= self.commission_per_trade  # Account for commission
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.daily_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update drawdown/runup
        if self.total_pnl > self.max_runup:
            self.max_runup = self.total_pnl
        
        drawdown = self.max_runup - self.total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[trade_id]
        
        if self.on_trade_closed:
            self.on_trade_closed(trade)
        
        self.show_message(system, f"Trade closed: {reason} P&L: {trade.pnl:.2f}")
    
    def _create_trade_record(self, system: SystemProtocol, signal_info: SignalInfo) -> None:
        """Create a trade record when opening position."""
        if signal_info.signal_type not in [SignalType.BUY, SignalType.SELL]:
            return
        
        # Create trade record
        trade_id = f"TRADE_{len(self.closed_trades) + len(self.open_trades) + 1}"
        trade = TradeInfo(
            trade_id=trade_id,
            direction=signal_info.direction,
            entry_price=signal_info.price,
            quantity=1.0,  # Default quantity
            entry_bar=signal_info.bar_number,
            entry_time=datetime.now()
        )
        
        self.open_trades[trade_id] = trade
        self.show_message(system, f"Trade opened: {signal_info.direction.value} @ {signal_info.price:.2f}")
    
    def _check_risk_management(self, system: SystemProtocol, bar_number: int) -> None:
        """Check risk management rules and trigger stops if needed."""
        if self.signals.position.is_flat:
            return
        
        position = self.signals.position
        current_price = self.close[bar_number]
        
        # Take profit check
        if self.risk_settings.take_profit_points > 0:
            if position.is_long:
                tp_price = position.entry_price + self.risk_settings.take_profit_points
                if current_price >= tp_price:
                    self.close_position(system, bar_number, "TakeProfit", tp_price)
                    return
            elif position.is_short:
                tp_price = position.entry_price - self.risk_settings.take_profit_points
                if current_price <= tp_price:
                    self.close_position(system, bar_number, "TakeProfit", tp_price)
                    return
        
        # Stop loss check
        if self.risk_settings.stop_loss_points > 0:
            if position.is_long:
                sl_price = position.entry_price - self.risk_settings.stop_loss_points
                if current_price <= sl_price:
                    self.close_position(system, bar_number, "StopLoss", sl_price)
                    return
            elif position.is_short:
                sl_price = position.entry_price + self.risk_settings.stop_loss_points
                if current_price >= sl_price:
                    self.close_position(system, bar_number, "StopLoss", sl_price)
                    return
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new trading day)."""
        self.daily_pnl = 0.0
    
    def ResetDateTimes(self, sistem=None):
        """Reset date times - Python equivalent of C# ResetDateTimes"""
        from datetime import datetime
        
        use_last_bar_datetime = True
        
        # V array equivalent - assuming we have access to bar data
        if hasattr(self, '_bar_data') and self._bar_data is not None and len(self._bar_data) > 0:
            self.StartDateTime = self._bar_data[0].get('date', datetime.now())
            if use_last_bar_datetime and hasattr(sistem, 'BarSayisi'):
                self.StopDateTime = self._bar_data[sistem.BarSayisi - 1].get('date', datetime.now())
            else:
                self.StopDateTime = datetime.now()
        else:
            # Default to current time if no data available
            self.StartDateTime = datetime.now()
            self.StopDateTime = datetime.now()
        
        # Set date and time components
        self.StartDate = self.StartDateTime.date()
        self.StopDate = self.StopDateTime.date()
        self.StartTime = self.StartDateTime.time()
        self.StopTime = self.StopDateTime.time()
        
        # String formats
        datetime_format = "%Y.%m.%d %H:%M:%S"
        date_format = "%Y.%m.%d"
        time_format = "%H:%M:%S"
        
        self.StartDateTimeStr = self.StartDateTime.strftime(datetime_format)
        self.StopDateTimeStr = self.StopDateTime.strftime(datetime_format)
        self.StartDateStr = self.StartDate.strftime(date_format)
        self.StopDateStr = self.StopDate.strftime(date_format)
        self.StartTimeStr = self.StartTime.strftime(time_format)
        self.StopTimeStr = self.StopTime.strftime(time_format)
        
        return self
    
    def SetDateTimes(self, *args):
        """Set date times - Python equivalent of C# SetDateTimes (overloaded)"""
        from datetime import datetime
        
        # Handle different argument patterns
        if len(args) == 2:
            # SetDateTimes(StartDateTime, StopDateTime)
            start_datetime, stop_datetime = args
            return self._set_date_times_simple(start_datetime, stop_datetime)
        elif len(args) == 3 and args[0] is not None:
            # SetDateTimes(sistem, StartDateTime, StopDateTime) - with sistem parameter
            sistem, start_datetime, stop_datetime = args
            return self._set_date_times_simple(start_datetime, stop_datetime)
        elif len(args) == 4:
            # SetDateTimes(StartDate, StartTime, StopDate, StopTime)
            start_date, start_time, stop_date, stop_time = args
            return self._set_date_times_detailed(start_date, start_time, stop_date, stop_time)
        elif len(args) == 5:
            # SetDateTimes(sistem, StartDate, StartTime, StopDate, StopTime)
            sistem, start_date, start_time, stop_date, stop_time = args
            return self._set_date_times_detailed(start_date, start_time, stop_date, stop_time)
        else:
            raise ValueError(f"Invalid number of arguments for SetDateTimes: {len(args)}")
    
    def _set_date_times_detailed(self, start_date: str, start_time: str, stop_date: str, stop_time: str):
        """SetDateTimes with separate date and time parameters"""
        from datetime import datetime
        
        date1 = start_date.strip()
        time1 = start_time.strip()
        date2 = stop_date.strip()
        time2 = stop_time.strip()
        datetime1 = date1 + " " + time1
        datetime2 = date2 + " " + time2
        suffix_date = "09:30:00"
        prefix_time = "01.01.1900"
        
        # Parse date times (simplified TimeUtils.GetDateTime equivalent)
        self.StartDateTime = self._parse_datetime(date1, time1)
        self.StopDateTime = self._parse_datetime(date2, time2)
        self.StartDate = self._parse_datetime(date1 + " " + suffix_date)
        self.StopDate = self._parse_datetime(date2 + " " + suffix_date)
        self.StartTime = self._parse_datetime(prefix_time + " " + time1)
        self.StopTime = self._parse_datetime(prefix_time + " " + time2)
        
        # String formats
        datetime_format = "%Y.%m.%d %H:%M:%S"
        date_format = "%Y.%m.%d"
        time_format = "%H:%M:%S"
        
        self.StartDateTimeStr = self.StartDateTime.strftime(datetime_format)
        self.StopDateTimeStr = self.StopDateTime.strftime(datetime_format)
        self.StartDateStr = self.StartDate.strftime(date_format)
        self.StopDateStr = self.StopDate.strftime(date_format)
        self.StartTimeStr = self.StartTime.strftime(time_format)
        self.StopTimeStr = self.StopTime.strftime(time_format)
        
        return self
    
    def _set_date_times_simple(self, start_datetime: str, stop_datetime: str):
        """SetDateTimes with datetime strings"""
        from datetime import datetime
        
        datetime1 = start_datetime.strip()
        datetime2 = stop_datetime.strip()
        
        # Extract date and time parts (assuming format: "dd.MM.yyyy HH:mm:ss")
        if len(datetime1) >= 10:
            date1 = datetime1[:10]
            time1 = datetime1[11:] if len(datetime1) > 11 else "00:00:00"
        else:
            date1 = datetime1
            time1 = "00:00:00"
            
        if len(datetime2) >= 10:
            date2 = datetime2[:10]
            time2 = datetime2[11:] if len(datetime2) > 11 else "00:00:00"
        else:
            date2 = datetime2
            time2 = "00:00:00"
            
        suffix_date = "09:30:00"
        prefix_time = "01.01.1900"
        
        # Parse date times
        self.StartDateTime = self._parse_datetime(datetime1)
        self.StopDateTime = self._parse_datetime(datetime2)
        self.StartDate = self._parse_datetime(date1 + " " + suffix_date)
        self.StopDate = self._parse_datetime(date2 + " " + suffix_date)
        self.StartTime = self._parse_datetime(prefix_time + " " + time1)
        self.StopTime = self._parse_datetime(prefix_time + " " + time2)
        
        # String formats
        datetime_format = "%Y.%m.%d %H:%M:%S"
        date_format = "%Y.%m.%d"
        time_format = "%H:%M:%S"
        
        self.StartDateTimeStr = self.StartDateTime.strftime(datetime_format)
        self.StopDateTimeStr = self.StopDateTime.strftime(datetime_format)
        self.StartDateStr = self.StartDate.strftime(date_format)
        self.StopDateStr = self.StopDate.strftime(date_format)
        self.StartTimeStr = self.StartTime.strftime(time_format)
        self.StopTimeStr = self.StopTime.strftime(time_format)
        
        return self
    
    def SetDateTime(self, *args):
        """Set date time - Python equivalent of C# SetDateTime (overloaded)"""
        from datetime import datetime
        
        if len(args) == 1:
            # SetDateTime(StartDateTime)
            start_datetime = args[0]
            return self.SetDateTimes(start_datetime, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        elif len(args) == 2:
            # SetDateTime(StartDate, StartTime) or SetDateTime(sistem, StartDateTime)
            if ":" in args[1]:  # Second arg is time
                start_date, start_time = args
                start_datetime = start_date + " " + start_time
                return self.SetDateTime(start_datetime)
            else:  # Second arg is datetime string
                sistem, start_datetime = args
                return self.SetDateTimes(start_datetime, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        elif len(args) == 3:
            # SetDateTime(sistem, StartDate, StartTime)
            sistem, start_date, start_time = args
            start_datetime = start_date + " " + start_time
            return self.SetDateTime(start_datetime)
        else:
            raise ValueError(f"Invalid number of arguments for SetDateTime: {len(args)}")
    
    def _parse_datetime(self, *args):
        """Parse datetime string(s) - simplified TimeUtils.GetDateTime equivalent"""
        from datetime import datetime
        
        if len(args) == 1:
            # Single datetime string
            datetime_str = args[0]
            # Try different formats commonly used
            formats = [
                "%d.%m.%Y %H:%M:%S",  # dd.MM.yyyy HH:mm:ss
                "%Y.%m.%d %H:%M:%S",  # yyyy.MM.dd HH:mm:ss
                "%d.%m.%Y",           # dd.MM.yyyy
                "%Y.%m.%d",           # yyyy.MM.dd
                "%H:%M:%S"            # HH:mm:ss
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            
            # If no format matches, try default parsing
            try:
                return datetime.fromisoformat(datetime_str.replace('.', '-'))
            except:
                return datetime.now()
                
        elif len(args) == 2:
            # Date and time strings
            date_str, time_str = args
            datetime_str = date_str + " " + time_str
            return self._parse_datetime(datetime_str)
        else:
            return datetime.now()
    
    # Compatibility methods for main.py
    def IsSonYonA(self) -> bool:
        """Check if current position is long - compatibility method."""
        return self.signals.is_son_yon_a()
    
    def IsSonYonS(self) -> bool:
        """Check if current position is short - compatibility method."""
        return self.signals.is_son_yon_s()
    
    def IsSonYonF(self) -> bool:
        """Check if current position is flat - compatibility method."""
        return self.signals.is_son_yon_f()
    
    # Compatibility method used in main.py for accessing KarAlZararKes
    @property
    def KarAlZararKes(self):
        """Property to access kar_al_zarar_kes for compatibility with main.py."""
        return self.kar_al_zarar_kes
    
    def SonFiyataGoreKarAlSeviyeHesapla(self, i: int, param1: float, param2: float, param3: float) -> float:
        """Compatibility method for main.py KarAlZararKes calls."""
        return self.kar_al_zarar_kes.son_fiyata_gore_kar_al_seviye_hesapla_range(None, i, int(param1), int(param2), int(param3))
    
    def SonFiyataGoreZararKesSeviyeHesapla(self, i: int, param1: float, param2: float, param3: float) -> float:
        """Compatibility method for main.py KarAlZararKes calls."""
        return self.kar_al_zarar_kes.son_fiyata_gore_zarar_kes_seviye_hesapla_range(None, i, int(param1), int(param2), int(param3))
    
    def EmirleriSetle(self, sistem=None, bar_index: int = 0, al: bool = False, sat: bool = False, 
                      flat_ol: bool = False, pas_gec: bool = False, kar_al: bool = False, zarar_kes: bool = False) -> int:
        """
        Set trading orders - Python equivalent of C# EmirleriSetle method.
        
        Args:
            sistem: System interface (for compatibility)
            bar_index: Current bar index
            al: Buy signal
            sat: Sell signal
            flat_ol: Flatten position signal (default False)
            pas_gec: Pass signal (default False)
            kar_al: Take profit signal (default False)
            zarar_kes: Stop loss signal (default False)
            
        Returns:
            int: Result status (0 for success)
        """
        result = 0
        i = bar_index
        
        # Set signals in the signals object
        self.signals.al = al
        self.signals.sat = sat
        self.signals.flat_ol = flat_ol
        self.signals.pas_gec = pas_gec
        self.signals.kar_al = kar_al
        self.signals.zarar_kes = zarar_kes
        
        return result
    
    def IslemZamanFiltresiUygula(self, sistem=None, bar_index: int = 0, filter_mode: int = 0) -> tuple[bool, bool, int]:
        """
        Apply time filtering - Python equivalent of C# IslemZamanFiltresiUygula method.
        
        Args:
            sistem: System interface (for compatibility)
            bar_index: Current bar index
            filter_mode: Filter mode (0-6)
                0: No filtering (always enabled)
                1: Time range filtering (startTime to stopTime)
                2: Date range filtering (startDate to stopDate)
                3: DateTime range filtering (startDateTime to stopDateTime)
                4: Start time only filtering (from startTime onwards)
                5: Start date only filtering (from startDate onwards)
                6: Start datetime only filtering (from startDateTime onwards)
                
        Returns:
            tuple[bool, bool, int]: (is_trade_enabled, is_poz_kapat_enabled, check_result)
                check_result: -1 (before range), 0 (in range), 1 (after range)
        """
        from datetime import datetime
        
        i = bar_index
        is_trade_enabled = False
        is_poz_kapat_enabled = False
        check_result = 0
        
        # Get bar datetime (placeholder - in real implementation this would come from V[i].Date)
        bar_datetime = datetime.now()  # This should be replaced with actual bar datetime
        
        # Get datetime strings
        start_date_time_str = getattr(self, 'StartDateTimeStr', '01.01.1900 00:00:00')
        stop_date_time_str = getattr(self, 'StopDateTimeStr', '01.01.2100 23:59:59')
        start_date_str = getattr(self, 'StartDateStr', '01.01.1900')
        stop_date_str = getattr(self, 'StopDateStr', '01.01.2100')
        start_time_str = getattr(self, 'StartTimeStr', '00:00:00')
        stop_time_str = getattr(self, 'StopTimeStr', '23:59:59')
        
        now_date_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        now_date = datetime.now().strftime("%d.%m.%Y")
        now_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if time filtering is enabled
        use_time_filtering = self.signals.time_filtering_enabled if hasattr(self.signals, 'time_filtering_enabled') else False
        
        if use_time_filtering:
            # Debug info for last bar (placeholder for Sistem.BarSayisi check)
            if i == 999:  # Assuming last bar for debug
                debug_info = f"""
  {start_date_time_str}
  {stop_date_time_str}
  {start_date_str}
  {stop_date_str}
  {start_time_str}
  {stop_time_str}
  {now_date_time}
  {now_date}
  {now_time}
  FilterMode = {filter_mode}
  CTrader::IslemZamanFiltresiUygula
"""
                # In C# this would be Sistem.Mesaj(debug_info) - placeholder for message display
                
            if filter_mode == 0:
                # No filtering - always enabled
                is_trade_enabled = True
                check_result = 0
                
            elif filter_mode == 1:
                # Time range filtering (startTime to stopTime)
                start_time_check = self._check_bar_time_with(sistema=sistem, bar_index=i, time_str=start_time_str)
                stop_time_check = self._check_bar_time_with(sistema=sistem, bar_index=i, time_str=stop_time_str)
                
                if start_time_check >= 0 and stop_time_check < 0:
                    is_trade_enabled = True
                    check_result = 0
                elif start_time_check < 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = -1
                elif stop_time_check >= 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = 1
                    
            elif filter_mode == 2:
                # Date range filtering (startDate to stopDate)
                start_date_check = self._check_bar_date_with(sistema=sistem, bar_index=i, date_str=start_date_str)
                stop_date_check = self._check_bar_date_with(sistema=sistem, bar_index=i, date_str=stop_date_str)
                
                if start_date_check >= 0 and stop_date_check < 0:
                    is_trade_enabled = True
                    check_result = 0
                elif start_date_check < 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = -1
                elif stop_date_check >= 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = 1
                    
            elif filter_mode == 3:
                # DateTime range filtering (startDateTime to stopDateTime)
                start_datetime_check = self._check_bar_datetime_with(sistema=sistem, bar_index=i, datetime_str=start_date_time_str)
                stop_datetime_check = self._check_bar_datetime_with(sistema=sistem, bar_index=i, datetime_str=stop_date_time_str)
                
                if start_datetime_check >= 0 and stop_datetime_check < 0:
                    is_trade_enabled = True
                    check_result = 0
                elif start_datetime_check < 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = -1
                elif stop_datetime_check >= 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = 1
                    
            elif filter_mode == 4:
                # Start time only filtering (from startTime onwards)
                start_time_check = self._check_bar_time_with(sistema=sistem, bar_index=i, time_str=start_time_str)
                
                if start_time_check >= 0:
                    is_trade_enabled = True
                    check_result = 0
                elif start_time_check < 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = -1
                    
            elif filter_mode == 5:
                # Start date only filtering (from startDate onwards)
                start_date_check = self._check_bar_date_with(sistema=sistem, bar_index=i, date_str=start_date_str)
                
                if start_date_check >= 0:
                    is_trade_enabled = True
                    check_result = 0
                elif start_date_check < 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = -1
                    
            elif filter_mode == 6:
                # Start datetime only filtering (from startDateTime onwards)
                start_datetime_check = self._check_bar_datetime_with(sistema=sistem, bar_index=i, datetime_str=start_date_time_str)
                
                if start_datetime_check >= 0:
                    is_trade_enabled = True
                    check_result = 0
                elif start_datetime_check < 0:
                    if not self.IsSonYonF():
                        is_poz_kapat_enabled = True
                    check_result = -1
        else:
            # Time filtering disabled - always allow trading
            is_trade_enabled = True
            check_result = 0
        
        return is_trade_enabled, is_poz_kapat_enabled, check_result
    
    def _check_bar_time_with(self, sistema=None, bar_index: int = 0, time_str: str = "00:00:00") -> int:
        """
        Check bar time against reference time - placeholder for TimeUtils.CheckBarTimeWith.
        
        Args:
            sistema: System interface
            bar_index: Bar index
            time_str: Time string to compare
            
        Returns:
            int: -1 (before), 0 (equal), 1 (after)
        """
        # Placeholder implementation - in real system this would use TimeUtils
        from datetime import datetime, time
        
        try:
            # Parse reference time
            ref_time = datetime.strptime(time_str, "%H:%M:%S").time()
            
            # Get current bar time (placeholder - should come from actual bar data)
            current_time = datetime.now().time()
            
            if current_time < ref_time:
                return -1
            elif current_time == ref_time:
                return 0
            else:
                return 1
        except:
            return 0
    
    def _check_bar_date_with(self, sistema=None, bar_index: int = 0, date_str: str = "01.01.1900") -> int:
        """
        Check bar date against reference date - placeholder for TimeUtils.CheckBarDateWith.
        
        Args:
            sistema: System interface
            bar_index: Bar index
            date_str: Date string to compare
            
        Returns:
            int: -1 (before), 0 (equal), 1 (after)
        """
        # Placeholder implementation - in real system this would use TimeUtils
        from datetime import datetime
        
        try:
            # Parse reference date
            ref_date = datetime.strptime(date_str, "%d.%m.%Y").date()
            
            # Get current bar date (placeholder - should come from actual bar data)
            current_date = datetime.now().date()
            
            if current_date < ref_date:
                return -1
            elif current_date == ref_date:
                return 0
            else:
                return 1
        except:
            return 0
    
    def _check_bar_datetime_with(self, sistema=None, bar_index: int = 0, datetime_str: str = "01.01.1900 00:00:00") -> int:
        """
        Check bar datetime against reference datetime - placeholder for TimeUtils.CheckBarDateTimeWith.
        
        Args:
            sistema: System interface
            bar_index: Bar index
            datetime_str: DateTime string to compare
            
        Returns:
            int: -1 (before), 0 (equal), 1 (after)
        """
        # Placeholder implementation - in real system this would use TimeUtils
        from datetime import datetime
        
        try:
            # Parse reference datetime
            ref_datetime = datetime.strptime(datetime_str, "%d.%m.%Y %H:%M:%S")
            
            # Get current bar datetime (placeholder - should come from actual bar data)
            current_datetime = datetime.now()
            
            if current_datetime < ref_datetime:
                return -1
            elif current_datetime == ref_datetime:
                return 0
            else:
                return 1
        except:
            return 0

    def __repr__(self) -> str:
        """String representation of trader."""
        position_info = self.get_position_info()
        return f"CTrader(name='{self.name}', position={position_info['direction']}, pnl={self.total_pnl:.2f})"