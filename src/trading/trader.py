"""
Trading logic implementation for the algorithmic trading system.

This module contains the CTrader class which manages the trading operations,
position management, order execution, and integration with signals and indicators.
"""

from typing import Optional, Dict, Any, List, Protocol, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from ..core.base import CBase, SystemProtocol
from ..utils.utils import CUtils
from ..indicators.indicator_manager import CIndicatorManager
from .signals import CSignals, Direction, SignalType, SignalInfo


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
    
    def __repr__(self) -> str:
        """String representation of trader."""
        position_info = self.get_position_info()
        return f"CTrader(name='{self.name}', position={position_info['direction']}, pnl={self.total_pnl:.2f})"