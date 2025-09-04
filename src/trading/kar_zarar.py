"""
Profit & Loss (P&L) calculation system for algorithmic trading.

This module contains the CKarZarar class which handles comprehensive
profit and loss calculations, trade tracking, performance metrics,
and advanced P&L analysis for trading systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

from ..core.base import SystemProtocol, PositionSide, TradingDecision
from ..portfolio.asset_manager import CVarlikManager, AssetType, CurrencyType


class PnLCalculationMethod(Enum):
    """P&L calculation methods."""
    FIFO = "FIFO"                   # First In, First Out
    LIFO = "LIFO"                   # Last In, First Out
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"  # Weighted Average Cost
    SPECIFIC_IDENTIFICATION = "SPECIFIC_IDENTIFICATION"  # Specific lot identification


class PnLType(Enum):
    """P&L types."""
    REALIZED = "REALIZED"           # Closed positions
    UNREALIZED = "UNREALIZED"       # Open positions
    TOTAL = "TOTAL"                 # Realized + Unrealized


@dataclass
class TradeRecord:
    """Individual trade record."""
    
    trade_id: str
    timestamp: datetime
    symbol: str
    side: PositionSide
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    # Trade value calculations
    gross_value: float = 0.0
    net_value: float = 0.0
    
    # Matching information for P&L calculation
    matched_trades: List[str] = field(default_factory=list)
    remaining_quantity: float = 0.0
    
    def __post_init__(self):
        """Calculate trade values after initialization."""
        self.gross_value = abs(self.quantity * self.price)
        self.net_value = self.gross_value + self.commission + self.slippage
        self.remaining_quantity = abs(self.quantity)


@dataclass
class PnLRecord:
    """P&L calculation record."""
    
    pnl_id: str
    timestamp: datetime
    symbol: str
    
    # Trade information
    entry_trade_id: str
    exit_trade_id: str
    quantity: float
    
    # Price information
    entry_price: float
    exit_price: float
    
    # P&L calculations
    gross_pnl: float = 0.0
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    
    # Performance metrics
    pnl_percentage: float = 0.0
    holding_period: timedelta = timedelta()
    
    def __post_init__(self):
        """Calculate P&L values after initialization."""
        price_diff = self.exit_price - self.entry_price
        self.gross_pnl = price_diff * self.quantity
        self.net_pnl = self.gross_pnl - self.commission_cost - self.slippage_cost
        
        if self.entry_price > 0:
            self.pnl_percentage = (self.net_pnl / (self.entry_price * abs(self.quantity))) * 100


@dataclass
class PositionSummary:
    """Position summary for a symbol."""
    
    symbol: str
    
    # Position information
    net_quantity: float = 0.0
    average_price: float = 0.0
    total_cost: float = 0.0
    
    # P&L information
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Risk metrics
    max_position_size: float = 0.0
    max_unrealized_loss: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.net_quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.net_quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.net_quantity) < 1e-8


class CKarZarar:
    """
    Comprehensive Profit & Loss calculation system.
    
    Features:
    - Multiple P&L calculation methods (FIFO, LIFO, etc.)
    - Realized and unrealized P&L tracking
    - Commission and slippage integration
    - Multi-currency support
    - Advanced performance metrics
    - Position tracking and management
    - Tax reporting capabilities
    """
    
    def __init__(self, asset_manager: Optional[CVarlikManager] = None):
        """Initialize P&L calculator."""
        self.asset_manager = asset_manager
        self.calculation_method = PnLCalculationMethod.FIFO
        self.is_initialized = False
        
        # Trade and P&L storage
        self.trades: Dict[str, TradeRecord] = {}
        self.pnl_records: Dict[str, PnLRecord] = {}
        self.positions: Dict[str, PositionSummary] = {}
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Trade counters
        self.trade_counter = 0
        self.pnl_counter = 0
        
        # Currency conversion (if needed)
        self.base_currency = CurrencyType.TL
        self.currency_rates: Dict[CurrencyType, float] = {
            CurrencyType.TL: 1.0,
            CurrencyType.USD: 30.0,  # Example rate
            CurrencyType.EUR: 33.0   # Example rate
        }
    
    def initialize(self, system: SystemProtocol) -> 'CKarZarar':
        """
        Initialize P&L calculator.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        if self.asset_manager:
            self.asset_manager.initialize(system)
        
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CKarZarar':
        """
        Reset P&L calculator.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.trades.clear()
        self.pnl_records.clear()
        self.positions.clear()
        
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        self.trade_counter = 0
        self.pnl_counter = 0
        
        return self
    
    # ========== Configuration Methods ==========
    
    def set_calculation_method(self, method: PnLCalculationMethod) -> 'CKarZarar':
        """Set P&L calculation method."""
        self.calculation_method = method
        return self
    
    def set_base_currency(self, currency: CurrencyType) -> 'CKarZarar':
        """Set base currency for P&L calculations."""
        self.base_currency = currency
        return self
    
    def update_currency_rate(self, currency: CurrencyType, rate: float) -> 'CKarZarar':
        """Update currency conversion rate."""
        self.currency_rates[currency] = rate
        return self
    
    # ========== Trade Recording Methods ==========
    
    def record_trade(self, symbol: str, side: PositionSide, quantity: float,
                    price: float, timestamp: Optional[datetime] = None,
                    commission: float = 0.0, slippage: float = 0.0) -> str:
        """
        Record a new trade.
        
        Args:
            symbol: Trading symbol
            side: Position side (BUY/SELL)
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp (default: now)
            commission: Commission cost
            slippage: Slippage cost
            
        Returns:
            Trade ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.trade_counter += 1
        trade_id = f"T{self.trade_counter:06d}"
        
        # Auto-calculate commission and slippage if asset manager available
        if self.asset_manager and commission == 0.0:
            commission = self.asset_manager.calculate_commission(price, side == PositionSide.BUY)
        
        if self.asset_manager and slippage == 0.0:
            slippage = self.asset_manager.calculate_slippage(price)
        
        # Create trade record
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage
        )
        
        self.trades[trade_id] = trade
        
        # Update totals
        self.total_commission += commission
        self.total_slippage += slippage
        
        # Process trade for P&L calculation
        self._process_trade(trade)
        
        return trade_id
    
    def _process_trade(self, trade: TradeRecord) -> None:
        """Process trade for P&L calculation."""
        symbol = trade.symbol
        
        # Initialize position if doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = PositionSummary(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Get opposing trades for matching
        opposing_trades = self._get_opposing_trades(trade)
        
        if opposing_trades:
            # Match with opposing trades to calculate P&L
            self._match_trades(trade, opposing_trades)
        
        # Update position summary
        self._update_position_summary(trade, position)
    
    def _get_opposing_trades(self, trade: TradeRecord) -> List[TradeRecord]:
        """Get opposing trades for P&L matching."""
        symbol = trade.symbol
        opposing_side = PositionSide.SELL if trade.side == PositionSide.BUY else PositionSide.BUY
        
        # Get all opposing trades with remaining quantity
        opposing_trades = []
        for existing_trade in self.trades.values():
            if (existing_trade.symbol == symbol and 
                existing_trade.side == opposing_side and 
                existing_trade.remaining_quantity > 1e-8):
                opposing_trades.append(existing_trade)
        
        # Sort based on calculation method
        if self.calculation_method == PnLCalculationMethod.FIFO:
            opposing_trades.sort(key=lambda x: x.timestamp)
        elif self.calculation_method == PnLCalculationMethod.LIFO:
            opposing_trades.sort(key=lambda x: x.timestamp, reverse=True)
        
        return opposing_trades
    
    def _match_trades(self, new_trade: TradeRecord, opposing_trades: List[TradeRecord]) -> None:
        """Match trades and calculate P&L."""
        remaining_quantity = abs(new_trade.quantity)
        
        for opposing_trade in opposing_trades:
            if remaining_quantity <= 1e-8:
                break
            
            # Calculate match quantity
            available_quantity = opposing_trade.remaining_quantity
            match_quantity = min(remaining_quantity, available_quantity)
            
            # Create P&L record
            pnl_record = self._create_pnl_record(new_trade, opposing_trade, match_quantity)
            self.pnl_records[pnl_record.pnl_id] = pnl_record
            
            # Update realized P&L
            self.total_realized_pnl += pnl_record.net_pnl
            
            # Update trade records
            opposing_trade.remaining_quantity -= match_quantity
            opposing_trade.matched_trades.append(new_trade.trade_id)
            new_trade.matched_trades.append(opposing_trade.trade_id)
            
            remaining_quantity -= match_quantity
        
        # Update remaining quantity for new trade
        new_trade.remaining_quantity = remaining_quantity
    
    def _create_pnl_record(self, entry_trade: TradeRecord, exit_trade: TradeRecord,
                          quantity: float) -> PnLRecord:
        """Create P&L record from matched trades."""
        self.pnl_counter += 1
        pnl_id = f"P{self.pnl_counter:06d}"
        
        # Determine entry and exit based on chronological order
        if entry_trade.timestamp <= exit_trade.timestamp:
            entry_price = entry_trade.price
            exit_price = exit_trade.price
            entry_id = entry_trade.trade_id
            exit_id = exit_trade.trade_id
            holding_period = exit_trade.timestamp - entry_trade.timestamp
        else:
            entry_price = exit_trade.price
            exit_price = entry_trade.price
            entry_id = exit_trade.trade_id
            exit_id = entry_trade.trade_id
            holding_period = entry_trade.timestamp - exit_trade.timestamp
        
        # Calculate proportional costs
        entry_commission = entry_trade.commission * (quantity / abs(entry_trade.quantity))
        exit_commission = exit_trade.commission * (quantity / abs(exit_trade.quantity))
        entry_slippage = entry_trade.slippage * (quantity / abs(entry_trade.quantity))
        exit_slippage = exit_trade.slippage * (quantity / abs(exit_trade.quantity))
        
        return PnLRecord(
            pnl_id=pnl_id,
            timestamp=max(entry_trade.timestamp, exit_trade.timestamp),
            symbol=entry_trade.symbol,
            entry_trade_id=entry_id,
            exit_trade_id=exit_id,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            commission_cost=entry_commission + exit_commission,
            slippage_cost=entry_slippage + exit_slippage,
            holding_period=holding_period
        )
    
    def _update_position_summary(self, trade: TradeRecord, position: PositionSummary) -> None:
        """Update position summary with new trade."""
        trade_quantity = trade.quantity if trade.side == PositionSide.BUY else -trade.quantity
        
        # Update position quantity and average price
        if position.net_quantity == 0:
            # Opening new position
            position.net_quantity = trade_quantity
            position.average_price = trade.price
            position.total_cost = trade.net_value
        else:
            # Adding to existing position
            total_value = position.total_cost + (trade_quantity * trade.price)
            position.net_quantity += trade_quantity
            
            if position.net_quantity != 0:
                position.average_price = total_value / position.net_quantity
                position.total_cost = total_value
        
        # Update statistics
        position.total_trades += 1
        position.max_position_size = max(position.max_position_size, abs(position.net_quantity))
        
        # Update realized P&L from latest P&L records
        symbol_pnl_records = [p for p in self.pnl_records.values() if p.symbol == trade.symbol]
        position.realized_pnl = sum(p.net_pnl for p in symbol_pnl_records)
        
        # Count winning/losing trades
        for pnl_record in symbol_pnl_records:
            if pnl_record.net_pnl > 0:
                position.winning_trades += 1
            elif pnl_record.net_pnl < 0:
                position.losing_trades += 1
    
    # ========== P&L Calculation Methods ==========
    
    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """
        Calculate unrealized P&L for a symbol.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        if position.is_flat:
            return 0.0
        
        # Calculate unrealized P&L
        price_diff = current_price - position.average_price
        unrealized_pnl = price_diff * position.net_quantity
        
        # Update position
        position.unrealized_pnl = unrealized_pnl
        position.total_pnl = position.realized_pnl + unrealized_pnl
        
        return unrealized_pnl
    
    def calculate_total_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total unrealized P&L across all positions.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total unrealized P&L
        """
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices and not position.is_flat:
                unrealized = self.calculate_unrealized_pnl(symbol, current_prices[symbol])
                total_unrealized += unrealized
        
        self.total_unrealized_pnl = total_unrealized
        return total_unrealized
    
    def get_realized_pnl(self, symbol: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> float:
        """
        Get realized P&L with optional filters.
        
        Args:
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Realized P&L
        """
        filtered_records = self.pnl_records.values()
        
        if symbol:
            filtered_records = [p for p in filtered_records if p.symbol == symbol]
        
        if start_date:
            filtered_records = [p for p in filtered_records if p.timestamp >= start_date]
        
        if end_date:
            filtered_records = [p for p in filtered_records if p.timestamp <= end_date]
        
        return sum(p.net_pnl for p in filtered_records)
    
    def get_total_pnl(self, pnl_type: PnLType = PnLType.TOTAL) -> float:
        """
        Get total P&L by type.
        
        Args:
            pnl_type: Type of P&L to return
            
        Returns:
            P&L amount
        """
        if pnl_type == PnLType.REALIZED:
            return self.total_realized_pnl
        elif pnl_type == PnLType.UNREALIZED:
            return self.total_unrealized_pnl
        else:  # TOTAL
            return self.total_realized_pnl + self.total_unrealized_pnl
    
    # ========== Performance Analysis ==========
    
    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Performance metrics dictionary
        """
        # Filter P&L records
        pnl_records = list(self.pnl_records.values())
        if symbol:
            pnl_records = [p for p in pnl_records if p.symbol == symbol]
        
        if not pnl_records:
            return {}
        
        # Basic metrics
        total_trades = len(pnl_records)
        winning_trades = len([p for p in pnl_records if p.net_pnl > 0])
        losing_trades = len([p for p in pnl_records if p.net_pnl < 0])
        
        gross_profit = sum(p.net_pnl for p in pnl_records if p.net_pnl > 0)
        gross_loss = abs(sum(p.net_pnl for p in pnl_records if p.net_pnl < 0))
        net_profit = gross_profit - gross_loss
        
        # Advanced metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        pnl_values = [p.net_pnl for p in pnl_records]
        max_win = max(pnl_values) if pnl_values else 0
        max_loss = min(pnl_values) if pnl_values else 0
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'total_commission': sum(p.commission_cost for p in pnl_records),
            'total_slippage': sum(p.slippage_cost for p in pnl_records)
        }
    
    def get_monthly_pnl_summary(self, year: int) -> Dict[str, float]:
        """Get monthly P&L summary for a specific year."""
        monthly_pnl = {}
        
        for month in range(1, 13):
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            month_pnl = self.get_realized_pnl(start_date=start_date, end_date=end_date)
            monthly_pnl[f"{year}-{month:02d}"] = month_pnl
        
        return monthly_pnl
    
    # ========== Reporting Methods ==========
    
    def get_position_report(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed position report."""
        report = {}
        
        for symbol, position in self.positions.items():
            report[symbol] = {
                'net_quantity': position.net_quantity,
                'average_price': position.average_price,
                'total_cost': position.total_cost,
                'realized_pnl': position.realized_pnl,
                'unrealized_pnl': position.unrealized_pnl,
                'total_pnl': position.total_pnl,
                'total_trades': position.total_trades,
                'win_rate': position.win_rate,
                'max_position_size': position.max_position_size,
                'position_status': 'LONG' if position.is_long else 'SHORT' if position.is_short else 'FLAT'
            }
        
        return report
    
    def export_trades_to_dataframe(self) -> pd.DataFrame:
        """Export trades to pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades.values():
            trade_data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'gross_value': trade.gross_value,
                'net_value': trade.net_value,
                'remaining_quantity': trade.remaining_quantity
            })
        
        return pd.DataFrame(trade_data)
    
    def export_pnl_to_dataframe(self) -> pd.DataFrame:
        """Export P&L records to pandas DataFrame."""
        if not self.pnl_records:
            return pd.DataFrame()
        
        pnl_data = []
        for pnl in self.pnl_records.values():
            pnl_data.append({
                'pnl_id': pnl.pnl_id,
                'timestamp': pnl.timestamp,
                'symbol': pnl.symbol,
                'entry_trade_id': pnl.entry_trade_id,
                'exit_trade_id': pnl.exit_trade_id,
                'quantity': pnl.quantity,
                'entry_price': pnl.entry_price,
                'exit_price': pnl.exit_price,
                'gross_pnl': pnl.gross_pnl,
                'commission_cost': pnl.commission_cost,
                'slippage_cost': pnl.slippage_cost,
                'net_pnl': pnl.net_pnl,
                'pnl_percentage': pnl.pnl_percentage,
                'holding_period_hours': pnl.holding_period.total_seconds() / 3600
            })
        
        return pd.DataFrame(pnl_data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CKarZarar(trades={len(self.trades)}, positions={len(self.positions)}, "
                f"realized_pnl={self.total_realized_pnl:.2f}, method={self.calculation_method.value})")