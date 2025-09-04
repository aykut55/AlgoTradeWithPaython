"""
Balance management system for algorithmic trading.

This module contains the CBakiye class which handles comprehensive
balance tracking, margin calculations, risk management, and
multi-currency account management for trading systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import threading
from collections import defaultdict

from ..core.base import SystemProtocol, PositionSide
from ..portfolio.asset_manager import AssetType, CurrencyType
from ..trading.kar_zarar import CKarZarar, PnLType
from ..trading.komisyon import CKomisyon


class BalanceType(Enum):
    """Balance types."""
    CASH = "CASH"                       # Available cash
    MARGIN_USED = "MARGIN_USED"         # Used margin
    MARGIN_AVAILABLE = "MARGIN_AVAILABLE"  # Available margin
    UNREALIZED_PNL = "UNREALIZED_PNL"   # Unrealized P&L
    TOTAL_EQUITY = "TOTAL_EQUITY"       # Total account equity
    FREE_MARGIN = "FREE_MARGIN"         # Free margin for new positions


class MarginMode(Enum):
    """Margin calculation modes."""
    GROSS = "GROSS"                     # Gross margin (each position separate)
    NET = "NET"                         # Net margin (netting allowed)
    HEDGED = "HEDGED"                   # Hedged margin (opposite positions offset)


class RiskLevel(Enum):
    """Risk level indicators."""
    LOW = "LOW"                         # Low risk (< 30% margin usage)
    MEDIUM = "MEDIUM"                   # Medium risk (30-60% margin usage)
    HIGH = "HIGH"                       # High risk (60-80% margin usage)
    CRITICAL = "CRITICAL"               # Critical risk (80-95% margin usage)
    MARGIN_CALL = "MARGIN_CALL"         # Margin call (95%+ margin usage)


@dataclass
class AccountBalance:
    """Account balance information."""
    
    currency: CurrencyType
    
    # Core balances
    initial_balance: float = 0.0
    current_cash: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Derived values
    total_equity: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    
    # Risk metrics
    risk_level: RiskLevel = RiskLevel.LOW
    margin_usage_percentage: float = 0.0
    
    def calculate_derived_values(self) -> None:
        """Calculate derived balance values."""
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.total_equity = self.current_cash + self.unrealized_pnl
        self.free_margin = self.total_equity - self.margin_used
        
        # Calculate margin level (equity/margin * 100)
        if self.margin_used > 0:
            self.margin_level = (self.total_equity / self.margin_used) * 100
            self.margin_usage_percentage = (self.margin_used / self.total_equity) * 100 if self.total_equity > 0 else 0
        else:
            self.margin_level = float('inf')
            self.margin_usage_percentage = 0
        
        # Determine risk level
        if self.margin_usage_percentage >= 95:
            self.risk_level = RiskLevel.MARGIN_CALL
        elif self.margin_usage_percentage >= 80:
            self.risk_level = RiskLevel.CRITICAL
        elif self.margin_usage_percentage >= 60:
            self.risk_level = RiskLevel.HIGH
        elif self.margin_usage_percentage >= 30:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW


@dataclass
class MarginRequirement:
    """Margin requirement for a position."""
    
    symbol: str
    asset_type: AssetType
    position_size: float
    position_value: float
    margin_rate: float
    required_margin: float
    currency: CurrencyType = CurrencyType.TL


@dataclass
class BalanceTransaction:
    """Balance transaction record."""
    
    transaction_id: str
    timestamp: datetime
    transaction_type: str  # DEPOSIT, WITHDRAWAL, TRADE, PNL, COMMISSION, etc.
    amount: float
    currency: CurrencyType
    balance_after: float
    description: str = ""
    related_trade_id: Optional[str] = None


class CBakiye:
    """
    Comprehensive balance management system.
    
    Features:
    - Multi-currency balance tracking
    - Margin calculation and monitoring
    - Risk level assessment
    - Balance transaction history
    - Automated margin calls and stop-outs
    - Real-time balance updates
    - Currency conversion
    - Balance reporting and analytics
    """
    
    def __init__(self, pnl_calculator: Optional[CKarZarar] = None,
                 commission_calculator: Optional[CKomisyon] = None):
        """Initialize balance manager."""
        self.pnl_calculator = pnl_calculator
        self.commission_calculator = commission_calculator
        self.is_initialized = False
        
        # Balance tracking
        self.balances: Dict[CurrencyType, AccountBalance] = {}
        self.base_currency = CurrencyType.TL
        
        # Margin settings
        self.margin_mode = MarginMode.GROSS
        self.margin_call_level = 100.0  # Margin call at 100% margin level
        self.stop_out_level = 50.0      # Stop out at 50% margin level
        
        # Transaction history
        self.transactions: List[BalanceTransaction] = []
        self.transaction_counter = 0
        
        # Position tracking for margin calculation
        self.position_margins: Dict[str, MarginRequirement] = {}
        
        # Currency conversion rates
        self.currency_rates: Dict[CurrencyType, float] = {
            CurrencyType.TL: 1.0,
            CurrencyType.USD: 30.0,
            CurrencyType.EUR: 33.0
        }
        
        # Risk management
        self.max_margin_usage = 0.8  # 80% max margin usage
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.account_risk_limits = {
            'max_positions': 10,
            'max_exposure_per_symbol': 0.2,  # 20% of equity per symbol
            'max_leverage': 10.0
        }
        
        # Thread safety
        self.balance_lock = threading.RLock()
    
    def initialize(self, system: SystemProtocol, 
                   initial_balance: float = 100000.0,
                   currency: CurrencyType = CurrencyType.TL) -> 'CBakiye':
        """
        Initialize balance manager.
        
        Args:
            system: System protocol interface
            initial_balance: Initial account balance
            currency: Base currency
            
        Returns:
            Self for method chaining
        """
        with self.balance_lock:
            self.base_currency = currency
            
            # Initialize base currency balance
            self.balances[currency] = AccountBalance(
                currency=currency,
                initial_balance=initial_balance,
                current_cash=initial_balance,
                margin_available=initial_balance
            )
            self.balances[currency].calculate_derived_values()
            
            # Record initial deposit
            self._record_transaction(
                transaction_type="INITIAL_DEPOSIT",
                amount=initial_balance,
                currency=currency,
                description=f"Initial account funding"
            )
        
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CBakiye':
        """
        Reset balance manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.balance_lock:
            self.balances.clear()
            self.transactions.clear()
            self.position_margins.clear()
            self.transaction_counter = 0
        return self
    
    # ========== Balance Management ==========
    
    def deposit(self, amount: float, currency: CurrencyType = None,
                description: str = "Deposit") -> bool:
        """
        Deposit funds to account.
        
        Args:
            amount: Deposit amount
            currency: Deposit currency (uses base if None)
            description: Transaction description
            
        Returns:
            True if successful
        """
        if amount <= 0:
            return False
        
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            # Initialize currency balance if doesn't exist
            if currency not in self.balances:
                self.balances[currency] = AccountBalance(currency=currency)
            
            balance = self.balances[currency]
            balance.current_cash += amount
            balance.margin_available += amount
            balance.calculate_derived_values()
            
            # Record transaction
            self._record_transaction(
                transaction_type="DEPOSIT",
                amount=amount,
                currency=currency,
                description=description
            )
        
        return True
    
    def withdraw(self, amount: float, currency: CurrencyType = None,
                 description: str = "Withdrawal") -> bool:
        """
        Withdraw funds from account.
        
        Args:
            amount: Withdrawal amount
            currency: Withdrawal currency
            description: Transaction description
            
        Returns:
            True if successful
        """
        if amount <= 0:
            return False
        
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if currency not in self.balances:
                return False
            
            balance = self.balances[currency]
            
            # Check if withdrawal is allowed
            if balance.free_margin < amount:
                return False
            
            balance.current_cash -= amount
            balance.margin_available -= amount
            balance.calculate_derived_values()
            
            # Record transaction
            self._record_transaction(
                transaction_type="WITHDRAWAL",
                amount=-amount,
                currency=currency,
                description=description
            )
        
        return True
    
    def update_balance_from_trade(self, symbol: str, pnl: float, commission: float,
                                 currency: CurrencyType = None, trade_id: str = "") -> None:
        """
        Update balance from trade results.
        
        Args:
            symbol: Trading symbol
            pnl: Trade P&L
            commission: Commission cost
            currency: Trade currency
            trade_id: Trade ID for tracking
        """
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if currency not in self.balances:
                self.balances[currency] = AccountBalance(currency=currency)
            
            balance = self.balances[currency]
            
            # Update realized P&L
            balance.realized_pnl += pnl
            balance.current_cash += pnl
            
            # Subtract commission
            balance.current_cash -= commission
            
            balance.calculate_derived_values()
            
            # Record transactions
            if pnl != 0:
                self._record_transaction(
                    transaction_type="TRADE_PNL",
                    amount=pnl,
                    currency=currency,
                    description=f"P&L from {symbol}",
                    related_trade_id=trade_id
                )
            
            if commission != 0:
                self._record_transaction(
                    transaction_type="COMMISSION",
                    amount=-commission,
                    currency=currency,
                    description=f"Commission for {symbol}",
                    related_trade_id=trade_id
                )
    
    def update_unrealized_pnl(self, symbol_pnl: Dict[str, float],
                             currency: CurrencyType = None) -> None:
        """
        Update unrealized P&L from open positions.
        
        Args:
            symbol_pnl: Dictionary of symbol -> unrealized P&L
            currency: P&L currency
        """
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if currency not in self.balances:
                self.balances[currency] = AccountBalance(currency=currency)
            
            balance = self.balances[currency]
            balance.unrealized_pnl = sum(symbol_pnl.values())
            balance.calculate_derived_values()
    
    # ========== Margin Management ==========
    
    def calculate_required_margin(self, symbol: str, asset_type: AssetType,
                                position_size: float, price: float,
                                leverage: float = 1.0) -> float:
        """
        Calculate required margin for a position.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            position_size: Position size
            price: Current price
            leverage: Leverage ratio
            
        Returns:
            Required margin amount
        """
        position_value = abs(position_size * price)
        
        # Get margin rate based on asset type
        if asset_type == AssetType.BIST_STOCK:
            margin_rate = 0.5  # 50% margin for stocks
        elif asset_type in [AssetType.VIOP_INDEX, AssetType.VIOP_STOCK]:
            margin_rate = 0.1  # 10% margin for VIOP
        elif asset_type in [AssetType.FX_CURRENCY, AssetType.FX_GOLD_MICRO]:
            margin_rate = 1.0 / leverage if leverage > 0 else 0.01  # Based on leverage
        else:
            margin_rate = 0.2  # 20% default margin
        
        return position_value * margin_rate
    
    def reserve_margin(self, symbol: str, asset_type: AssetType,
                      position_size: float, price: float,
                      leverage: float = 1.0, currency: CurrencyType = None) -> bool:
        """
        Reserve margin for a new position.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            position_size: Position size
            price: Position price
            leverage: Leverage ratio
            currency: Margin currency
            
        Returns:
            True if margin reserved successfully
        """
        if currency is None:
            currency = self.base_currency
        
        required_margin = self.calculate_required_margin(symbol, asset_type, position_size, price, leverage)
        
        with self.balance_lock:
            if currency not in self.balances:
                return False
            
            balance = self.balances[currency]
            
            # Check if enough free margin is available
            if balance.free_margin < required_margin:
                return False
            
            # Reserve margin
            balance.margin_used += required_margin
            balance.calculate_derived_values()
            
            # Store margin requirement
            position_value = abs(position_size * price)
            self.position_margins[symbol] = MarginRequirement(
                symbol=symbol,
                asset_type=asset_type,
                position_size=position_size,
                position_value=position_value,
                margin_rate=required_margin / position_value if position_value > 0 else 0,
                required_margin=required_margin,
                currency=currency
            )
            
            # Record transaction
            self._record_transaction(
                transaction_type="MARGIN_RESERVE",
                amount=-required_margin,
                currency=currency,
                description=f"Margin reserved for {symbol}"
            )
        
        return True
    
    def release_margin(self, symbol: str, currency: CurrencyType = None) -> bool:
        """
        Release margin from closed position.
        
        Args:
            symbol: Trading symbol
            currency: Margin currency
            
        Returns:
            True if margin released
        """
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if symbol not in self.position_margins:
                return False
            
            margin_req = self.position_margins[symbol]
            required_margin = margin_req.required_margin
            
            if currency not in self.balances:
                return False
            
            balance = self.balances[currency]
            balance.margin_used -= required_margin
            balance.calculate_derived_values()
            
            # Remove margin requirement
            del self.position_margins[symbol]
            
            # Record transaction
            self._record_transaction(
                transaction_type="MARGIN_RELEASE",
                amount=required_margin,
                currency=currency,
                description=f"Margin released from {symbol}"
            )
        
        return True
    
    def check_margin_requirements(self, currency: CurrencyType = None) -> Tuple[bool, RiskLevel]:
        """
        Check current margin requirements and risk level.
        
        Args:
            currency: Currency to check
            
        Returns:
            (is_safe, risk_level)
        """
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if currency not in self.balances:
                return True, RiskLevel.LOW
            
            balance = self.balances[currency]
            
            # Check margin level
            is_safe = balance.margin_level >= self.margin_call_level
            
            return is_safe, balance.risk_level
    
    def get_available_leverage(self, symbol: str, price: float,
                              currency: CurrencyType = None) -> float:
        """
        Get maximum available leverage for a position.
        
        Args:
            symbol: Trading symbol
            price: Position price
            currency: Currency
            
        Returns:
            Maximum leverage available
        """
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if currency not in self.balances:
                return 1.0
            
            balance = self.balances[currency]
            
            # Calculate max position size based on free margin
            if balance.free_margin <= 0 or price <= 0:
                return 1.0
            
            # Assume 1% margin requirement as base
            base_margin_rate = 0.01
            max_position_value = balance.free_margin / base_margin_rate
            max_leverage = max_position_value / price if price > 0 else 1.0
            
            return min(max_leverage, self.account_risk_limits['max_leverage'])
    
    # ========== Risk Management ==========
    
    def check_position_risk(self, symbol: str, position_size: float, price: float,
                           currency: CurrencyType = None) -> Dict[str, Any]:
        """
        Check risk for a potential position.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            price: Position price
            currency: Currency
            
        Returns:
            Risk analysis results
        """
        if currency is None:
            currency = self.base_currency
        
        position_value = abs(position_size * price)
        
        with self.balance_lock:
            if currency not in self.balances:
                return {'allowed': False, 'reason': 'Currency not available'}
            
            balance = self.balances[currency]
            
            # Check exposure limits
            max_exposure = balance.total_equity * self.account_risk_limits['max_exposure_per_symbol']
            if position_value > max_exposure:
                return {
                    'allowed': False,
                    'reason': f'Position exceeds max exposure limit ({max_exposure:.2f})',
                    'max_allowed_value': max_exposure
                }
            
            # Check margin availability
            required_margin = self.calculate_required_margin(symbol, AssetType.BIST_STOCK, position_size, price)
            if required_margin > balance.free_margin:
                return {
                    'allowed': False,
                    'reason': f'Insufficient margin (required: {required_margin:.2f}, available: {balance.free_margin:.2f})',
                    'required_margin': required_margin,
                    'available_margin': balance.free_margin
                }
            
            # Check maximum positions limit
            if len(self.position_margins) >= self.account_risk_limits['max_positions']:
                return {
                    'allowed': False,
                    'reason': f'Maximum positions limit reached ({self.account_risk_limits["max_positions"]})',
                    'current_positions': len(self.position_margins)
                }
            
            return {
                'allowed': True,
                'position_value': position_value,
                'required_margin': required_margin,
                'exposure_percentage': (position_value / balance.total_equity) * 100,
                'margin_usage_after': ((balance.margin_used + required_margin) / balance.total_equity) * 100
            }
    
    def check_daily_loss_limit(self, currency: CurrencyType = None) -> Dict[str, Any]:
        """Check daily loss limit compliance."""
        if currency is None:
            currency = self.base_currency
        
        today = datetime.now().date()
        daily_pnl = 0.0
        
        # Calculate today's P&L from transactions
        for transaction in self.transactions:
            if (transaction.timestamp.date() == today and
                transaction.currency == currency and
                transaction.transaction_type == "TRADE_PNL"):
                daily_pnl += transaction.amount
        
        with self.balance_lock:
            if currency not in self.balances:
                return {'compliant': True, 'daily_pnl': 0}
            
            balance = self.balances[currency]
            max_daily_loss = balance.initial_balance * self.daily_loss_limit
            
            return {
                'compliant': daily_pnl >= -max_daily_loss,
                'daily_pnl': daily_pnl,
                'daily_loss_limit': max_daily_loss,
                'remaining_loss_allowance': max_daily_loss + daily_pnl if daily_pnl < 0 else max_daily_loss
            }
    
    # ========== Currency Conversion ==========
    
    def convert_currency(self, amount: float, from_currency: CurrencyType,
                        to_currency: CurrencyType) -> float:
        """Convert amount between currencies."""
        if from_currency == to_currency:
            return amount
        
        from_rate = self.currency_rates.get(from_currency, 1.0)
        to_rate = self.currency_rates.get(to_currency, 1.0)
        
        # Convert to base (TL) then to target currency
        tl_amount = amount * from_rate
        return tl_amount / to_rate
    
    def update_currency_rate(self, currency: CurrencyType, rate: float) -> 'CBakiye':
        """Update currency conversion rate."""
        self.currency_rates[currency] = rate
        return self
    
    def get_total_equity_in_currency(self, target_currency: CurrencyType) -> float:
        """Get total equity across all currencies in target currency."""
        total_equity = 0.0
        
        with self.balance_lock:
            for currency, balance in self.balances.items():
                equity_in_target = self.convert_currency(
                    balance.total_equity, currency, target_currency
                )
                total_equity += equity_in_target
        
        return total_equity
    
    # ========== Reporting and Analytics ==========
    
    def get_balance_summary(self, currency: CurrencyType = None) -> Dict[str, Any]:
        """Get comprehensive balance summary."""
        if currency is None:
            currency = self.base_currency
        
        with self.balance_lock:
            if currency not in self.balances:
                return {}
            
            balance = self.balances[currency]
            
            return {
                'currency': currency.value,
                'initial_balance': balance.initial_balance,
                'current_cash': balance.current_cash,
                'margin_used': balance.margin_used,
                'margin_available': balance.margin_available,
                'realized_pnl': balance.realized_pnl,
                'unrealized_pnl': balance.unrealized_pnl,
                'total_pnl': balance.total_pnl,
                'total_equity': balance.total_equity,
                'free_margin': balance.free_margin,
                'margin_level': balance.margin_level,
                'margin_usage_percentage': balance.margin_usage_percentage,
                'risk_level': balance.risk_level.value,
                'open_positions': len(self.position_margins),
                'total_transactions': len(self.transactions)
            }
    
    def get_margin_breakdown(self) -> Dict[str, Any]:
        """Get detailed margin breakdown by position."""
        margin_breakdown = {}
        total_margin_used = 0.0
        
        for symbol, margin_req in self.position_margins.items():
            margin_breakdown[symbol] = {
                'asset_type': margin_req.asset_type.value,
                'position_size': margin_req.position_size,
                'position_value': margin_req.position_value,
                'margin_rate': margin_req.margin_rate,
                'required_margin': margin_req.required_margin,
                'currency': margin_req.currency.value
            }
            total_margin_used += margin_req.required_margin
        
        return {
            'positions': margin_breakdown,
            'total_margin_used': total_margin_used,
            'position_count': len(self.position_margins)
        }
    
    def get_transaction_history(self, limit: int = 100,
                               transaction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get transaction history."""
        transactions = self.transactions[-limit:] if limit else self.transactions
        
        if transaction_type:
            transactions = [t for t in transactions if t.transaction_type == transaction_type]
        
        return [
            {
                'transaction_id': t.transaction_id,
                'timestamp': t.timestamp.isoformat(),
                'type': t.transaction_type,
                'amount': t.amount,
                'currency': t.currency.value,
                'balance_after': t.balance_after,
                'description': t.description,
                'related_trade_id': t.related_trade_id
            }
            for t in transactions
        ]
    
    def _record_transaction(self, transaction_type: str, amount: float,
                           currency: CurrencyType, description: str = "",
                           related_trade_id: Optional[str] = None) -> None:
        """Record a balance transaction."""
        self.transaction_counter += 1
        
        # Get current balance
        current_balance = self.balances[currency].current_cash if currency in self.balances else 0.0
        
        transaction = BalanceTransaction(
            transaction_id=f"TX{self.transaction_counter:06d}",
            timestamp=datetime.now(),
            transaction_type=transaction_type,
            amount=amount,
            currency=currency,
            balance_after=current_balance,
            description=description,
            related_trade_id=related_trade_id
        )
        
        self.transactions.append(transaction)
    
    def __repr__(self) -> str:
        """String representation."""
        total_equity = sum(b.total_equity for b in self.balances.values())
        return (f"CBakiye(currencies={len(self.balances)}, total_equity={total_equity:.2f}, "
                f"positions={len(self.position_margins)}, transactions={len(self.transactions)})")