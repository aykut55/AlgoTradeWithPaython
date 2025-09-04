"""
Commission calculation system for algorithmic trading.

This module contains the CKomisyon class which handles comprehensive
commission calculations for different asset types, brokers, and trading
scenarios with advanced commission structures and optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import math

from ..core.base import SystemProtocol, PositionSide
from ..portfolio.asset_manager import AssetType, CurrencyType


class CommissionStructure(Enum):
    """Commission structure types."""
    FIXED = "FIXED"                     # Fixed amount per trade
    PERCENTAGE = "PERCENTAGE"           # Percentage of trade value
    TIERED = "TIERED"                   # Tiered based on volume
    PER_SHARE = "PER_SHARE"            # Per share/contract
    SPREAD_BASED = "SPREAD_BASED"       # Based on bid-ask spread
    HYBRID = "HYBRID"                   # Combination of methods


class BrokerType(Enum):
    """Broker types with different commission structures."""
    TURKISH_BANK = "TURKISH_BANK"       # Turkish bank brokers
    TURKISH_BROKERAGE = "TURKISH_BROKERAGE"  # Turkish brokerage firms
    INTERNATIONAL_DISCOUNT = "INTERNATIONAL_DISCOUNT"  # International discount brokers
    INTERNATIONAL_FULL_SERVICE = "INTERNATIONAL_FULL_SERVICE"  # Full service brokers
    ECN_BROKER = "ECN_BROKER"          # ECN brokers
    MARKET_MAKER = "MARKET_MAKER"       # Market maker brokers


@dataclass
class CommissionTier:
    """Commission tier definition."""
    
    min_volume: float           # Minimum volume for this tier
    max_volume: float           # Maximum volume for this tier
    rate: float                 # Commission rate for this tier
    fixed_fee: float = 0.0      # Additional fixed fee
    
    def applies_to_volume(self, volume: float) -> bool:
        """Check if tier applies to given volume."""
        return self.min_volume <= volume <= self.max_volume


@dataclass
class CommissionRule:
    """Commission calculation rule."""
    
    asset_type: AssetType
    broker_type: BrokerType
    structure: CommissionStructure
    
    # Basic rates
    base_rate: float = 0.0
    min_commission: float = 0.0
    max_commission: float = 0.0
    
    # Additional fees
    exchange_fee: float = 0.0
    regulatory_fee: float = 0.0
    clearing_fee: float = 0.0
    
    # Tiered structure
    tiers: List[CommissionTier] = field(default_factory=list)
    
    # Special conditions
    volume_discount_threshold: float = 0.0
    volume_discount_rate: float = 0.0
    
    # Currency
    currency: CurrencyType = CurrencyType.TL
    
    def calculate_commission(self, trade_value: float, quantity: float,
                           monthly_volume: float = 0.0) -> float:
        """Calculate commission based on rule parameters."""
        commission = 0.0
        
        if self.structure == CommissionStructure.FIXED:
            commission = self.base_rate
            
        elif self.structure == CommissionStructure.PERCENTAGE:
            commission = trade_value * (self.base_rate / 100.0)
            
        elif self.structure == CommissionStructure.PER_SHARE:
            commission = abs(quantity) * self.base_rate
            
        elif self.structure == CommissionStructure.TIERED:
            commission = self._calculate_tiered_commission(monthly_volume)
            
        elif self.structure == CommissionStructure.SPREAD_BASED:
            # For spread-based, commission is typically zero but spread is wider
            commission = 0.0
        
        # Apply volume discounts
        if monthly_volume >= self.volume_discount_threshold and self.volume_discount_rate > 0:
            discount = commission * (self.volume_discount_rate / 100.0)
            commission -= discount
        
        # Add additional fees
        commission += self.exchange_fee + self.regulatory_fee + self.clearing_fee
        
        # Apply min/max limits
        if self.min_commission > 0:
            commission = max(commission, self.min_commission)
        if self.max_commission > 0:
            commission = min(commission, self.max_commission)
        
        return commission
    
    def _calculate_tiered_commission(self, volume: float) -> float:
        """Calculate tiered commission."""
        for tier in self.tiers:
            if tier.applies_to_volume(volume):
                return tier.fixed_fee + (volume * tier.rate / 100.0)
        
        # If no tier applies, use base rate
        return volume * (self.base_rate / 100.0)


@dataclass
class CommissionCalculation:
    """Commission calculation result."""
    
    trade_id: str
    timestamp: datetime
    symbol: str
    asset_type: AssetType
    broker_type: BrokerType
    
    # Trade details
    quantity: float
    price: float
    trade_value: float
    side: PositionSide
    
    # Commission breakdown
    base_commission: float = 0.0
    exchange_fee: float = 0.0
    regulatory_fee: float = 0.0
    clearing_fee: float = 0.0
    volume_discount: float = 0.0
    
    total_commission: float = 0.0
    commission_percentage: float = 0.0
    
    # Metadata
    rule_applied: str = ""
    currency: CurrencyType = CurrencyType.TL
    
    def __post_init__(self):
        """Calculate derived values."""
        self.total_commission = (self.base_commission + self.exchange_fee + 
                               self.regulatory_fee + self.clearing_fee - self.volume_discount)
        
        if self.trade_value > 0:
            self.commission_percentage = (self.total_commission / self.trade_value) * 100


class CKomisyon:
    """
    Comprehensive commission calculation system.
    
    Features:
    - Multiple commission structures (fixed, percentage, tiered, etc.)
    - Support for different broker types and asset classes
    - Volume-based discounts and rebates
    - Regulatory and exchange fee calculations
    - Commission optimization and analysis
    - Multi-currency support
    - Historical commission tracking
    """
    
    def __init__(self):
        """Initialize commission calculator."""
        self.is_initialized = False
        
        # Commission rules by asset type and broker
        self.commission_rules: Dict[Tuple[AssetType, BrokerType], CommissionRule] = {}
        
        # Default broker and settings
        self.default_broker = BrokerType.TURKISH_BROKERAGE
        self.default_currency = CurrencyType.TL
        
        # Volume tracking for discounts
        self.monthly_volumes: Dict[str, Dict[str, float]] = {}  # {month: {asset_type: volume}}
        self.yearly_volumes: Dict[str, Dict[str, float]] = {}   # {year: {asset_type: volume}}
        
        # Commission history
        self.commission_history: List[CommissionCalculation] = []
        self.total_commissions_paid = 0.0
        
        # Currency rates for conversion
        self.currency_rates: Dict[CurrencyType, float] = {
            CurrencyType.TL: 1.0,
            CurrencyType.USD: 30.0,
            CurrencyType.EUR: 33.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def initialize(self, system: SystemProtocol) -> 'CKomisyon':
        """
        Initialize commission calculator.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CKomisyon':
        """
        Reset commission calculator.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.commission_history.clear()
        self.monthly_volumes.clear()
        self.yearly_volumes.clear()
        self.total_commissions_paid = 0.0
        return self
    
    def _initialize_default_rules(self) -> None:
        """Initialize default commission rules for common scenarios."""
        
        # BIST Stocks - Turkish Brokerage
        self.add_commission_rule(
            asset_type=AssetType.BIST_STOCK,
            broker_type=BrokerType.TURKISH_BROKERAGE,
            structure=CommissionStructure.PERCENTAGE,
            base_rate=0.08,  # 0.08% typical rate
            min_commission=8.0,  # 8 TL minimum
            exchange_fee=0.005,  # 0.005% exchange fee
            regulatory_fee=0.002  # 0.002% regulatory fee
        )
        
        # VIOP Index - Turkish Brokerage
        self.add_commission_rule(
            asset_type=AssetType.VIOP_INDEX,
            broker_type=BrokerType.TURKISH_BROKERAGE,
            structure=CommissionStructure.FIXED,
            base_rate=3.0,  # 3 TL per contract
            min_commission=3.0,
            exchange_fee=0.5,
            regulatory_fee=0.2
        )
        
        # VIOP Stock - Turkish Brokerage
        self.add_commission_rule(
            asset_type=AssetType.VIOP_STOCK,
            broker_type=BrokerType.TURKISH_BROKERAGE,
            structure=CommissionStructure.FIXED,
            base_rate=2.5,  # 2.5 TL per contract
            min_commission=2.5,
            exchange_fee=0.3,
            regulatory_fee=0.1
        )
        
        # VIOP Currency - Turkish Brokerage
        self.add_commission_rule(
            asset_type=AssetType.VIOP_CURRENCY,
            broker_type=BrokerType.TURKISH_BROKERAGE,
            structure=CommissionStructure.FIXED,
            base_rate=4.0,  # 4 TL per contract
            min_commission=4.0,
            exchange_fee=0.6,
            regulatory_fee=0.2
        )
        
        # FX - International ECN Broker
        self.add_commission_rule(
            asset_type=AssetType.FX_CURRENCY,
            broker_type=BrokerType.ECN_BROKER,
            structure=CommissionStructure.SPREAD_BASED,
            base_rate=0.0,  # Commission-free, profit from spread
            min_commission=0.0,
            currency=CurrencyType.USD
        )
        
        # Bank Currency - Turkish Bank
        self.add_commission_rule(
            asset_type=AssetType.BANK_CURRENCY,
            broker_type=BrokerType.TURKISH_BANK,
            structure=CommissionStructure.PERCENTAGE,
            base_rate=0.15,  # 0.15% typical bank rate
            min_commission=5.0,  # 5 TL minimum
            max_commission=100.0  # 100 TL maximum
        )
    
    # ========== Rule Management ==========
    
    def add_commission_rule(self, asset_type: AssetType, broker_type: BrokerType,
                           structure: CommissionStructure, base_rate: float,
                           min_commission: float = 0.0, max_commission: float = 0.0,
                           exchange_fee: float = 0.0, regulatory_fee: float = 0.0,
                           clearing_fee: float = 0.0, currency: CurrencyType = CurrencyType.TL,
                           tiers: Optional[List[CommissionTier]] = None) -> 'CKomisyon':
        """
        Add or update commission rule.
        
        Args:
            asset_type: Asset type
            broker_type: Broker type
            structure: Commission structure
            base_rate: Base commission rate
            min_commission: Minimum commission
            max_commission: Maximum commission
            exchange_fee: Exchange fee
            regulatory_fee: Regulatory fee
            clearing_fee: Clearing fee
            currency: Commission currency
            tiers: Tiered structure (for tiered commissions)
            
        Returns:
            Self for method chaining
        """
        rule = CommissionRule(
            asset_type=asset_type,
            broker_type=broker_type,
            structure=structure,
            base_rate=base_rate,
            min_commission=min_commission,
            max_commission=max_commission,
            exchange_fee=exchange_fee,
            regulatory_fee=regulatory_fee,
            clearing_fee=clearing_fee,
            currency=currency,
            tiers=tiers or []
        )
        
        self.commission_rules[(asset_type, broker_type)] = rule
        return self
    
    def add_tiered_commission_rule(self, asset_type: AssetType, broker_type: BrokerType,
                                  tiers: List[Tuple[float, float, float]],
                                  currency: CurrencyType = CurrencyType.TL) -> 'CKomisyon':
        """
        Add tiered commission rule.
        
        Args:
            asset_type: Asset type
            broker_type: Broker type
            tiers: List of (min_volume, max_volume, rate) tuples
            currency: Commission currency
            
        Returns:
            Self for method chaining
        """
        tier_objects = []
        for min_vol, max_vol, rate in tiers:
            tier_objects.append(CommissionTier(
                min_volume=min_vol,
                max_volume=max_vol,
                rate=rate
            ))
        
        return self.add_commission_rule(
            asset_type=asset_type,
            broker_type=broker_type,
            structure=CommissionStructure.TIERED,
            base_rate=0.0,
            currency=currency,
            tiers=tier_objects
        )
    
    def set_default_broker(self, broker_type: BrokerType) -> 'CKomisyon':
        """Set default broker type."""
        self.default_broker = broker_type
        return self
    
    # ========== Commission Calculation ==========
    
    def calculate_commission(self, symbol: str, asset_type: AssetType,
                           quantity: float, price: float, side: PositionSide,
                           broker_type: Optional[BrokerType] = None,
                           trade_id: Optional[str] = None) -> CommissionCalculation:
        """
        Calculate commission for a trade.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            quantity: Trade quantity
            price: Trade price
            side: Position side
            broker_type: Broker type (uses default if None)
            trade_id: Trade ID for tracking
            
        Returns:
            Commission calculation result
        """
        if broker_type is None:
            broker_type = self.default_broker
        
        if trade_id is None:
            trade_id = f"T{len(self.commission_history) + 1:06d}"
        
        # Get commission rule
        rule = self.commission_rules.get((asset_type, broker_type))
        if not rule:
            # Fallback to basic percentage rule
            rule = CommissionRule(
                asset_type=asset_type,
                broker_type=broker_type,
                structure=CommissionStructure.PERCENTAGE,
                base_rate=0.1,
                min_commission=1.0
            )
        
        # Calculate trade value
        trade_value = abs(quantity * price)
        
        # Get monthly volume for discounts
        current_month = datetime.now().strftime("%Y-%m")
        monthly_volume = self.monthly_volumes.get(current_month, {}).get(asset_type.value, 0.0)
        
        # Calculate commission using rule
        base_commission = rule.calculate_commission(trade_value, quantity, monthly_volume)
        
        # Create calculation result
        calculation = CommissionCalculation(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            asset_type=asset_type,
            broker_type=broker_type,
            quantity=quantity,
            price=price,
            trade_value=trade_value,
            side=side,
            base_commission=base_commission,
            exchange_fee=rule.exchange_fee,
            regulatory_fee=rule.regulatory_fee,
            clearing_fee=rule.clearing_fee,
            rule_applied=f"{asset_type.value}_{broker_type.value}",
            currency=rule.currency
        )
        
        # Store in history
        self.commission_history.append(calculation)
        self.total_commissions_paid += calculation.total_commission
        
        # Update volume tracking
        self._update_volume_tracking(asset_type, trade_value)
        
        return calculation
    
    def calculate_round_trip_commission(self, symbol: str, asset_type: AssetType,
                                      quantity: float, entry_price: float, exit_price: float,
                                      broker_type: Optional[BrokerType] = None) -> Tuple[float, CommissionCalculation, CommissionCalculation]:
        """
        Calculate round-trip commission (entry + exit).
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            quantity: Trade quantity
            entry_price: Entry price
            exit_price: Exit price
            broker_type: Broker type
            
        Returns:
            (total_commission, entry_calculation, exit_calculation)
        """
        if broker_type is None:
            broker_type = self.default_broker
        
        # Calculate entry commission
        entry_calc = self.calculate_commission(
            symbol=symbol,
            asset_type=asset_type,
            quantity=quantity,
            price=entry_price,
            side=PositionSide.BUY,
            broker_type=broker_type,
            trade_id=f"ENTRY_{len(self.commission_history) + 1:06d}"
        )
        
        # Calculate exit commission
        exit_calc = self.calculate_commission(
            symbol=symbol,
            asset_type=asset_type,
            quantity=quantity,
            price=exit_price,
            side=PositionSide.SELL,
            broker_type=broker_type,
            trade_id=f"EXIT_{len(self.commission_history) + 1:06d}"
        )
        
        total_commission = entry_calc.total_commission + exit_calc.total_commission
        
        return total_commission, entry_calc, exit_calc
    
    def _update_volume_tracking(self, asset_type: AssetType, trade_value: float) -> None:
        """Update volume tracking for discounts."""
        current_month = datetime.now().strftime("%Y-%m")
        current_year = datetime.now().strftime("%Y")
        
        # Update monthly volume
        if current_month not in self.monthly_volumes:
            self.monthly_volumes[current_month] = {}
        
        asset_key = asset_type.value
        self.monthly_volumes[current_month][asset_key] = self.monthly_volumes[current_month].get(asset_key, 0.0) + trade_value
        
        # Update yearly volume
        if current_year not in self.yearly_volumes:
            self.yearly_volumes[current_year] = {}
        
        self.yearly_volumes[current_year][asset_key] = self.yearly_volumes[current_year].get(asset_key, 0.0) + trade_value
    
    # ========== Analysis and Optimization ==========
    
    def analyze_commission_costs(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Analyze commission costs over a period.
        
        Args:
            period_days: Analysis period in days
            
        Returns:
            Commission analysis results
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_commissions = [c for c in self.commission_history if c.timestamp >= cutoff_date]
        
        if not recent_commissions:
            return {}
        
        total_commission = sum(c.total_commission for c in recent_commissions)
        total_trade_value = sum(c.trade_value for c in recent_commissions)
        
        # Breakdown by asset type
        by_asset = {}
        for calc in recent_commissions:
            asset_key = calc.asset_type.value
            if asset_key not in by_asset:
                by_asset[asset_key] = {
                    'count': 0,
                    'total_commission': 0.0,
                    'total_trade_value': 0.0,
                    'avg_commission': 0.0
                }
            
            by_asset[asset_key]['count'] += 1
            by_asset[asset_key]['total_commission'] += calc.total_commission
            by_asset[asset_key]['total_trade_value'] += calc.trade_value
        
        # Calculate averages
        for asset_data in by_asset.values():
            if asset_data['count'] > 0:
                asset_data['avg_commission'] = asset_data['total_commission'] / asset_data['count']
        
        return {
            'period_days': period_days,
            'total_trades': len(recent_commissions),
            'total_commission': total_commission,
            'total_trade_value': total_trade_value,
            'avg_commission_per_trade': total_commission / len(recent_commissions),
            'commission_percentage': (total_commission / total_trade_value * 100) if total_trade_value > 0 else 0,
            'by_asset_type': by_asset
        }
    
    def suggest_broker_optimization(self, monthly_volume: float, 
                                  primary_assets: List[AssetType]) -> Dict[str, Any]:
        """
        Suggest broker optimization based on trading volume and assets.
        
        Args:
            monthly_volume: Expected monthly trading volume
            primary_assets: Primary asset types traded
            
        Returns:
            Optimization suggestions
        """
        suggestions = {}
        
        for asset_type in primary_assets:
            best_broker = None
            lowest_cost = float('inf')
            
            # Test different brokers for this asset type
            for broker_type in BrokerType:
                rule_key = (asset_type, broker_type)
                if rule_key in self.commission_rules:
                    rule = self.commission_rules[rule_key]
                    
                    # Estimate monthly commission cost
                    estimated_trades = monthly_volume / 10000  # Assume avg 10k per trade
                    monthly_commission = rule.calculate_commission(monthly_volume, estimated_trades, monthly_volume)
                    
                    if monthly_commission < lowest_cost:
                        lowest_cost = monthly_commission
                        best_broker = broker_type
            
            suggestions[asset_type.value] = {
                'recommended_broker': best_broker.value if best_broker else None,
                'estimated_monthly_cost': lowest_cost if lowest_cost != float('inf') else 0,
                'potential_savings': 0  # Could calculate vs current broker
            }
        
        return suggestions
    
    def calculate_breakeven_volume(self, asset_type: AssetType, 
                                 broker1: BrokerType, broker2: BrokerType) -> float:
        """
        Calculate breakeven volume between two brokers.
        
        Args:
            asset_type: Asset type to analyze
            broker1: First broker
            broker2: Second broker
            
        Returns:
            Breakeven volume (0 if rules don't exist)
        """
        rule1 = self.commission_rules.get((asset_type, broker1))
        rule2 = self.commission_rules.get((asset_type, broker2))
        
        if not rule1 or not rule2:
            return 0.0
        
        # For simplicity, assume percentage-based calculation
        if (rule1.structure == CommissionStructure.PERCENTAGE and 
            rule2.structure == CommissionStructure.PERCENTAGE):
            
            rate_diff = abs(rule1.base_rate - rule2.base_rate)
            if rate_diff > 0:
                # Very simplified calculation
                return 1000000 / rate_diff  # Placeholder logic
        
        return 0.0
    
    # ========== Reporting ==========
    
    def get_commission_summary(self, start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get commission summary for a period."""
        filtered_history = self.commission_history
        
        if start_date:
            filtered_history = [c for c in filtered_history if c.timestamp >= start_date]
        if end_date:
            filtered_history = [c for c in filtered_history if c.timestamp <= end_date]
        
        if not filtered_history:
            return {}
        
        total_commission = sum(c.total_commission for c in filtered_history)
        total_trades = len(filtered_history)
        total_volume = sum(c.trade_value for c in filtered_history)
        
        return {
            'total_trades': total_trades,
            'total_commission': total_commission,
            'total_volume': total_volume,
            'avg_commission_per_trade': total_commission / total_trades,
            'commission_as_percentage': (total_commission / total_volume * 100) if total_volume > 0 else 0,
            'period_start': start_date,
            'period_end': end_date
        }
    
    def export_commission_history(self) -> List[Dict[str, Any]]:
        """Export commission history for external analysis."""
        return [
            {
                'trade_id': calc.trade_id,
                'timestamp': calc.timestamp.isoformat(),
                'symbol': calc.symbol,
                'asset_type': calc.asset_type.value,
                'broker_type': calc.broker_type.value,
                'quantity': calc.quantity,
                'price': calc.price,
                'trade_value': calc.trade_value,
                'side': calc.side.value,
                'base_commission': calc.base_commission,
                'exchange_fee': calc.exchange_fee,
                'regulatory_fee': calc.regulatory_fee,
                'clearing_fee': calc.clearing_fee,
                'total_commission': calc.total_commission,
                'commission_percentage': calc.commission_percentage,
                'currency': calc.currency.value
            }
            for calc in self.commission_history
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CKomisyon(rules={len(self.commission_rules)}, history={len(self.commission_history)}, "
                f"total_paid={self.total_commissions_paid:.2f})")