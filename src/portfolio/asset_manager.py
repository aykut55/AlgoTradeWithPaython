"""
Asset management system for algorithmic trading.

This module contains the CVarlikManager class which manages asset configuration,
position sizing, commission calculations, and multi-asset support for various
financial instruments including stocks, futures, forex, and commodities.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

from ..core.base import SystemProtocol


class AssetType(Enum):
    """Asset type enumeration."""
    BIST_STOCK = "BIST_STOCK"           # BIST Hisse
    VIOP_INDEX = "VIOP_INDEX"           # VIOP Endeks
    VIOP_STOCK = "VIOP_STOCK"           # VIOP Hisse
    VIOP_CURRENCY = "VIOP_CURRENCY"     # VIOP Dolar/Euro
    VIOP_GOLD_GRAM = "VIOP_GOLD_GRAM"   # VIOP Gram Altın
    VIOP_GOLD_OUNCE = "VIOP_GOLD_OUNCE" # VIOP Ons Altın
    BANK_CURRENCY = "BANK_CURRENCY"     # Banka Döviz
    BANK_GOLD = "BANK_GOLD"             # Banka Gram Altın
    FX_GOLD_MICRO = "FX_GOLD_MICRO"     # FX Ons Altın (Micro)
    FX_CURRENCY = "FX_CURRENCY"         # FX Parite
    FX_INDEX = "FX_INDEX"               # FX Endeks


class CurrencyType(Enum):
    """Currency type enumeration."""
    TL = "TL"       # Turkish Lira
    USD = "USD"     # US Dollar
    EUR = "EUR"     # Euro
    
    def __str__(self):
        return self.value


@dataclass
class AssetConfiguration:
    """Asset configuration parameters."""
    
    # Asset identification
    asset_type: AssetType
    symbol: str = ""
    
    # Contract specifications
    contract_count: int = 1
    asset_quantity_multiplier: int = 1
    shares_count: int = 0  # For stocks
    
    # Calculated quantities
    total_asset_quantity: int = 0
    commission_asset_quantity: int = 0
    
    # Pricing
    currency_type: CurrencyType = CurrencyType.TL
    
    # Commission settings
    commission_multiplier: float = 0.0
    include_commission: bool = False
    
    # Slippage settings
    slippage_amount: float = 0.0
    include_slippage: bool = False
    
    # Balance settings
    initial_balance_price: float = 100000.0
    initial_balance_points: float = 0.0
    
    def calculate_quantities(self) -> None:
        """Calculate total quantities based on configuration."""
        if self.asset_type == AssetType.BIST_STOCK:
            self.total_asset_quantity = self.shares_count * self.asset_quantity_multiplier
            self.commission_asset_quantity = self.shares_count
        else:
            self.total_asset_quantity = self.contract_count * self.asset_quantity_multiplier
            self.commission_asset_quantity = self.contract_count
        
        # Update commission flag
        self.include_commission = self.commission_multiplier > 0.0
        self.include_slippage = self.slippage_amount > 0.0
    
    def validate(self) -> bool:
        """Validate asset configuration."""
        if self.asset_type == AssetType.BIST_STOCK:
            return self.shares_count > 0 and self.asset_quantity_multiplier > 0
        else:
            return self.contract_count > 0 and self.asset_quantity_multiplier > 0


class CVarlikManager:
    """
    Comprehensive asset manager for trading system.
    
    Manages:
    - Asset configuration for different instrument types
    - Position sizing calculations
    - Commission and slippage calculations
    - Multi-asset portfolio management
    - Cost analysis
    """
    
    def __init__(self):
        """Initialize asset manager."""
        self.config = AssetConfiguration(
            asset_type=AssetType.VIOP_INDEX,
            currency_type=CurrencyType.TL
        )
        self.is_initialized: bool = False
        self._market_index: int = 0
    
    def initialize(self, system: SystemProtocol) -> 'CVarlikManager':
        """
        Initialize asset manager with default settings.
        
        Args:
            system: System interface
            
        Returns:
            Self for method chaining
        """
        self.set_commission_params(system)
        self.set_slippage_params(system)
        self.set_balance_params(system)
        self.reset(system)
        
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CVarlikManager':
        """
        Reset asset manager state.
        
        Args:
            system: System interface
            
        Returns:
            Self for method chaining
        """
        # Reset can include clearing temporary calculations, etc.
        return self
    
    # ========== Parameter Setting Methods ==========
    
    def set_commission_params(self, system: SystemProtocol, 
                            commission_multiplier: float = 3.0) -> 'CVarlikManager':
        """
        Set commission parameters.
        
        Args:
            system: System interface
            commission_multiplier: Commission multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.commission_multiplier = commission_multiplier
        self.config.include_commission = commission_multiplier > 0.0
        return self
    
    def set_slippage_params(self, system: SystemProtocol,
                          slippage_amount: float = 0.0) -> 'CVarlikManager':
        """
        Set slippage parameters.
        
        Args:
            system: System interface
            slippage_amount: Slippage amount
            
        Returns:
            Self for method chaining
        """
        self.config.slippage_amount = slippage_amount
        self.config.include_slippage = slippage_amount > 0.0
        return self
    
    def set_balance_params(self, system: SystemProtocol,
                         initial_balance: float = 100000.0,
                         initial_balance_points: float = 0.0) -> 'CVarlikManager':
        """
        Set balance parameters.
        
        Args:
            system: System interface
            initial_balance: Initial balance amount
            initial_balance_points: Initial balance in points
            
        Returns:
            Self for method chaining
        """
        self.config.initial_balance_price = initial_balance
        self.config.initial_balance_points = initial_balance_points
        return self
    
    # ========== Asset Configuration Methods ==========
    
    def set_bist_stock_params(self, system: SystemProtocol,
                            shares_count: int = 1000,
                            asset_multiplier: int = 1) -> 'CVarlikManager':
        """
        Configure for BIST stocks.
        
        Args:
            system: System interface
            shares_count: Number of shares
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.BIST_STOCK
        self.config.shares_count = shares_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_viop_index_params(self, system: SystemProtocol,
                            contract_count: int = 1,
                            asset_multiplier: int = 10) -> 'CVarlikManager':
        """
        Configure for VIOP Index contracts.
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.VIOP_INDEX
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_viop_stock_params(self, system: SystemProtocol,
                            contract_count: int = 1,
                            asset_multiplier: int = 100) -> 'CVarlikManager':
        """
        Configure for VIOP Stock contracts.
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.VIOP_STOCK
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_viop_currency_params(self, system: SystemProtocol,
                               contract_count: int = 1,
                               asset_multiplier: int = 1000,
                               currency: str = "USD") -> 'CVarlikManager':
        """
        Configure for VIOP Currency contracts (USD/EUR).
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            currency: Currency type (USD/EUR)
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.VIOP_CURRENCY
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_viop_gold_gram_params(self, system: SystemProtocol,
                                contract_count: int = 1,
                                asset_multiplier: int = 1) -> 'CVarlikManager':
        """
        Configure for VIOP Gram Gold contracts.
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.VIOP_GOLD_GRAM
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_bank_currency_params(self, system: SystemProtocol,
                               contract_count: int = 1,
                               asset_multiplier: int = 1,
                               currency: str = "USD") -> 'CVarlikManager':
        """
        Configure for Bank Currency trading.
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            currency: Currency type
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.BANK_CURRENCY
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_bank_gold_params(self, system: SystemProtocol,
                           contract_count: int = 1,
                           asset_multiplier: int = 1) -> 'CVarlikManager':
        """
        Configure for Bank Gram Gold trading.
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.BANK_GOLD
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.TL
        self.config.calculate_quantities()
        return self
    
    def set_fx_gold_micro_params(self, system: SystemProtocol,
                               contract_count: int = 1,
                               asset_multiplier: int = 1) -> 'CVarlikManager':
        """
        Configure for FX Ounce Gold Micro (0.01 lot increments).
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.FX_GOLD_MICRO
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.USD
        self.config.calculate_quantities()
        return self
    
    def set_fx_currency_params(self, system: SystemProtocol,
                             contract_count: int = 1,
                             asset_multiplier: int = 1) -> 'CVarlikManager':
        """
        Configure for FX Currency pairs.
        
        Args:
            system: System interface
            contract_count: Number of contracts
            asset_multiplier: Asset multiplier
            
        Returns:
            Self for method chaining
        """
        self.config.asset_type = AssetType.FX_CURRENCY
        self.config.contract_count = contract_count
        self.config.asset_quantity_multiplier = asset_multiplier
        self.config.currency_type = CurrencyType.USD
        self.config.calculate_quantities()
        return self
    
    # ========== Calculation Methods ==========
    
    def calculate_position_value(self, price: float) -> float:
        """
        Calculate total position value.
        
        Args:
            price: Current price
            
        Returns:
            Total position value
        """
        return price * self.config.total_asset_quantity
    
    def calculate_commission(self, price: float, is_buy: bool = True) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            price: Trade price
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Commission amount
        """
        if not self.config.include_commission or self.config.commission_multiplier == 0:
            return 0.0
        
        # Base commission calculation
        base_commission = self.config.commission_multiplier
        
        # Different commission structures for different asset types
        if self.config.asset_type == AssetType.BIST_STOCK:
            # BIST stocks: percentage-based commission
            return (price * self.config.commission_asset_quantity * 
                   self.config.commission_multiplier / 1000.0)  # Typical rate
        elif self.config.asset_type in [AssetType.VIOP_INDEX, AssetType.VIOP_STOCK, 
                                       AssetType.VIOP_CURRENCY, AssetType.VIOP_GOLD_GRAM]:
            # VIOP: fixed commission per contract
            return self.config.commission_asset_quantity * self.config.commission_multiplier
        elif self.config.asset_type in [AssetType.FX_GOLD_MICRO, AssetType.FX_CURRENCY]:
            # FX: spread-based (often zero for micro accounts)
            return 0.0 if self.config.commission_multiplier == 0 else base_commission
        else:
            # Bank instruments: minimal commission
            return base_commission
    
    def calculate_slippage(self, price: float) -> float:
        """
        Calculate slippage cost.
        
        Args:
            price: Trade price
            
        Returns:
            Slippage amount
        """
        if not self.config.include_slippage:
            return 0.0
        
        return self.config.slippage_amount * self.config.total_asset_quantity
    
    def calculate_total_trade_cost(self, price: float, is_buy: bool = True) -> float:
        """
        Calculate total cost of a trade including commission and slippage.
        
        Args:
            price: Trade price
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Total trade cost
        """
        position_value = self.calculate_position_value(price)
        commission = self.calculate_commission(price, is_buy)
        slippage = self.calculate_slippage(price)
        
        return position_value + commission + slippage
    
    def calculate_required_margin(self, price: float, margin_rate: float = 0.1) -> float:
        """
        Calculate required margin for the position.
        
        Args:
            price: Current price
            margin_rate: Margin requirement rate (default 10%)
            
        Returns:
            Required margin amount
        """
        position_value = self.calculate_position_value(price)
        
        # Different margin requirements for different asset types
        if self.config.asset_type == AssetType.BIST_STOCK:
            # Stocks: typically 50% margin
            return position_value * 0.5
        elif self.config.asset_type in [AssetType.VIOP_INDEX, AssetType.VIOP_STOCK]:
            # VIOP: defined margin rates
            return position_value * margin_rate
        elif self.config.asset_type in [AssetType.FX_GOLD_MICRO, AssetType.FX_CURRENCY]:
            # FX: leverage-based margin (1:100 leverage = 1% margin)
            return position_value * 0.01
        else:
            # Default margin
            return position_value * margin_rate
    
    def calculate_pnl(self, entry_price: float, exit_price: float, is_long: bool = True) -> float:
        """
        Calculate P&L for a completed trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            is_long: True for long positions, False for short
            
        Returns:
            P&L amount (positive = profit, negative = loss)
        """
        price_diff = exit_price - entry_price
        if not is_long:
            price_diff = -price_diff
        
        gross_pnl = price_diff * self.config.total_asset_quantity
        
        # Subtract costs
        entry_commission = self.calculate_commission(entry_price, is_long)
        exit_commission = self.calculate_commission(exit_price, not is_long)
        entry_slippage = self.calculate_slippage(entry_price)
        exit_slippage = self.calculate_slippage(exit_price)
        
        net_pnl = gross_pnl - entry_commission - exit_commission - entry_slippage - exit_slippage
        
        return net_pnl
    
    # ========== Information Methods ==========
    
    def get_asset_info(self) -> Dict[str, Any]:
        """Get comprehensive asset information."""
        return {
            "asset_type": self.config.asset_type.value,
            "currency_type": str(self.config.currency_type),
            "contract_count": self.config.contract_count,
            "shares_count": self.config.shares_count,
            "asset_quantity_multiplier": self.config.asset_quantity_multiplier,
            "total_asset_quantity": self.config.total_asset_quantity,
            "commission_asset_quantity": self.config.commission_asset_quantity,
            "commission_multiplier": self.config.commission_multiplier,
            "include_commission": self.config.include_commission,
            "slippage_amount": self.config.slippage_amount,
            "include_slippage": self.config.include_slippage,
            "initial_balance": self.config.initial_balance_price
        }
    
    def get_cost_breakdown(self, price: float) -> Dict[str, float]:
        """
        Get detailed cost breakdown for a trade.
        
        Args:
            price: Trade price
            
        Returns:
            Dictionary with cost breakdown
        """
        return {
            "position_value": self.calculate_position_value(price),
            "commission": self.calculate_commission(price),
            "slippage": self.calculate_slippage(price),
            "total_cost": self.calculate_total_trade_cost(price),
            "required_margin": self.calculate_required_margin(price)
        }
    
    def is_configuration_valid(self) -> bool:
        """Check if current configuration is valid."""
        return self.config.validate()
    
    def __repr__(self) -> str:
        """String representation of asset manager."""
        return (f"CVarlikManager(type={self.config.asset_type.value}, "
                f"quantity={self.config.total_asset_quantity}, "
                f"currency={self.config.currency_type.value})")


# ========== Utility Functions ==========

def create_preset_configurations() -> Dict[str, AssetConfiguration]:
    """Create preset configurations for common asset types."""
    presets = {}
    
    # VIOP Index (XU030)
    config = AssetConfiguration(
        asset_type=AssetType.VIOP_INDEX,
        contract_count=10,
        asset_quantity_multiplier=10,
        commission_multiplier=3.0,
        currency_type=CurrencyType.TL
    )
    config.calculate_quantities()
    presets["VIOP_XU030"] = config
    
    # BIST Stock
    config = AssetConfiguration(
        asset_type=AssetType.BIST_STOCK,
        shares_count=1000,
        asset_quantity_multiplier=1,
        commission_multiplier=1.0,
        currency_type=CurrencyType.TL
    )
    config.calculate_quantities()
    presets["BIST_STOCK"] = config
    
    # FX Gold Micro
    config = AssetConfiguration(
        asset_type=AssetType.FX_GOLD_MICRO,
        contract_count=1,
        asset_quantity_multiplier=1,
        commission_multiplier=0.0,
        currency_type=CurrencyType.USD
    )
    config.calculate_quantities()
    presets["FX_GOLD"] = config
    
    return presets


def get_recommended_settings(asset_type: AssetType) -> Dict[str, Any]:
    """Get recommended settings for different asset types."""
    recommendations = {
        AssetType.VIOP_INDEX: {
            "contract_count": 10,
            "asset_multiplier": 10,
            "commission_multiplier": 3.0,
            "initial_balance": 100000.0
        },
        AssetType.BIST_STOCK: {
            "shares_count": 1000,
            "asset_multiplier": 1,
            "commission_multiplier": 1.0,
            "initial_balance": 50000.0
        },
        AssetType.FX_GOLD_MICRO: {
            "contract_count": 1,
            "asset_multiplier": 1,
            "commission_multiplier": 0.0,
            "initial_balance": 10000.0
        }
    }
    
    return recommendations.get(asset_type, {})