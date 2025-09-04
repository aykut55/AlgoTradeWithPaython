#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for the portfolio management module.

Tests include:
- Asset configuration and management
- Position sizing calculations
- Commission and slippage calculations
- Multi-asset support
- Cost analysis
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.portfolio.asset_manager import (
    CVarlikManager, AssetConfiguration, AssetType, CurrencyType,
    create_preset_configurations, get_recommended_settings
)


class MockSystem:
    """Mock system for testing."""
    
    def __init__(self):
        self.messages = []
    
    def mesaj(self, message: str) -> None:
        self.messages.append(message)


class TestAssetConfiguration:
    """Test cases for AssetConfiguration class."""
    
    def test_default_configuration(self):
        """Test default asset configuration."""
        config = AssetConfiguration(asset_type=AssetType.VIOP_INDEX)
        
        assert config.asset_type == AssetType.VIOP_INDEX
        assert config.contract_count == 1
        assert config.asset_quantity_multiplier == 1
        assert config.currency_type == CurrencyType.TL
        assert config.commission_multiplier == 0.0
        assert not config.include_commission
    
    def test_quantity_calculations_viop(self):
        """Test quantity calculations for VIOP contracts."""
        config = AssetConfiguration(
            asset_type=AssetType.VIOP_INDEX,
            contract_count=10,
            asset_quantity_multiplier=10
        )
        config.calculate_quantities()
        
        assert config.total_asset_quantity == 100  # 10 * 10
        assert config.commission_asset_quantity == 10  # contract_count
    
    def test_quantity_calculations_bist(self):
        """Test quantity calculations for BIST stocks."""
        config = AssetConfiguration(
            asset_type=AssetType.BIST_STOCK,
            shares_count=1000,
            asset_quantity_multiplier=1
        )
        config.calculate_quantities()
        
        assert config.total_asset_quantity == 1000  # 1000 * 1
        assert config.commission_asset_quantity == 1000  # shares_count
    
    def test_commission_flags(self):
        """Test commission and slippage flag calculations."""
        config = AssetConfiguration(asset_type=AssetType.VIOP_INDEX)
        
        # Test commission flag
        config.commission_multiplier = 3.0
        config.calculate_quantities()
        assert config.include_commission
        
        config.commission_multiplier = 0.0
        config.calculate_quantities()
        assert not config.include_commission
        
        # Test slippage flag
        config.slippage_amount = 1.5
        config.calculate_quantities()
        assert config.include_slippage
        
        config.slippage_amount = 0.0
        config.calculate_quantities()
        assert not config.include_slippage
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid VIOP configuration
        config = AssetConfiguration(
            asset_type=AssetType.VIOP_INDEX,
            contract_count=5,
            asset_quantity_multiplier=10
        )
        assert config.validate()
        
        # Invalid VIOP configuration
        config.contract_count = 0
        assert not config.validate()
        
        # Valid BIST configuration
        config = AssetConfiguration(
            asset_type=AssetType.BIST_STOCK,
            shares_count=1000,
            asset_quantity_multiplier=1
        )
        assert config.validate()
        
        # Invalid BIST configuration
        config.shares_count = 0
        assert not config.validate()


class TestCVarlikManager:
    """Test cases for CVarlikManager class."""
    
    def test_initialization(self):
        """Test asset manager initialization."""
        manager = CVarlikManager()
        
        assert not manager.is_initialized
        assert manager.config.asset_type == AssetType.VIOP_INDEX
        assert manager.config.currency_type == CurrencyType.TL
    
    def test_system_initialization(self):
        """Test system initialization."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.initialize(system)
        
        assert manager.is_initialized
    
    def test_commission_params(self):
        """Test commission parameter setting."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Set commission params
        result = manager.set_commission_params(system, 5.0)
        
        assert result == manager  # Method chaining
        assert manager.config.commission_multiplier == 5.0
        assert manager.config.include_commission
        
        # Set zero commission
        manager.set_commission_params(system, 0.0)
        assert not manager.config.include_commission
    
    def test_slippage_params(self):
        """Test slippage parameter setting."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Set slippage params
        result = manager.set_slippage_params(system, 2.5)
        
        assert result == manager  # Method chaining
        assert manager.config.slippage_amount == 2.5
        assert manager.config.include_slippage
    
    def test_balance_params(self):
        """Test balance parameter setting."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_balance_params(system, 50000.0, 100.0)
        
        assert manager.config.initial_balance_price == 50000.0
        assert manager.config.initial_balance_points == 100.0
    
    def test_viop_index_configuration(self):
        """Test VIOP index configuration."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_viop_index_params(system, contract_count=5, asset_multiplier=20)
        
        assert manager.config.asset_type == AssetType.VIOP_INDEX
        assert manager.config.contract_count == 5
        assert manager.config.asset_quantity_multiplier == 20
        assert manager.config.total_asset_quantity == 100  # 5 * 20
        assert manager.config.currency_type == CurrencyType.TL
    
    def test_bist_stock_configuration(self):
        """Test BIST stock configuration."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_bist_stock_params(system, shares_count=2000, asset_multiplier=1)
        
        assert manager.config.asset_type == AssetType.BIST_STOCK
        assert manager.config.shares_count == 2000
        assert manager.config.total_asset_quantity == 2000
        assert manager.config.currency_type == CurrencyType.TL
    
    def test_fx_gold_configuration(self):
        """Test FX gold micro configuration."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_fx_gold_micro_params(system, contract_count=2, asset_multiplier=1)
        
        assert manager.config.asset_type == AssetType.FX_GOLD_MICRO
        assert manager.config.contract_count == 2
        assert manager.config.total_asset_quantity == 2
        assert manager.config.currency_type == CurrencyType.USD
    
    def test_position_value_calculation(self):
        """Test position value calculation."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure for VIOP index: 10 contracts * 10 multiplier = 100 quantity
        manager.set_viop_index_params(system, contract_count=10, asset_multiplier=10)
        
        # Test calculation
        price = 1500.0
        position_value = manager.calculate_position_value(price)
        
        expected_value = 1500.0 * 100  # price * total_asset_quantity
        assert position_value == expected_value
    
    def test_commission_calculation_viop(self):
        """Test commission calculation for VIOP contracts."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure VIOP with commission
        manager.set_viop_index_params(system, contract_count=5, asset_multiplier=10)
        manager.set_commission_params(system, 4.0)
        
        commission = manager.calculate_commission(1000.0)
        
        # VIOP: commission_asset_quantity * commission_multiplier = 5 * 4.0 = 20
        assert commission == 20.0
    
    def test_commission_calculation_bist(self):
        """Test commission calculation for BIST stocks."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure BIST stock with commission
        manager.set_bist_stock_params(system, shares_count=1000, asset_multiplier=1)
        manager.set_commission_params(system, 2.0)  # 2 per mille
        
        price = 50.0
        commission = manager.calculate_commission(price)
        
        # BIST: (price * commission_asset_quantity * commission_multiplier) / 1000
        expected = (50.0 * 1000 * 2.0) / 1000.0  # 100.0
        assert commission == expected
    
    def test_commission_calculation_fx(self):
        """Test commission calculation for FX instruments."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure FX with zero commission (typical for micro accounts)
        manager.set_fx_gold_micro_params(system, contract_count=1, asset_multiplier=1)
        manager.set_commission_params(system, 0.0)
        
        commission = manager.calculate_commission(1800.0)
        
        # FX with zero commission
        assert commission == 0.0
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure with slippage
        manager.set_viop_index_params(system, contract_count=10, asset_multiplier=10)
        manager.set_slippage_params(system, 1.5)
        
        slippage = manager.calculate_slippage(1000.0)
        
        # slippage_amount * total_asset_quantity = 1.5 * 100 = 150.0
        assert slippage == 150.0
    
    def test_total_trade_cost(self):
        """Test total trade cost calculation."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure with all costs
        manager.set_viop_index_params(system, contract_count=5, asset_multiplier=10)
        manager.set_commission_params(system, 3.0)
        manager.set_slippage_params(system, 2.0)
        
        price = 1000.0
        total_cost = manager.calculate_total_trade_cost(price)
        
        # position_value + commission + slippage
        position_value = 1000.0 * 50  # 50000
        commission = 5 * 3.0  # 15
        slippage = 2.0 * 50  # 100
        expected_total = position_value + commission + slippage  # 50115
        
        assert total_cost == expected_total
    
    def test_required_margin_viop(self):
        """Test required margin calculation for VIOP."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_viop_index_params(system, contract_count=10, asset_multiplier=10)
        
        price = 1500.0
        margin = manager.calculate_required_margin(price, margin_rate=0.1)
        
        position_value = 1500.0 * 100  # 150000
        expected_margin = position_value * 0.1  # 15000
        
        assert margin == expected_margin
    
    def test_required_margin_bist(self):
        """Test required margin calculation for BIST stocks."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_bist_stock_params(system, shares_count=1000, asset_multiplier=1)
        
        price = 25.0
        margin = manager.calculate_required_margin(price)
        
        position_value = 25.0 * 1000  # 25000
        expected_margin = position_value * 0.5  # 12500 (50% for stocks)
        
        assert margin == expected_margin
    
    def test_required_margin_fx(self):
        """Test required margin calculation for FX."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_fx_gold_micro_params(system, contract_count=1, asset_multiplier=1)
        
        price = 1800.0
        margin = manager.calculate_required_margin(price)
        
        position_value = 1800.0 * 1  # 1800
        expected_margin = position_value * 0.01  # 18 (1% for FX)
        
        assert margin == expected_margin
    
    def test_pnl_calculation_long(self):
        """Test P&L calculation for long positions."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_viop_index_params(system, contract_count=5, asset_multiplier=10)
        manager.set_commission_params(system, 2.0)
        manager.set_slippage_params(system, 0.5)
        
        entry_price = 1000.0
        exit_price = 1050.0
        
        pnl = manager.calculate_pnl(entry_price, exit_price, is_long=True)
        
        # Gross P&L: (1050 - 1000) * 50 = 2500
        # Costs: entry_commission (10) + exit_commission (10) + entry_slippage (25) + exit_slippage (25) = 70
        # Net P&L: 2500 - 70 = 2430
        
        expected_pnl = 2430.0
        assert pnl == expected_pnl
    
    def test_pnl_calculation_short(self):
        """Test P&L calculation for short positions."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_viop_index_params(system, contract_count=5, asset_multiplier=10)
        manager.set_commission_params(system, 2.0)
        
        entry_price = 1000.0
        exit_price = 950.0
        
        pnl = manager.calculate_pnl(entry_price, exit_price, is_long=False)
        
        # Gross P&L for short: -(950 - 1000) * 50 = 2500
        # Costs: entry_commission (10) + exit_commission (10) = 20
        # Net P&L: 2500 - 20 = 2480
        
        expected_pnl = 2480.0
        assert pnl == expected_pnl
    
    def test_asset_info(self):
        """Test asset information retrieval."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_viop_index_params(system, contract_count=8, asset_multiplier=15)
        manager.set_commission_params(system, 4.5)
        
        info = manager.get_asset_info()
        
        assert info['asset_type'] == 'VIOP_INDEX'
        assert info['currency_type'] == 'TL'
        assert info['contract_count'] == 8
        assert info['asset_quantity_multiplier'] == 15
        assert info['total_asset_quantity'] == 120  # 8 * 15
        assert info['commission_multiplier'] == 4.5
        assert info['include_commission'] == True
    
    def test_cost_breakdown(self):
        """Test detailed cost breakdown."""
        manager = CVarlikManager()
        system = MockSystem()
        
        manager.set_viop_index_params(system, contract_count=10, asset_multiplier=10)
        manager.set_commission_params(system, 3.0)
        manager.set_slippage_params(system, 1.0)
        
        price = 1200.0
        breakdown = manager.get_cost_breakdown(price)
        
        assert 'position_value' in breakdown
        assert 'commission' in breakdown
        assert 'slippage' in breakdown
        assert 'total_cost' in breakdown
        assert 'required_margin' in breakdown
        
        assert breakdown['position_value'] == 1200.0 * 100  # 120000
        assert breakdown['commission'] == 10 * 3.0  # 30
        assert breakdown['slippage'] == 1.0 * 100  # 100
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Valid configuration
        manager.set_viop_index_params(system, contract_count=5, asset_multiplier=10)
        assert manager.is_configuration_valid()
        
        # Invalid configuration (manually set to test)
        manager.config.contract_count = 0
        assert not manager.is_configuration_valid()
    
    def test_method_chaining(self):
        """Test method chaining functionality."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Test method chaining
        result = (manager.set_viop_index_params(system, 10, 10)
                        .set_commission_params(system, 3.0)
                        .set_slippage_params(system, 1.5)
                        .set_balance_params(system, 100000.0))
        
        assert result == manager
        assert manager.config.contract_count == 10
        assert manager.config.commission_multiplier == 3.0
        assert manager.config.slippage_amount == 1.5
        assert manager.config.initial_balance_price == 100000.0


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_preset_configurations(self):
        """Test preset configuration creation."""
        presets = create_preset_configurations()
        
        assert 'VIOP_XU030' in presets
        assert 'BIST_STOCK' in presets
        assert 'FX_GOLD' in presets
        
        # Test VIOP preset
        viop_config = presets['VIOP_XU030']
        assert viop_config.asset_type == AssetType.VIOP_INDEX
        assert viop_config.contract_count == 10
        assert viop_config.total_asset_quantity == 100  # 10 * 10
        
        # Test BIST preset
        bist_config = presets['BIST_STOCK']
        assert bist_config.asset_type == AssetType.BIST_STOCK
        assert bist_config.shares_count == 1000
        
        # Test FX preset
        fx_config = presets['FX_GOLD']
        assert fx_config.asset_type == AssetType.FX_GOLD_MICRO
        assert fx_config.currency_type == CurrencyType.USD
    
    def test_recommended_settings(self):
        """Test recommended settings retrieval."""
        # Test VIOP recommendations
        viop_rec = get_recommended_settings(AssetType.VIOP_INDEX)
        assert 'contract_count' in viop_rec
        assert 'asset_multiplier' in viop_rec
        assert 'commission_multiplier' in viop_rec
        
        # Test BIST recommendations
        bist_rec = get_recommended_settings(AssetType.BIST_STOCK)
        assert 'shares_count' in bist_rec
        assert 'asset_multiplier' in bist_rec
        
        # Test FX recommendations
        fx_rec = get_recommended_settings(AssetType.FX_GOLD_MICRO)
        assert 'contract_count' in fx_rec
        assert 'commission_multiplier' in fx_rec
        
        # Test unknown asset type
        unknown_rec = get_recommended_settings(AssetType.VIOP_GOLD_OUNCE)
        assert unknown_rec == {}


class TestAssetTypes:
    """Test cases for different asset types."""
    
    def test_all_viop_configurations(self):
        """Test all VIOP asset configurations."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Test VIOP Index
        manager.set_viop_index_params(system, 10, 10)
        assert manager.config.asset_type == AssetType.VIOP_INDEX
        assert manager.config.currency_type == CurrencyType.TL
        
        # Test VIOP Stock
        manager.set_viop_stock_params(system, 5, 100)
        assert manager.config.asset_type == AssetType.VIOP_STOCK
        assert manager.config.total_asset_quantity == 500
        
        # Test VIOP Currency
        manager.set_viop_currency_params(system, 2, 1000, "USD")
        assert manager.config.asset_type == AssetType.VIOP_CURRENCY
        assert manager.config.total_asset_quantity == 2000
        
        # Test VIOP Gold Gram
        manager.set_viop_gold_gram_params(system, 3, 1)
        assert manager.config.asset_type == AssetType.VIOP_GOLD_GRAM
        assert manager.config.total_asset_quantity == 3
    
    def test_bank_configurations(self):
        """Test bank instrument configurations."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Test Bank Currency
        manager.set_bank_currency_params(system, 5, 1, "EUR")
        assert manager.config.asset_type == AssetType.BANK_CURRENCY
        assert manager.config.total_asset_quantity == 5
        
        # Test Bank Gold
        manager.set_bank_gold_params(system, 10, 1)
        assert manager.config.asset_type == AssetType.BANK_GOLD
        assert manager.config.total_asset_quantity == 10
    
    def test_fx_configurations(self):
        """Test FX instrument configurations."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Test FX Gold Micro
        manager.set_fx_gold_micro_params(system, 2, 1)
        assert manager.config.asset_type == AssetType.FX_GOLD_MICRO
        assert manager.config.currency_type == CurrencyType.USD
        assert manager.config.total_asset_quantity == 2
        
        # Test FX Currency
        manager.set_fx_currency_params(system, 1, 1)
        assert manager.config.asset_type == AssetType.FX_CURRENCY
        assert manager.config.currency_type == CurrencyType.USD


class TestRealWorldScenarios:
    """Test cases for real-world trading scenarios."""
    
    def test_viop_xu030_trading_scenario(self):
        """Test realistic VIOP XU030 trading scenario."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure for XU030 trading
        manager.set_viop_index_params(system, contract_count=10, asset_multiplier=10)
        manager.set_commission_params(system, 3.0)
        manager.set_balance_params(system, 100000.0)
        
        # Trade scenario
        entry_price = 1500.0
        exit_price = 1520.0
        
        # Calculate costs and P&L
        entry_cost = manager.calculate_total_trade_cost(entry_price)
        required_margin = manager.calculate_required_margin(entry_price)
        pnl = manager.calculate_pnl(entry_price, exit_price, is_long=True)
        
        # Verify calculations
        assert entry_cost == 150000.0 + 30.0  # position + commission
        assert required_margin == 15000.0  # 10% margin
        assert pnl == 2000.0 - 60.0  # gross profit - total commissions
    
    def test_bist_stock_trading_scenario(self):
        """Test realistic BIST stock trading scenario."""
        manager = CVarlikManager()
        system = MockSystem()
        
        # Configure for BIST stock trading
        manager.set_bist_stock_params(system, shares_count=1000, asset_multiplier=1)
        manager.set_commission_params(system, 1.5)  # 1.5 per mille
        manager.set_balance_params(system, 50000.0)
        
        # Trade scenario
        entry_price = 25.0
        exit_price = 27.5
        
        # Calculate values
        position_value = manager.calculate_position_value(entry_price)
        commission = manager.calculate_commission(entry_price)
        required_margin = manager.calculate_required_margin(entry_price)
        pnl = manager.calculate_pnl(entry_price, exit_price, is_long=True)
        
        # Verify calculations
        assert position_value == 25000.0  # 25 * 1000
        assert commission == 37.5  # (25 * 1000 * 1.5) / 1000
        assert required_margin == 12500.0  # 50% margin for stocks
        # P&L: (27.5 - 25) * 1000 - entry_commission - exit_commission
        exit_commission = (exit_price * 1000 * 1.5) / 1000  # 41.25
        expected_pnl = 2500.0 - 37.5 - 41.25  # 2421.25
        assert pnl == expected_pnl


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])