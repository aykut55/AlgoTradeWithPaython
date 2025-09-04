#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for the trading module.

Tests include:
- Signal generation and processing
- Position management
- Order execution
- Risk management
- Trading statistics
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.trading.signals import Direction, SignalType, SignalInfo, PositionInfo, CSignals
from src.trading.trader import OrderType, OrderStatus, OrderInfo, TradeInfo, RiskSettings, CTrader


class MockSystem:
    """Mock system for testing."""
    
    def __init__(self):
        self.messages = []
    
    def mesaj(self, message: str) -> None:
        self.messages.append(message)


class TestCSignals:
    """Test cases for CSignals class."""
    
    def test_initialization(self):
        """Test CSignals initialization."""
        signals = CSignals()
        
        # Check initial state
        assert not signals.al
        assert not signals.sat
        assert not signals.flat_ol
        assert not signals.kar_al
        assert not signals.zarar_kes
        
        assert signals.son_yon == Direction.FLAT.value
        assert signals.position.direction == Direction.FLAT
        assert len(signals.signal_history) == 0
    
    def test_reset_signals(self):
        """Test signal reset functionality."""
        signals = CSignals()
        
        # Set some signals
        signals.al = True
        signals.sat = True
        signals.kar_al = True
        
        # Reset
        signals.reset_signals()
        
        # Check all signals are reset
        assert not signals.al
        assert not signals.sat  
        assert not signals.flat_ol
        assert not signals.kar_al
        assert not signals.zarar_kes
    
    def test_buy_signal_processing(self):
        """Test buy signal processing."""
        signals = CSignals()
        
        # Set buy signal and process
        signals.al = True
        signal_info = signals.process_signals(1, 100.0)
        
        # Check signal was processed
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.BUY
        assert signal_info.direction == Direction.LONG
        assert signal_info.price == 100.0
        assert signal_info.bar_number == 1
        
        # Check position updated
        assert signals.position.direction == Direction.LONG
        assert signals.position.entry_price == 100.0
        assert signals.position.entry_bar == 1
        assert signals.son_yon == Direction.LONG.value
        
        # Check signal in history
        assert len(signals.signal_history) == 1
        assert signals.signal_history[0] == signal_info
    
    def test_sell_signal_processing(self):
        """Test sell signal processing."""
        signals = CSignals()
        
        # Set sell signal and process
        signals.sat = True
        signal_info = signals.process_signals(1, 100.0)
        
        # Check signal was processed
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.SELL
        assert signal_info.direction == Direction.SHORT
        assert signal_info.price == 100.0
        
        # Check position updated
        assert signals.position.direction == Direction.SHORT
        assert signals.position.entry_price == 100.0
        assert signals.son_yon == Direction.SHORT.value
    
    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions."""
        signals = CSignals()
        
        # Test long position P&L
        signals.al = True
        signals.process_signals(1, 100.0)
        
        # Update current price
        signals.update_current_data(2, 110.0)
        
        # Check unrealized P&L
        expected_pnl = (110.0 - 100.0) * 1.0  # (current - entry) * quantity
        assert abs(signals.position.unrealized_pnl - expected_pnl) < 1e-6
        
        # Test short position P&L
        signals.sat = True
        signals.process_signals(3, 110.0)
        
        signals.update_current_data(4, 105.0)
        expected_pnl = (110.0 - 105.0) * 1.0  # (entry - current) * quantity
        assert abs(signals.position.unrealized_pnl - expected_pnl) < 1e-6
    
    def test_take_profit_signal(self):
        """Test take profit signal processing."""
        signals = CSignals()
        
        # Open long position
        signals.al = True
        signals.process_signals(1, 100.0)
        
        # Take profit
        signals.kar_al = True
        signal_info = signals.process_signals(2, 110.0)
        
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.TAKE_PROFIT
        assert signals.position.is_flat
        assert signals.kar_alindi
    
    def test_stop_loss_signal(self):
        """Test stop loss signal processing."""
        signals = CSignals()
        
        # Open long position
        signals.al = True
        signals.process_signals(1, 100.0)
        
        # Stop loss
        signals.zarar_kes = True
        signal_info = signals.process_signals(2, 90.0)
        
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.STOP_LOSS
        assert signals.position.is_flat
        assert signals.zarar_kesildi
    
    def test_signal_history_management(self):
        """Test signal history tracking."""
        signals = CSignals()
        
        # Generate multiple signals
        signals.al = True
        signals.process_signals(1, 100.0)
        
        signals.sat = True
        signals.process_signals(2, 105.0)
        
        signals.flat_ol = True
        signals.process_signals(3, 102.0)
        
        # Check history
        assert len(signals.signal_history) == 3
        
        # Get last signals
        last_2 = signals.get_last_signals(2)
        assert len(last_2) == 2
        assert last_2[0].signal_type == SignalType.SELL
        assert last_2[1].signal_type == SignalType.FLAT


class TestCTrader:
    """Test cases for CTrader class."""
    
    def create_sample_trader(self) -> CTrader:
        """Create a sample trader with market data."""
        trader = CTrader(id_value=1, name="Test Trader")
        
        # Create sample OHLCV data
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100), 
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'lot': [1.0] * 100
        })
        
        trader.set_data_from_dataframe(data)
        return trader
    
    def test_trader_initialization(self):
        """Test trader initialization."""
        trader = CTrader(id_value=1, name="Test Trader")
        
        assert trader.id == 1
        assert trader.name == "Test Trader"
        assert not trader.is_initialized
        assert trader.current_bar == 0
        assert trader.total_pnl == 0.0
        assert len(trader.pending_orders) == 0
        assert len(trader.open_trades) == 0
    
    def test_trader_data_initialization(self):
        """Test trader initialization with market data."""
        trader = self.create_sample_trader()
        system = MockSystem()
        
        trader.initialize(system, None)
        
        assert trader.is_initialized
        assert trader.indicators is not None
        assert len(system.messages) > 0
        assert "initialized" in system.messages[-1].lower()
    
    def test_buy_signal_generation(self):
        """Test buy signal generation."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Generate buy signal
        signal_info = trader.generate_buy_signal(system, 10, 100.0)
        
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.BUY
        assert signal_info.direction == Direction.LONG
        assert signal_info.price == 100.0
        assert trader.signals.position.is_long
    
    def test_sell_signal_generation(self):
        """Test sell signal generation."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Generate sell signal
        signal_info = trader.generate_sell_signal(system, 10, 100.0)
        
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.SELL
        assert signal_info.direction == Direction.SHORT
        assert signal_info.price == 100.0
        assert trader.signals.position.is_short
    
    def test_position_closing(self):
        """Test position closing functionality."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Open position
        trader.generate_buy_signal(system, 10, 100.0)
        assert trader.signals.position.is_long
        
        # Close position
        signal_info = trader.close_position(system, 15, "Manual", 105.0)
        
        assert signal_info is not None
        assert signal_info.signal_type == SignalType.FLAT
        assert trader.signals.position.is_flat
        assert trader.total_trades == 1
        assert trader.total_pnl > 0  # Should be profitable
    
    def test_risk_settings(self):
        """Test risk settings functionality."""
        trader = self.create_sample_trader()
        
        # Create risk settings
        risk_settings = RiskSettings(
            max_position_size=2.0,
            take_profit_points=10.0,
            stop_loss_points=5.0,
            max_daily_loss=100.0
        )
        
        assert risk_settings.is_valid()
        
        trader.set_risk_settings(risk_settings)
        assert trader.risk_settings.take_profit_points == 10.0
        assert trader.risk_settings.stop_loss_points == 5.0
    
    def test_risk_management_take_profit(self):
        """Test automatic take profit execution."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Set risk settings
        risk_settings = RiskSettings(take_profit_points=10.0)
        trader.set_risk_settings(risk_settings)
        
        # Open long position at 100.0
        trader.generate_buy_signal(system, 10, 100.0)
        
        # Update bar where take profit should trigger (price >= 110.0)
        trader.close[15] = 110.0  # Modify test data
        trader.update_bar(system, 15)
        
        # Position should be closed by take profit
        assert trader.signals.position.is_flat
        assert trader.signals.kar_alindi
    
    def test_risk_management_stop_loss(self):
        """Test automatic stop loss execution."""
        trader = self.create_sample_trader()
        system = MockSystem()  
        trader.initialize(system, None)
        
        # Set risk settings
        risk_settings = RiskSettings(stop_loss_points=5.0)
        trader.set_risk_settings(risk_settings)
        
        # Open long position at 100.0
        trader.generate_buy_signal(system, 10, 100.0)
        
        # Update bar where stop loss should trigger (price <= 95.0)
        trader.close[15] = 95.0  # Modify test data
        trader.update_bar(system, 15)
        
        # Position should be closed by stop loss
        assert trader.signals.position.is_flat
        assert trader.signals.zarar_kesildi
    
    def test_trading_statistics(self):
        """Test trading statistics calculation."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Make some trades
        # Winning trade
        trader.generate_buy_signal(system, 10, 100.0)
        trader.close_position(system, 15, "Manual", 105.0)
        
        # Losing trade
        trader.generate_sell_signal(system, 20, 100.0)
        trader.close_position(system, 25, "Manual", 102.0)
        
        stats = trader.get_trading_statistics()
        
        assert stats['total_trades'] == 2
        assert stats['winning_trades'] == 1
        assert stats['losing_trades'] == 1
        assert stats['win_rate'] == 50.0
        assert 'total_pnl' in stats
        assert 'average_win' in stats
        assert 'average_loss' in stats
    
    def test_position_info(self):
        """Test position information retrieval."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Test flat position
        pos_info = trader.get_position_info()
        assert pos_info['direction'] == Direction.FLAT.value
        assert pos_info['size'] == 1.0
        assert pos_info['unrealized_pnl'] == 0.0
        
        # Test long position
        trader.generate_buy_signal(system, 10, 100.0)
        trader.update_bar(system, 15)  # Update current bar
        
        pos_info = trader.get_position_info()
        assert pos_info['direction'] == Direction.LONG.value
        assert pos_info['entry_price'] == 100.0
        assert pos_info['bars_in_trade'] == 5
    
    def test_signal_history_retrieval(self):
        """Test signal history retrieval."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Generate signals
        trader.generate_buy_signal(system, 10, 100.0)
        trader.close_position(system, 15, "Manual", 105.0)
        
        history = trader.get_signal_history(5)
        assert len(history) == 2
        assert history[0].signal_type == SignalType.BUY
        assert history[1].signal_type == SignalType.FLAT
    
    def test_position_constraints(self):
        """Test position opening constraints."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Open long position
        signal1 = trader.generate_buy_signal(system, 10, 100.0)
        assert signal1 is not None
        
        # Try to open another long position (should be blocked)
        signal2 = trader.generate_buy_signal(system, 11, 101.0)
        assert signal2 is None  # Should not generate signal when already long
        
        # Try to open short position (should close long first)
        signal3 = trader.generate_sell_signal(system, 12, 99.0)
        assert signal3 is not None
        assert trader.signals.position.is_short
    
    def test_daily_loss_limit(self):
        """Test daily loss limit constraint."""
        trader = self.create_sample_trader()
        system = MockSystem()
        trader.initialize(system, None)
        
        # Set low daily loss limit
        risk_settings = RiskSettings(max_daily_loss=5.0)
        trader.set_risk_settings(risk_settings)
        
        # Make a losing trade
        trader.generate_buy_signal(system, 10, 100.0)
        trader.close_position(system, 15, "Manual", 94.0)  # Loss of 6.0
        
        # Try to open new position (should be blocked due to daily loss)
        signal = trader.generate_buy_signal(system, 20, 95.0)
        assert signal is None
    
    def test_order_management(self):
        """Test order management functionality."""
        order = OrderInfo(
            order_id="TEST001",
            order_type=OrderType.MARKET,
            direction=Direction.LONG,
            quantity=1.0,
            price=100.0
        )
        
        assert not order.is_filled
        assert order.is_pending
        
        order.status = OrderStatus.FILLED
        assert order.is_filled
        assert not order.is_pending
    
    def test_trade_info(self):
        """Test trade information functionality."""
        trade = TradeInfo(
            trade_id="TRADE001",
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            entry_bar=10
        )
        
        assert not trade.is_closed
        assert trade.unrealized_pnl == 0.0
        
        trade.exit_price = 105.0
        trade.pnl = 5.0
        assert trade.is_closed


class TestRiskSettings:
    """Test cases for RiskSettings class."""
    
    def test_valid_risk_settings(self):
        """Test valid risk settings."""
        settings = RiskSettings(
            max_position_size=2.0,
            take_profit_points=10.0,
            stop_loss_points=5.0,
            max_daily_loss=100.0
        )
        
        assert settings.is_valid()
    
    def test_invalid_risk_settings(self):
        """Test invalid risk settings."""
        # Negative position size
        settings = RiskSettings(max_position_size=-1.0)
        assert not settings.is_valid()
        
        # Negative take profit
        settings = RiskSettings(take_profit_points=-5.0)
        assert not settings.is_valid()
        
        # Negative stop loss
        settings = RiskSettings(stop_loss_points=-5.0)
        assert not settings.is_valid()


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])