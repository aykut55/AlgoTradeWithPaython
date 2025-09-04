#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for the system module.

Tests include:
- System wrapper functionality
- Component integration
- Strategy execution
- Backtest engine
- Performance metrics
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from src.system.system_wrapper import (
    CSystemWrapper, SystemConfiguration, StrategySignals, 
    ExecutionMode, ReportingLevel, ExecutionStatistics
)
from src.system.backtest_engine import (
    CBacktestEngine, BacktestConfiguration, BacktestMode,
    BacktestMetrics, BacktestResults
)
from src.trading.trader import RiskSettings
from src.trading.signals import Direction


class MockSystem:
    """Mock system for testing."""
    
    def __init__(self):
        self.messages = []
        self.Sembol = "TEST"
        self.Periyot = "1D"
        self.Name = "TestSystem"
    
    def mesaj(self, message: str) -> None:
        self.messages.append(message)


class TestSystemConfiguration:
    """Test cases for SystemConfiguration class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = SystemConfiguration()
        
        assert config.symbol == ""
        assert config.period == ""
        assert config.system_name == ""
        assert config.execution_mode == ExecutionMode.SINGLE_RUN
        assert config.reporting_level == ReportingLevel.STANDARD
        assert config.contract_count == 10
        assert config.calculate_statistics == True
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = SystemConfiguration(
            symbol="EURUSD",
            system_name="TestStrategy",
            contract_count=1
        )
        assert config.validate()
        
        # Invalid configuration - empty symbol
        config.symbol = ""
        assert not config.validate()
        
        # Invalid configuration - zero contract count
        config.symbol = "EURUSD"
        config.contract_count = 0
        assert not config.validate()


class TestStrategySignals:
    """Test cases for StrategySignals class."""
    
    def test_signals_initialization(self):
        """Test signals initialization."""
        signals = StrategySignals()
        
        assert not signals.buy
        assert not signals.sell
        assert not signals.flat
        assert not signals.pass_signal
        assert not signals.take_profit
        assert not signals.stop_loss
        assert not signals.has_signal()
    
    def test_signals_reset(self):
        """Test signals reset functionality."""
        signals = StrategySignals()
        
        # Set some signals
        signals.buy = True
        signals.take_profit = True
        assert signals.has_signal()
        
        # Reset signals
        signals.reset()
        assert not signals.has_signal()
        assert not signals.buy
        assert not signals.take_profit
    
    def test_has_signal_detection(self):
        """Test signal detection."""
        signals = StrategySignals()
        
        # Test individual signals
        signals.buy = True
        assert signals.has_signal()
        signals.reset()
        
        signals.sell = True
        assert signals.has_signal()
        signals.reset()
        
        signals.flat = True
        assert signals.has_signal()
        signals.reset()
        
        signals.take_profit = True
        assert signals.has_signal()


class TestExecutionStatistics:
    """Test cases for ExecutionStatistics class."""
    
    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = ExecutionStatistics()
        
        assert stats.start_time is None
        assert stats.end_time is None
        assert stats.total_bars_processed == 0
        assert stats.signals_generated == 0
        assert stats.execution_time_ms == 0.0
    
    def test_execution_time_calculations(self):
        """Test execution time calculations."""
        stats = ExecutionStatistics()
        stats.execution_time_ms = 5000.0  # 5 seconds
        stats.total_bars_processed = 100
        
        assert stats.execution_time_seconds == 5.0
        assert stats.bars_per_second == 20.0


class TestCSystemWrapper:
    """Test cases for CSystemWrapper class."""
    
    def create_sample_system_wrapper(self) -> CSystemWrapper:
        """Create a sample system wrapper with market data."""
        wrapper = CSystemWrapper(id_value=1, system_name="TestSystem")
        
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
        
        wrapper.set_data_from_dataframe(data)
        return wrapper
    
    def test_system_wrapper_initialization(self):
        """Test system wrapper initialization."""
        wrapper = CSystemWrapper(id_value=1, system_name="TestSystem")
        
        assert wrapper.id == 1
        assert wrapper.system_name == "TestSystem"
        assert not wrapper.is_initialized
        assert wrapper.current_bar == 0
        assert wrapper.trader is None
        assert wrapper.indicators is None
        assert wrapper.utils is None
    
    def test_create_modules(self):
        """Test module creation."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        
        wrapper.create_modules(system)
        
        assert wrapper.trader is not None
        assert wrapper.indicators is not None
        assert wrapper.utils is not None
    
    def test_system_initialization(self):
        """Test complete system initialization."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        
        wrapper.initialize_system(system, None)
        
        assert wrapper.is_initialized
        assert wrapper.trader is not None
        assert wrapper.indicators is not None
        assert wrapper.utils is not None
        assert wrapper.stats.start_time is not None
        assert len(system.messages) > 0
        assert "initialized successfully" in system.messages[-1]
    
    def test_system_configuration(self):
        """Test system configuration."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        
        config = SystemConfiguration(
            symbol="EURUSD",
            system_name="TestStrategy",
            contract_count=5,
            calculate_statistics=True
        )
        
        wrapper.configure_system(config)
        assert wrapper.config.symbol == "EURUSD"
        assert wrapper.config.contract_count == 5
    
    def test_strategy_signals_setting(self):
        """Test strategy signals setting."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Set buy signal
        wrapper.set_strategy_signals(system, 10, buy=True)
        assert wrapper.signals.buy
        assert not wrapper.signals.sell
        assert wrapper.stats.signals_generated == 1
        
        # Reset and set sell signal
        wrapper.signals.reset()
        wrapper.set_strategy_signals(system, 11, sell=True, take_profit=True)
        assert wrapper.signals.sell
        assert wrapper.signals.take_profit
        assert wrapper.stats.signals_generated == 2
    
    def test_single_bar_execution(self):
        """Test single bar strategy execution."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Test valid bar execution
        success = wrapper.execute_strategy_bar(system, 10)
        assert success
        assert wrapper.current_bar == 10
        assert wrapper.stats.total_bars_processed == 1
        
        # Test invalid bar execution
        success = wrapper.execute_strategy_bar(system, 200)  # Beyond data range
        assert not success
    
    def test_complete_strategy_execution(self):
        """Test complete strategy execution."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Simple strategy that buys on bar 10
        def simple_strategy(bar_index: int) -> None:
            if bar_index == 10:
                wrapper.set_strategy_signals(system, bar_index, buy=True)
            elif bar_index == 20:
                wrapper.set_strategy_signals(system, bar_index, sell=True)
        
        wrapper.on_bar_update = simple_strategy
        
        # Execute strategy
        results = wrapper.execute_complete_strategy(system, start_bar=1, end_bar=50)
        
        assert results['success']
        assert results['bars_processed'] == 50
        assert results['execution_time_ms'] > 0
        assert wrapper.stats.signals_generated >= 2  # At least buy and sell
    
    def test_system_reset(self):
        """Test system reset functionality."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Execute some bars and set some state
        wrapper.execute_strategy_bar(system, 10)
        wrapper.set_strategy_signals(system, 10, buy=True)
        
        # Reset system
        wrapper.reset_system(system)
        
        assert wrapper.current_bar == 0
        assert not wrapper.signals.has_signal()
        assert wrapper.stats.total_bars_processed == 0
    
    def test_component_access(self):
        """Test component access methods."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Test component access
        trader = wrapper.get_component('trader')
        assert trader is not None
        assert trader == wrapper.trader
        
        indicators = wrapper.get_component('indicators')
        assert indicators is not None
        assert indicators == wrapper.indicators
        
        utils = wrapper.get_component('utils')
        assert utils is not None
        assert utils == wrapper.utils
        
        # Test invalid component
        invalid = wrapper.get_component('invalid')
        assert invalid is None
    
    def test_trading_statistics_integration(self):
        """Test trading statistics integration."""
        wrapper = self.create_sample_system_wrapper()
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Execute some trading
        wrapper.set_strategy_signals(system, 10, buy=True)
        wrapper.execute_strategy_bar(system, 10)
        
        wrapper.set_strategy_signals(system, 20, sell=True)
        wrapper.execute_strategy_bar(system, 20)
        
        # Get statistics
        stats = wrapper.get_trading_statistics()
        assert 'execution_time_ms' in stats
        assert 'bars_processed' in stats
        assert 'signals_generated' in stats


class TestBacktestConfiguration:
    """Test cases for BacktestConfiguration class."""
    
    def test_default_configuration(self):
        """Test default backtest configuration."""
        config = BacktestConfiguration()
        
        assert config.mode == BacktestMode.FULL_HISTORY
        assert config.start_bar == 1
        assert config.initial_capital == 100000.0
        assert config.commission_per_trade == 0.0
        assert config.generate_trade_log == True
    
    def test_configuration_validation(self):
        """Test backtest configuration validation."""
        # Full history mode - should always be valid
        config = BacktestConfiguration(mode=BacktestMode.FULL_HISTORY)
        assert config.validate()
        
        # Bar range mode - valid
        config = BacktestConfiguration(
            mode=BacktestMode.BAR_RANGE,
            start_bar=10,
            end_bar=100
        )
        assert config.validate()
        
        # Bar range mode - invalid (end before start)
        config.end_bar = 5
        assert not config.validate()
        
        # Date range mode - valid
        config = BacktestConfiguration(
            mode=BacktestMode.DATE_RANGE,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        assert config.validate()
        
        # Date range mode - invalid (missing dates)
        config.start_date = None
        assert not config.validate()


class TestBacktestMetrics:
    """Test cases for BacktestMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = BacktestMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
    
    def test_metrics_to_dict(self):
        """Test metrics dictionary conversion."""
        metrics = BacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            total_trades=10,
            win_rate=60.0
        )
        
        dict_metrics = metrics.to_dict()
        
        assert dict_metrics['total_return'] == 0.15
        assert dict_metrics['sharpe_ratio'] == 1.2
        assert dict_metrics['max_drawdown'] == -0.05
        assert dict_metrics['total_trades'] == 10
        assert dict_metrics['win_rate'] == 60.0


class TestCBacktestEngine:
    """Test cases for CBacktestEngine class."""
    
    def create_sample_backtest_setup(self):
        """Create sample backtest setup."""
        wrapper = CSystemWrapper(system_name="TestStrategy")
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100), 
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'lot': [1.0] * 100
        })
        
        wrapper.set_data_from_dataframe(data)
        
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        engine = CBacktestEngine(wrapper)
        
        return wrapper, system, engine
    
    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization."""
        wrapper, system, engine = self.create_sample_backtest_setup()
        
        assert engine.system_wrapper == wrapper
        assert engine.results is None
    
    def test_simple_backtest_execution(self):
        """Test simple backtest execution."""
        wrapper, system, engine = self.create_sample_backtest_setup()
        
        # Simple buy and hold strategy
        def simple_strategy(sys, wrap, bar_index):
            if bar_index == 10:
                wrap.set_strategy_signals(sys, bar_index, buy=True)
            elif bar_index == 80:
                wrap.set_strategy_signals(sys, bar_index, flat=True)
        
        # Configure backtest
        config = BacktestConfiguration(
            mode=BacktestMode.BAR_RANGE,
            start_bar=1,
            end_bar=90,
            initial_capital=100000.0
        )
        
        # Run backtest
        results = engine.run_backtest(system, config, simple_strategy)
        
        # Verify results
        assert results is not None
        assert results.bars_processed > 0
        assert results.execution_time_seconds > 0
        assert len(results.equity_curve) > 0
        assert results.config == config
        assert results.start_time <= results.end_time
    
    def test_backtest_metrics_calculation(self):
        """Test backtest metrics calculation."""
        wrapper, system, engine = self.create_sample_backtest_setup()
        
        # Strategy that generates some trades
        def trading_strategy(sys, wrap, bar_index):
            if bar_index == 20:
                wrap.set_strategy_signals(sys, bar_index, buy=True)
            elif bar_index == 40:
                wrap.set_strategy_signals(sys, bar_index, sell=True)
            elif bar_index == 60:
                wrap.set_strategy_signals(sys, bar_index, flat=True)
        
        config = BacktestConfiguration(
            mode=BacktestMode.FULL_HISTORY,
            initial_capital=100000.0
        )
        
        results = engine.run_backtest(system, config, trading_strategy)
        
        # Check that metrics are calculated
        metrics = results.metrics
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.total_trades, int)
        assert len(results.drawdown_curve) > 0
    
    def test_backtest_results_dataframe(self):
        """Test backtest results DataFrame conversion."""
        wrapper, system, engine = self.create_sample_backtest_setup()
        
        def simple_strategy(sys, wrap, bar_index):
            pass  # No trading
        
        config = BacktestConfiguration(mode=BacktestMode.BAR_RANGE, start_bar=1, end_bar=50)
        results = engine.run_backtest(system, config, simple_strategy)
        
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'equity' in df.columns
        assert 'drawdown' in df.columns
        assert len(df) > 0
    
    def test_walk_forward_analysis(self):
        """Test walk-forward analysis."""
        wrapper, system, engine = self.create_sample_backtest_setup()
        
        def simple_strategy(sys, wrap, bar_index):
            if bar_index % 20 == 0:  # Trade every 20 bars
                wrap.set_strategy_signals(sys, bar_index, buy=True)
            elif bar_index % 20 == 10:
                wrap.set_strategy_signals(sys, bar_index, flat=True)
        
        config = BacktestConfiguration(
            mode=BacktestMode.WALK_FORWARD,
            start_bar=1,
            training_period_bars=30,
            test_period_bars=10,
            walk_forward_step=5,
            initial_capital=100000.0
        )
        
        # Run walk-forward analysis
        results_list = engine.run_walk_forward_analysis(system, config, simple_strategy)
        
        # Should have multiple results
        assert len(results_list) > 0
        for results in results_list:
            assert isinstance(results, BacktestResults)
            assert results.bars_processed > 0


class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_trading_system(self):
        """Test complete end-to-end trading system."""
        # Create system wrapper
        wrapper = CSystemWrapper(system_name="IntegrationTest")
        
        # Load sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 200),
            'high': np.random.uniform(100, 102, 200),
            'low': np.random.uniform(98, 100, 200), 
            'close': np.random.uniform(99, 101, 200),
            'volume': np.random.randint(1000, 10000, 200),
            'lot': [1.0] * 200
        })
        wrapper.set_data_from_dataframe(data)
        
        # Initialize system
        system = MockSystem()
        wrapper.initialize_system(system, None)
        
        # Configure system
        config = SystemConfiguration(
            symbol="TESTPAIR",
            system_name="IntegrationTest",
            execution_mode=ExecutionMode.BACKTEST,
            calculate_statistics=True
        )
        wrapper.configure_system(config)
        
        # Create backtest engine
        engine = CBacktestEngine(wrapper)
        
        # Define a complete trading strategy
        def ma_crossover_strategy(sys, wrap, bar_index):
            if bar_index < 20:  # Need history for MA calculation
                return
            
            # Calculate simple moving averages
            fast_ma = np.mean(wrap.close[bar_index-5:bar_index])
            slow_ma = np.mean(wrap.close[bar_index-20:bar_index])
            
            # Current position
            current_position = wrap.trader.signals.position.direction
            
            # Generate signals
            if fast_ma > slow_ma and current_position != Direction.LONG:
                wrap.set_strategy_signals(sys, bar_index, buy=True)
            elif fast_ma < slow_ma and current_position != Direction.SHORT:
                wrap.set_strategy_signals(sys, bar_index, sell=True)
        
        # Run backtest
        backtest_config = BacktestConfiguration(
            mode=BacktestMode.FULL_HISTORY,
            initial_capital=100000.0,
            commission_per_trade=2.0
        )
        
        results = engine.run_backtest(system, backtest_config, ma_crossover_strategy)
        
        # Verify comprehensive results
        assert results.bars_processed > 0
        assert results.execution_time_seconds > 0
        assert len(results.equity_curve) > 0
        assert len(results.drawdown_curve) > 0
        
        # Verify statistics were calculated
        stats = wrapper.get_trading_statistics()
        assert 'total_trades' in stats
        assert 'execution_time_ms' in stats
        assert 'bars_processed' in stats
        
        # Verify metrics
        metrics = results.metrics
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.sharpe_ratio, float)
        
        # Test results export functionality
        assert results.to_dataframe() is not None


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])