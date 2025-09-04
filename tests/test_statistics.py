"""
Comprehensive tests for the statistics module.

Tests cover:
- Basic statistics calculations (trades, wins, losses)
- Performance metrics (Sharpe, Sortino, Profit Factor)
- Drawdown analysis and risk metrics
- Edge cases and error handling
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from src.statistics.statistics_manager import (
    CStatistics,
    TradingStatistics,
    PerformanceMetrics,
    DrawdownAnalysis,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_profit_factor
)
from src.trading.trader import TradeInfo, Direction
from src.trading.signals import SignalType
from src.core.base import MarketData


class MockSystem:
    """Mock system for testing."""
    def __init__(self):
        self.id = 1
        self.market_data = MarketData()
        # Create sample market data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        prices = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(100)]
        
        self.market_data.dates = dates
        self.market_data.open = np.array([p + np.random.normal(0, 0.5) for p in prices])
        self.market_data.high = np.array([p + abs(np.random.normal(0, 1)) for p in prices])
        self.market_data.low = np.array([p - abs(np.random.normal(0, 1)) for p in prices])
        self.market_data.close = np.array(prices)
        self.market_data.volume = np.array([1000 + np.random.randint(-200, 200) for _ in range(100)])


class TestTradingStatistics:
    """Test TradingStatistics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of TradingStatistics."""
        stats = TradingStatistics()
        
        assert stats.system_id == 0
        assert stats.system_name == ""
        assert stats.total_trades == 0
        assert stats.winning_trades == 0
        assert stats.losing_trades == 0
        assert stats.winning_trade_ratio == 0.0
        assert stats.initial_balance_price == 0.0
        assert stats.return_price == 0.0
    
    def test_custom_initialization(self):
        """Test custom initialization values."""
        stats = TradingStatistics(
            system_id=123,
            system_name="TestStrategy",
            chart_symbol="EURUSD",
            total_trades=50,
            winning_trades=30,
            losing_trades=20
        )
        
        assert stats.system_id == 123
        assert stats.system_name == "TestStrategy"
        assert stats.chart_symbol == "EURUSD"
        assert stats.total_trades == 50
        assert stats.winning_trades == 30
        assert stats.losing_trades == 20


class TestDrawdownAnalysis:
    """Test DrawdownAnalysis dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of DrawdownAnalysis."""
        dd = DrawdownAnalysis()
        
        assert dd.max_drawdown == 0.0
        assert dd.max_drawdown_date is None
        assert dd.max_drawdown_index == 0
        assert dd.current_drawdown == 0.0
        assert dd.recovery_factor == 0.0
        assert dd.underwater_periods == []


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of PerformanceMetrics."""
        pm = PerformanceMetrics()
        
        assert pm.profit_factor == 0.0
        assert pm.sharpe_ratio == 0.0
        assert pm.sortino_ratio == 0.0
        assert pm.volatility == 0.0
        assert pm.avg_win == 0.0
        assert pm.avg_loss == 0.0
        assert pm.expectancy == 0.0


class TestCStatistics:
    """Test CStatistics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.statistics = CStatistics(system_id=1)
        self.mock_system = MockSystem()
    
    def test_initialization(self):
        """Test CStatistics initialization."""
        stats = CStatistics(123)
        
        assert stats.id == 123
        assert isinstance(stats.statistics, TradingStatistics)
        assert isinstance(stats.drawdown_analysis, DrawdownAnalysis)
        assert isinstance(stats.performance_metrics, PerformanceMetrics)
        assert stats.balance_history == []
        assert stats.trade_records == []
        assert stats.statistics_dict == {}
        assert not stats._is_calculated
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        # Add some data
        self.statistics.add_balance_snapshot(1000.0)
        self.statistics.trade_records.append(self._create_sample_trade())
        self.statistics.statistics_dict["test"] = "value"
        self.statistics._is_calculated = True
        
        # Reset
        self.statistics.reset_statistics()
        
        # Verify reset
        assert self.statistics.balance_history == []
        assert self.statistics.trade_records == []
        assert self.statistics.statistics_dict == {}
        assert not self.statistics._is_calculated
        assert self.statistics.statistics.total_trades == 0
    
    def test_set_system_info(self):
        """Test setting system information."""
        result = self.statistics.set_system_info(
            self.mock_system,
            system_name="TestStrategy",
            chart_symbol="EURUSD",
            chart_period="1H"
        )
        
        # Should return self for method chaining
        assert result is self.statistics
        
        # Verify values
        assert self.statistics.statistics.system_id == 1
        assert self.statistics.statistics.system_name == "TestStrategy"
        assert self.statistics.statistics.chart_symbol == "EURUSD"
        assert self.statistics.statistics.chart_period == "1H"
    
    def test_set_initial_balance(self):
        """Test setting initial balance."""
        result = self.statistics.set_initial_balance(10000.0, 500.0)
        
        # Should return self for method chaining
        assert result is self.statistics
        
        # Verify values
        assert self.statistics.statistics.initial_balance_price == 10000.0
        assert self.statistics.statistics.initial_balance_points == 500.0
    
    def test_set_initial_balance_without_points(self):
        """Test setting initial balance without points parameter."""
        self.statistics.set_initial_balance(10000.0)
        
        assert self.statistics.statistics.initial_balance_price == 10000.0
        assert self.statistics.statistics.initial_balance_points == 10000.0
    
    def test_add_trade_record(self):
        """Test adding trade records."""
        trade = self._create_sample_trade()
        
        self.statistics.add_trade_record(trade)
        
        assert len(self.statistics.trade_records) == 1
        assert self.statistics.trade_records[0] == trade
        assert not self.statistics._is_calculated
    
    def test_add_balance_snapshot(self):
        """Test adding balance snapshots."""
        self.statistics.add_balance_snapshot(1000.0, 5)
        self.statistics.add_balance_snapshot(1100.0, 6)
        
        assert len(self.statistics.balance_history) == 2
        assert self.statistics.balance_history == [1000.0, 1100.0]
        assert self.statistics.statistics.current_balance_price == 1100.0
        assert self.statistics.statistics.last_bar_index == 6
        assert not self.statistics._is_calculated
    
    def test_add_pnl_snapshot(self):
        """Test adding P&L snapshots."""
        self.statistics.add_pnl_snapshot(50.0)
        self.statistics.add_pnl_snapshot(-25.0)
        
        assert len(self.statistics.pnl_history) == 2
        assert self.statistics.pnl_history == [50.0, -25.0]
        assert not self.statistics._is_calculated
    
    def test_calculate_all_statistics_empty_data(self):
        """Test calculating statistics with empty data."""
        result = self.statistics.calculate_all_statistics(self.mock_system)
        
        assert isinstance(result, dict)
        assert self.statistics._is_calculated
        assert self.statistics._last_calculation_time is not None
        
        # Should have basic dictionary structure
        assert "SistemId" in result
        assert "IslemSayisi" in result
        assert result["IslemSayisi"] == "0"
    
    def test_calculate_all_statistics_with_trades(self):
        """Test calculating statistics with sample trades."""
        # Set up data
        self.statistics.set_initial_balance(10000.0)
        self.statistics.set_system_info(self.mock_system, "TestStrategy", "EURUSD", "1H")
        
        # Add winning trade
        winning_trade = TradeInfo(
            trade_id="W001",
            direction=Direction.LONG,
            entry_price=1.1000,
            exit_price=1.1050,
            quantity=100000,
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            pnl=500.0,  # 50 pips * 100k lot = $500
            commission=5.0
        )
        
        # Add losing trade
        losing_trade = TradeInfo(
            trade_id="L001", 
            direction=Direction.SHORT,
            entry_price=1.1040,
            exit_price=1.1060,
            quantity=100000,
            entry_time=datetime(2024, 1, 3),
            exit_time=datetime(2024, 1, 4),
            pnl=-200.0,  # -20 pips * 100k lot = -$200
            commission=5.0
        )
        
        self.statistics.add_trade_record(winning_trade)
        self.statistics.add_trade_record(losing_trade)
        
        # Add balance snapshots
        balances = [10000, 10500, 10300]
        for i, balance in enumerate(balances):
            self.statistics.add_balance_snapshot(balance, i)
        
        # Calculate statistics
        result = self.statistics.calculate_all_statistics(self.mock_system)
        
        # Verify basic statistics
        assert result["IslemSayisi"] == "2"
        assert result["KazandiranIslemSayisi"] == "1"
        assert result["KaybettirenIslemSayisi"] == "1"
        assert float(result["KarliIslemOrani"]) == 50.0
        
        # Verify balance calculations
        assert float(result["IlkBakiyeFiyat"]) == 10000.0
        assert float(result["BakiyeFiyat"]) == 10300.0
        assert float(result["GetiriFiyat"]) == 300.0
        assert float(result["GetiriFiyatYuzde"]) == 3.0
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation logic."""
        # Set up balance history with drawdown pattern
        balances = [1000, 1100, 1200, 1000, 800, 900, 1300]
        
        for balance in balances:
            self.statistics.add_balance_snapshot(balance)
        
        self.statistics._calculate_drawdown_analysis()
        
        # Max drawdown should be from 1200 to 800 = 400
        assert self.statistics.drawdown_analysis.max_drawdown == 400.0
        # Percentage: 400/1200 * 100 = 33.33%
        assert abs(self.statistics.drawdown_analysis.max_drawdown_percent - 33.33) < 0.1
        
        # Should have underwater periods
        assert len(self.statistics.drawdown_analysis.underwater_periods) > 0
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Add sample trades
        trades = [
            (1.1000, 1.1050, Direction.LONG),   # +50 pips profit
            (1.1040, 1.1020, Direction.SHORT),  # +20 pips profit  
            (1.1060, 1.1030, Direction.LONG),   # -30 pips loss
            (1.1070, 1.1080, Direction.SHORT),  # -10 pips loss
        ]
        
        for i, (entry, exit, direction) in enumerate(trades):
            if direction == Direction.LONG:
                pnl = (exit - entry) * 100000  # 50 pips * 100k = 500$
            else:
                pnl = (entry - exit) * 100000
            
            
            trade = TradeInfo(
                trade_id=f"T{i+1:03d}",
                direction=direction,
                entry_price=entry,
                exit_price=exit,
                quantity=100000,
                entry_time=datetime.now(),
                exit_time=datetime.now(), 
                pnl=pnl
            )
            self.statistics.add_trade_record(trade)
        
        # Add balance history for returns calculation
        balances = [10000, 10500, 10700, 10400, 10300]
        for balance in balances:
            self.statistics.add_balance_snapshot(balance)
        
        # First calculate basic statistics to get profit/loss values
        self.statistics._calculate_basic_statistics(self.mock_system)
        self.statistics._calculate_performance_metrics()
        
        # Profit factor = total profit / total loss  
        # Trade PnLs: +500, +200, -300, -100
        # Total profit: 500 + 200 = 700
        # Total loss: 300 + 100 = 400
        # Profit factor: 700 / 400 = 1.75
        
        # Profit factor should be 1.75 (700/400)
        assert abs(self.statistics.performance_metrics.profit_factor - 1.75) < 0.01
        
        # Should calculate other metrics
        assert self.statistics.performance_metrics.avg_win > 0
        assert self.statistics.performance_metrics.avg_loss > 0
        assert self.statistics.performance_metrics.volatility >= 0
    
    def test_statistics_dict_generation(self):
        """Test statistics dictionary generation."""
        # Set up some data
        self.statistics.statistics.system_name = "TestStrategy"
        self.statistics.statistics.total_trades = 10
        self.statistics.statistics.winning_trades = 6
        self.statistics.statistics.winning_trade_ratio = 60.0
        self.statistics.performance_metrics.profit_factor = 1.5
        self.statistics.performance_metrics.sharpe_ratio = 1.2
        
        self.statistics._generate_statistics_dict()
        
        # Verify key statistics are present and formatted correctly
        assert self.statistics.statistics_dict["SistemName"] == "TestStrategy"
        assert self.statistics.statistics_dict["IslemSayisi"] == "10"
        assert self.statistics.statistics_dict["KazandiranIslemSayisi"] == "6"
        assert self.statistics.statistics_dict["KarliIslemOrani"] == "60.00"
        assert self.statistics.statistics_dict["ProfitFactor"] == "1.50"
        assert self.statistics.statistics_dict["SharpeRatio"] == "1.20"
    
    def test_get_statistics_summary_not_calculated(self):
        """Test getting summary when statistics not calculated."""
        summary = self.statistics.get_statistics_summary()
        
        assert "not calculated" in summary.lower()
        assert "calculate_all_statistics" in summary
    
    def test_get_statistics_summary_calculated(self):
        """Test getting summary when statistics are calculated."""
        # Set up basic data
        self.statistics.statistics.system_name = "TestStrategy"
        self.statistics.statistics.chart_symbol = "EURUSD"
        self.statistics.statistics.total_trades = 10
        self.statistics.statistics.winning_trades = 6
        self.statistics.statistics.winning_trade_ratio = 60.0
        self.statistics._is_calculated = True
        
        summary = self.statistics.get_statistics_summary()
        
        assert "Trading Statistics Summary" in summary
        assert "TestStrategy" in summary
        assert "EURUSD" in summary
        assert "Total Trades: 10" in summary
        assert "Winning Trades: 6" in summary
        assert "60.00%" in summary
    
    def _create_sample_trade(self) -> TradeInfo:
        """Create a sample trade record for testing."""
        return TradeInfo(
            trade_id="SAMPLE_001",
            direction=Direction.LONG,
            entry_price=1.1000,
            exit_price=1.1050,
            quantity=100000,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 11, 0),
            pnl=500.0,
            commission=5.0
        )


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_sharpe_ratio_empty_returns(self):
        """Test Sharpe ratio calculation with empty returns."""
        result = calculate_sharpe_ratio([])
        assert result == 0.0
    
    def test_calculate_sharpe_ratio_valid_returns(self):
        """Test Sharpe ratio calculation with valid returns."""
        # Returns with positive mean and some volatility
        returns = [0.01, -0.005, 0.02, 0.005, -0.01, 0.015, 0.008]
        result = calculate_sharpe_ratio(returns, 0.0)
        
        # Should return a reasonable positive value
        assert result > 0
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]  # No volatility
        result = calculate_sharpe_ratio(returns, 0.0)
        
        assert result == 0.0
    
    def test_calculate_sortino_ratio_empty_returns(self):
        """Test Sortino ratio calculation with empty returns."""
        result = calculate_sortino_ratio([])
        assert result == 0.0
    
    def test_calculate_sortino_ratio_valid_returns(self):
        """Test Sortino ratio calculation with valid returns."""
        returns = [0.02, -0.01, 0.015, -0.008, 0.025, -0.005]
        result = calculate_sortino_ratio(returns, 0.0)
        
        # Should return a reasonable value
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_calculate_sortino_ratio_no_negative_returns(self):
        """Test Sortino ratio with no negative returns."""
        returns = [0.01, 0.02, 0.015, 0.008, 0.025]  # All positive
        result = calculate_sortino_ratio(returns, 0.0)
        
        assert result == 0.0
    
    def test_calculate_profit_factor_empty_trades(self):
        """Test profit factor with empty trades."""
        result = calculate_profit_factor([], [])
        assert result == 0.0
        
        result = calculate_profit_factor([100, 50], [])
        assert result == 0.0
        
        result = calculate_profit_factor([], [-30, -20])
        assert result == 0.0
    
    def test_calculate_profit_factor_valid_trades(self):
        """Test profit factor with valid trades."""
        winning_trades = [100, 50, 75]  # Total: 225
        losing_trades = [-30, -20, -25]  # Total: -75 (abs = 75)
        
        result = calculate_profit_factor(winning_trades, losing_trades)
        
        # 225 / 75 = 3.0
        assert abs(result - 3.0) < 0.001
    
    def test_calculate_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        winning_trades = [100, 50]
        losing_trades = []
        
        result = calculate_profit_factor(winning_trades, losing_trades)
        assert result == 0.0  # Our implementation returns 0 for no losses
    
    def test_calculate_profit_factor_zero_losses(self):
        """Test profit factor with zero total losses."""
        winning_trades = [100, 50]
        losing_trades = [0, 0]  # Zero losses
        
        result = calculate_profit_factor(winning_trades, losing_trades)
        assert result == float('inf')


class TestRealWorldScenarios:
    """Test real-world trading scenarios."""
    
    def setup_method(self):
        """Set up realistic trading scenario."""
        self.statistics = CStatistics(1)
        self.mock_system = MockSystem()
    
    def test_profitable_strategy_scenario(self):
        """Test a profitable trading strategy scenario."""
        # Set up system
        self.statistics.set_system_info(
            self.mock_system,
            "Moving Average Crossover",
            "EURUSD", 
            "1H"
        )
        self.statistics.set_initial_balance(10000.0)
        
        # Create realistic trade sequence
        trade_data = [
            # (entry, exit, direction, profit/loss)
            (1.1000, 1.1050, Direction.LONG),    # +50 pips
            (1.1080, 1.1060, Direction.SHORT),   # +20 pips
            (1.1040, 1.1070, Direction.LONG),    # -30 pips
            (1.1100, 1.1120, Direction.LONG),    # +20 pips
            (1.1150, 1.1130, Direction.SHORT),   # +20 pips
            (1.1110, 1.1140, Direction.LONG),    # -30 pips
            (1.1160, 1.1200, Direction.LONG),    # +40 pips
            (1.1220, 1.1180, Direction.SHORT),   # +40 pips
        ]
        
        balance = 10000.0
        for i, (entry, exit, direction) in enumerate(trade_data):
            # Calculate P&L
            if direction == Direction.LONG:
                pnl = (exit - entry) * 100000  # 100k lot size
            else:
                pnl = (entry - exit) * 100000
            
            trade = TradeInfo(
                trade_id=f"REAL_{i+1:03d}",
                direction=direction,
                entry_price=entry,
                exit_price=exit,
                quantity=100000,
                entry_time=datetime(2024, 1, 1) + timedelta(hours=i),
                exit_time=datetime(2024, 1, 1) + timedelta(hours=i+1),
                pnl=pnl,
                commission=5.0
            )
            
            self.statistics.add_trade_record(trade)
            
            # Update balance
            balance += pnl - 5.0  # Subtract commission
            self.statistics.add_balance_snapshot(balance, i)
        
        # Calculate all statistics
        result = self.statistics.calculate_all_statistics(self.mock_system)
        
        # Verify results
        assert int(result["IslemSayisi"]) == 8
        assert int(result["KazandiranIslemSayisi"]) == 8  # All trades are winning
        assert int(result["KaybettirenIslemSayisi"]) == 0  # No losing trades
        assert float(result["KarliIslemOrani"]) == 100.0  # 100% win rate
        
        # Should be profitable overall
        assert float(result["GetiriFiyat"]) > 0
        assert float(result["GetiriFiyatYuzde"]) > 0
        # Profit factor is 0.0 when there are no losing trades (division by zero case)
        assert float(result["ProfitFactor"]) == 0.0
        
        # Get summary
        summary = self.statistics.get_statistics_summary()
        assert "Moving Average Crossover" in summary
        assert "EURUSD" in summary
        assert "100.00%" in summary  # Win rate
    
    def test_losing_strategy_scenario(self):
        """Test a losing trading strategy scenario."""
        self.statistics.set_initial_balance(10000.0)
        
        # Create losing trade sequence
        losing_trades = [
            (1.1000, 1.0980, Direction.LONG),    # -20 pips
            (1.0970, 1.0990, Direction.SHORT),   # -20 pips
            (1.1010, 1.0990, Direction.LONG),    # -20 pips
            (1.0980, 1.1000, Direction.SHORT),   # -20 pips
        ]
        
        balance = 10000.0
        for entry, exit, direction in losing_trades:
            if direction == Direction.LONG:
                pnl = (exit - entry) * 100000
            else:
                pnl = (entry - exit) * 100000
            
            trade = TradeInfo(
                trade_id=f"LOSE_{len(losing_trades)+1:03d}",
                direction=direction,
                entry_price=entry,
                exit_price=exit,
                quantity=100000,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                pnl=pnl
            )
            
            self.statistics.add_trade_record(trade)
            balance += pnl
            self.statistics.add_balance_snapshot(balance)
        
        result = self.statistics.calculate_all_statistics(self.mock_system)
        
        # All trades should be losing
        assert result["KazandiranIslemSayisi"] == "0"
        assert result["KaybettirenIslemSayisi"] == "4"
        assert result["KarliIslemOrani"] == "0.00"
        
        # Should show overall loss
        assert float(result["GetiriFiyat"]) < 0
        assert float(result["GetiriFiyatYuzde"]) < 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])