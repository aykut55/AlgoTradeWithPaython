"""
Backtesting engine for algorithmic trading strategies.

This module provides comprehensive backtesting functionality including
historical strategy execution, performance analysis, and result reporting.
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

from .system_wrapper import CSystemWrapper, SystemConfiguration, ExecutionMode
from ..core.base import SystemProtocol


class BacktestMode(Enum):
    """Backtest execution mode."""
    FULL_HISTORY = "FULL_HISTORY"
    DATE_RANGE = "DATE_RANGE"
    BAR_RANGE = "BAR_RANGE"
    WALK_FORWARD = "WALK_FORWARD"


@dataclass
class BacktestConfiguration:
    """Backtesting configuration parameters."""
    
    # Date range settings
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Bar range settings
    start_bar: int = 1
    end_bar: Optional[int] = None
    
    # Walk forward settings
    training_period_bars: int = 252  # 1 year of daily bars
    test_period_bars: int = 63       # 1 quarter of daily bars
    walk_forward_step: int = 21      # 1 month step
    
    # Execution settings
    mode: BacktestMode = BacktestMode.FULL_HISTORY
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0
    slippage_points: float = 0.0
    
    # Reporting settings
    generate_trade_log: bool = True
    generate_equity_curve: bool = True
    calculate_drawdown_periods: bool = True
    export_results_csv: bool = True
    
    def validate(self) -> bool:
        """Validate backtest configuration."""
        if self.mode == BacktestMode.DATE_RANGE:
            return self.start_date is not None and self.end_date is not None
        elif self.mode == BacktestMode.BAR_RANGE:
            return self.start_bar >= 0 and (self.end_bar is None or self.end_bar > self.start_bar)
        return True


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting performance metrics."""
    
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    recovery_factor: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Average trade metrics
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_winning_trade: float = 0.0
    largest_losing_trade: float = 0.0
    
    # Risk metrics
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    
    # Efficiency metrics
    calmar_ratio: float = 0.0  # Annual return / Max DD
    sterling_ratio: float = 0.0
    information_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration_days': self.max_drawdown_duration_days,
            'recovery_factor': self.recovery_factor,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'calmar_ratio': self.calmar_ratio,
            'value_at_risk_95': self.value_at_risk_95
        }


@dataclass
class BacktestResults:
    """Complete backtesting results."""
    
    # Configuration
    config: BacktestConfiguration
    start_time: datetime
    end_time: datetime
    
    # Performance metrics
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    
    # Time series data
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trade_returns: List[float] = field(default_factory=list)
    
    # Trade log
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # System statistics
    execution_time_seconds: float = 0.0
    bars_processed: int = 0
    
    @property
    def duration_seconds(self) -> float:
        """Get backtest duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'equity': self.equity_curve,
            'drawdown': self.drawdown_curve
        })


class CBacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    
    Provides:
    - Historical strategy execution
    - Performance metrics calculation
    - Risk analysis
    - Result reporting and visualization
    """
    
    def __init__(self, system_wrapper: CSystemWrapper):
        """
        Initialize backtest engine.
        
        Args:
            system_wrapper: System wrapper instance
        """
        self.system_wrapper = system_wrapper
        self.results: Optional[BacktestResults] = None
        
    def run_backtest(self, system: SystemProtocol, 
                    config: BacktestConfiguration,
                    strategy_function: Callable[[SystemProtocol, CSystemWrapper, int], None]) -> BacktestResults:
        """
        Run complete backtest.
        
        Args:
            system: System interface
            config: Backtest configuration
            strategy_function: User strategy function that sets signals
            
        Returns:
            Backtest results
        """
        if not config.validate():
            raise ValueError("Invalid backtest configuration")
        
        start_time = datetime.now()
        
        # Initialize results
        results = BacktestResults(
            config=config,
            start_time=start_time,
            end_time=start_time  # Will be updated at the end
        )
        
        # Configure system wrapper
        system_config = SystemConfiguration(
            symbol="BACKTEST",
            system_name="BacktestEngine",
            execution_mode=ExecutionMode.BACKTEST,
            calculate_statistics=True,
            print_statistics=config.generate_trade_log
        )
        self.system_wrapper.configure_system(system_config)
        
        # Determine execution range
        start_bar, end_bar = self._determine_execution_range(config)
        
        # Set up strategy callback
        def strategy_callback(bar_index: int) -> None:
            strategy_function(system, self.system_wrapper, bar_index)
        
        self.system_wrapper.on_bar_update = strategy_callback
        
        # Initialize equity tracking
        initial_equity = config.initial_capital
        current_equity = initial_equity
        equity_curve = [initial_equity]
        
        # Execute backtest
        for bar_index in range(start_bar, min(end_bar + 1, self.system_wrapper.bar_count)):
            # Execute strategy for this bar
            success = self.system_wrapper.execute_strategy_bar(system, bar_index)
            
            if success:
                # Update equity based on trading results
                if self.system_wrapper.trader:
                    trader_pnl = self.system_wrapper.trader.total_pnl
                    current_equity = initial_equity + trader_pnl
                    equity_curve.append(current_equity)
                
                results.bars_processed += 1
        
        # Finalize results
        end_time = datetime.now()
        results.end_time = end_time
        results.execution_time_seconds = (end_time - start_time).total_seconds()
        results.equity_curve = equity_curve
        
        # Calculate performance metrics
        results.metrics = self._calculate_metrics(results, config)
        results.drawdown_curve = self._calculate_drawdown_curve(equity_curve)
        
        # Extract trade information
        if self.system_wrapper.trader:
            results.trades = self._extract_trade_information()
            results.trade_returns = [trade.get('pnl', 0.0) for trade in results.trades]
        
        self.results = results
        return results
    
    def run_walk_forward_analysis(self, system: SystemProtocol,
                                 config: BacktestConfiguration,
                                 strategy_function: Callable,
                                 optimization_function: Optional[Callable] = None) -> List[BacktestResults]:
        """
        Run walk-forward analysis.
        
        Args:
            system: System interface
            config: Backtest configuration
            strategy_function: Strategy function
            optimization_function: Optional parameter optimization function
            
        Returns:
            List of backtest results for each period
        """
        results = []
        total_bars = self.system_wrapper.bar_count
        
        current_start = config.start_bar
        while current_start + config.training_period_bars + config.test_period_bars < total_bars:
            # Training period
            training_end = current_start + config.training_period_bars
            
            # Test period
            test_start = training_end
            test_end = test_start + config.test_period_bars
            
            # Create configuration for this period
            period_config = BacktestConfiguration(
                mode=BacktestMode.BAR_RANGE,
                start_bar=test_start,
                end_bar=test_end,
                initial_capital=config.initial_capital,
                commission_per_trade=config.commission_per_trade
            )
            
            # Run optimization on training data if provided
            if optimization_function:
                optimal_params = optimization_function(system, self.system_wrapper, 
                                                     current_start, training_end)
                # Apply optimal parameters to strategy
            
            # Run backtest on test period
            period_results = self.run_backtest(system, period_config, strategy_function)
            period_results.config.start_bar = current_start  # Store training start for reference
            results.append(period_results)
            
            # Move forward
            current_start += config.walk_forward_step
        
        return results
    
    def _determine_execution_range(self, config: BacktestConfiguration) -> Tuple[int, int]:
        """Determine bar range for execution."""
        total_bars = self.system_wrapper.bar_count
        
        if config.mode == BacktestMode.FULL_HISTORY:
            return 1, total_bars - 1
        elif config.mode == BacktestMode.BAR_RANGE:
            end_bar = config.end_bar if config.end_bar is not None else total_bars - 1
            return config.start_bar, min(end_bar, total_bars - 1)
        elif config.mode == BacktestMode.DATE_RANGE:
            # For date range, we'd need timestamp data
            # For now, fall back to full history
            return 1, total_bars - 1
        else:
            return 1, total_bars - 1
    
    def _calculate_metrics(self, results: BacktestResults, config: BacktestConfiguration) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = BacktestMetrics()
        
        if not results.equity_curve or len(results.equity_curve) < 2:
            return metrics
        
        equity = np.array(results.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Basic return metrics
        initial_equity = config.initial_capital
        final_equity = equity[-1]
        
        metrics.total_return = (final_equity - initial_equity) / initial_equity
        
        # Annualized return (assuming daily data)
        trading_days = len(equity) - 1
        if trading_days > 0:
            metrics.annualized_return = (final_equity / initial_equity) ** (252 / trading_days) - 1
        
        # Volatility
        if len(returns) > 1:
            metrics.volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if metrics.volatility > 0:
            metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            if downside_deviation > 0:
                metrics.sortino_ratio = metrics.annualized_return / downside_deviation
        
        # Drawdown analysis
        metrics.max_drawdown, metrics.max_drawdown_duration_days = self._calculate_drawdown_metrics(equity)
        
        # Recovery factor
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.total_return / abs(metrics.max_drawdown)
        
        # Trade metrics from trader
        if self.system_wrapper.trader:
            trader_stats = self.system_wrapper.trader.get_trading_statistics()
            metrics.total_trades = trader_stats.get('total_trades', 0)
            metrics.winning_trades = trader_stats.get('winning_trades', 0)
            metrics.losing_trades = trader_stats.get('losing_trades', 0)
            metrics.win_rate = trader_stats.get('win_rate', 0.0)
            metrics.profit_factor = trader_stats.get('profit_factor', 0.0)
            
            # Calculate average trade metrics
            if results.trade_returns:
                metrics.avg_trade_return = np.mean(results.trade_returns)
                winning_trades = [r for r in results.trade_returns if r > 0]
                losing_trades = [r for r in results.trade_returns if r < 0]
                
                if winning_trades:
                    metrics.avg_winning_trade = np.mean(winning_trades)
                    metrics.largest_winning_trade = np.max(winning_trades)
                
                if losing_trades:
                    metrics.avg_losing_trade = np.mean(losing_trades)
                    metrics.largest_losing_trade = np.min(losing_trades)
        
        # Risk metrics
        if len(returns) > 20:  # Need sufficient data
            metrics.value_at_risk_95 = np.percentile(returns, 5)
            var_threshold = metrics.value_at_risk_95
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) > 0:
                metrics.conditional_var_95 = np.mean(tail_returns)
        
        # Calmar ratio
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        return metrics
    
    def _calculate_drawdown_curve(self, equity_curve: List[float]) -> List[float]:
        """Calculate drawdown curve."""
        if not equity_curve:
            return []
        
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown.tolist()
    
    def _calculate_drawdown_metrics(self, equity: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(equity) < 2:
            return 0.0, 0
        
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        
        # Calculate maximum drawdown duration
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_drawdown, max_duration
    
    def _extract_trade_information(self) -> List[Dict[str, Any]]:
        """Extract trade information from trader."""
        trades = []
        
        if not self.system_wrapper.trader:
            return trades
        
        # Get closed trades
        for trade in self.system_wrapper.trader.closed_trades:
            trade_info = {
                'trade_id': trade.trade_id,
                'direction': trade.direction.value,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'entry_bar': trade.entry_bar,
                'exit_bar': trade.exit_bar,
                'pnl': trade.pnl,
                'commission': trade.commission,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time
            }
            trades.append(trade_info)
        
        return trades
    
    def export_results_to_csv(self, filepath: str) -> None:
        """Export backtest results to CSV file."""
        if not self.results:
            raise ValueError("No results to export")
        
        # Create results DataFrame
        equity_df = self.results.to_dataframe()
        
        # Add metrics as first row
        metrics_dict = self.results.metrics.to_dict()
        
        # Export equity curve
        equity_path = filepath.replace('.csv', '_equity.csv')
        equity_df.to_csv(equity_path, index=False)
        
        # Export metrics
        metrics_path = filepath.replace('.csv', '_metrics.csv')
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_df.to_csv(metrics_path, index=False)
        
        # Export trades if available
        if self.results.trades:
            trades_path = filepath.replace('.csv', '_trades.csv')
            trades_df = pd.DataFrame(self.results.trades)
            trades_df.to_csv(trades_path, index=False)
    
    def print_summary(self, system: SystemProtocol) -> None:
        """Print backtest summary."""
        if not self.results:
            return
        
        metrics = self.results.metrics
        
        system.mesaj("=" * 80)
        system.mesaj("BACKTEST RESULTS SUMMARY")
        system.mesaj("=" * 80)
        
        # Performance metrics
        system.mesaj(f"Total Return: {metrics.total_return:.2%}")
        system.mesaj(f"Annualized Return: {metrics.annualized_return:.2%}")
        system.mesaj(f"Volatility: {metrics.volatility:.2%}")
        system.mesaj(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        system.mesaj(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        
        # Drawdown metrics
        system.mesaj(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        system.mesaj(f"Max DD Duration: {metrics.max_drawdown_duration_days} days")
        system.mesaj(f"Recovery Factor: {metrics.recovery_factor:.2f}")
        system.mesaj(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
        
        # Trade metrics
        system.mesaj(f"Total Trades: {metrics.total_trades}")
        system.mesaj(f"Win Rate: {metrics.win_rate:.1f}%")
        system.mesaj(f"Profit Factor: {metrics.profit_factor:.2f}")
        system.mesaj(f"Avg Trade Return: {metrics.avg_trade_return:.2f}")
        
        # Execution stats
        system.mesaj(f"Bars Processed: {self.results.bars_processed}")
        system.mesaj(f"Execution Time: {self.results.execution_time_seconds:.2f} seconds")
        
        system.mesaj("=" * 80)