"""
Advanced trading statistics and performance analysis module.

This module implements comprehensive trading performance metrics including:
- Profit/Loss analysis with risk-adjusted returns
- Drawdown analysis and recovery metrics
- Win/Loss ratios and trade quality assessment
- Advanced performance metrics (Sharpe, Sortino, Profit Factor)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from ..core.base import CBase, SystemProtocol
from ..trading.trader import TradeInfo
from ..trading.signals import Direction


@dataclass
class TradingStatistics:
    """Core trading statistics data structure."""
    
    # System Information
    system_id: int = 0
    system_name: str = ""
    chart_symbol: str = ""
    chart_period: str = ""
    last_bar_index: int = 0
    
    # Time Analysis
    total_elapsed_months: float = 0.0
    total_elapsed_days: float = 0.0
    total_elapsed_hours: float = 0.0
    total_elapsed_minutes: float = 0.0
    total_bars: int = 0
    
    # Bar Information
    first_bar_date: Optional[datetime] = None
    last_bar_date: Optional[datetime] = None
    first_bar_index: int = 0
    last_bar_index: int = 0
    
    # Balance Tracking
    initial_balance_price: float = 0.0
    initial_balance_points: float = 0.0
    current_balance_price: float = 0.0
    current_balance_points: float = 0.0
    
    # Returns Analysis
    return_price: float = 0.0
    return_points: float = 0.0
    return_price_percent: float = 0.0
    return_points_percent: float = 0.0
    
    # Net Returns (after commission)
    net_balance_price: float = 0.0
    net_balance_points: float = 0.0
    net_return_price: float = 0.0
    net_return_points: float = 0.0
    net_return_price_percent: float = 0.0
    net_return_points_percent: float = 0.0
    
    # Min/Max Balance
    min_balance_price: float = 0.0
    max_balance_price: float = 0.0
    min_balance_points: float = 0.0
    max_balance_points: float = 0.0
    min_balance_price_index: int = 0
    max_balance_price_index: int = 0
    min_balance_points_index: int = 0
    max_balance_points_index: int = 0
    
    # Net Min/Max Balance
    min_net_balance_price: float = 0.0
    max_net_balance_price: float = 0.0
    min_net_balance_price_index: int = 0
    max_net_balance_price_index: int = 0
    
    # Trade Counts
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    flat_count: int = 0
    pass_count: int = 0
    
    # Win/Loss Analysis
    winning_trades: int = 0
    losing_trades: int = 0
    neutral_trades: int = 0
    winning_buys: int = 0
    losing_buys: int = 0
    neutral_buys: int = 0
    winning_sells: int = 0
    losing_sells: int = 0
    neutral_sells: int = 0
    
    # Command Counts
    buy_commands: int = 0
    sell_commands: int = 0
    pass_commands: int = 0
    take_profit_commands: int = 0
    stop_loss_commands: int = 0
    flat_commands: int = 0
    
    # Commission Analysis
    commission_trades: int = 0
    commission_asset_count: int = 1
    commission_multiplier: float = 0.0
    commission_price: float = 0.0
    commission_percent: float = 0.0
    include_commission: bool = False
    
    # P&L Analysis
    total_pnl_price: float = 0.0
    total_pnl_points: float = 0.0
    total_pnl_price_percent: float = 0.0
    net_profit_price: float = 0.0
    total_profit_price: float = 0.0
    total_loss_price: float = 0.0
    net_profit_points: float = 0.0
    total_profit_points: float = 0.0
    total_loss_points: float = 0.0
    max_profit_price: float = 0.0
    max_loss_price: float = 0.0
    max_profit_points: float = 0.0
    max_loss_points: float = 0.0
    max_profit_price_index: int = 0
    max_loss_price_index: int = 0
    max_profit_points_index: int = 0
    max_loss_points_index: int = 0
    
    # Performance Ratios
    winning_trade_ratio: float = 0.0
    bars_in_profit: int = 0
    bars_in_loss: int = 0
    
    # Average Trade Frequency
    avg_monthly_trades: float = 0.0
    avg_weekly_trades: float = 0.0
    avg_daily_trades: float = 0.0
    avg_hourly_trades: float = 0.0


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis metrics."""
    
    max_drawdown: float = 0.0
    max_drawdown_date: Optional[datetime] = None
    max_drawdown_index: int = 0
    max_loss: float = 0.0
    current_drawdown: float = 0.0
    drawdown_duration: int = 0
    recovery_factor: float = 0.0
    underwater_periods: List[Tuple[int, int, float]] = field(default_factory=list)
    
    # Drawdown in different metrics
    max_drawdown_percent: float = 0.0
    max_drawdown_points: float = 0.0


@dataclass
class PerformanceMetrics:
    """Advanced performance metrics."""
    
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    
    # Trade quality metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Expectancy
    expectancy: float = 0.0
    expectancy_percent: float = 0.0


class CStatistics(CBase):
    """
    Comprehensive trading statistics calculator.
    
    Provides detailed analysis of trading performance including:
    - Basic trade statistics (wins, losses, counts)
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis and recovery metrics
    - Trade quality and expectancy calculations
    """
    
    def __init__(self, system_id: int = 0):
        super().__init__(system_id)
        self.statistics = TradingStatistics()
        self.drawdown_analysis = DrawdownAnalysis()
        self.performance_metrics = PerformanceMetrics()
        
        # Historical data for calculations
        self.balance_history: List[float] = []
        self.pnl_history: List[float] = []
        self.returns_history: List[float] = []
        self.trade_records: List[TradeInfo] = []
        
        # Statistics dictionary for compatibility
        self.statistics_dict: Dict[str, Any] = {}
        
        # Calculation flags
        self._is_calculated: bool = False
        self._last_calculation_time: Optional[datetime] = None
    
    def reset_statistics(self) -> None:
        """Reset all statistics to initial values."""
        self.statistics = TradingStatistics()
        self.drawdown_analysis = DrawdownAnalysis()
        self.performance_metrics = PerformanceMetrics()
        self.balance_history.clear()
        self.pnl_history.clear()
        self.returns_history.clear()
        self.trade_records.clear()
        self.statistics_dict.clear()
        self._is_calculated = False
        self._last_calculation_time = None
    
    def set_system_info(self, system: SystemProtocol, 
                       system_name: str = "",
                       chart_symbol: str = "",
                       chart_period: str = "") -> 'CStatistics':
        """Set system information for statistics."""
        self.statistics.system_id = system.id if hasattr(system, 'id') else 0
        self.statistics.system_name = system_name
        self.statistics.chart_symbol = chart_symbol
        self.statistics.chart_period = chart_period
        return self
    
    def set_initial_balance(self, price_balance: float, 
                           points_balance: Optional[float] = None) -> 'CStatistics':
        """Set initial balance for return calculations."""
        self.statistics.initial_balance_price = price_balance
        self.statistics.initial_balance_points = points_balance or price_balance
        return self
    
    def add_trade_record(self, trade: TradeInfo) -> None:
        """Add a trade record for analysis."""
        self.trade_records.append(trade)
        self._is_calculated = False
    
    def add_balance_snapshot(self, balance: float, bar_index: int = 0) -> None:
        """Add balance snapshot for tracking."""
        self.balance_history.append(balance)
        
        # Update current balance
        self.statistics.current_balance_price = balance
        self.statistics.last_bar_index = bar_index
        self._is_calculated = False
    
    def add_pnl_snapshot(self, pnl: float) -> None:
        """Add P&L snapshot for analysis."""
        self.pnl_history.append(pnl)
        self._is_calculated = False
    
    def calculate_all_statistics(self, system: SystemProtocol) -> Dict[str, Any]:
        """
        Calculate comprehensive trading statistics.
        
        Args:
            system: Trading system protocol instance
            
        Returns:
            Dictionary of all calculated statistics
        """
        # Calculate basic statistics
        self._calculate_basic_statistics(system)
        
        # Calculate drawdown analysis
        self._calculate_drawdown_analysis()
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Generate statistics dictionary
        self._generate_statistics_dict()
        
        self._is_calculated = True
        self._last_calculation_time = datetime.now()
        
        return self.statistics_dict
    
    def _calculate_basic_statistics(self, system: SystemProtocol) -> None:
        """Calculate basic trading statistics."""
        if not self.trade_records:
            return
        
        # Time analysis
        if hasattr(system, 'market_data') and len(system.market_data.close) > 0:
            start_time = getattr(system.market_data, 'dates', [datetime.now()])[0]
            end_time = getattr(system.market_data, 'dates', [datetime.now()])[-1]
            time_diff = end_time - start_time
            
            self.statistics.total_elapsed_days = time_diff.total_seconds() / 86400
            self.statistics.total_elapsed_hours = time_diff.total_seconds() / 3600
            self.statistics.total_elapsed_minutes = time_diff.total_seconds() / 60
            self.statistics.total_elapsed_months = self.statistics.total_elapsed_days / 30.4
            
            self.statistics.first_bar_date = start_time
            self.statistics.last_bar_date = end_time
            self.statistics.total_bars = len(system.market_data.close)
        
        # Trade counts
        self.statistics.total_trades = len(self.trade_records)
        
        winning_trades = []
        losing_trades = []
        neutral_trades = []
        
        for trade in self.trade_records:
            if not trade.is_closed:
                continue
                
            pnl = trade.pnl  # Use stored PnL
            
            if pnl > 0:
                winning_trades.append(trade)
            elif pnl < 0:
                losing_trades.append(trade)
            else:
                neutral_trades.append(trade)
            
            # Count by direction
            if trade.direction == Direction.LONG:
                self.statistics.buy_count += 1
                if pnl > 0:
                    self.statistics.winning_buys += 1
                elif pnl < 0:
                    self.statistics.losing_buys += 1
                else:
                    self.statistics.neutral_buys += 1
            else:
                self.statistics.sell_count += 1
                if pnl > 0:
                    self.statistics.winning_sells += 1
                elif pnl < 0:
                    self.statistics.losing_sells += 1
                else:
                    self.statistics.neutral_sells += 1
        
        self.statistics.winning_trades = len(winning_trades)
        self.statistics.losing_trades = len(losing_trades)
        self.statistics.neutral_trades = len(neutral_trades)
        
        # Win ratio
        if self.statistics.total_trades > 0:
            self.statistics.winning_trade_ratio = self.statistics.winning_trades / self.statistics.total_trades * 100
        
        # P&L Analysis
        if self.trade_records:
            pnls = [trade.pnl for trade in self.trade_records if trade.is_closed]
            
            self.statistics.total_pnl_price = sum(pnls)
            self.statistics.total_profit_price = sum(p for p in pnls if p > 0)
            self.statistics.total_loss_price = abs(sum(p for p in pnls if p < 0))
            self.statistics.net_profit_price = self.statistics.total_profit_price - self.statistics.total_loss_price
            
            if pnls:
                self.statistics.max_profit_price = max(pnls)
                self.statistics.max_loss_price = abs(min(pnls))
        
        # Balance analysis
        if self.balance_history:
            self.statistics.min_balance_price = min(self.balance_history)
            self.statistics.max_balance_price = max(self.balance_history)
            self.statistics.min_balance_price_index = self.balance_history.index(self.statistics.min_balance_price)
            self.statistics.max_balance_price_index = self.balance_history.index(self.statistics.max_balance_price)
        
        # Return calculations
        if self.statistics.initial_balance_price > 0:
            self.statistics.return_price = self.statistics.current_balance_price - self.statistics.initial_balance_price
            self.statistics.return_price_percent = (self.statistics.return_price / self.statistics.initial_balance_price) * 100
        
        # Average trade frequency
        if self.statistics.total_elapsed_days > 0:
            self.statistics.avg_daily_trades = self.statistics.total_trades / self.statistics.total_elapsed_days
            self.statistics.avg_monthly_trades = self.statistics.total_trades / max(1, self.statistics.total_elapsed_months)
            self.statistics.avg_weekly_trades = self.statistics.avg_daily_trades * 7
            if self.statistics.total_elapsed_hours > 0:
                self.statistics.avg_hourly_trades = self.statistics.total_trades / self.statistics.total_elapsed_hours
    
    def _calculate_drawdown_analysis(self) -> None:
        """Calculate drawdown metrics."""
        if not self.balance_history:
            return
        
        balance_array = np.array(self.balance_history)
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(balance_array)
        
        # Calculate drawdown
        drawdown = balance_array - running_max
        drawdown_percent = (drawdown / running_max) * 100
        
        # Maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        self.drawdown_analysis.max_drawdown = abs(drawdown[max_dd_idx])
        self.drawdown_analysis.max_drawdown_percent = abs(drawdown_percent[max_dd_idx])
        self.drawdown_analysis.max_drawdown_index = max_dd_idx
        
        # Current drawdown
        self.drawdown_analysis.current_drawdown = abs(drawdown[-1])
        
        # Recovery factor
        if self.drawdown_analysis.max_drawdown > 0:
            total_return = self.statistics.return_price
            self.drawdown_analysis.recovery_factor = total_return / self.drawdown_analysis.max_drawdown
        
        # Find underwater periods
        underwater_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                max_dd_in_period = abs(min(drawdown[start_idx:i+1]))
                underwater_periods.append((start_idx, i, max_dd_in_period))
        
        # Handle case where we end in drawdown
        if in_drawdown:
            max_dd_in_period = abs(min(drawdown[start_idx:]))
            underwater_periods.append((start_idx, len(drawdown)-1, max_dd_in_period))
        
        self.drawdown_analysis.underwater_periods = underwater_periods
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate advanced performance metrics."""
        if not self.trade_records:
            return
        
        # Calculate returns for risk metrics
        if len(self.balance_history) > 1:
            returns = np.diff(self.balance_history) / np.array(self.balance_history[:-1])
            self.returns_history = returns.tolist()
        
        # Profit factor
        if self.statistics.total_loss_price > 0:
            self.performance_metrics.profit_factor = self.statistics.total_profit_price / self.statistics.total_loss_price
        
        # Risk metrics
        if self.returns_history:
            returns_array = np.array(self.returns_history)
            
            # Volatility
            self.performance_metrics.volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio (assuming risk-free rate = 0)
            if self.performance_metrics.volatility > 0:
                avg_return = np.mean(returns_array) * 252  # Annualized
                self.performance_metrics.sharpe_ratio = avg_return / self.performance_metrics.volatility
            
            # Sortino ratio
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                self.performance_metrics.downside_deviation = downside_deviation
                if downside_deviation > 0:
                    avg_return = np.mean(returns_array) * 252
                    self.performance_metrics.sortino_ratio = avg_return / downside_deviation
            
            # Calmar ratio
            if self.drawdown_analysis.max_drawdown > 0:
                annual_return = np.mean(returns_array) * 252
                max_dd_percent = self.drawdown_analysis.max_drawdown_percent / 100
                if max_dd_percent > 0:
                    self.performance_metrics.calmar_ratio = annual_return / max_dd_percent
            
            # VaR and CVaR (95% confidence)
            self.performance_metrics.var_95 = np.percentile(returns_array, 5) * 100
            var_threshold = np.percentile(returns_array, 5)
            tail_returns = returns_array[returns_array <= var_threshold]
            if len(tail_returns) > 0:
                self.performance_metrics.cvar_95 = np.mean(tail_returns) * 100
        
        # Trade quality metrics
        if self.trade_records:
            pnls = [trade.pnl for trade in self.trade_records if trade.is_closed]
            
            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            if winning_pnls:
                self.performance_metrics.avg_win = np.mean(winning_pnls)
                self.performance_metrics.largest_win = max(winning_pnls)
            
            if losing_pnls:
                self.performance_metrics.avg_loss = abs(np.mean(losing_pnls))
                self.performance_metrics.largest_loss = abs(min(losing_pnls))
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for pnl in pnls:
                if pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                elif pnl < 0:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_wins = 0
                    consecutive_losses = 0
            
            self.performance_metrics.max_consecutive_wins = max_consecutive_wins
            self.performance_metrics.max_consecutive_losses = max_consecutive_losses
            
            # Expectancy
            if self.statistics.total_trades > 0:
                win_rate = self.statistics.winning_trades / self.statistics.total_trades
                loss_rate = self.statistics.losing_trades / self.statistics.total_trades
                
                if self.performance_metrics.avg_win > 0 and self.performance_metrics.avg_loss > 0:
                    self.performance_metrics.expectancy = (win_rate * self.performance_metrics.avg_win) - (loss_rate * self.performance_metrics.avg_loss)
                    
                    # Expectancy as percentage
                    avg_trade_size = np.mean([abs(p) for p in pnls]) if pnls else 1
                    if avg_trade_size > 0:
                        self.performance_metrics.expectancy_percent = (self.performance_metrics.expectancy / avg_trade_size) * 100
    
    def _generate_statistics_dict(self) -> None:
        """Generate statistics dictionary for compatibility."""
        self.statistics_dict.clear()
        
        # System info
        self.statistics_dict["SistemId"] = str(self.statistics.system_id)
        self.statistics_dict["SistemName"] = self.statistics.system_name
        self.statistics_dict["GrafikSembol"] = self.statistics.chart_symbol
        self.statistics_dict["GrafikPeriyot"] = self.statistics.chart_period
        
        # Time analysis
        self.statistics_dict["ToplamGecenSureAy"] = f"{self.statistics.total_elapsed_months:.2f}"
        self.statistics_dict["ToplamGecenSureGun"] = f"{self.statistics.total_elapsed_days:.0f}"
        self.statistics_dict["ToplamGecenSureSaat"] = f"{self.statistics.total_elapsed_hours:.0f}"
        self.statistics_dict["ToplamGecenSureDakika"] = f"{self.statistics.total_elapsed_minutes:.0f}"
        self.statistics_dict["ToplamBarSayisi"] = str(self.statistics.total_bars)
        
        # Balance and returns
        self.statistics_dict["IlkBakiyeFiyat"] = f"{self.statistics.initial_balance_price:.2f}"
        self.statistics_dict["BakiyeFiyat"] = f"{self.statistics.current_balance_price:.2f}"
        self.statistics_dict["GetiriFiyat"] = f"{self.statistics.return_price:.2f}"
        self.statistics_dict["GetiriFiyatYuzde"] = f"{self.statistics.return_price_percent:.2f}"
        self.statistics_dict["MinBakiyeFiyat"] = f"{self.statistics.min_balance_price:.2f}"
        self.statistics_dict["MaxBakiyeFiyat"] = f"{self.statistics.max_balance_price:.2f}"
        
        # Trade statistics
        self.statistics_dict["IslemSayisi"] = str(self.statistics.total_trades)
        self.statistics_dict["AlisSayisi"] = str(self.statistics.buy_count)
        self.statistics_dict["SatisSayisi"] = str(self.statistics.sell_count)
        self.statistics_dict["KazandiranIslemSayisi"] = str(self.statistics.winning_trades)
        self.statistics_dict["KaybettirenIslemSayisi"] = str(self.statistics.losing_trades)
        self.statistics_dict["NotrIslemSayisi"] = str(self.statistics.neutral_trades)
        self.statistics_dict["KarliIslemOrani"] = f"{self.statistics.winning_trade_ratio:.2f}"
        
        # P&L analysis
        self.statistics_dict["ToplamKarFiyat"] = f"{self.statistics.total_profit_price:.2f}"
        self.statistics_dict["ToplamZararFiyat"] = f"{self.statistics.total_loss_price:.2f}"
        self.statistics_dict["NetKarFiyat"] = f"{self.statistics.net_profit_price:.2f}"
        self.statistics_dict["MaxKarFiyat"] = f"{self.statistics.max_profit_price:.2f}"
        self.statistics_dict["MaxZararFiyat"] = f"{self.statistics.max_loss_price:.2f}"
        
        # Drawdown
        self.statistics_dict["GetiriMaxDD"] = f"{self.drawdown_analysis.max_drawdown:.2f}"
        self.statistics_dict["GetiriMaxDDYuzde"] = f"{self.drawdown_analysis.max_drawdown_percent:.2f}"
        if self.drawdown_analysis.max_drawdown_date:
            self.statistics_dict["GetiriMaxDDTarih"] = self.drawdown_analysis.max_drawdown_date.strftime("%d.%m.%Y")
        
        # Performance metrics
        self.statistics_dict["ProfitFactor"] = f"{self.performance_metrics.profit_factor:.2f}"
        self.statistics_dict["SharpeRatio"] = f"{self.performance_metrics.sharpe_ratio:.2f}"
        self.statistics_dict["SortinoRatio"] = f"{self.performance_metrics.sortino_ratio:.2f}"
        self.statistics_dict["CalmarRatio"] = f"{self.performance_metrics.calmar_ratio:.2f}"
        self.statistics_dict["Volatilite"] = f"{self.performance_metrics.volatility:.2f}"
        self.statistics_dict["Beklenti"] = f"{self.performance_metrics.expectancy:.2f}"
        self.statistics_dict["BeklentiYuzde"] = f"{self.performance_metrics.expectancy_percent:.2f}"
        
        # Trade frequency
        self.statistics_dict["OrtAylikIslemSayisi"] = f"{self.statistics.avg_monthly_trades:.2f}"
        self.statistics_dict["OrtGunlukIslemSayisi"] = f"{self.statistics.avg_daily_trades:.2f}"
        self.statistics_dict["OrtSaatlikIslemSayisi"] = f"{self.statistics.avg_hourly_trades:.2f}"
    
    def get_statistics_summary(self) -> str:
        """Get formatted statistics summary."""
        if not self._is_calculated:
            return "Statistics not calculated. Call calculate_all_statistics() first."
        
        summary = []
        summary.append("=== Trading Statistics Summary ===")
        summary.append(f"System: {self.statistics.system_name} ({self.statistics.chart_symbol})")
        summary.append(f"Period: {self.statistics.chart_period}")
        summary.append("")
        
        summary.append("--- Basic Statistics ---")
        summary.append(f"Total Trades: {self.statistics.total_trades}")
        summary.append(f"Winning Trades: {self.statistics.winning_trades}")
        summary.append(f"Losing Trades: {self.statistics.losing_trades}")
        summary.append(f"Win Rate: {self.statistics.winning_trade_ratio:.2f}%")
        summary.append("")
        
        summary.append("--- Performance ---")
        summary.append(f"Total Return: {self.statistics.return_price:.2f} ({self.statistics.return_price_percent:.2f}%)")
        summary.append(f"Profit Factor: {self.performance_metrics.profit_factor:.2f}")
        summary.append(f"Sharpe Ratio: {self.performance_metrics.sharpe_ratio:.2f}")
        summary.append(f"Max Drawdown: {self.drawdown_analysis.max_drawdown:.2f} ({self.drawdown_analysis.max_drawdown_percent:.2f}%)")
        summary.append("")
        
        summary.append("--- Risk Metrics ---")
        summary.append(f"Volatility: {self.performance_metrics.volatility:.2f}")
        summary.append(f"Sortino Ratio: {self.performance_metrics.sortino_ratio:.2f}")
        summary.append(f"Calmar Ratio: {self.performance_metrics.calmar_ratio:.2f}")
        summary.append(f"VaR (95%): {self.performance_metrics.var_95:.2f}%")
        summary.append("")
        
        return "\n".join(summary)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns series."""
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio from returns series."""
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0 or np.std(negative_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)


def calculate_profit_factor(winning_trades: List[float], losing_trades: List[float]) -> float:
    """Calculate profit factor from winning and losing trades."""
    if not winning_trades or not losing_trades:
        return 0.0
    
    total_profit = sum(winning_trades)
    total_loss = abs(sum(losing_trades))
    
    if total_loss == 0:
        return float('inf') if total_profit > 0 else 0.0
    
    return total_profit / total_loss