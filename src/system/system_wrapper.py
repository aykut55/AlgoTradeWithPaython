"""
System wrapper for algorithmic trading system orchestration.

This module contains the CSystemWrapper class which serves as the main
orchestrator for the trading system, managing all components and coordinating
strategy execution, data flow, and reporting.
"""

from typing import Optional, Dict, Any, List, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
import os
from pathlib import Path

from ..core.base import CBase, SystemProtocol
from ..utils.utils import CUtils
from ..indicators.indicator_manager import CIndicatorManager
from ..trading.trader import CTrader, RiskSettings
from ..trading.signals import Direction, SignalType, SignalInfo


class ExecutionMode(Enum):
    """Strategy execution mode enumeration."""
    SINGLE_RUN = "SINGLE_RUN"
    BACKTEST = "BACKTEST"
    OPTIMIZATION = "OPTIMIZATION"
    LIVE_TRADING = "LIVE_TRADING"


class ReportingLevel(Enum):
    """Reporting detail level enumeration."""
    MINIMAL = "MINIMAL"
    STANDARD = "STANDARD"
    DETAILED = "DETAILED"
    VERBOSE = "VERBOSE"


@dataclass
class SystemConfiguration:
    """System configuration parameters."""
    
    # Symbol and market info
    symbol: str = ""
    period: str = ""
    system_name: str = ""
    
    # Execution parameters
    execution_mode: ExecutionMode = ExecutionMode.SINGLE_RUN
    reporting_level: ReportingLevel = ReportingLevel.STANDARD
    
    # Asset configuration
    contract_count: int = 10
    asset_multiplier: int = 1
    commission_multiplier: float = 0.0
    
    # File paths
    inputs_dir: str = "data/inputs/"
    outputs_dir: str = "data/outputs/"
    params_input_filename: str = ""
    statistics_output_filename: str = ""
    
    # Execution flags
    calculate_ideal_return: bool = True
    calculate_statistics: bool = True
    print_statistics: bool = True
    print_return_statistics: bool = True
    write_statistics_to_file: bool = True
    draw_signals_on_chart: bool = True
    
    # Optimization parameters
    optimization_enabled: bool = False
    current_run_index: int = 0
    total_run_count: int = 1
    
    # Input parameters for strategy
    input_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (
            len(self.symbol) > 0 and
            len(self.system_name) > 0 and
            self.contract_count > 0 and
            self.asset_multiplier > 0
        )


@dataclass 
class StrategySignals:
    """Current strategy signals state."""
    buy: bool = False
    sell: bool = False
    flat: bool = False
    pass_signal: bool = False
    take_profit: bool = False
    stop_loss: bool = False
    
    def reset(self) -> None:
        """Reset all signals to False."""
        self.buy = False
        self.sell = False
        self.flat = False
        self.pass_signal = False
        self.take_profit = False
        self.stop_loss = False
    
    def has_signal(self) -> bool:
        """Check if any signal is active."""
        return any([self.buy, self.sell, self.flat, self.take_profit, self.stop_loss])


@dataclass
class ExecutionStatistics:
    """System execution statistics."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_bars_processed: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    execution_time_ms: float = 0.0
    
    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time_ms / 1000.0
    
    @property
    def bars_per_second(self) -> float:
        """Calculate processing speed in bars per second."""
        if self.execution_time_seconds > 0:
            return self.total_bars_processed / self.execution_time_seconds
        return 0.0


class CSystemWrapper(CBase):
    """
    Main system wrapper that orchestrates all trading system components.
    
    Manages:
    - Component initialization and integration
    - Strategy execution workflow
    - Signal processing and order management
    - Statistics calculation and reporting
    - Configuration and parameter management
    """
    
    def __init__(self, id_value: int = 0, system_name: str = "SystemWrapper"):
        """
        Initialize system wrapper.
        
        Args:
            id_value: Unique identifier
            system_name: Name of the trading system
        """
        super().__init__(id_value)
        self.system_name = system_name
        
        # Configuration
        self.config = SystemConfiguration(system_name=system_name)
        
        # Core components
        self.trader: Optional[CTrader] = None
        self.indicators: Optional[CIndicatorManager] = None
        self.utils: Optional[CUtils] = None
        
        # System state
        self.is_initialized: bool = False
        self.current_bar: int = 0
        
        # Strategy signals
        self.signals = StrategySignals()
        
        # Execution statistics
        self.stats = ExecutionStatistics()
        
        # Strategy callback functions
        self.on_bar_update: Optional[Callable[[int], None]] = None
        self.on_signal_generated: Optional[Callable[[SignalInfo], None]] = None
        self.on_trade_completed: Optional[Callable[[Any], None]] = None
        
        # Performance tracking
        self._bar_processing_times: List[float] = []
    
    def create_modules(self, system: SystemProtocol) -> 'CSystemWrapper':
        """
        Create and initialize all system modules.
        
        Args:
            system: System interface
            
        Returns:
            Self for method chaining
        """
        # Create core components
        self.trader = CTrader(id_value=1, name=f"{self.system_name}_Trader")
        self.indicators = CIndicatorManager()
        self.utils = CUtils()
        
        return self
    
    def initialize_system(self, system: SystemProtocol, v: Any = None) -> 'CSystemWrapper':
        """
        Initialize the complete trading system.
        
        Args:
            system: System interface
            v: Original data reference
            
        Returns:
            Self for method chaining
        """
        if not self.market_data.validate():
            raise ValueError("Invalid market data for system initialization")
        
        # Set system configuration from system interface if available
        if hasattr(system, 'Sembol'):
            self.config.symbol = system.Sembol
        if hasattr(system, 'Periyot'):
            self.config.period = system.Periyot
        if hasattr(system, 'Name'):
            self.config.system_name = system.Name
        
        # Create modules if not already created
        if not all([self.trader, self.indicators, self.utils]):
            self.create_modules(system)
        
        # Initialize components with market data
        self.utils.initialize(system)
        
        self.indicators.initialize(
            system, v,
            self.open, self.high, self.low, self.close,
            self.volume, self.lot
        )
        
        self.trader.set_data_from_dataframe(self.get_ohlcv_dataframe())
        self.trader.initialize(system, v)
        
        # Initialize execution statistics
        self.stats.start_time = datetime.now()
        self.stats.total_bars_processed = 0
        self.stats.signals_generated = 0
        
        self.is_initialized = True
        self.show_message(system, f"System '{self.system_name}' initialized successfully")
        
        return self
    
    def configure_system(self, config: SystemConfiguration) -> 'CSystemWrapper':
        """
        Configure system parameters.
        
        Args:
            config: System configuration
            
        Returns:
            Self for method chaining
        """
        if not config.validate():
            raise ValueError("Invalid system configuration")
        
        self.config = config
        
        # Apply configuration to components if they exist
        if self.trader:
            # Configure risk settings
            risk_settings = RiskSettings(
                max_position_size=float(config.contract_count),
                risk_per_trade=0.02  # Default 2% risk
            )
            self.trader.set_risk_settings(risk_settings)
        
        return self
    
    def reset_system(self, system: SystemProtocol) -> 'CSystemWrapper':
        """
        Reset system state for new execution.
        
        Args:
            system: System interface
            
        Returns:
            Self for method chaining
        """
        # Reset signals
        self.signals.reset()
        
        # Reset components if they exist
        if self.trader:
            self.trader.reset_daily_stats()
        
        # Reset execution state
        self.current_bar = 0
        self.stats = ExecutionStatistics()
        self.stats.start_time = datetime.now()
        
        # Clear performance tracking
        self._bar_processing_times.clear()
        
        self.show_message(system, "System reset completed")
        return self
    
    def set_strategy_signals(self, system: SystemProtocol, bar_index: int,
                           buy: bool = False, sell: bool = False, 
                           flat: bool = False, pass_signal: bool = False,
                           take_profit: bool = False, stop_loss: bool = False) -> None:
        """
        Set strategy signals for the current bar.
        
        Args:
            system: System interface
            bar_index: Current bar index
            buy: Buy signal
            sell: Sell signal
            flat: Flatten position signal
            pass_signal: Pass signal
            take_profit: Take profit signal
            stop_loss: Stop loss signal
        """
        self.signals.buy = buy
        self.signals.sell = sell
        self.signals.flat = flat
        self.signals.pass_signal = pass_signal
        self.signals.take_profit = take_profit
        self.signals.stop_loss = stop_loss
        
        if self.signals.has_signal():
            self.stats.signals_generated += 1
    
    def execute_strategy_bar(self, system: SystemProtocol, bar_index: int) -> bool:
        """
        Execute strategy for a single bar.
        
        Args:
            system: System interface
            bar_index: Bar index to process
            
        Returns:
            True if bar processed successfully, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        if bar_index < 0 or bar_index >= self.bar_count:
            return False
        
        bar_start_time = datetime.now()
        self.current_bar = bar_index
        
        try:
            # Pre-processing: Update trader state
            if self.trader:
                self.trader.update_bar(system, bar_index)
            
            # Reset signals for this bar
            self.signals.reset()
            
            # Call user strategy callback if provided
            if self.on_bar_update:
                self.on_bar_update(bar_index)
            
            # Process strategy signals
            self._process_strategy_signals(system, bar_index)
            
            # Update statistics
            self.stats.total_bars_processed += 1
            
            # Track processing time
            processing_time = (datetime.now() - bar_start_time).total_seconds() * 1000
            self._bar_processing_times.append(processing_time)
            
            return True
            
        except Exception as e:
            self.show_message(system, f"Error processing bar {bar_index}: {str(e)}")
            return False
    
    def execute_complete_strategy(self, system: SystemProtocol, 
                                start_bar: int = 1, end_bar: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute strategy for a range of bars.
        
        Args:
            system: System interface
            start_bar: Starting bar index
            end_bar: Ending bar index (None for all bars)
            
        Returns:
            Execution results dictionary
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        if end_bar is None:
            end_bar = self.bar_count - 1
        
        self.show_message(system, f"Starting strategy execution: bars {start_bar} to {end_bar}")
        
        # Start timing
        execution_start = datetime.now()
        
        # Execute strategy for each bar
        successful_bars = 0
        for bar_index in range(start_bar, min(end_bar + 1, self.bar_count)):
            if self.execute_strategy_bar(system, bar_index):
                successful_bars += 1
            else:
                self.show_message(system, f"Failed to process bar {bar_index}")
        
        # Finalize execution
        execution_end = datetime.now()
        self.stats.end_time = execution_end
        self.stats.execution_time_ms = (execution_end - execution_start).total_seconds() * 1000
        
        # Calculate final statistics
        results = self._calculate_execution_results(system)
        
        self.show_message(system, f"Strategy execution completed: {successful_bars}/{end_bar - start_bar + 1} bars processed")
        
        return results
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        if not self.trader:
            return {}
        
        trader_stats = self.trader.get_trading_statistics()
        system_stats = {
            "execution_time_ms": self.stats.execution_time_ms,
            "execution_time_seconds": self.stats.execution_time_seconds,
            "bars_processed": self.stats.total_bars_processed,
            "bars_per_second": self.stats.bars_per_second,
            "signals_generated": self.stats.signals_generated,
            "avg_bar_processing_time_ms": np.mean(self._bar_processing_times) if self._bar_processing_times else 0.0
        }
        
        # Combine statistics
        combined_stats = {**trader_stats, **system_stats}
        return combined_stats
    
    def _process_strategy_signals(self, system: SystemProtocol, bar_index: int) -> None:
        """Process the current strategy signals through the trader."""
        if not self.trader:
            return
        
        current_price = self.close[bar_index]
        
        # Process signals in priority order
        signal_processed = False
        
        # Take profit and stop loss have highest priority
        if self.signals.take_profit:
            signal_info = self.trader.close_position(system, bar_index, "TakeProfit", current_price)
            if signal_info:
                signal_processed = True
                if self.on_signal_generated:
                    self.on_signal_generated(signal_info)
        
        elif self.signals.stop_loss:
            signal_info = self.trader.close_position(system, bar_index, "StopLoss", current_price)
            if signal_info:
                signal_processed = True
                if self.on_signal_generated:
                    self.on_signal_generated(signal_info)
        
        # Flatten position
        elif self.signals.flat:
            signal_info = self.trader.close_position(system, bar_index, "Manual", current_price)
            if signal_info:
                signal_processed = True
                if self.on_signal_generated:
                    self.on_signal_generated(signal_info)
        
        # Entry signals
        elif self.signals.buy:
            signal_info = self.trader.generate_buy_signal(system, bar_index, current_price)
            if signal_info:
                signal_processed = True
                self.stats.trades_executed += 1
                if self.on_signal_generated:
                    self.on_signal_generated(signal_info)
        
        elif self.signals.sell:
            signal_info = self.trader.generate_sell_signal(system, bar_index, current_price)
            if signal_info:
                signal_processed = True
                self.stats.trades_executed += 1
                if self.on_signal_generated:
                    self.on_signal_generated(signal_info)
    
    def _calculate_execution_results(self, system: SystemProtocol) -> Dict[str, Any]:
        """Calculate final execution results."""
        results = {
            "success": True,
            "bars_processed": self.stats.total_bars_processed,
            "execution_time_ms": self.stats.execution_time_ms,
            "performance": {
                "bars_per_second": self.stats.bars_per_second,
                "avg_bar_time_ms": np.mean(self._bar_processing_times) if self._bar_processing_times else 0.0,
                "max_bar_time_ms": np.max(self._bar_processing_times) if self._bar_processing_times else 0.0
            },
            "trading_results": self.get_trading_statistics() if self.trader else {}
        }
        
        return results
    
    def print_execution_summary(self, system: SystemProtocol) -> None:
        """Print execution summary to system."""
        if not self.config.print_statistics:
            return
        
        stats = self.get_trading_statistics()
        
        self.show_message(system, "=" * 60)
        self.show_message(system, f"EXECUTION SUMMARY - {self.system_name}")
        self.show_message(system, "=" * 60)
        
        # Performance metrics
        self.show_message(system, f"Bars Processed: {self.stats.total_bars_processed}")
        self.show_message(system, f"Execution Time: {self.stats.execution_time_seconds:.2f} seconds")
        self.show_message(system, f"Processing Speed: {self.stats.bars_per_second:.0f} bars/sec")
        
        # Trading metrics
        if 'total_trades' in stats:
            self.show_message(system, f"Total Trades: {stats['total_trades']}")
            self.show_message(system, f"Win Rate: {stats.get('win_rate', 0):.1f}%")
            self.show_message(system, f"Total P&L: {stats.get('total_pnl', 0):.2f}")
        
        self.show_message(system, "=" * 60)
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a system component by name."""
        components = {
            'trader': self.trader,
            'indicators': self.indicators,
            'utils': self.utils
        }
        return components.get(component_name.lower())
    
    def __repr__(self) -> str:
        """String representation of system wrapper."""
        status = "initialized" if self.is_initialized else "not initialized"
        return f"CSystemWrapper(name='{self.system_name}', status={status}, bars={self.bar_count})"