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


class SystemWrapper(CBase):
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
        
        # Core components - initialize immediately for compatibility
        self.trader: Optional[CTrader] = CTrader(id_value=1, name=f"{system_name}_Trader")
        self.indicators: Optional[CIndicatorManager] = CIndicatorManager()
        self.utils: Optional[CUtils] = CUtils()
        
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
    
    def create_modules(self, system: SystemProtocol) -> 'SystemWrapper':
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
    
    # ===========================================================================================
    # COMPATIBILITY METHODS FROM MAIN.PY (for backward compatibility)
    # ===========================================================================================
    
    def __init_compat__(self):
        """Initialize compatibility attributes from main.py SystemWrapper."""
        # Add compatibility attributes
        self.results = []
        self._timing_report = {}
        self.InputParamsCount = 10
        self.InputParams = [""] * self.InputParamsCount
        
        # Add main.py specific attributes
        self.myVarlik = None
        self.myTrader = self.trader
        self.myUtils = None
        self.myTimeUtils = None
        self.myBarUtils = None
        self.myFileUtils = None
        self.myExcelUtils = None
        self.mySharedMemory = None
        self.myConfig = None
        self.myIndicators = None
        self.myDataManager = None
        
        # Trading parameters
        self.HisseSayisi = 0
        self.KontratSayisi = 10
        self.KomisyonCarpan = 0.0
        self.VarlikAdedCarpani = 1
        
        # File paths
        self.InputsDir = "Aykut/Exports/"
        self.OutputsDir = "Aykut/Exports/"
        self.ParamsInputFileName = ""
        self.IstatistiklerOutputFileName = ""
        self.IstatistiklerOptOutputFileName = ""
        
        # System flags
        self.bUseParamsFromInputFile = False
        self.bOptEnabled = False
        self.bIdealGetiriHesapla = True
        self.bIstatistikleriHesapla = True
        self.bIstatistikleriEkranaYaz = True
        self.bGetiriIstatistikleriEkranaYaz = True
        self.bIstatistikleriDosyayaYaz = True
        self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = False
        self.bOptimizasyonIstatistikleriniDosyayaYaz = False
        self.bSinyalleriEkranaCiz = True
        
        # Run control
        self.CurrentRunIndex = 0
        self.TotalRunCount = 1
        
        # Strategy parameters
        self.Parametreler = ["SMA", 20, 50]
        
        # Market data references
        self.V = None
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.Lot = None
    
    def CreateModules(self, dataManager=None, lib=None):
        """Create system modules - Python equivalent of C# CreateModules"""
        def _impl():
            # Initialize compatibility attributes if not done
            if not hasattr(self, 'results'):
                self.__init_compat__()
            
            # Initialize modules similar to C# version
            self.myVarlik = None  # Asset management (placeholder)
            self.myTrader = self.trader  # Already created in __init__
            self.myUtils = self.utils  # Use existing utils
            self.myTimeUtils = None  # Time utilities (placeholder)
            self.myBarUtils = None  # Bar utilities (placeholder)
            self.myFileUtils = None  # File utilities (placeholder)
            self.myExcelUtils = None  # Excel utilities (placeholder)
            self.mySharedMemory = None  # Shared memory (placeholder)
            self.myConfig = None  # Configuration (placeholder)
            self.myIndicators = self.indicators  # Use existing indicators
            self.myDataManager = dataManager  # Data manager
            
            return self
        
        return self._timeit("CreateModules", _impl)
    
    def Initialize(self, sistem=None, V=None, Open=None, High=None, Low=None, Close=None, Volume=None, Lot=None):
        """Initialize system with market data - Python equivalent of C# Initialize"""
        def _impl():
            # System properties (equivalent to C# version)
            if sistem:
                self.GrafikSembol = getattr(sistem, 'Sembol', 'BTCUSD')
                self.GrafikPeriyot = getattr(sistem, 'Periyot', '1D')
                self.SistemAdi = getattr(sistem, 'Name', 'TradingSystem')
            else:
                # Default values when no sistem object provided
                self.GrafikSembol = 'BTCUSD'
                self.GrafikPeriyot = '1D'
                self.SistemAdi = 'PythonTradingSystem'
            
            # Set data (equivalent to SetData method)
            self.SetData(sistem, V, Open, High, Low, Close, Volume, Lot)
            
            # Initialize modules (equivalent to C# module initialization)
            if hasattr(self, 'myVarlik') and self.myVarlik:
                pass
                
            if hasattr(self, 'myTrader') and self.myTrader:
                self.myTrader.position = "FLAT"  # Reset position
                
            if hasattr(self, 'myUtils') and self.myUtils:
                pass
                
            if hasattr(self, 'myTimeUtils') and self.myTimeUtils:
                pass
                
            if hasattr(self, 'myBarUtils') and self.myBarUtils:
                pass
                
            if hasattr(self, 'myIndicators') and self.myIndicators:
                pass
                
            return self
        
        return self._timeit("Initialize", _impl)
    
    def SetData(self, sistem, V, Open, High, Low, Close, Volume, Lot):
        """Set market data - equivalent to C# SetData method"""
        self.V = V
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Close = Close
        self.Volume = Volume
        self.Lot = Lot
    
    def Reset(self, sistem=None):
        """Reset system state - Python equivalent of C# Reset"""
        def _impl():
            # Initialize compatibility attributes if not done
            if not hasattr(self, 'results'):
                self.__init_compat__()
                
            # Reset all modules (equivalent to C# module Reset calls)
            if hasattr(self, 'myVarlik') and self.myVarlik:
                pass
                
            if hasattr(self, 'myTrader') and self.myTrader:
                self.myTrader.position = "FLAT"
                if hasattr(self.myTrader, 'datetime_start'):
                    self.myTrader.datetime_start = None
                if hasattr(self.myTrader, 'datetime_end'):
                    self.myTrader.datetime_end = None
                
            if hasattr(self, 'myUtils') and self.myUtils:
                pass
                
            if hasattr(self, 'myTimeUtils') and self.myTimeUtils:
                pass
                
            if hasattr(self, 'myBarUtils') and self.myBarUtils:
                pass
                
            if hasattr(self, 'myIndicators') and self.myIndicators:
                pass
            
            # Reset InputParams (equivalent to C# for loop)
            for i in range(self.InputParamsCount):
                self.InputParams[i] = ""
            
            # Reset other system properties
            if hasattr(self, 'Parametreler'):
                self.Parametreler = ["SMA", 20, 50]  # Reset to defaults
            
            # Clear results
            self.results = []
            
            return self
        
        return self._timeit("Reset", _impl)
    
    def InitializeParamsWithDefaults(self, sistem=None):
        """Initialize parameters with defaults - Python equivalent of C# InitializeParamsWithDefaults"""
        def _impl():
            # Initialize compatibility attributes if not done
            if not hasattr(self, 'results'):
                self.__init_compat__()
                
            # Trading parameters (equivalent to C# version)
            self.HisseSayisi = 0
            self.KontratSayisi = 10
            self.KomisyonCarpan = 0.0
            self.VarlikAdedCarpani = 1
            
            # File paths (equivalent to C# version)
            self.InputsDir = "Aykut/Exports/"
            self.OutputsDir = "Aykut/Exports/"
            sistem_adi = getattr(self, 'SistemAdi', 'PythonTradingSystem')
            self.ParamsInputFileName = f"{self.InputsDir}{sistem_adi}_params.txt"
            self.IstatistiklerOutputFileName = f"{self.OutputsDir}Istatistikler.csv"
            self.IstatistiklerOptOutputFileName = f"{self.OutputsDir}IstatistiklerOpt.csv"
            
            # System flags (equivalent to C# version)
            self.bUseParamsFromInputFile = False
            self.bOptEnabled = False
            self.bIdealGetiriHesapla = True
            self.bIstatistikleriHesapla = True
            self.bIstatistikleriEkranaYaz = True
            self.bGetiriIstatistikleriEkranaYaz = True
            self.bIstatistikleriDosyayaYaz = True
            self.bOptimizasyonIstatistiklerininBasliklariniDosyayaYaz = False
            self.bOptimizasyonIstatistikleriniDosyayaYaz = False
            self.bSinyalleriEkranaCiz = True
            
            # Run control (equivalent to C# version)
            self.CurrentRunIndex = 0
            self.TotalRunCount = 1
            
            # Asset configuration (equivalent to C# Fx Ons Altin Micro setup)
            if hasattr(self, 'myVarlik') and self.myVarlik:
                pass
            
            # Trading signals initialization (equivalent to C# version)
            if hasattr(self, 'myTrader') and self.myTrader and hasattr(self.myTrader, 'signals'):
                # Main control signals
                self.myTrader.signals.kar_al_enabled = False
                self.myTrader.signals.zarar_kes_enabled = False
                self.myTrader.signals.gun_sonu_poz_kapat_enabled = False
                self.myTrader.signals.time_filtering_enabled = False
                
                # State tracking signals
                if not hasattr(self.myTrader.signals, 'kar_alindi'):
                    self.myTrader.signals.kar_alindi = False
                if not hasattr(self.myTrader.signals, 'zarar_kesildi'):
                    self.myTrader.signals.zarar_kesildi = False
                if not hasattr(self.myTrader.signals, 'flat_olundu'):
                    self.myTrader.signals.flat_olundu = False
                if not hasattr(self.myTrader.signals, 'poz_acilabilir'):
                    self.myTrader.signals.poz_acilabilir = False
                if not hasattr(self.myTrader.signals, 'poz_acildi'):
                    self.myTrader.signals.poz_acildi = False
                if not hasattr(self.myTrader.signals, 'poz_kapatilabilir'):
                    self.myTrader.signals.poz_kapatilabilir = False
                if not hasattr(self.myTrader.signals, 'poz_kapatildi'):
                    self.myTrader.signals.poz_kapatildi = False
                if not hasattr(self.myTrader.signals, 'poz_acilabilir_alis'):
                    self.myTrader.signals.poz_acilabilir_alis = False
                if not hasattr(self.myTrader.signals, 'poz_acilabilir_satis'):
                    self.myTrader.signals.poz_acilabilir_satis = False
                if not hasattr(self.myTrader.signals, 'poz_acildi_alis'):
                    self.myTrader.signals.poz_acildi_alis = False
                if not hasattr(self.myTrader.signals, 'poz_acildi_satis'):
                    self.myTrader.signals.poz_acildi_satis = False
                if not hasattr(self.myTrader.signals, 'gun_sonu_poz_kapatildi'):
                    self.myTrader.signals.gun_sonu_poz_kapatildi = False
            
            # Default strategy parameters
            self.Parametreler = ["SMA", 20, 50]  # Method, Period1, Period2
            
            return self
        
        return self._timeit("InitializeParamsWithDefaults", _impl)
    
    def SetParamsForSingleRun(self, sistem=None, IdealGetiriHesapla=True, IstatistikleriHesapla=True,
                              IstatistikleriEkranaYaz=True, GetiriIstatistikleriEkranaYaz=True, 
                              IstatistikleriDosyayaYaz=True, SinyalleriEkranaCiz=True):
        """Set parameters for single run - Python equivalent of C# SetParamsForSingleRun"""
        def _impl():
            # Initialize compatibility attributes if not done
            if not hasattr(self, 'results'):
                self.__init_compat__()
                
            # Set boolean flags for single run configuration (equivalent to C# version)
            self.bIdealGetiriHesapla = IdealGetiriHesapla
            self.bIstatistikleriHesapla = IstatistikleriHesapla
            self.bIstatistikleriEkranaYaz = IstatistikleriEkranaYaz
            self.bGetiriIstatistikleriEkranaYaz = GetiriIstatistikleriEkranaYaz
            self.bIstatistikleriDosyayaYaz = IstatistikleriDosyayaYaz
            self.bSinyalleriEkranaCiz = SinyalleriEkranaCiz
            
            return self
        
        return self._timeit("SetParamsForSingleRun", _impl)
    
    def GetTrader(self):
        """Get trader instance"""
        return self.trader
    
    def Start(self):
        """Start trading system"""
        print("Trading system started...")
    
    def EmirleriResetle(self, i: int):
        """Reset orders for current bar"""
        pass
    
    def EmirOncesiDonguFoksiyonlariniCalistir(self, i: int):
        """Execute pre-order loop functions"""
        pass
    
    def EmirleriSetle(self, i: int, Al: bool, Sat: bool, FlatOl: bool, PasGec: bool, KarAl: bool, ZararKes: bool):
        """Set orders based on signals"""
        if Al and not self.trader.IsSonYonA():
            self.trader.position = "LONG"
            print(f"Bar {i}: BUY signal executed")
        elif Sat and not self.trader.IsSonYonS():
            self.trader.position = "SHORT" 
            print(f"Bar {i}: SELL signal executed")
        elif FlatOl:
            self.trader.position = "FLAT"
            print(f"Bar {i}: FLAT signal executed")
    
    def IslemZamanFiltresiUygula(self, sistem=None, bar_index: int = 0, filter_mode: int = 3):
        """
        Apply time filtering - Python equivalent of C# SystemWrapper IslemZamanFiltresiUygula method.
        
        This method modifies the trading signal variables (Al, Sat, FlatOl) based on time filtering.
        It calls the CTrader method to get filtering results and applies them to signals.
        
        Args:
            sistem: System interface (for compatibility)
            bar_index: Current bar index 
            filter_mode: Filter mode (default 3 - DateTime range)
        """
        use_time_filtering = self.trader.signals.time_filtering_enabled if hasattr(self.trader.signals, 'time_filtering_enabled') else False
        
        if use_time_filtering:
            # Get filtering results from CTrader
            is_trade_enabled, is_poz_kapat_enabled, check_result = self.trader.IslemZamanFiltresiUygula(sistem, bar_index, filter_mode)
            
            # Access the current signal values that were set by EmirleriSetle
            # These would normally be passed by reference in C# but we'll access them from trader signals
            al = self.trader.signals.al if hasattr(self.trader.signals, 'al') else False
            sat = self.trader.signals.sat if hasattr(self.trader.signals, 'sat') else False
            flat_ol = self.trader.signals.flat_ol if hasattr(self.trader.signals, 'flat_ol') else False
            
            # Modify signals based on time filtering results
            if not is_trade_enabled:
                self.trader.signals.al = False  # Al = false
                self.trader.signals.sat = False  # Sat = false
                
            if is_poz_kapat_enabled:
                self.trader.signals.flat_ol = True  # FlatOl = true
    
    def EmirSonrasiDonguFoksiyonlariniCalistir(self, i: int):
        """Execute post-order loop functions"""
        pass
    
    def Stop(self):
        """Stop trading system"""
        print("Trading system stopped.")
    
    def HesaplamalariYap(self):
        """Perform calculations"""
        def _impl():
            print("Performing final calculations...")
        
        return self._timeit("HesaplamalariYap", _impl)
    
    def SonuclariEkrandaGoster(self):
        """Show results on screen"""
        print("=== TRADING RESULTS ===")
        print(f"Final position: {self.trader.position}")
    
    def SonuclariDosyayaYaz(self):
        """Write results to file"""
        def _impl():
            with open("trading_results.txt", "w", encoding="utf-8") as f:
                f.write("=== TRADING RESULTS ===\n")
                f.write(f"Final position: {self.trader.position}\n")
                f.write(f"Total bars processed: {len(self.Open) if hasattr(self, 'Open') and self.Open is not None else 'N/A'}\n")
            print("Results written to trading_results.txt")
        
        return self._timeit("SonuclariDosyayaYaz", _impl)
    
    # Timing utilities
    def _timeit(self, name, func, *args, **kwargs):
        """Genel zaman ölçer"""
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if not hasattr(self, '_timing_report'):
            self._timing_report = {}
        self._timing_report[name] = elapsed
        return result
    
    def reportTimes(self):
        """Timer raporu"""
        print("\n=== SystemWrapper Timing Report ===")
        if not hasattr(self, '_timing_report') or not self._timing_report:
            print("No timing data collected.")
            return
        for k, v in self._timing_report.items():
            print(f"{k:25s}: {v:.6f} sec")
    
    # ===========================================================================================
    # END OF COMPATIBILITY METHODS
    # ===========================================================================================
    
    def initialize_system(self, system: SystemProtocol, v: Any = None) -> 'SystemWrapper':
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