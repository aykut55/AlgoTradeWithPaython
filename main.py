import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional
from src.data_management.data_manager import DataManager
from src.trading.trader import CTrader
from src.trading.signals import CSignals
from src.system.system_wrapper import SystemWrapper



class IndicatorManager:
    """Python equivalent of C# IndicatorManager"""
    
    def __init__(self):
        pass
    
    def MA(self, data: np.ndarray, method: str, period: int) -> np.ndarray:
        """Calculate Moving Average"""
        if method == "SMA":
            return self._sma(data, period)
        elif method == "EMA":
            return self._ema(data, period)
        else:
            return self._sma(data, period)  # Default to SMA
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average using numpy"""
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average using numpy"""
        alpha = 2.0 / (period + 1.0)
        result = np.full_like(data, np.nan)
        result[0] = data[0]
        for i in range(1, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = data[i]
        return result
    
    def RSI(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using numpy"""
        delta = np.diff(data, prepend=np.nan)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        result = np.full_like(data, np.nan)
        
        for i in range(period, len(data)):
            avg_gain = np.mean(gains[i - period + 1:i + 1])
            avg_loss = np.mean(losses[i - period + 1:i + 1])
            
            if avg_loss == 0:
                result[i] = 100
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        
        return result


class Utils:
    """Python equivalent of C# Utils class"""
    
    def __init__(self):
        pass
    
    def YukariKesti(self, i: int, series1: np.ndarray, series2: np.ndarray) -> bool:
        """Check if series1 crosses above series2"""
        if i < 1 or i >= len(series1) or i >= len(series2):
            return False
        return (series1[i-1] <= series2[i-1] and series1[i] > series2[i])
    
    def AsagiKesti(self, i: int, series1: np.ndarray, series2: np.ndarray) -> bool:
        """Check if series1 crosses below series2"""
        if i < 1 or i >= len(series1) or i >= len(series2):
            return False
        return (series1[i-1] >= series2[i-1] and series1[i] < series2[i])


class SystemWrapper_sil:
    """Python equivalent of C# SystemWrapper class - OLD VERSION TO BE REMOVED"""
    
    def __init__(self):
        self.trader = CTrader()
        self.results = []
        # timing bilgisi
        self._timing_report = {}
        # input parameters (equivalent to C# InputParams)
        self.InputParamsCount = 10  # Default parameter count
        self.InputParams = [""] * self.InputParamsCount
    
    def CreateModules(self, dataManager=None, lib=None):
        """Create system modules - Python equivalent of C# CreateModules"""
        def _impl():
            # Initialize modules similar to C# version
            self.myVarlik = None  # Asset management (placeholder)
            self.myTrader = self.trader  # Already created in __init__
            self.myUtils = Utils()  # Utils class
            self.myTimeUtils = None  # Time utilities (placeholder)
            self.myBarUtils = None  # Bar utilities (placeholder)
            self.myFileUtils = None  # File utilities (placeholder)
            self.myExcelUtils = None  # Excel utilities (placeholder)
            self.mySharedMemory = None  # Shared memory (placeholder)
            self.myConfig = None  # Configuration (placeholder)
            self.myIndicators = IndicatorManager()  # Indicator manager
            self.myDataManager = dataManager  # Data manager
            
            return self
        
        return self._timeit("CreateModules", _impl)
    
    def Initialize(self, sistem=None, V=None, Open=None, High=None, Low=None, Close=None, Volume=None, Lot=None):
        """Initialize system with market data - Python equivalent of C# Initialize"""
        def _impl():
            # System properties (equivalent to C# version)
            if sistem:
                self.GrafikSembol = getattr(sistem, 'Sembol', 'BTCUSD')  # Default symbol
                self.GrafikPeriyot = getattr(sistem, 'Periyot', '1D')    # Default period
                self.SistemAdi = getattr(sistem, 'Name', 'TradingSystem') # Default name
            else:
                # Default values when no sistem object provided
                self.GrafikSembol = 'BTCUSD'
                self.GrafikPeriyot = '1D'
                self.SistemAdi = 'PythonTradingSystem'
            
            # Set data (equivalent to SetData method)
            self.SetData(sistem, V, Open, High, Low, Close, Volume, Lot)
            
            # Initialize modules (equivalent to C# module initialization)
            if hasattr(self, 'myVarlik') and self.myVarlik:
                # myVarlik.Initialize(sistem) - placeholder
                pass
                
            if hasattr(self, 'myTrader') and self.myTrader:
                # myTrader.Initialize(sistem, V, Open, High, Low, Close, Volume, Lot, myVarlik)
                self.myTrader.position = "FLAT"  # Reset position
                
            if hasattr(self, 'myUtils') and self.myUtils:
                # myUtils.Initialize(sistem) - placeholder
                pass
                
            if hasattr(self, 'myTimeUtils') and self.myTimeUtils:
                # myTimeUtils.Initialize(sistem, V, Open, High, Low, Close, Volume, Lot) - placeholder
                pass
                
            if hasattr(self, 'myBarUtils') and self.myBarUtils:
                # myBarUtils.Initialize(sistem, V, Open, High, Low, Close, Volume, Lot) - placeholder
                pass
                
            if hasattr(self, 'myIndicators') and self.myIndicators:
                # myIndicators.Initialize(sistem, V, Open, High, Low, Close, Volume, Lot) - placeholder
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
            # Reset all modules (equivalent to C# module Reset calls)
            if hasattr(self, 'myVarlik') and self.myVarlik:
                # myVarlik.Reset(sistem) - placeholder
                pass
                
            if hasattr(self, 'myTrader') and self.myTrader:
                # myTrader.Reset(sistem)
                self.myTrader.position = "FLAT"
                self.myTrader.datetime_start = None
                self.myTrader.datetime_end = None
                
            if hasattr(self, 'myUtils') and self.myUtils:
                # myUtils.Reset(sistem) - placeholder
                pass
                
            if hasattr(self, 'myTimeUtils') and self.myTimeUtils:
                # myTimeUtils.Reset(sistem) - placeholder
                pass
                
            if hasattr(self, 'myBarUtils') and self.myBarUtils:
                # myBarUtils.Reset(sistem) - placeholder
                pass
                
            if hasattr(self, 'myIndicators') and self.myIndicators:
                # myIndicators.Reset(sistem) - placeholder
                pass
            
            # Reset InputParams (equivalent to C# for loop)
            for i in range(self.InputParamsCount):
                self.InputParams[i] = ""
            
            # Reset other system properties
            if hasattr(self, 'Parametreler'):
                self.Parametreler = []
            
            # Clear results
            self.results = []
            
            return self
        
        return self._timeit("Reset", _impl)
    
    def InitializeParamsWithDefaults(self, sistem=None):
        """Initialize parameters with defaults - Python equivalent of C# InitializeParamsWithDefaults"""
        def _impl():
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
                # myVarlik.SetKontratParamsFxOnsAltinMicro(sistem, KontratSayisi=1, VarlikAdedCarpani=1)
                # myVarlik.SetKomisyonParams(sistem, KomisyonCarpan=0.0)
                pass  # placeholder
            
            # Trading signals initialization (equivalent to C# version)
            if hasattr(self, 'myTrader') and self.myTrader:
                # Main control signals (marked in C# with <<--------------) 
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
    
    def IslemZamanFiltresiUygula_with_refs(self, sistem=None, bar_index: int = 0, 
                                          al_ref: list = None, sat_ref: list = None, flat_ol_ref: list = None,
                                          pas_gec_ref: list = None, kar_al_ref: list = None, zarar_kes_ref: list = None,
                                          filter_mode: int = 3):
        """
        Apply time filtering with reference parameters - Python equivalent of C# overloaded method.
        
        This version takes signal variables by reference (using lists as mutable containers)
        and modifies them based on time filtering results.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            al_ref: Buy signal reference (list with one element)
            sat_ref: Sell signal reference (list with one element) 
            flat_ol_ref: Flatten signal reference (list with one element)
            pas_gec_ref: Pass signal reference (list with one element)
            kar_al_ref: Take profit signal reference (list with one element)
            zarar_kes_ref: Stop loss signal reference (list with one element)
            filter_mode: Filter mode (default 3)
        """
        use_time_filtering = self.trader.signals.time_filtering_enabled if hasattr(self.trader.signals, 'time_filtering_enabled') else False
        
        if use_time_filtering:
            # Get filtering results from CTrader
            is_trade_enabled, is_poz_kapat_enabled, check_result = self.trader.IslemZamanFiltresiUygula(sistem, bar_index, filter_mode)
            
            # Modify signal references based on time filtering results
            if not is_trade_enabled:
                if al_ref: al_ref[0] = False
                if sat_ref: sat_ref[0] = False
                
            if is_poz_kapat_enabled:
                if flat_ol_ref: flat_ol_ref[0] = True
    
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
                f.write(f"Total bars processed: {len(self.Open) if hasattr(self, 'Open') else 'N/A'}\n")
            print("Results written to trading_results.txt")
        
        return self._timeit("SonuclariDosyayaYaz", _impl)
    
    # --------------------------------------------------------
    # Timing utilities
    def _timeit(self, name, func, *args, **kwargs):
        """Genel zaman ölçer"""
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self._timing_report[name] = elapsed
        return result
    
    def reportTimes(self):
        """Timer raporu"""
        print("\n=== SystemWrapper Timing Report ===")
        if not self._timing_report:
            print("No timing data collected.")
            return
        for k, v in self._timing_report.items():
            print(f"{k:25s}: {v:.6f} sec")


# Main execution starts here
# --------------------------------------------------------------
# Read market data (equivalent to Sistem.GrafikVerileri operations)
print("Loading market data...")

dataManager = DataManager()

dataManager.set_read_mode_last_n(1000)  # Son 2000 satırı okumaya ayarla
dataManager.load_prices_from_csv("data", "01", "BTCUSD.csv")

Df         = dataManager.get_dataframe()
Time       = dataManager.get_epoch_time_array()
Open       = dataManager.get_open_array()
High       = dataManager.get_high_array()
Low        = dataManager.get_low_array()
Close      = dataManager.get_close_array()
Volume     = dataManager.get_volume_array()
Lot        = dataManager.get_lot_array()
BarCount   = dataManager.get_bar_count()
ItemsCount = dataManager.get_items_count()
dataManager.add_time_columns()

print("========================")
print("BarCount    :", BarCount)
print("ItemsCount  :", ItemsCount)

print("InputTime   :", dataManager.get_timestamp_array()[-5:])
print("EpochTime   :", dataManager.get_epoch_time_array()[-5:])

print("DateTime    :", dataManager.get_date_time_array_as_str()[-5:])
print("Date        :", dataManager.get_date_array_as_str()[-5:])
print("Time        :", dataManager.get_time_array_as_str()[-5:])

print("Open        :", dataManager.get_open_array()[-5:])
print("High        :", dataManager.get_high_array()[-5:])
print("Low         :", dataManager.get_low_array()[-5:])
print("Close       :", dataManager.get_close_array()[-5:])
print("Volume      :", dataManager.get_volume_array()[-5:])
print("Lot         :", dataManager.get_lot_array()[-5:])
print("========================")


# --------------------------------------------------------------
# Initialize system components (equivalent to Lib.Get* methods)
myIndicators = IndicatorManager()
mySystem = SystemWrapper()
myUtils = Utils()

# --------------------------------------------------------------
# Initialize system
V = None  # Placeholder for GrafikVerileri
mySystem.CreateModules(dataManager=dataManager).Initialize(V, Open, High, Low, Close, Volume, Lot)

# --------------------------------------------------------------
# System configuration
mySystem.Reset()
mySystem.InitializeParamsWithDefaults()
mySystem.SetParamsForSingleRun()

# ----------------- Get parameters ---------------------------
Yontem = mySystem.Parametreler[0]    # "SMA"
Periyot1 = int(mySystem.Parametreler[1])  # 20
Periyot2 = int(mySystem.Parametreler[2])  # 50
# ----------------- Get parameters ---------------------------

# Calculate moving averages
MA1 = myIndicators.MA(Close, Yontem, Periyot1)
MA2 = myIndicators.MA(Close, Yontem, Periyot2)

# Calculate RSI
Rsi = myIndicators.RSI(Close, 14)

# --------------------------------------------------------------
# Set trading time filters
DateTimes = ["25.05.2025 14:30:00", "02.06.2025 14:00:00"]
Dates = ["01.01.1900", "01.01.2100"]
Times = ["09:30:00", "11:59:00"]

mySystem.GetTrader().ResetDateTimes()
mySystem.GetTrader().SetDateTimes(DateTimes[0], DateTimes[1])

# --------------------------------------------------------------
# Configure trading signals
mySystem.GetTrader().signals.kar_al_enabled = False
mySystem.GetTrader().signals.zarar_kes_enabled = False
mySystem.GetTrader().signals.gun_sonu_poz_kapat_enabled = False
mySystem.GetTrader().signals.time_filtering_enabled = True

# --------------------------------------------------------------
# Main trading loop
mySystem.Start()
for i in range(BarCount):
    # Initialize signal variables
    Al = False
    Sat = False
    FlatOl = False
    PasGec = False
    KarAl = False
    ZararKes = False
    isTradeEnabled = False
    isPozKapatEnabled = False
    
    # Reset orders for current bar
    mySystem.EmirleriResetle(i)
    
    # Execute pre-order functions
    mySystem.EmirOncesiDonguFoksiyonlariniCalistir(i)
    
    # Skip first bar
    if i < 1:
        continue
    
    # Generate trading signals based on MA crossover
    Al = myUtils.YukariKesti(i, MA1, MA2)
    Sat = myUtils.AsagiKesti(i, MA1, MA2)
    
    # Additional RSI-based signals
    rsi_50_line = np.full(len(Rsi), 50.0)
    Al = Al or myUtils.YukariKesti(i, Rsi, rsi_50_line)
    Sat = Sat or myUtils.AsagiKesti(i, Rsi, rsi_50_line)
    
    # Calculate take profit and stop loss
    KarAl = mySystem.GetTrader().KarAlZararKes.SonFiyataGoreKarAlSeviyeHesapla(i, 5, 50, 1000) != 0
    ZararKes = mySystem.GetTrader().KarAlZararKes.SonFiyataGoreZararKesSeviyeHesapla(i, -1, -10, 1000) != 0
    
    # Apply signal enablers
    KarAl = KarAl if mySystem.GetTrader().signals.kar_al_enabled else False
    ZararKes = ZararKes if mySystem.GetTrader().signals.zarar_kes_enabled else False
    
    # Get position status
    IsSonYonA = mySystem.GetTrader().IsSonYonA()
    IsSonYonS = mySystem.GetTrader().IsSonYonS()
    IsSonYonF = mySystem.GetTrader().IsSonYonF()
    
    useTimeFiltering = mySystem.GetTrader().signals.time_filtering_enabled
    
    # Set orders (order is important)
    mySystem.GetTrader().EmirleriSetle(None, i, Al, Sat, FlatOl, PasGec, KarAl, ZararKes)
    
    # Apply time filtering
    mySystem.IslemZamanFiltresiUygula(i)
    
    # Execute post-order functions
    mySystem.EmirSonrasiDonguFoksiyonlariniCalistir(i)

# Stop system
mySystem.Stop()

# --------------------------------------------------------------
# Perform final calculations and display results
mySystem.HesaplamalariYap()
mySystem.SonuclariEkrandaGoster()
mySystem.SonuclariDosyayaYaz()

# --------------------------------------------------------------
# Show timing reports
dataManager.reportTimes()
mySystem.reportTimes()

# --------------------------------------------------------------
k = 0

# --------------------------------------------------------------
# Cleanup (equivalent to Lib.DeleteSystemWrapper)
del mySystem

print("Main execution completed successfully!")

