import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional
from src.data_management.data_manager import DataManager



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


class KarAlZararKes:
    """Take profit and stop loss management"""
    
    def __init__(self):
        pass
    
    def SonFiyataGoreKarAlSeviyeHesapla(self, i: int, param1: float, param2: float, param3: float) -> float:
        """Calculate take profit level based on last price"""
        # Simplified implementation
        return param1 if param1 > 0 else 0
    
    def SonFiyataGoreZararKesSeviyeHesapla(self, i: int, param1: float, param2: float, param3: float) -> float:
        """Calculate stop loss level based on last price"""
        # Simplified implementation
        return param1 if param1 < 0 else 0


class Signals:
    """Trading signals configuration"""
    
    def __init__(self):
        self.KarAlEnabled = False
        self.ZararKesEnabled = False
        self.GunSonuPozKapatEnabled = False
        self.TimeFilteringEnabled = True


class Trader:
    """Python equivalent of C# Trader class"""
    
    def __init__(self):
        self.Signals = Signals()
        self.KarAlZararKes = KarAlZararKes()
        self.position = "FLAT"  # "LONG", "SHORT", "FLAT"
        self.datetime_start = None
        self.datetime_end = None
    
    def ResetDateTimes(self):
        """Reset trading time filters"""
        self.datetime_start = None
        self.datetime_end = None
    
    def SetDateTimes(self, start_datetime: str, end_datetime: str):
        """Set trading time filters"""
        self.datetime_start = start_datetime
        self.datetime_end = end_datetime
    
    def IsSonYonA(self) -> bool:
        """Check if last direction was buy"""
        return self.position == "LONG"
    
    def IsSonYonS(self) -> bool:
        """Check if last direction was sell"""
        return self.position == "SHORT"
    
    def IsSonYonF(self) -> bool:
        """Check if last direction was flat"""
        return self.position == "FLAT"


class SystemWrapper:
    """Python equivalent of C# SystemWrapper class"""
    
    def __init__(self):
        self.trader = Trader()
        self.results = []
    
    def CreateModules(self):
        """Create system modules"""
        return self
    
    def Initialize(self, V, Open, High, Low, Close, Volume, Lot):
        """Initialize system with market data"""
        self.V = V
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Close = Close
        self.Volume = Volume
        self.Lot = Lot
    
    def Reset(self):
        """Reset system state"""
        pass
    
    def InitializeParamsWithDefaults(self):
        """Initialize parameters with defaults"""
        self.Parametreler = ["SMA", 20, 50]  # Method, Period1, Period2
    
    def SetParamsForSingleRun(self):
        """Set parameters for single run"""
        pass
    
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
    
    def IslemZamanFiltresiUygula(self, i: int):
        """Apply time filtering"""
        pass
    
    def EmirSonrasiDonguFoksiyonlariniCalistir(self, i: int):
        """Execute post-order loop functions"""
        pass
    
    def Stop(self):
        """Stop trading system"""
        print("Trading system stopped.")
    
    def HesaplamalariYap(self):
        """Perform calculations"""
        print("Performing final calculations...")
    
    def SonuclariEkrandaGoster(self):
        """Show results on screen"""
        print("=== TRADING RESULTS ===")
        print(f"Final position: {self.trader.position}")
    
    def SonuclariDosyayaYaz(self):
        """Write results to file"""
        with open("trading_results.txt", "w", encoding="utf-8") as f:
            f.write("=== TRADING RESULTS ===\n")
            f.write(f"Final position: {self.trader.position}\n")
            f.write(f"Total bars processed: {len(self.Open) if hasattr(self, 'Open') else 'N/A'}\n")
        print("Results written to trading_results.txt")


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
mySystem.CreateModules().Initialize(V, Open, High, Low, Close, Volume, Lot)

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
mySystem.GetTrader().Signals.KarAlEnabled = False
mySystem.GetTrader().Signals.ZararKesEnabled = False
mySystem.GetTrader().Signals.GunSonuPozKapatEnabled = False
mySystem.GetTrader().Signals.TimeFilteringEnabled = True

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
    KarAl = KarAl if mySystem.GetTrader().Signals.KarAlEnabled else False
    ZararKes = ZararKes if mySystem.GetTrader().Signals.ZararKesEnabled else False
    
    # Get position status
    IsSonYonA = mySystem.GetTrader().IsSonYonA()
    IsSonYonS = mySystem.GetTrader().IsSonYonS()
    IsSonYonF = mySystem.GetTrader().IsSonYonF()
    
    useTimeFiltering = mySystem.GetTrader().Signals.TimeFilteringEnabled
    
    # Set orders (order is important)
    mySystem.EmirleriSetle(i, Al, Sat, FlatOl, PasGec, KarAl, ZararKes)
    
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
k = 0

# --------------------------------------------------------------
# Cleanup (equivalent to Lib.DeleteSystemWrapper)
del mySystem

print("Main execution completed successfully!")

# Show timing report
dataManager.reportTimes()