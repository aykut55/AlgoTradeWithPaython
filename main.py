import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional
from src.data_management.data_manager import DataManager
from src.trading.trader import CTrader
from src.trading.signals import CSignals
from src.system.system_wrapper import SystemWrapper
from src.utils.utils import CUtils
from src.indicators.indicator_manager import CIndicatorManager

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
myIndicators = CIndicatorManager()
mySystem = SystemWrapper()
myUtils = CUtils()

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
MA1 = myIndicators.calculate_ma(None, Close, Yontem, Periyot1)
MA2 = myIndicators.calculate_ma(None, Close, Yontem, Periyot2)

# Calculate RSI
Rsi = myIndicators.calculate_rsi(None, Close, 14)

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
mySystem.Start(None)  # Pass None for sistem parameter
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
    mySystem.EmirleriResetle(None, i)
    
    # Execute pre-order functions
    mySystem.EmirOncesiDonguFoksiyonlariniCalistir(None, i)
    
    # Skip first bar
    if i < 1:
        continue
    
    # Generate trading signals based on MA crossover
    Al = myUtils.yukari_kesti(None, i, MA1, MA2)
    Sat = myUtils.asagi_kesti(None, i, MA1, MA2)
    
    # Additional RSI-based signals
    rsi_50_line = np.full(len(Rsi), 50.0)
    Al = Al or myUtils.yukari_kesti(None, i, Rsi, rsi_50_line)
    Sat = Sat or myUtils.asagi_kesti(None, i, Rsi, rsi_50_line)
    
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
    mySystem.EmirSonrasiDonguFoksiyonlariniCalistir(None, i)

# Stop system
mySystem.Stop(None)

# --------------------------------------------------------------
# Perform final calculations and display results
mySystem.HesaplamalariYap(None)
mySystem.SonuclariEkrandaGoster()
mySystem.SonuclariDosyayaYaz(None)

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

