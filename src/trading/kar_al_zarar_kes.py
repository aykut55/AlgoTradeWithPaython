"""
Take profit and stop loss calculation system for algorithmic trading.

This module contains the CKarAlZararKes class which provides various methods
for calculating take profit and stop loss levels using different approaches:
- Percentage-based calculations
- Level-based calculations  
- Trailing stop calculations
"""

from typing import Optional, List
import numpy as np
from ..utils.utils import CUtils


class CKarAlZararKes:
    """
    Take profit and stop loss manager.
    
    Provides multiple calculation methods for take profit and stop loss:
    1. Percentage-based calculations with reference arrays
    2. Price-based percentage calculations from last price
    3. Range-based percentage calculations
    4. Fixed level calculations
    5. Range-based level calculations
    """
    
    def __init__(self):
        """Initialize take profit/stop loss manager."""
        self.trader = None
    
    def __del__(self):
        """Destructor - Python equivalent of C# finalizer."""
        pass
    
    def initialize(self, sistem=None, trader=None):
        """
        Initialize with trader reference.
        
        Args:
            sistem: System interface (for compatibility)
            trader: CTrader instance
            
        Returns:
            Self for method chaining
        """
        self.trader = trader
        return self
    
    def reset(self, sistem=None):
        """
        Reset the manager state.
        
        Args:
            sistem: System interface (for compatibility)
            
        Returns:
            Self for method chaining
        """
        return self
    
    # Type 1: Percentage-based calculations with reference array
    def kar_al_yuzde_hesapla(self, sistem, bar_index: int, kar_al_yuzdesi: float, ref: np.ndarray) -> int:
        """
        Calculate take profit based on percentage with reference array.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            kar_al_yuzdesi: Take profit percentage
            ref: Reference price array
            
        Returns:
            1 for long position take profit, -1 for short position take profit, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'kar_al_yuzde_hesapla_enabled') and self.trader.signals.kar_al_yuzde_hesapla_enabled:
            # Calculate take profit level using system method
            if hasattr(sistem, 'KarAlYuzde'):
                kar_al_level = sistem.KarAlYuzde(kar_al_yuzdesi, i)
            else:
                kar_al_level = 0
            
            # Store in trader's kar_al_list if exists
            if hasattr(self.trader, 'lists') and hasattr(self.trader.lists, 'kar_al_list'):
                self.trader.lists.kar_al_list[i] = kar_al_level
            
            if kar_al_level == 0:
                kar_al_level = ref[i]
            
            # Check conditions
            if self.trader.is_son_yon_a(sistem) and ref[i] > kar_al_level:
                result = 1
            elif self.trader.is_son_yon_s(sistem) and ref[i] < kar_al_level:
                result = -1
        
        return result
    
    def izleyen_stop_yuzde_hesapla(self, sistem, bar_index: int, izleyen_stop_yuzdesi: float, ref: np.ndarray) -> int:
        """
        Calculate trailing stop based on percentage with reference array.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            izleyen_stop_yuzdesi: Trailing stop percentage
            ref: Reference price array
            
        Returns:
            1 for long position stop, -1 for short position stop, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'izleyen_stop_yuzde_hesapla_enabled') and self.trader.signals.izleyen_stop_yuzde_hesapla_enabled:
            # Calculate trailing stop level using system method
            if hasattr(sistem, 'IzleyenStopYuzde'):
                stop_level = sistem.IzleyenStopYuzde(izleyen_stop_yuzdesi, i)
            else:
                stop_level = 0
            
            # Store in trader's izleyen_stop_list if exists
            if hasattr(self.trader, 'lists') and hasattr(self.trader.lists, 'izleyen_stop_list'):
                self.trader.lists.izleyen_stop_list[i] = stop_level
            
            if stop_level == 0:
                stop_level = ref[i]
            
            # Check conditions
            if self.trader.is_son_yon_a(sistem) and ref[i] < stop_level:
                result = 1
            elif self.trader.is_son_yon_s(sistem) and ref[i] > stop_level:
                result = -1
        
        return result
    
    def kar_al_yuzde_hesapla_simple(self, sistem, bar_index: int, kar_al_yuzdesi: float = 2.0) -> int:
        """
        Calculate take profit based on percentage using close prices.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            kar_al_yuzdesi: Take profit percentage (default 2.0%)
            
        Returns:
            1 for long position take profit, -1 for short position take profit, 0 for no action
        """
        return self.kar_al_yuzde_hesapla(sistem, bar_index, kar_al_yuzdesi, self.trader.close)
    
    def izleyen_stop_yuzde_hesapla_simple(self, sistem, bar_index: int, izleyen_stop_yuzdesi: float = 1.0) -> int:
        """
        Calculate trailing stop based on percentage using close prices.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            izleyen_stop_yuzdesi: Trailing stop percentage (default 1.0%)
            
        Returns:
            1 for long position stop, -1 for short position stop, 0 for no action
        """
        return self.izleyen_stop_yuzde_hesapla(sistem, bar_index, izleyen_stop_yuzdesi, self.trader.close)
    
    # Type 2: Price-based percentage calculations
    def son_fiyata_gore_kar_al_yuzde_hesapla(self, sistem, bar_index: int, kar_al_yuzdesi: float = 2.0) -> int:
        """
        Calculate take profit based on percentage from last price.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            kar_al_yuzdesi: Take profit percentage (default 2.0%)
            
        Returns:
            1 for long position take profit, -1 for short position take profit, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'kar_al_yuzde_hesapla_enabled') and self.trader.signals.kar_al_yuzde_hesapla_enabled:
            son_fiyat = self.trader.signals.son_fiyat
            
            if self.trader.is_son_yon_a(sistem) and self.trader.close[i] > son_fiyat * (1.0 + kar_al_yuzdesi * 0.01):
                result = 1
            elif self.trader.is_son_yon_s(sistem) and self.trader.close[i] < son_fiyat * (1.0 - kar_al_yuzdesi * 0.01):
                result = -1
        
        return result
    
    def son_fiyata_gore_zarar_kes_yuzde_hesapla(self, sistem, bar_index: int, zarar_kes_yuzdesi: float = -1.0) -> int:
        """
        Calculate stop loss based on percentage from last price.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            zarar_kes_yuzdesi: Stop loss percentage (default -1.0%, negative value)
            
        Returns:
            1 for long position stop, -1 for short position stop, 0 for no action
        """
        result = 0
        i = bar_index
        
        zarar_kes_yuzdesi_ = -1.0 * zarar_kes_yuzdesi  # Convert to positive
        
        if hasattr(self.trader.signals, 'zarar_kes_yuzde_hesapla_enabled') and self.trader.signals.zarar_kes_yuzde_hesapla_enabled:
            son_fiyat = self.trader.signals.son_fiyat
            
            if self.trader.is_son_yon_a(sistem) and self.trader.close[i] < son_fiyat * (1.0 - zarar_kes_yuzdesi_ * 0.01):
                result = 1
            elif self.trader.is_son_yon_s(sistem) and self.trader.close[i] > son_fiyat * (1.0 + zarar_kes_yuzdesi_ * 0.01):
                result = -1
        
        return result
    
    # Type 3: Range-based percentage calculations
    def son_fiyata_gore_kar_al_yuzde_hesapla_range(self, sistem, bar_index: int, 
                                                   seviye_bas: int = 2, seviye_son: int = 10, 
                                                   carpan: float = 0.01) -> int:
        """
        Calculate take profit using range of percentage levels.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            seviye_bas: Starting level (default 2)
            seviye_son: Ending level (default 10)
            carpan: Multiplier (default 0.01 for 1%)
            
        Returns:
            1 for long position take profit, -1 for short position take profit, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'kar_al_yuzde_hesapla_enabled') and self.trader.signals.kar_al_yuzde_hesapla_enabled:
            my_utils = CUtils()
            kar_al = False
            son_fiyat = self.trader.signals.son_fiyat
            
            if self.trader.is_son_yon_a(sistem):
                for m in range(seviye_bas, seviye_son):
                    target_level = np.full(len(self.trader.close), son_fiyat * (1.0 + m * carpan))
                    kar_al = kar_al or my_utils.AsagiKesti(sistem, i, self.trader.close, target_level)
                    if kar_al:
                        break
                
                if kar_al:
                    result = 1
            
            elif self.trader.is_son_yon_s(sistem):
                for m in range(seviye_bas, seviye_son):
                    target_level = np.full(len(self.trader.close), son_fiyat * (1.0 - m * carpan))
                    kar_al = kar_al or my_utils.YukariKesti(sistem, i, self.trader.close, target_level)
                    if kar_al:
                        break
                
                if kar_al:
                    result = -1
        
        return result
    
    def son_fiyata_gore_zarar_kes_yuzde_hesapla_range(self, sistem, bar_index: int,
                                                      seviye_bas: int = -2, seviye_son: int = -10,
                                                      carpan: float = 0.01) -> int:
        """
        Calculate stop loss using range of percentage levels.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            seviye_bas: Starting level (default -2, negative)
            seviye_son: Ending level (default -10, negative)
            carpan: Multiplier (default 0.01 for 1%)
            
        Returns:
            1 for long position stop, -1 for short position stop, 0 for no action
        """
        result = 0
        i = bar_index
        
        seviye_bas_ = -1 * seviye_bas  # Convert to positive
        seviye_son_ = -1 * seviye_son  # Convert to positive
        
        if hasattr(self.trader.signals, 'zarar_kes_yuzde_hesapla_enabled') and self.trader.signals.zarar_kes_yuzde_hesapla_enabled:
            my_utils = CUtils()
            zarar_kes = False
            son_fiyat = self.trader.signals.son_fiyat
            
            if self.trader.is_son_yon_a(sistem):
                for m in range(seviye_bas_, seviye_son_):
                    target_level = np.full(len(self.trader.close), son_fiyat * (1.0 - m * carpan))
                    zarar_kes = zarar_kes or my_utils.AsagiKesti(sistem, i, self.trader.close, target_level)
                    if zarar_kes:
                        break
                
                if zarar_kes:
                    result = 1
            
            elif self.trader.is_son_yon_s(sistem):
                for m in range(seviye_bas_, seviye_son_):
                    target_level = np.full(len(self.trader.close), son_fiyat * (1.0 + m * carpan))
                    zarar_kes = zarar_kes or my_utils.YukariKesti(sistem, i, self.trader.close, target_level)
                    if zarar_kes:
                        break
                
                if zarar_kes:
                    result = -1
        
        return result
    
    # Type 4: Fixed level calculations
    def son_fiyata_gore_kar_al_seviye_hesapla(self, sistem, bar_index: int, kar_al_seviyesi: float = 2000.0) -> int:
        """
        Calculate take profit based on fixed profit level.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            kar_al_seviyesi: Take profit level (default 2000.0)
            
        Returns:
            1 if take profit triggered, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'kar_al_seviye_hesapla_enabled') and self.trader.signals.kar_al_seviye_hesapla_enabled:
            my_utils = CUtils()
            
            # Use kar_zarar_fiyat_list if exists on trader
            if hasattr(self.trader, 'lists') and hasattr(self.trader.lists, 'kar_zarar_fiyat_list'):
                kar_al_level_array = np.full(len(self.trader.lists.kar_zarar_fiyat_list), kar_al_seviyesi)
                result = 1 if my_utils.YukariKesti(sistem, i, self.trader.lists.kar_zarar_fiyat_list, kar_al_level_array) else 0
        
        return result
    
    def son_fiyata_gore_zarar_kes_seviye_hesapla(self, sistem, bar_index: int, zarar_kes_seviyesi: float = -1000.0) -> int:
        """
        Calculate stop loss based on fixed loss level.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            zarar_kes_seviyesi: Stop loss level (default -1000.0)
            
        Returns:
            1 if stop loss triggered, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'zarar_kes_seviye_hesapla_enabled') and self.trader.signals.zarar_kes_seviye_hesapla_enabled:
            my_utils = CUtils()
            
            # Use kar_zarar_fiyat_list if exists on trader
            if hasattr(self.trader, 'lists') and hasattr(self.trader.lists, 'kar_zarar_fiyat_list'):
                zarar_kes_level_array = np.full(len(self.trader.lists.kar_zarar_fiyat_list), zarar_kes_seviyesi)
                result = 1 if my_utils.AsagiKesti(sistem, i, self.trader.lists.kar_zarar_fiyat_list, zarar_kes_level_array) else 0
        
        return result
    
    # Type 5: Range-based level calculations
    def son_fiyata_gore_kar_al_seviye_hesapla_range(self, sistem, bar_index: int,
                                                    seviye_bas: int = 5, seviye_son: int = 50, 
                                                    carpan: int = 1000) -> int:
        """
        Calculate take profit using range of fixed levels.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            seviye_bas: Starting level (default 5)
            seviye_son: Ending level (default 50)
            carpan: Level multiplier (default 1000)
            
        Returns:
            1 if take profit triggered, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'kar_al_seviye_hesapla_enabled') and self.trader.signals.kar_al_seviye_hesapla_enabled:
            my_utils = CUtils()
            kar_al = False
            
            if hasattr(self.trader, 'lists') and hasattr(self.trader.lists, 'kar_zarar_fiyat_list'):
                for m in range(seviye_bas, seviye_son):
                    target_level = np.full(len(self.trader.lists.kar_zarar_fiyat_list), m * carpan)
                    kar_al = kar_al or my_utils.AsagiKesti(sistem, i, self.trader.lists.kar_zarar_fiyat_list, target_level)
                    if kar_al:
                        break
                
                if kar_al:
                    result = 1
        
        return result
    
    def son_fiyata_gore_zarar_kes_seviye_hesapla_range(self, sistem, bar_index: int,
                                                       seviye_bas: int = -1, seviye_son: int = -10,
                                                       carpan: int = 1000) -> int:
        """
        Calculate stop loss using range of fixed levels.
        
        Args:
            sistem: System interface
            bar_index: Current bar index
            seviye_bas: Starting level (default -1)
            seviye_son: Ending level (default -10)
            carpan: Level multiplier (default 1000)
            
        Returns:
            1 if stop loss triggered, 0 for no action
        """
        result = 0
        i = bar_index
        
        if hasattr(self.trader.signals, 'zarar_kes_seviye_hesapla_enabled') and self.trader.signals.zarar_kes_seviye_hesapla_enabled:
            my_utils = CUtils()
            zarar_kes = False
            
            if hasattr(self.trader, 'lists') and hasattr(self.trader.lists, 'kar_zarar_fiyat_list'):
                for m in range(seviye_bas, seviye_son - 1, -1):  # Reverse range for negative values
                    target_level = np.full(len(self.trader.lists.kar_zarar_fiyat_list), m * carpan)
                    zarar_kes = zarar_kes or my_utils.AsagiKesti(sistem, i, self.trader.lists.kar_zarar_fiyat_list, target_level)
                    if zarar_kes:
                        break
                
                if zarar_kes:
                    result = 1
        
        return result
    
    # Compatibility methods for main.py (C# style naming)
    def SonFiyataGoreKarAlSeviyeHesapla(self, i: int, param1: float, param2: float, param3: float) -> float:
        """Compatibility method for main.py - C# style naming"""
        return self.son_fiyata_gore_kar_al_seviye_hesapla_range(None, i, int(param1), int(param2), int(param3))
    
    def SonFiyataGoreZararKesSeviyeHesapla(self, i: int, param1: float, param2: float, param3: float) -> float:
        """Compatibility method for main.py - C# style naming"""
        return self.son_fiyata_gore_zarar_kes_seviye_hesapla_range(None, i, int(param1), int(param2), int(param3))


# Usage Examples (commented for reference):
"""
# Method 1 - Basic percentage
if trader.signals.kar_al_enabled:
    trader.signals.kar_al = kar_zarar.kar_al_yuzde_hesapla_simple(sistem, i, 0.7) != 0

if trader.signals.zarar_kes_enabled:
    trader.signals.zarar_kes = kar_zarar.izleyen_stop_yuzde_hesapla_simple(sistem, i, 0.3) != 0

# Method 2 - Price-based percentage
if trader.signals.kar_al_enabled:
    trader.signals.kar_al = kar_zarar.son_fiyata_gore_kar_al_yuzde_hesapla(sistem, i, 1.0) != 0

if trader.signals.zarar_kes_enabled:
    trader.signals.zarar_kes = kar_zarar.son_fiyata_gore_zarar_kes_yuzde_hesapla(sistem, i, -0.5) != 0

# Method 3 - Range-based percentage
if trader.signals.kar_al_enabled:
    trader.signals.kar_al = kar_zarar.son_fiyata_gore_kar_al_yuzde_hesapla_range(sistem, i, 2, 10) != 0

if trader.signals.zarar_kes_enabled:
    trader.signals.zarar_kes = kar_zarar.son_fiyata_gore_zarar_kes_yuzde_hesapla_range(sistem, i, -2, -10) != 0

# Method 4 - Fixed levels
if trader.signals.kar_al_enabled:
    trader.signals.kar_al = kar_zarar.son_fiyata_gore_kar_al_seviye_hesapla(sistem, i, 2000.0) != 0

if trader.signals.zarar_kes_enabled:
    trader.signals.zarar_kes = kar_zarar.son_fiyata_gore_zarar_kes_seviye_hesapla(sistem, i, -1500.0) != 0

# Method 5 - Range-based levels
if trader.signals.kar_al_enabled:
    trader.signals.kar_al = kar_zarar.son_fiyata_gore_kar_al_seviye_hesapla_range(sistem, i, 5, 50) != 0

if trader.signals.zarar_kes_enabled:
    trader.signals.zarar_kes = kar_zarar.son_fiyata_gore_zarar_kes_seviye_hesapla_range(sistem, i, -3, -5) != 0
"""