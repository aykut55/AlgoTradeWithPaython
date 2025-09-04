"""
Trading signals management for the algorithmic trading system.

This module contains the CSignals class which manages all trading signals
including buy/sell decisions, position tracking, and signal history.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    """Trading direction enumeration."""
    LONG = "A"  # Al (Buy)
    SHORT = "S"  # Sat (Sell)  
    FLAT = "F"  # Flat
    PASS = "P"  # Pas (Pass)


class SignalType(Enum):
    """Signal type enumeration."""
    BUY = "AL"
    SELL = "SAT" 
    FLAT = "FLAT"
    PASS = "PAS"
    TAKE_PROFIT = "KAR_AL"
    STOP_LOSS = "ZARAR_KES"


@dataclass
class SignalInfo:
    """Information about a trading signal."""
    signal_type: SignalType
    direction: Direction
    price: float
    bar_number: int
    timestamp: Optional[datetime] = None
    
    def __str__(self) -> str:
        return f"{self.signal_type.value}@{self.price} (Bar:{self.bar_number})"


@dataclass  
class PositionInfo:
    """Information about current position."""
    direction: Direction = Direction.FLAT
    entry_price: float = 0.0
    entry_bar: int = 0
    current_price: float = 0.0
    current_bar: int = 0
    quantity: float = 1.0
    
    @property
    def is_long(self) -> bool:
        return self.direction == Direction.LONG
    
    @property 
    def is_short(self) -> bool:
        return self.direction == Direction.SHORT
    
    @property
    def is_flat(self) -> bool:
        return self.direction == Direction.FLAT
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.is_flat:
            return 0.0
        elif self.is_long:
            return (self.current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - self.current_price) * self.quantity


class CSignals:
    """
    Trading signals manager.
    
    Manages all trading signals including:
    - Buy/Sell signals
    - Position tracking
    - Signal history
    - Take profit/Stop loss signals
    """
    
    def __init__(self):
        """Initialize trading signals."""
        # Current signal flags
        self.al: bool = False  # Buy signal
        self.sat: bool = False  # Sell signal  
        self.flat_ol: bool = False  # Flatten position signal
        self.pas_gec: bool = False  # Pass signal
        self.kar_al: bool = False  # Take profit signal
        self.zarar_kes: bool = False  # Stop loss signal
        
        # Signal information
        self.sinyal: str = ""  # Current signal description
        self.son_yon: str = Direction.FLAT.value  # Current direction
        self.prev_yon: str = Direction.FLAT.value  # Previous direction
        
        # Current position prices
        self.son_fiyat: float = 0.0  # Current price
        self.son_a_fiyat: float = 0.0  # Last long entry price
        self.son_s_fiyat: float = 0.0  # Last short entry price  
        self.son_f_fiyat: float = 0.0  # Last flat price
        self.son_p_fiyat: float = 0.0  # Last pass price
        
        # Previous position prices
        self.prev_fiyat: float = 0.0  # Previous price
        self.prev_a_fiyat: float = 0.0  # Previous long entry price
        self.prev_s_fiyat: float = 0.0  # Previous short entry price
        self.prev_f_fiyat: float = 0.0  # Previous flat price
        self.prev_p_fiyat: float = 0.0  # Previous pass price
        
        # Bar numbers
        self.son_bar_no: int = 0  # Current bar
        self.son_a_bar_no: int = 0  # Last long entry bar
        self.son_s_bar_no: int = 0  # Last short entry bar
        self.son_f_bar_no: int = 0  # Last flat bar
        self.son_p_bar_no: int = 0  # Last pass bar
        
        self.prev_bar_no: int = 0  # Previous bar
        self.prev_a_bar_no: int = 0  # Previous long entry bar
        self.prev_s_bar_no: int = 0  # Previous short entry bar
        self.prev_f_bar_no: int = 0  # Previous flat bar
        self.prev_p_bar_no: int = 0  # Previous pass bar
        
        # Order status
        self.emir_komut: float = 0.0  # Order command
        self.emir_status: float = 0.0  # Order status
        
        # Risk management flags
        self.kar_al_enabled: bool = False  # Take profit enabled
        self.zarar_kes_enabled: bool = False  # Stop loss enabled
        self.gun_sonu_poz_kapat_enabled: bool = False  # End of day flatten enabled
        self.time_filtering_enabled: bool = False  # Time filtering enabled
        
        # Position status flags
        self.kar_alindi: bool = False  # Take profit executed
        self.zarar_kesildi: bool = False  # Stop loss executed
        self.flat_olundu: bool = False  # Position flattened
        self.poz_acilabilir: bool = False  # Position can be opened
        self.poz_acildi: bool = False  # Position opened
        self.poz_kapatilabilir: bool = False  # Position can be closed
        self.poz_kapatildi: bool = False  # Position closed
        self.poz_acilabilir_alis: bool = False  # Can open long position
        self.poz_acilabilir_satis: bool = False  # Can open short position
        self.poz_acildi_alis: bool = False  # Long position opened
        self.poz_acildi_satis: bool = False  # Short position opened
        self.gun_sonu_poz_kapatildi: bool = False  # End of day position closed
        
        # Signal history
        self.signal_history: list[SignalInfo] = []
        
        # Current position
        self.position = PositionInfo()
    
    def reset_signals(self) -> None:
        """Reset all current signals to False."""
        self.al = False
        self.sat = False
        self.flat_ol = False
        self.pas_gec = False
        self.kar_al = False
        self.zarar_kes = False
    
    def reset_position_flags(self) -> None:
        """Reset position status flags."""
        self.kar_alindi = False
        self.zarar_kesildi = False
        self.flat_olundu = False
        self.poz_acildi = False
    
    def update_current_data(self, bar_number: int, price: float) -> None:
        """
        Update current bar and price data.
        
        Args:
            bar_number: Current bar number
            price: Current price
        """
        self.son_bar_no = bar_number
        self.son_fiyat = price
        self.position.current_bar = bar_number
        self.position.current_price = price
    
    def process_signals(self, bar_number: int, price: float) -> Optional[SignalInfo]:
        """
        Process current signals and update position.
        
        Args:
            bar_number: Current bar number
            price: Current price
            
        Returns:
            SignalInfo if a signal was processed, None otherwise
        """
        self.update_current_data(bar_number, price)
        
        # Store previous state
        self.prev_yon = self.son_yon
        self.prev_fiyat = self.son_fiyat
        self.prev_bar_no = self.son_bar_no
        
        signal_info = None
        
        # Process take profit signal
        if self.kar_al and not self.position.is_flat:
            signal_info = self._process_take_profit(bar_number, price)
            
        # Process stop loss signal  
        elif self.zarar_kes and not self.position.is_flat:
            signal_info = self._process_stop_loss(bar_number, price)
            
        # Process flatten signal
        elif self.flat_ol and not self.position.is_flat:
            signal_info = self._process_flatten(bar_number, price)
            
        # Process buy signal
        elif self.al and not self.position.is_long:
            signal_info = self._process_buy_signal(bar_number, price)
            
        # Process sell signal
        elif self.sat and not self.position.is_short:
            signal_info = self._process_sell_signal(bar_number, price)
        
        # Add signal to history if processed
        if signal_info:
            self.signal_history.append(signal_info)
            self.sinyal = str(signal_info)
        
        return signal_info
    
    def _process_buy_signal(self, bar_number: int, price: float) -> SignalInfo:
        """Process buy signal and update position."""
        # Close short position if exists
        if self.position.is_short:
            self._close_position(bar_number, price)
        
        # Open long position
        self.position.direction = Direction.LONG
        self.position.entry_price = price
        self.position.entry_bar = bar_number
        
        # Update tracking variables
        self.son_yon = Direction.LONG.value
        self.son_a_fiyat = price
        self.son_a_bar_no = bar_number
        self.poz_acildi = True
        
        return SignalInfo(
            signal_type=SignalType.BUY,
            direction=Direction.LONG,
            price=price,
            bar_number=bar_number
        )
    
    def _process_sell_signal(self, bar_number: int, price: float) -> SignalInfo:
        """Process sell signal and update position."""
        # Close long position if exists
        if self.position.is_long:
            self._close_position(bar_number, price)
        
        # Open short position
        self.position.direction = Direction.SHORT
        self.position.entry_price = price
        self.position.entry_bar = bar_number
        
        # Update tracking variables
        self.son_yon = Direction.SHORT.value
        self.son_s_fiyat = price
        self.son_s_bar_no = bar_number
        self.poz_acildi = True
        
        return SignalInfo(
            signal_type=SignalType.SELL,
            direction=Direction.SHORT,
            price=price,
            bar_number=bar_number
        )
    
    def _process_flatten(self, bar_number: int, price: float) -> SignalInfo:
        """Process flatten signal and close position."""
        signal_type = SignalType.FLAT
        direction = self.position.direction
        
        self._close_position(bar_number, price)
        self.flat_olundu = True
        
        return SignalInfo(
            signal_type=signal_type,
            direction=direction,
            price=price,
            bar_number=bar_number
        )
    
    def _process_take_profit(self, bar_number: int, price: float) -> SignalInfo:
        """Process take profit signal."""
        direction = self.position.direction
        
        self._close_position(bar_number, price)
        self.kar_alindi = True
        
        return SignalInfo(
            signal_type=SignalType.TAKE_PROFIT,
            direction=direction,
            price=price,
            bar_number=bar_number
        )
    
    def _process_stop_loss(self, bar_number: int, price: float) -> SignalInfo:
        """Process stop loss signal."""
        direction = self.position.direction
        
        self._close_position(bar_number, price)
        self.zarar_kesildi = True
        
        return SignalInfo(
            signal_type=SignalType.STOP_LOSS,
            direction=direction,
            price=price,
            bar_number=bar_number
        )
    
    def _close_position(self, bar_number: int, price: float) -> None:
        """Close current position."""
        self.position.direction = Direction.FLAT
        self.position.entry_price = 0.0
        self.position.entry_bar = 0
        
        # Update tracking variables
        self.son_yon = Direction.FLAT.value
        self.son_f_fiyat = price
        self.son_f_bar_no = bar_number
    
    def is_son_yon_a(self) -> bool:
        """Check if current direction is long."""
        return self.son_yon == Direction.LONG.value
    
    def is_son_yon_s(self) -> bool:
        """Check if current direction is short."""
        return self.son_yon == Direction.SHORT.value
    
    def is_son_yon_f(self) -> bool:
        """Check if current direction is flat."""
        return self.son_yon == Direction.FLAT.value
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signals and position."""
        return {
            'signals': {
                'al': self.al,
                'sat': self.sat, 
                'flat_ol': self.flat_ol,
                'kar_al': self.kar_al,
                'zarar_kes': self.zarar_kes
            },
            'position': {
                'direction': self.position.direction.value,
                'entry_price': self.position.entry_price,
                'current_price': self.position.current_price,
                'unrealized_pnl': self.position.unrealized_pnl,
                'entry_bar': self.position.entry_bar,
                'current_bar': self.position.current_bar
            },
            'status': {
                'son_yon': self.son_yon,
                'kar_alindi': self.kar_alindi,
                'zarar_kesildi': self.zarar_kesildi,
                'flat_olundu': self.flat_olundu
            },
            'history_count': len(self.signal_history)
        }
    
    def get_last_signals(self, count: int = 5) -> list[SignalInfo]:
        """Get last N signals from history."""
        return self.signal_history[-count:] if self.signal_history else []
    
    def __repr__(self) -> str:
        """String representation of signals."""
        return f"CSignals(position={self.position.direction.value}, signals={self.get_signal_summary()['signals']})"