"""
Composite System Manager for algorithmic trading.

This module contains the CBirlesikSistemManager class which orchestrates
multiple trading systems, manages system combinations, portfolio allocation,
and provides unified system execution and monitoring capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict
import numpy as np
import pandas as pd

from ..core.base import SystemProtocol, MarketData, TradingDecision, SystemState, PositionSide
from ..trading.trader import CTrader
from ..trading.signals import CSignals
from ..system.system_wrapper import CSystemWrapper
from ..portfolio.asset_manager import CVarlikManager


class SystemCombinationMode(Enum):
    """System combination modes."""
    PARALLEL = "PARALLEL"           # Systems run in parallel, independent decisions
    SEQUENTIAL = "SEQUENTIAL"       # Systems run sequentially, cascading decisions
    CONSENSUS = "CONSENSUS"         # Require consensus between systems
    WEIGHTED_VOTE = "WEIGHTED_VOTE" # Weighted voting system
    MASTER_SLAVE = "MASTER_SLAVE"   # One master system, others provide confirmation
    PORTFOLIO = "PORTFOLIO"         # Portfolio allocation across systems


class SystemPriority(Enum):
    """System priority levels."""
    CRITICAL = 1    # Highest priority - can override others
    HIGH = 2        # High priority
    NORMAL = 3      # Normal priority
    LOW = 4         # Low priority
    MONITOR = 5     # Monitor only, no trading decisions


@dataclass
class SystemConfiguration:
    """Configuration for individual system in composite manager."""
    
    system_id: str
    system: CSystemWrapper
    priority: SystemPriority = SystemPriority.NORMAL
    weight: float = 1.0
    allocation_percentage: float = 0.0
    is_active: bool = True
    is_master: bool = False
    
    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk limits
    max_position_size: float = 0.0
    daily_loss_limit: float = 0.0
    max_trades_per_day: int = 0
    
    # Timing
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    def update_performance(self, trade_pnl: float, is_winning: bool) -> None:
        """Update system performance metrics."""
        self.total_trades += 1
        if is_winning:
            self.winning_trades += 1
        self.total_pnl += trade_pnl


@dataclass
class CompositeSignal:
    """Composite trading signal from multiple systems."""
    
    timestamp: datetime
    symbol: str
    composite_decision: TradingDecision
    confidence: float = 0.0
    
    # Individual system signals
    system_signals: Dict[str, TradingDecision] = field(default_factory=dict)
    system_weights: Dict[str, float] = field(default_factory=dict)
    system_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Consensus information
    consensus_strength: float = 0.0
    participating_systems: List[str] = field(default_factory=list)
    conflicting_systems: List[str] = field(default_factory=list)


class CBirlesikSistemManager:
    """
    Composite System Manager - orchestrates multiple trading systems.
    
    Features:
    - Multiple system combination modes
    - Portfolio allocation across systems
    - Consensus-based decision making
    - Risk management across systems
    - Performance monitoring and optimization
    - Dynamic system weighting
    """
    
    def __init__(self):
        """Initialize composite system manager."""
        self.systems: Dict[str, SystemConfiguration] = {}
        self.combination_mode = SystemCombinationMode.PARALLEL
        self.is_initialized = False
        self.is_running = False
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.system_lock = threading.RLock()
        
        # Performance tracking
        self.composite_performance = {
            'total_signals': 0,
            'executed_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        
        # Risk management
        self.global_risk_limits = {
            'max_total_exposure': 1000000.0,
            'max_systems_active': 10,
            'min_consensus_threshold': 0.6,
            'max_daily_trades': 100
        }
        
        # Signal history
        self.signal_history: List[CompositeSignal] = []
        self.max_history_size = 1000
    
    def initialize(self, system: SystemProtocol) -> 'CBirlesikSistemManager':
        """
        Initialize composite system manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CBirlesikSistemManager':
        """
        Reset composite system manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.system_lock:
            self.signal_history.clear()
            self.composite_performance = {
                'total_signals': 0,
                'executed_trades': 0,
                'total_pnl': 0.0,
                'start_time': datetime.now(),
                'last_update': datetime.now()
            }
        return self
    
    # ========== System Management ==========
    
    def add_system(self, system_id: str, system: CSystemWrapper,
                   priority: SystemPriority = SystemPriority.NORMAL,
                   weight: float = 1.0, allocation: float = 0.0) -> 'CBirlesikSistemManager':
        """
        Add a trading system to composite manager.
        
        Args:
            system_id: Unique identifier for system
            system: System wrapper instance
            priority: System priority level
            weight: System weight for voting
            allocation: Portfolio allocation percentage
            
        Returns:
            Self for method chaining
        """
        with self.system_lock:
            config = SystemConfiguration(
                system_id=system_id,
                system=system,
                priority=priority,
                weight=weight,
                allocation_percentage=allocation
            )
            self.systems[system_id] = config
        
        return self
    
    def remove_system(self, system_id: str) -> bool:
        """
        Remove a system from composite manager.
        
        Args:
            system_id: System to remove
            
        Returns:
            True if removed successfully
        """
        with self.system_lock:
            return self.systems.pop(system_id, None) is not None
    
    def activate_system(self, system_id: str) -> bool:
        """Activate a system."""
        if system_id in self.systems:
            self.systems[system_id].is_active = True
            return True
        return False
    
    def deactivate_system(self, system_id: str) -> bool:
        """Deactivate a system."""
        if system_id in self.systems:
            self.systems[system_id].is_active = False
            return True
        return False
    
    def set_master_system(self, system_id: str) -> bool:
        """
        Set a system as master (for master-slave mode).
        
        Args:
            system_id: System to set as master
            
        Returns:
            True if successful
        """
        if system_id not in self.systems:
            return False
        
        with self.system_lock:
            # Remove master status from all systems
            for config in self.systems.values():
                config.is_master = False
            
            # Set new master
            self.systems[system_id].is_master = True
            self.combination_mode = SystemCombinationMode.MASTER_SLAVE
        
        return True
    
    def set_combination_mode(self, mode: SystemCombinationMode) -> 'CBirlesikSistemManager':
        """Set system combination mode."""
        self.combination_mode = mode
        return self
    
    # ========== Signal Generation ==========
    
    def generate_composite_signal(self, market_data: MarketData) -> Optional[CompositeSignal]:
        """
        Generate composite signal from all active systems.
        
        Args:
            market_data: Current market data
            
        Returns:
            Composite signal or None
        """
        if not self.is_initialized or not self.systems:
            return None
        
        active_systems = {sid: config for sid, config in self.systems.items() 
                         if config.is_active}
        
        if not active_systems:
            return None
        
        # Collect signals from all systems
        system_signals = self._collect_system_signals(market_data, active_systems)
        
        if not system_signals:
            return None
        
        # Generate composite decision based on combination mode
        composite_signal = self._combine_signals(system_signals, market_data)
        
        if composite_signal:
            self._store_signal_history(composite_signal)
            self.composite_performance['total_signals'] += 1
        
        return composite_signal
    
    def _collect_system_signals(self, market_data: MarketData, 
                               active_systems: Dict[str, SystemConfiguration]) -> Dict[str, TradingDecision]:
        """Collect signals from all active systems."""
        system_signals = {}
        
        if self.combination_mode == SystemCombinationMode.PARALLEL:
            # Collect signals in parallel
            futures = {}
            for system_id, config in active_systems.items():
                future = self.executor.submit(self._get_system_signal, config.system, market_data)
                futures[system_id] = future
            
            for system_id, future in futures.items():
                try:
                    signal = future.result(timeout=1.0)  # 1 second timeout
                    if signal and signal != TradingDecision.HOLD:
                        system_signals[system_id] = signal
                        active_systems[system_id].last_signal_time = datetime.now()
                except Exception:
                    continue  # Skip failed systems
        
        else:
            # Collect signals sequentially
            for system_id, config in active_systems.items():
                try:
                    signal = self._get_system_signal(config.system, market_data)
                    if signal and signal != TradingDecision.HOLD:
                        system_signals[system_id] = signal
                        config.last_signal_time = datetime.now()
                except Exception:
                    continue
        
        return system_signals
    
    def _get_system_signal(self, system: CSystemWrapper, market_data: MarketData) -> Optional[TradingDecision]:
        """Get signal from individual system."""
        try:
            return system.get_trading_decision(market_data)
        except Exception:
            return None
    
    def _combine_signals(self, system_signals: Dict[str, TradingDecision], 
                        market_data: MarketData) -> Optional[CompositeSignal]:
        """Combine individual system signals into composite signal."""
        if not system_signals:
            return None
        
        composite_signal = CompositeSignal(
            timestamp=datetime.now(),
            symbol=market_data.symbol,
            composite_decision=TradingDecision.HOLD,
            system_signals=system_signals.copy()
        )
        
        if self.combination_mode == SystemCombinationMode.PARALLEL:
            composite_signal.composite_decision = self._parallel_combination(system_signals)
        
        elif self.combination_mode == SystemCombinationMode.CONSENSUS:
            composite_signal = self._consensus_combination(system_signals, composite_signal)
        
        elif self.combination_mode == SystemCombinationMode.WEIGHTED_VOTE:
            composite_signal = self._weighted_vote_combination(system_signals, composite_signal)
        
        elif self.combination_mode == SystemCombinationMode.MASTER_SLAVE:
            composite_signal = self._master_slave_combination(system_signals, composite_signal)
        
        elif self.combination_mode == SystemCombinationMode.PORTFOLIO:
            composite_signal = self._portfolio_combination(system_signals, composite_signal)
        
        # Calculate composite confidence
        composite_signal.confidence = self._calculate_confidence(system_signals)
        composite_signal.participating_systems = list(system_signals.keys())
        
        return composite_signal
    
    def _parallel_combination(self, system_signals: Dict[str, TradingDecision]) -> TradingDecision:
        """Simple majority vote for parallel combination."""
        buy_count = sum(1 for signal in system_signals.values() if signal == TradingDecision.BUY)
        sell_count = sum(1 for signal in system_signals.values() if signal == TradingDecision.SELL)
        
        if buy_count > sell_count:
            return TradingDecision.BUY
        elif sell_count > buy_count:
            return TradingDecision.SELL
        else:
            return TradingDecision.HOLD
    
    def _consensus_combination(self, system_signals: Dict[str, TradingDecision], 
                             composite_signal: CompositeSignal) -> CompositeSignal:
        """Consensus-based combination requiring minimum agreement."""
        total_systems = len(system_signals)
        buy_count = sum(1 for signal in system_signals.values() if signal == TradingDecision.BUY)
        sell_count = sum(1 for signal in system_signals.values() if signal == TradingDecision.SELL)
        
        buy_consensus = buy_count / total_systems
        sell_consensus = sell_count / total_systems
        
        min_threshold = self.global_risk_limits['min_consensus_threshold']
        
        if buy_consensus >= min_threshold:
            composite_signal.composite_decision = TradingDecision.BUY
            composite_signal.consensus_strength = buy_consensus
        elif sell_consensus >= min_threshold:
            composite_signal.composite_decision = TradingDecision.SELL
            composite_signal.consensus_strength = sell_consensus
        else:
            composite_signal.composite_decision = TradingDecision.HOLD
            composite_signal.consensus_strength = 0.0
        
        return composite_signal
    
    def _weighted_vote_combination(self, system_signals: Dict[str, TradingDecision],
                                 composite_signal: CompositeSignal) -> CompositeSignal:
        """Weighted voting combination based on system weights and performance."""
        total_weight = 0.0
        weighted_buy = 0.0
        weighted_sell = 0.0
        
        for system_id, signal in system_signals.items():
            if system_id in self.systems:
                config = self.systems[system_id]
                # Dynamic weight based on performance
                performance_multiplier = max(0.1, min(2.0, 1.0 + config.sharpe_ratio))
                effective_weight = config.weight * performance_multiplier
                
                composite_signal.system_weights[system_id] = effective_weight
                total_weight += effective_weight
                
                if signal == TradingDecision.BUY:
                    weighted_buy += effective_weight
                elif signal == TradingDecision.SELL:
                    weighted_sell += effective_weight
        
        if total_weight > 0:
            buy_strength = weighted_buy / total_weight
            sell_strength = weighted_sell / total_weight
            
            if buy_strength > sell_strength and buy_strength > 0.5:
                composite_signal.composite_decision = TradingDecision.BUY
                composite_signal.confidence = buy_strength
            elif sell_strength > buy_strength and sell_strength > 0.5:
                composite_signal.composite_decision = TradingDecision.SELL
                composite_signal.confidence = sell_strength
        
        return composite_signal
    
    def _master_slave_combination(self, system_signals: Dict[str, TradingDecision],
                                composite_signal: CompositeSignal) -> CompositeSignal:
        """Master-slave combination where master system dominates."""
        master_system_id = None
        master_signal = None
        
        # Find master system
        for system_id, config in self.systems.items():
            if config.is_master and system_id in system_signals:
                master_system_id = system_id
                master_signal = system_signals[system_id]
                break
        
        if master_signal:
            composite_signal.composite_decision = master_signal
            
            # Check for slave confirmation
            slave_confirmations = 0
            slave_conflicts = 0
            
            for system_id, signal in system_signals.items():
                if system_id != master_system_id:
                    if signal == master_signal:
                        slave_confirmations += 1
                    else:
                        slave_conflicts += 1
                        composite_signal.conflicting_systems.append(system_id)
            
            # Adjust confidence based on slave confirmations
            total_slaves = len(system_signals) - 1
            if total_slaves > 0:
                confirmation_ratio = slave_confirmations / total_slaves
                composite_signal.confidence = 0.7 + (0.3 * confirmation_ratio)
            else:
                composite_signal.confidence = 0.7
        
        return composite_signal
    
    def _portfolio_combination(self, system_signals: Dict[str, TradingDecision],
                             composite_signal: CompositeSignal) -> CompositeSignal:
        """Portfolio-based combination using allocation percentages."""
        total_buy_allocation = 0.0
        total_sell_allocation = 0.0
        
        for system_id, signal in system_signals.items():
            if system_id in self.systems:
                allocation = self.systems[system_id].allocation_percentage
                if signal == TradingDecision.BUY:
                    total_buy_allocation += allocation
                elif signal == TradingDecision.SELL:
                    total_sell_allocation += allocation
        
        if total_buy_allocation > total_sell_allocation and total_buy_allocation > 0.3:
            composite_signal.composite_decision = TradingDecision.BUY
            composite_signal.confidence = min(1.0, total_buy_allocation)
        elif total_sell_allocation > total_buy_allocation and total_sell_allocation > 0.3:
            composite_signal.composite_decision = TradingDecision.SELL
            composite_signal.confidence = min(1.0, total_sell_allocation)
        
        return composite_signal
    
    def _calculate_confidence(self, system_signals: Dict[str, TradingDecision]) -> float:
        """Calculate overall confidence in composite signal."""
        if not system_signals:
            return 0.0
        
        total_systems = len(system_signals)
        signal_counts = defaultdict(int)
        
        for signal in system_signals.values():
            signal_counts[signal] += 1
        
        if not signal_counts:
            return 0.0
        
        max_count = max(signal_counts.values())
        return max_count / total_systems
    
    # ========== Performance Monitoring ==========
    
    def update_system_performance(self, system_id: str, trade_pnl: float, is_winning: bool) -> None:
        """Update performance metrics for a system."""
        if system_id in self.systems:
            self.systems[system_id].update_performance(trade_pnl, is_winning)
            self.composite_performance['total_pnl'] += trade_pnl
            self.composite_performance['executed_trades'] += 1
    
    def get_system_performance(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a system."""
        if system_id not in self.systems:
            return None
        
        config = self.systems[system_id]
        win_rate = config.winning_trades / max(1, config.total_trades)
        
        return {
            'system_id': system_id,
            'total_trades': config.total_trades,
            'winning_trades': config.winning_trades,
            'win_rate': win_rate,
            'total_pnl': config.total_pnl,
            'sharpe_ratio': config.sharpe_ratio,
            'max_drawdown': config.max_drawdown,
            'is_active': config.is_active,
            'weight': config.weight,
            'allocation': config.allocation_percentage
        }
    
    def get_composite_performance(self) -> Dict[str, Any]:
        """Get overall composite system performance."""
        runtime = datetime.now() - self.composite_performance['start_time']
        
        return {
            'total_signals': self.composite_performance['total_signals'],
            'executed_trades': self.composite_performance['executed_trades'],
            'total_pnl': self.composite_performance['total_pnl'],
            'runtime_hours': runtime.total_seconds() / 3600,
            'signals_per_hour': self.composite_performance['total_signals'] / max(1, runtime.total_seconds() / 3600),
            'active_systems': len([s for s in self.systems.values() if s.is_active]),
            'total_systems': len(self.systems),
            'combination_mode': self.combination_mode.value
        }
    
    def optimize_system_weights(self) -> 'CBirlesikSistemManager':
        """Optimize system weights based on historical performance."""
        total_performance_score = 0.0
        system_scores = {}
        
        # Calculate performance scores
        for system_id, config in self.systems.items():
            if config.total_trades > 10:  # Minimum trades for statistical significance
                win_rate = config.winning_trades / config.total_trades
                avg_pnl_per_trade = config.total_pnl / config.total_trades
                
                # Combined score: win rate + average P&L + Sharpe ratio
                performance_score = (win_rate * 0.4) + (avg_pnl_per_trade * 0.4) + (config.sharpe_ratio * 0.2)
                system_scores[system_id] = max(0.1, performance_score)  # Minimum weight
                total_performance_score += system_scores[system_id]
        
        # Normalize weights
        if total_performance_score > 0:
            for system_id, score in system_scores.items():
                self.systems[system_id].weight = score / total_performance_score
        
        return self
    
    # ========== Utility Methods ==========
    
    def _store_signal_history(self, signal: CompositeSignal) -> None:
        """Store signal in history with size limit."""
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history_size:
            self.signal_history.pop(0)
    
    def get_signal_history(self, limit: int = 100) -> List[CompositeSignal]:
        """Get recent signal history."""
        return self.signal_history[-limit:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'combination_mode': self.combination_mode.value,
            'total_systems': len(self.systems),
            'active_systems': len([s for s in self.systems.values() if s.is_active]),
            'master_system': next((sid for sid, cfg in self.systems.items() if cfg.is_master), None),
            'risk_limits': self.global_risk_limits,
            'signal_history_size': len(self.signal_history)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export system configuration for persistence."""
        return {
            'combination_mode': self.combination_mode.value,
            'risk_limits': self.global_risk_limits,
            'systems': {
                system_id: {
                    'priority': config.priority.value,
                    'weight': config.weight,
                    'allocation_percentage': config.allocation_percentage,
                    'is_active': config.is_active,
                    'is_master': config.is_master,
                    'performance': {
                        'total_trades': config.total_trades,
                        'winning_trades': config.winning_trades,
                        'total_pnl': config.total_pnl,
                        'sharpe_ratio': config.sharpe_ratio
                    }
                }
                for system_id, config in self.systems.items()
            }
        }
    
    def start_monitoring(self) -> bool:
        """Start system monitoring."""
        if not self.is_initialized:
            return False
        
        self.is_running = True
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop system monitoring."""
        self.is_running = False
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CBirlesikSistemManager(systems={len(self.systems)}, "
                f"mode={self.combination_mode.value}, active={len([s for s in self.systems.values() if s.is_active])})")