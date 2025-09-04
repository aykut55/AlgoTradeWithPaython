"""
Data Lists Management for algorithmic trading.

This module contains the CLists class which handles comprehensive
data list management, collections, watchlists, symbol management,
and various list operations for trading systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Union, Callable, Iterator, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import json
import pickle
from collections import defaultdict, deque, OrderedDict
import heapq
from abc import ABC, abstractmethod

from ..core.base import SystemProtocol

T = TypeVar('T')

class ListType(Enum):
    """List type enumeration."""
    WATCHLIST = "WATCHLIST"             # Symbol watchlists
    PORTFOLIO = "PORTFOLIO"             # Portfolio holdings
    UNIVERSE = "UNIVERSE"               # Trading universe
    BLACKLIST = "BLACKLIST"             # Blacklisted symbols
    WHITELIST = "WHITELIST"             # Whitelisted symbols
    CUSTOM = "CUSTOM"                   # Custom lists
    TEMPORARY = "TEMPORARY"             # Temporary lists
    HISTORICAL = "HISTORICAL"           # Historical data lists


class ListOperation(Enum):
    """List operation types."""
    ADD = "ADD"
    REMOVE = "REMOVE"
    UPDATE = "UPDATE"
    CLEAR = "CLEAR"
    SORT = "SORT"
    FILTER = "FILTER"


class SortOrder(Enum):
    """Sort order enumeration."""
    ASCENDING = "ASC"
    DESCENDING = "DESC"


@dataclass
class ListItem(Generic[T]):
    """Generic list item with metadata."""
    
    value: T
    added_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    
    def update_value(self, new_value: T) -> None:
        """Update item value."""
        self.value = new_value
        self.updated_at = datetime.now()


@dataclass
class SymbolInfo:
    """Symbol information for watchlists."""
    
    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market: str = ""
    currency: str = "TL"
    
    # Price information
    last_price: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    
    # Technical indicators
    rsi: float = 0.0
    ma_20: float = 0.0
    ma_50: float = 0.0
    
    # Fundamental data
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    
    # List metadata
    added_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def update_price_data(self, price: float, change_pct: float, volume: int) -> None:
        """Update price data."""
        self.last_price = price
        self.change_percent = change_pct
        self.volume = volume
        self.last_updated = datetime.now()


@dataclass
class ListOperation:
    """List operation record."""
    
    operation_id: str
    timestamp: datetime
    list_name: str
    operation_type: str
    item: Any
    details: Dict[str, Any] = field(default_factory=dict)


class ManagedList(Generic[T]):
    """Generic managed list with advanced features."""
    
    def __init__(self, name: str, list_type: ListType):
        """Initialize managed list."""
        self.name = name
        self.list_type = list_type
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Data storage
        self._items: OrderedDict[str, ListItem[T]] = OrderedDict()
        self._indices: Dict[str, Dict[Any, List[str]]] = {}
        
        # Configuration
        self.max_size: Optional[int] = None
        self.auto_sort: bool = False
        self.sort_key: Optional[Callable[[T], Any]] = None
        self.sort_order = SortOrder.ASCENDING
        
        # Operations history
        self.operations: deque = deque(maxlen=100)
        self.operation_counter = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_operations': 0,
            'additions': 0,
            'removals': 0,
            'updates': 0,
            'sorts': 0,
            'filters': 0
        }
    
    def add(self, key: str, value: T, metadata: Optional[Dict[str, Any]] = None,
            priority: int = 0, tags: Optional[List[str]] = None) -> bool:
        """Add item to list."""
        with self.lock:
            # Check size limit
            if self.max_size and len(self._items) >= self.max_size:
                if key not in self._items:  # Don't reject updates
                    return False
            
            item = ListItem(
                value=value,
                metadata=metadata or {},
                priority=priority,
                tags=tags or []
            )
            
            is_update = key in self._items
            self._items[key] = item
            
            # Update indices
            self._update_indices(key, item)
            
            # Auto-sort if enabled
            if self.auto_sort and self.sort_key:
                self._sort_internal()
            
            # Record operation
            self._record_operation("ADD" if not is_update else "UPDATE", key, value)
            
            # Update statistics
            if is_update:
                self.stats['updates'] += 1
            else:
                self.stats['additions'] += 1
            
            self.updated_at = datetime.now()
            return True
    
    def remove(self, key: str) -> bool:
        """Remove item from list."""
        with self.lock:
            if key not in self._items:
                return False
            
            item = self._items[key]
            del self._items[key]
            
            # Update indices
            self._remove_from_indices(key, item)
            
            # Record operation
            self._record_operation("REMOVE", key, item.value)
            
            self.stats['removals'] += 1
            self.updated_at = datetime.now()
            return True
    
    def get(self, key: str) -> Optional[T]:
        """Get item by key."""
        with self.lock:
            item = self._items.get(key)
            return item.value if item else None
    
    def get_item(self, key: str) -> Optional[ListItem[T]]:
        """Get list item with metadata."""
        return self._items.get(key)
    
    def contains(self, key: str) -> bool:
        """Check if list contains key."""
        return key in self._items
    
    def size(self) -> int:
        """Get list size."""
        return len(self._items)
    
    def is_empty(self) -> bool:
        """Check if list is empty."""
        return len(self._items) == 0
    
    def clear(self) -> None:
        """Clear all items from list."""
        with self.lock:
            self._items.clear()
            self._indices.clear()
            
            self._record_operation("CLEAR", "", None)
            self.updated_at = datetime.now()
    
    def keys(self) -> List[str]:
        """Get all keys."""
        return list(self._items.keys())
    
    def values(self) -> List[T]:
        """Get all values."""
        return [item.value for item in self._items.values()]
    
    def items(self) -> List[Tuple[str, T]]:
        """Get all key-value pairs."""
        return [(key, item.value) for key, item in self._items.items()]
    
    def sort(self, key_func: Optional[Callable[[T], Any]] = None,
             reverse: bool = False) -> None:
        """Sort list items."""
        with self.lock:
            sort_key = key_func or self.sort_key
            if not sort_key:
                return
            
            # Sort items
            sorted_items = sorted(
                self._items.items(),
                key=lambda x: sort_key(x[1].value),
                reverse=reverse
            )
            
            # Rebuild ordered dict
            self._items = OrderedDict(sorted_items)
            
            self._record_operation("SORT", "", None)
            self.stats['sorts'] += 1
            self.updated_at = datetime.now()
    
    def _sort_internal(self) -> None:
        """Internal sort using configured parameters."""
        if self.sort_key:
            reverse = self.sort_order == SortOrder.DESCENDING
            self.sort(self.sort_key, reverse)
    
    def filter(self, predicate: Callable[[T], bool]) -> 'ManagedList[T]':
        """Filter list items and return new list."""
        filtered_list = ManagedList(f"{self.name}_filtered", self.list_type)
        
        with self.lock:
            for key, item in self._items.items():
                if predicate(item.value):
                    filtered_list.add(key, item.value, item.metadata, item.priority, item.tags)
        
        self.stats['filters'] += 1
        return filtered_list
    
    def find_by_tag(self, tag: str) -> List[Tuple[str, T]]:
        """Find items by tag."""
        results = []
        with self.lock:
            for key, item in self._items.items():
                if tag in item.tags:
                    results.append((key, item.value))
        return results
    
    def find_by_metadata(self, metadata_key: str, metadata_value: Any) -> List[Tuple[str, T]]:
        """Find items by metadata."""
        results = []
        with self.lock:
            for key, item in self._items.items():
                if item.metadata.get(metadata_key) == metadata_value:
                    results.append((key, item.value))
        return results
    
    def _update_indices(self, key: str, item: ListItem[T]) -> None:
        """Update search indices."""
        # Tag indices
        for tag in item.tags:
            if 'tags' not in self._indices:
                self._indices['tags'] = defaultdict(list)
            if key not in self._indices['tags'][tag]:
                self._indices['tags'][tag].append(key)
        
        # Metadata indices
        for meta_key, meta_value in item.metadata.items():
            index_name = f"meta_{meta_key}"
            if index_name not in self._indices:
                self._indices[index_name] = defaultdict(list)
            if key not in self._indices[index_name][meta_value]:
                self._indices[index_name][meta_value].append(key)
    
    def _remove_from_indices(self, key: str, item: ListItem[T]) -> None:
        """Remove from search indices."""
        # Tag indices
        for tag in item.tags:
            if 'tags' in self._indices and tag in self._indices['tags']:
                try:
                    self._indices['tags'][tag].remove(key)
                except ValueError:
                    pass
        
        # Metadata indices
        for meta_key, meta_value in item.metadata.items():
            index_name = f"meta_{meta_key}"
            if index_name in self._indices and meta_value in self._indices[index_name]:
                try:
                    self._indices[index_name][meta_value].remove(key)
                except ValueError:
                    pass
    
    def _record_operation(self, op_type: str, key: str, value: Any) -> None:
        """Record list operation."""
        self.operation_counter += 1
        
        operation = ListOperation(
            operation_id=f"OP_{self.operation_counter:06d}",
            timestamp=datetime.now(),
            list_name=self.name,
            operation_type=op_type,
            item=value,
            details={'key': key}
        )
        
        self.operations.append(operation)
        self.stats['total_operations'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get list statistics."""
        return {
            'name': self.name,
            'type': self.list_type.value,
            'size': len(self._items),
            'max_size': self.max_size,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'operations': self.stats.copy()
        }


class WatchList(ManagedList[SymbolInfo]):
    """Specialized watchlist for trading symbols."""
    
    def __init__(self, name: str):
        """Initialize watchlist."""
        super().__init__(name, ListType.WATCHLIST)
        self.max_size = 100  # Default watchlist size limit
        
        # Watchlist-specific settings
        self.price_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.auto_update_prices = True
        self.update_interval = 60  # seconds
    
    def add_symbol(self, symbol: str, name: str = "", sector: str = "",
                   notes: str = "", tags: Optional[List[str]] = None) -> bool:
        """Add symbol to watchlist."""
        symbol_info = SymbolInfo(
            symbol=symbol.upper(),
            name=name,
            sector=sector,
            notes=notes
        )
        
        return self.add(symbol.upper(), symbol_info, tags=tags)
    
    def update_symbol_price(self, symbol: str, price: float, change_pct: float,
                           volume: int = 0) -> bool:
        """Update symbol price data."""
        with self.lock:
            item = self.get_item(symbol.upper())
            if item and isinstance(item.value, SymbolInfo):
                item.value.update_price_data(price, change_pct, volume)
                item.updated_at = datetime.now()
                return True
        return False
    
    def add_price_alert(self, symbol: str, alert_type: str, threshold: float,
                       callback: Optional[Callable] = None) -> bool:
        """Add price alert for symbol."""
        if symbol.upper() not in self._items:
            return False
        
        alert = {
            'type': alert_type,  # 'above', 'below', 'change'
            'threshold': threshold,
            'callback': callback,
            'created_at': datetime.now(),
            'active': True
        }
        
        self.price_alerts[symbol.upper()].append(alert)
        return True
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check and trigger price alerts."""
        triggered_alerts = []
        
        with self.lock:
            for symbol, alerts in self.price_alerts.items():
                item = self.get_item(symbol)
                if not item or not isinstance(item.value, SymbolInfo):
                    continue
                
                symbol_info = item.value
                
                for alert in alerts:
                    if not alert['active']:
                        continue
                    
                    triggered = False
                    
                    if alert['type'] == 'above' and symbol_info.last_price > alert['threshold']:
                        triggered = True
                    elif alert['type'] == 'below' and symbol_info.last_price < alert['threshold']:
                        triggered = True
                    elif alert['type'] == 'change' and abs(symbol_info.change_percent) > alert['threshold']:
                        triggered = True
                    
                    if triggered:
                        alert['active'] = False
                        triggered_alerts.append({
                            'symbol': symbol,
                            'alert_type': alert['type'],
                            'threshold': alert['threshold'],
                            'current_price': symbol_info.last_price,
                            'change_percent': symbol_info.change_percent
                        })
                        
                        if alert['callback']:
                            try:
                                alert['callback'](symbol, symbol_info)
                            except Exception:
                                pass
        
        return triggered_alerts
    
    def get_top_movers(self, count: int = 10, by_change: bool = True) -> List[Tuple[str, SymbolInfo]]:
        """Get top moving symbols."""
        symbols = [(key, item.value) for key, item in self._items.items()
                  if isinstance(item.value, SymbolInfo)]
        
        if by_change:
            symbols.sort(key=lambda x: abs(x[1].change_percent), reverse=True)
        else:
            symbols.sort(key=lambda x: x[1].volume, reverse=True)
        
        return symbols[:count]
    
    def get_symbols_by_sector(self, sector: str) -> List[Tuple[str, SymbolInfo]]:
        """Get symbols filtered by sector."""
        return [(key, item.value) for key, item in self._items.items()
                if isinstance(item.value, SymbolInfo) and item.value.sector == sector]


class CLists:
    """
    Comprehensive data lists management system.
    
    Features:
    - Multiple list types (watchlists, portfolios, universes, etc.)
    - Generic managed lists with metadata
    - Symbol watchlists with price alerts
    - List operations and history tracking
    - Search and filtering capabilities
    - Import/export functionality
    - Thread-safe operations
    - Performance analytics
    """
    
    def __init__(self):
        """Initialize lists manager."""
        self.is_initialized = False
        
        # List storage
        self.managed_lists: Dict[str, ManagedList] = {}
        self.watchlists: Dict[str, WatchList] = {}
        
        # Built-in lists
        self.blacklist: ManagedList[str] = ManagedList("system_blacklist", ListType.BLACKLIST)
        self.whitelist: ManagedList[str] = ManagedList("system_whitelist", ListType.WHITELIST)
        self.universe: ManagedList[str] = ManagedList("trading_universe", ListType.UNIVERSE)
        
        # Thread safety
        self.lists_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_lists': 0,
            'total_items': 0,
            'watchlists': 0,
            'operations': 0,
            'alerts_triggered': 0
        }
        
        # Configuration
        self.auto_save = True
        self.save_interval = 300  # 5 minutes
        self.max_lists = 100
    
    def initialize(self, system: SystemProtocol) -> 'CLists':
        """
        Initialize lists manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.lists_lock:
            # Register built-in lists
            self.managed_lists["system_blacklist"] = self.blacklist
            self.managed_lists["system_whitelist"] = self.whitelist
            self.managed_lists["trading_universe"] = self.universe
            
            # Create default watchlist
            default_watchlist = self.create_watchlist("default")
            
            self.is_initialized = True
        
        return self
    
    def reset(self, system: SystemProtocol) -> 'CLists':
        """
        Reset lists manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.lists_lock:
            # Clear all lists
            for managed_list in self.managed_lists.values():
                managed_list.clear()
            
            for watchlist in self.watchlists.values():
                watchlist.clear()
            
            # Reset statistics
            self.stats = {
                'total_lists': len(self.managed_lists) + len(self.watchlists),
                'total_items': 0,
                'watchlists': len(self.watchlists),
                'operations': 0,
                'alerts_triggered': 0
            }
        
        return self
    
    # ========== List Management ==========
    
    def create_list(self, name: str, list_type: ListType,
                   max_size: Optional[int] = None) -> Optional[ManagedList]:
        """Create a new managed list."""
        with self.lists_lock:
            if name in self.managed_lists or len(self.managed_lists) >= self.max_lists:
                return None
            
            managed_list = ManagedList(name, list_type)
            if max_size:
                managed_list.max_size = max_size
            
            self.managed_lists[name] = managed_list
            self.stats['total_lists'] += 1
            
            return managed_list
    
    def create_watchlist(self, name: str) -> Optional[WatchList]:
        """Create a new watchlist."""
        with self.lists_lock:
            if name in self.watchlists:
                return None
            
            watchlist = WatchList(name)
            self.watchlists[name] = watchlist
            self.stats['total_lists'] += 1
            self.stats['watchlists'] += 1
            
            return watchlist
    
    def get_list(self, name: str) -> Optional[ManagedList]:
        """Get managed list by name."""
        return self.managed_lists.get(name)
    
    def get_watchlist(self, name: str) -> Optional[WatchList]:
        """Get watchlist by name."""
        return self.watchlists.get(name)
    
    def remove_list(self, name: str) -> bool:
        """Remove a list."""
        with self.lists_lock:
            removed = False
            
            if name in self.managed_lists:
                del self.managed_lists[name]
                removed = True
            
            if name in self.watchlists:
                del self.watchlists[name]
                removed = True
                self.stats['watchlists'] -= 1
            
            if removed:
                self.stats['total_lists'] -= 1
            
            return removed
    
    def list_exists(self, name: str) -> bool:
        """Check if list exists."""
        return name in self.managed_lists or name in self.watchlists
    
    def get_list_names(self, list_type: Optional[ListType] = None) -> List[str]:
        """Get names of all lists, optionally filtered by type."""
        names = []
        
        for name, managed_list in self.managed_lists.items():
            if list_type is None or managed_list.list_type == list_type:
                names.append(name)
        
        if list_type is None or list_type == ListType.WATCHLIST:
            names.extend(self.watchlists.keys())
        
        return names
    
    # ========== Built-in Lists Operations ==========
    
    def add_to_blacklist(self, symbol: str, reason: str = "") -> bool:
        """Add symbol to blacklist."""
        return self.blacklist.add(
            symbol.upper(), 
            symbol.upper(),
            metadata={'reason': reason}
        )
    
    def add_to_whitelist(self, symbol: str, reason: str = "") -> bool:
        """Add symbol to whitelist."""
        return self.whitelist.add(
            symbol.upper(),
            symbol.upper(),
            metadata={'reason': reason}
        )
    
    def add_to_universe(self, symbol: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add symbol to trading universe."""
        return self.universe.add(symbol.upper(), symbol.upper(), metadata)
    
    def is_blacklisted(self, symbol: str) -> bool:
        """Check if symbol is blacklisted."""
        return self.blacklist.contains(symbol.upper())
    
    def is_whitelisted(self, symbol: str) -> bool:
        """Check if symbol is whitelisted."""
        return self.whitelist.contains(symbol.upper())
    
    def is_in_universe(self, symbol: str) -> bool:
        """Check if symbol is in trading universe."""
        return self.universe.contains(symbol.upper())
    
    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if symbol can be traded (not blacklisted)."""
        return not self.is_blacklisted(symbol)
    
    # ========== Bulk Operations ==========
    
    def add_symbols_to_watchlist(self, watchlist_name: str,
                                symbols: List[str]) -> Dict[str, bool]:
        """Add multiple symbols to watchlist."""
        watchlist = self.get_watchlist(watchlist_name)
        if not watchlist:
            return {}
        
        results = {}
        for symbol in symbols:
            results[symbol] = watchlist.add_symbol(symbol)
        
        return results
    
    def remove_symbols_from_watchlist(self, watchlist_name: str,
                                    symbols: List[str]) -> Dict[str, bool]:
        """Remove multiple symbols from watchlist."""
        watchlist = self.get_watchlist(watchlist_name)
        if not watchlist:
            return {}
        
        results = {}
        for symbol in symbols:
            results[symbol] = watchlist.remove(symbol.upper())
        
        return results
    
    def update_watchlist_prices(self, watchlist_name: str,
                               price_data: Dict[str, Tuple[float, float, int]]) -> int:
        """Update prices for multiple symbols in watchlist."""
        watchlist = self.get_watchlist(watchlist_name)
        if not watchlist:
            return 0
        
        updated_count = 0
        for symbol, (price, change_pct, volume) in price_data.items():
            if watchlist.update_symbol_price(symbol, price, change_pct, volume):
                updated_count += 1
        
        return updated_count
    
    # ========== Import/Export ==========
    
    def export_list_to_dict(self, name: str) -> Optional[Dict[str, Any]]:
        """Export list to dictionary format."""
        managed_list = self.get_list(name) or self.get_watchlist(name)
        if not managed_list:
            return None
        
        export_data = {
            'name': managed_list.name,
            'type': managed_list.list_type.value,
            'created_at': managed_list.created_at.isoformat(),
            'updated_at': managed_list.updated_at.isoformat(),
            'items': {}
        }
        
        for key, item in managed_list._items.items():
            if isinstance(item.value, SymbolInfo):
                # Special handling for SymbolInfo
                export_data['items'][key] = {
                    'symbol': item.value.symbol,
                    'name': item.value.name,
                    'sector': item.value.sector,
                    'last_price': item.value.last_price,
                    'change_percent': item.value.change_percent,
                    'added_at': item.added_at.isoformat(),
                    'metadata': item.metadata,
                    'tags': item.tags
                }
            else:
                export_data['items'][key] = {
                    'value': item.value,
                    'added_at': item.added_at.isoformat(),
                    'metadata': item.metadata,
                    'tags': item.tags,
                    'priority': item.priority
                }
        
        return export_data
    
    def import_list_from_dict(self, data: Dict[str, Any]) -> bool:
        """Import list from dictionary format."""
        try:
            name = data['name']
            list_type = ListType(data['type'])
            
            # Create list
            if list_type == ListType.WATCHLIST:
                new_list = self.create_watchlist(name)
            else:
                new_list = self.create_list(name, list_type)
            
            if not new_list:
                return False
            
            # Import items
            for key, item_data in data['items'].items():
                if 'symbol' in item_data:  # SymbolInfo
                    symbol_info = SymbolInfo(
                        symbol=item_data['symbol'],
                        name=item_data.get('name', ''),
                        sector=item_data.get('sector', ''),
                        last_price=item_data.get('last_price', 0.0),
                        change_percent=item_data.get('change_percent', 0.0)
                    )
                    new_list.add(key, symbol_info, 
                                item_data.get('metadata', {}),
                                tags=item_data.get('tags', []))
                else:
                    new_list.add(key, item_data['value'],
                                item_data.get('metadata', {}),
                                item_data.get('priority', 0),
                                item_data.get('tags', []))
            
            return True
            
        except Exception:
            return False
    
    def export_to_json(self, name: str) -> Optional[str]:
        """Export list to JSON string."""
        data = self.export_list_to_dict(name)
        if data:
            return json.dumps(data, indent=2, default=str)
        return None
    
    def import_from_json(self, json_str: str) -> bool:
        """Import list from JSON string."""
        try:
            data = json.loads(json_str)
            return self.import_list_from_dict(data)
        except Exception:
            return False
    
    # ========== Information and Analytics ==========
    
    def get_lists_summary(self) -> Dict[str, Any]:
        """Get comprehensive lists summary."""
        with self.lists_lock:
            total_items = sum(lst.size() for lst in self.managed_lists.values())
            total_items += sum(wl.size() for wl in self.watchlists.values())
            
            lists_by_type = defaultdict(int)
            for managed_list in self.managed_lists.values():
                lists_by_type[managed_list.list_type.value] += 1
            lists_by_type[ListType.WATCHLIST.value] = len(self.watchlists)
            
            return {
                'total_lists': len(self.managed_lists) + len(self.watchlists),
                'total_items': total_items,
                'lists_by_type': dict(lists_by_type),
                'built_in_lists': {
                    'blacklist_size': self.blacklist.size(),
                    'whitelist_size': self.whitelist.size(),
                    'universe_size': self.universe.size()
                },
                'statistics': self.stats.copy()
            }
    
    def get_list_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific list."""
        managed_list = self.get_list(name) or self.get_watchlist(name)
        if not managed_list:
            return None
        
        return managed_list.get_statistics()
    
    def search_across_lists(self, query: str, list_types: Optional[List[ListType]] = None) -> Dict[str, List[str]]:
        """Search for items across multiple lists."""
        results = defaultdict(list)
        query_lower = query.lower()
        
        # Search managed lists
        for name, managed_list in self.managed_lists.items():
            if list_types and managed_list.list_type not in list_types:
                continue
            
            for key, item in managed_list._items.items():
                if (query_lower in key.lower() or 
                    query_lower in str(item.value).lower() or
                    any(query_lower in tag.lower() for tag in item.tags)):
                    results[name].append(key)
        
        # Search watchlists
        if not list_types or ListType.WATCHLIST in list_types:
            for name, watchlist in self.watchlists.items():
                for key, item in watchlist._items.items():
                    symbol_info = item.value
                    if (query_lower in symbol_info.symbol.lower() or
                        query_lower in symbol_info.name.lower() or
                        query_lower in symbol_info.sector.lower()):
                        results[name].append(key)
        
        return dict(results)
    
    def get_duplicate_items(self) -> Dict[str, List[str]]:
        """Find duplicate items across all lists."""
        item_locations = defaultdict(list)
        
        # Check managed lists
        for list_name, managed_list in self.managed_lists.items():
            for key in managed_list.keys():
                item_locations[key].append(list_name)
        
        # Check watchlists
        for list_name, watchlist in self.watchlists.items():
            for key in watchlist.keys():
                item_locations[key].append(list_name)
        
        # Return only duplicates
        return {item: locations for item, locations in item_locations.items() 
                if len(locations) > 1}
    
    def __repr__(self) -> str:
        """String representation."""
        total_lists = len(self.managed_lists) + len(self.watchlists)
        return f"CLists(lists={total_lists}, watchlists={len(self.watchlists)})"