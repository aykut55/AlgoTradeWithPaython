"""
Shared Memory operations for algorithmic trading.

This module contains the CSharedMemory class which handles inter-process
communication, data sharing between components, memory-mapped operations,
and high-performance data exchange for trading systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Union, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import multiprocessing
import mmap
import struct
import json
import pickle
import hashlib
from collections import defaultdict, deque
import time

from ..core.base import SystemProtocol

T = TypeVar('T')

class SharedMemoryType(Enum):
    """Shared memory types."""
    MARKET_DATA = "MARKET_DATA"         # Real-time market data
    TRADING_SIGNALS = "TRADING_SIGNALS" # Trading signals
    POSITIONS = "POSITIONS"             # Current positions
    ORDERS = "ORDERS"                   # Order book
    SYSTEM_STATUS = "SYSTEM_STATUS"     # System status information
    PERFORMANCE = "PERFORMANCE"         # Performance metrics
    CONFIGURATION = "CONFIGURATION"    # System configuration
    ALERTS = "ALERTS"                   # System alerts
    LOGS = "LOGS"                       # System logs


class AccessMode(Enum):
    """Memory access modes."""
    READ_ONLY = "READ_ONLY"
    WRITE_ONLY = "WRITE_ONLY"
    READ_WRITE = "READ_WRITE"


class SyncMode(Enum):
    """Synchronization modes."""
    NONE = "NONE"                       # No synchronization
    MUTEX = "MUTEX"                     # Mutex locking
    SEMAPHORE = "SEMAPHORE"             # Semaphore counting
    EVENT = "EVENT"                     # Event signaling
    CONDITION = "CONDITION"             # Condition variables


@dataclass
class SharedMemorySegment:
    """Shared memory segment configuration."""
    
    name: str
    memory_type: SharedMemoryType
    size: int
    access_mode: AccessMode = AccessMode.READ_WRITE
    sync_mode: SyncMode = SyncMode.MUTEX
    
    # Memory management
    auto_resize: bool = True
    max_size: int = 0  # 0 = unlimited
    cleanup_on_exit: bool = True
    
    # Performance settings
    buffer_size: int = 4096
    write_cache: bool = True
    compression: bool = False
    
    # Metadata
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.max_size == 0:
            self.max_size = self.size * 10  # Default 10x expansion


@dataclass
class MemoryBlock(Generic[T]):
    """Memory block for typed data storage."""
    
    block_id: str
    data_type: str
    data: T
    timestamp: datetime
    size: int = 0
    checksum: str = ""
    
    def __post_init__(self):
        if isinstance(self.data, (str, bytes)):
            self.size = len(self.data)
        elif hasattr(self.data, '__sizeof__'):
            self.size = self.data.__sizeof__()
        
        # Calculate checksum for data integrity
        if self.data is not None:
            data_str = str(self.data).encode('utf-8')
            self.checksum = hashlib.md5(data_str).hexdigest()


class CSharedMemory:
    """
    Comprehensive shared memory management system.
    
    Features:
    - Inter-process communication
    - Memory-mapped file operations
    - Thread-safe data sharing
    - High-performance data exchange
    - Automatic memory management
    - Data compression and integrity checking
    - Multiple synchronization modes
    - Real-time market data sharing
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize shared memory manager."""
        self.is_initialized = False
        
        # Memory segments
        self.segments: Dict[str, SharedMemorySegment] = {}
        self.memory_maps: Dict[str, mmap.mmap] = {}
        self.memory_files: Dict[str, Any] = {}
        
        # Synchronization primitives
        self.locks: Dict[str, threading.RLock] = {}
        self.semaphores: Dict[str, threading.Semaphore] = {}
        self.events: Dict[str, threading.Event] = {}
        self.conditions: Dict[str, threading.Condition] = {}
        
        # Process management
        self.process_manager = multiprocessing.Manager()
        self.shared_dicts: Dict[str, Any] = {}
        self.shared_queues: Dict[str, multiprocessing.Queue] = {}
        
        # Data storage
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.access_stats = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0,
            'last_operation_time': 0.0
        }
        
        # Configuration
        self.default_segment_size = 1024 * 1024  # 1MB
        self.enable_compression = False
        self.enable_integrity_check = True
        self.cleanup_interval = 300  # 5 minutes
        
        # Background tasks
        self.cleanup_thread: Optional[threading.Thread] = None
        self.running = False
    
    def initialize(self, system: SystemProtocol) -> 'CSharedMemory':
        """
        Initialize shared memory manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        # Create default memory segments
        self._create_default_segments()
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        self.is_initialized = True
        return self
    
    def reset(self, system: SystemProtocol) -> 'CSharedMemory':
        """
        Reset shared memory manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        self.running = False
        
        # Close all memory maps
        for mmap_obj in self.memory_maps.values():
            mmap_obj.close()
        
        # Close memory files
        for file_obj in self.memory_files.values():
            file_obj.close()
        
        # Clear all data
        self.segments.clear()
        self.memory_maps.clear()
        self.memory_files.clear()
        self.memory_blocks.clear()
        self.subscribers.clear()
        
        return self
    
    def _create_default_segments(self) -> None:
        """Create default memory segments for common use cases."""
        default_segments = [
            SharedMemorySegment(
                name="market_data",
                memory_type=SharedMemoryType.MARKET_DATA,
                size=self.default_segment_size * 2,  # 2MB for market data
                access_mode=AccessMode.READ_WRITE,
                sync_mode=SyncMode.MUTEX
            ),
            SharedMemorySegment(
                name="trading_signals",
                memory_type=SharedMemoryType.TRADING_SIGNALS,
                size=self.default_segment_size // 2,  # 512KB for signals
                access_mode=AccessMode.READ_WRITE,
                sync_mode=SyncMode.EVENT
            ),
            SharedMemorySegment(
                name="positions",
                memory_type=SharedMemoryType.POSITIONS,
                size=self.default_segment_size // 4,  # 256KB for positions
                access_mode=AccessMode.READ_WRITE,
                sync_mode=SyncMode.MUTEX
            ),
            SharedMemorySegment(
                name="system_status",
                memory_type=SharedMemoryType.SYSTEM_STATUS,
                size=self.default_segment_size // 8,  # 128KB for status
                access_mode=AccessMode.READ_WRITE,
                sync_mode=SyncMode.NONE
            )
        ]
        
        for segment in default_segments:
            self.create_segment(segment)
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self) -> None:
        """Background worker for memory cleanup."""
        while self.running:
            try:
                self._perform_cleanup()
                time.sleep(self.cleanup_interval)
            except Exception:
                pass  # Ignore cleanup errors
    
    def _perform_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        current_time = datetime.now()
        cleanup_threshold = timedelta(hours=1)
        
        # Clean up old memory blocks
        blocks_to_remove = []
        for block_id, block in self.memory_blocks.items():
            if current_time - block.timestamp > cleanup_threshold:
                blocks_to_remove.append(block_id)
        
        for block_id in blocks_to_remove:
            del self.memory_blocks[block_id]
        
        # Update access statistics
        for segment in self.segments.values():
            if segment.last_accessed:
                time_since_access = current_time - segment.last_accessed
                if time_since_access > cleanup_threshold:
                    # Consider compressing or paging out unused segments
                    pass
    
    # ========== Segment Management ==========
    
    def create_segment(self, segment: SharedMemorySegment) -> bool:
        """
        Create a new shared memory segment.
        
        Args:
            segment: Segment configuration
            
        Returns:
            True if created successfully
        """
        if segment.name in self.segments:
            return False
        
        try:
            # Create memory-mapped file
            filename = f"shm_{segment.name}_{os.getpid()}.dat"
            filepath = os.path.join(os.path.dirname(__file__), filename)
            
            # Create file with specified size
            with open(filepath, 'wb') as f:
                f.write(b'\x00' * segment.size)
            
            # Open for memory mapping
            file_obj = open(filepath, 'r+b')
            mmap_obj = mmap.mmap(file_obj.fileno(), segment.size)
            
            # Store references
            self.segments[segment.name] = segment
            self.memory_files[segment.name] = file_obj
            self.memory_maps[segment.name] = mmap_obj
            
            # Create synchronization primitives
            if segment.sync_mode == SyncMode.MUTEX:
                self.locks[segment.name] = threading.RLock()
            elif segment.sync_mode == SyncMode.SEMAPHORE:
                self.semaphores[segment.name] = threading.Semaphore(1)
            elif segment.sync_mode == SyncMode.EVENT:
                self.events[segment.name] = threading.Event()
            elif segment.sync_mode == SyncMode.CONDITION:
                self.conditions[segment.name] = threading.Condition()
            
            # Create process-shared dict if needed
            if segment.access_mode == AccessMode.READ_WRITE:
                self.shared_dicts[segment.name] = self.process_manager.dict()
                self.shared_queues[segment.name] = multiprocessing.Queue()
            
            return True
            
        except Exception:
            return False
    
    def remove_segment(self, segment_name: str) -> bool:
        """Remove a shared memory segment."""
        if segment_name not in self.segments:
            return False
        
        try:
            # Close memory map
            if segment_name in self.memory_maps:
                self.memory_maps[segment_name].close()
                del self.memory_maps[segment_name]
            
            # Close file
            if segment_name in self.memory_files:
                self.memory_files[segment_name].close()
                del self.memory_files[segment_name]
            
            # Remove synchronization primitives
            self.locks.pop(segment_name, None)
            self.semaphores.pop(segment_name, None)
            self.events.pop(segment_name, None)
            self.conditions.pop(segment_name, None)
            
            # Remove process-shared objects
            self.shared_dicts.pop(segment_name, None)
            self.shared_queues.pop(segment_name, None)
            
            # Remove segment
            del self.segments[segment_name]
            
            return True
            
        except Exception:
            return False
    
    def get_segment_info(self, segment_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a memory segment."""
        if segment_name not in self.segments:
            return None
        
        segment = self.segments[segment_name]
        mmap_obj = self.memory_maps.get(segment_name)
        
        return {
            'name': segment.name,
            'memory_type': segment.memory_type.value,
            'size': segment.size,
            'access_mode': segment.access_mode.value,
            'sync_mode': segment.sync_mode.value,
            'created_at': segment.created_at.isoformat() if segment.created_at else None,
            'last_accessed': segment.last_accessed.isoformat() if segment.last_accessed else None,
            'access_count': segment.access_count,
            'current_size': len(mmap_obj) if mmap_obj else 0,
            'auto_resize': segment.auto_resize,
            'cleanup_on_exit': segment.cleanup_on_exit
        }
    
    # ========== Data Operations ==========
    
    def write_data(self, segment_name: str, data: Any, block_id: Optional[str] = None) -> bool:
        """
        Write data to shared memory segment.
        
        Args:
            segment_name: Name of memory segment
            data: Data to write
            block_id: Optional block identifier
            
        Returns:
            True if written successfully
        """
        if segment_name not in self.segments:
            return False
        
        segment = self.segments[segment_name]
        
        # Check access mode
        if segment.access_mode == AccessMode.READ_ONLY:
            return False
        
        try:
            # Acquire synchronization primitive
            with self._acquire_sync(segment_name):
                # Serialize data
                if isinstance(data, (dict, list)):
                    serialized_data = json.dumps(data).encode('utf-8')
                elif isinstance(data, str):
                    serialized_data = data.encode('utf-8')
                elif isinstance(data, bytes):
                    serialized_data = data
                else:
                    serialized_data = pickle.dumps(data)
                
                # Compress if enabled
                if segment.compression:
                    import zlib
                    serialized_data = zlib.compress(serialized_data)
                
                # Check size limits
                if len(serialized_data) > segment.size:
                    if segment.auto_resize and len(serialized_data) <= segment.max_size:
                        self._resize_segment(segment_name, len(serialized_data))
                    else:
                        return False
                
                # Write to memory map
                mmap_obj = self.memory_maps[segment_name]
                mmap_obj.seek(0)
                mmap_obj.write(struct.pack('I', len(serialized_data)))
                mmap_obj.write(serialized_data)
                mmap_obj.flush()
                
                # Store in memory block
                if block_id is None:
                    block_id = f"{segment_name}_{int(time.time() * 1000)}"
                
                memory_block = MemoryBlock(
                    block_id=block_id,
                    data_type=type(data).__name__,
                    data=data,
                    timestamp=datetime.now(),
                    size=len(serialized_data)
                )
                
                self.memory_blocks[block_id] = memory_block
                
                # Update segment statistics
                segment.last_accessed = datetime.now()
                segment.access_count += 1
                
                # Update global statistics
                self.access_stats['writes'] += 1
                self.access_stats['total_bytes_written'] += len(serialized_data)
                
                # Notify subscribers
                self._notify_subscribers(segment_name, data)
                
                return True
                
        except Exception:
            return False
    
    def read_data(self, segment_name: str, block_id: Optional[str] = None) -> Optional[Any]:
        """
        Read data from shared memory segment.
        
        Args:
            segment_name: Name of memory segment
            block_id: Optional block identifier
            
        Returns:
            Data if found, None otherwise
        """
        if segment_name not in self.segments:
            return None
        
        segment = self.segments[segment_name]
        
        # Check access mode
        if segment.access_mode == AccessMode.WRITE_ONLY:
            return None
        
        try:
            # Check memory block cache first
            if block_id and block_id in self.memory_blocks:
                self.access_stats['cache_hits'] += 1
                return self.memory_blocks[block_id].data
            
            self.access_stats['cache_misses'] += 1
            
            # Acquire synchronization primitive
            with self._acquire_sync(segment_name):
                # Read from memory map
                mmap_obj = self.memory_maps[segment_name]
                mmap_obj.seek(0)
                
                # Read data length
                length_bytes = mmap_obj.read(4)
                if len(length_bytes) < 4:
                    return None
                
                data_length = struct.unpack('I', length_bytes)[0]
                if data_length == 0:
                    return None
                
                # Read data
                serialized_data = mmap_obj.read(data_length)
                if len(serialized_data) < data_length:
                    return None
                
                # Decompress if needed
                if segment.compression:
                    import zlib
                    serialized_data = zlib.decompress(serialized_data)
                
                # Deserialize data
                try:
                    # Try JSON first
                    data = json.loads(serialized_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        # Try string
                        data = serialized_data.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            # Try pickle
                            data = pickle.loads(serialized_data)
                        except:
                            # Return raw bytes
                            data = serialized_data
                
                # Update segment statistics
                segment.last_accessed = datetime.now()
                segment.access_count += 1
                
                # Update global statistics
                self.access_stats['reads'] += 1
                self.access_stats['total_bytes_read'] += len(serialized_data)
                
                return data
                
        except Exception:
            return None
    
    def _acquire_sync(self, segment_name: str):
        """Acquire synchronization primitive for segment."""
        segment = self.segments[segment_name]
        
        if segment.sync_mode == SyncMode.MUTEX and segment_name in self.locks:
            return self.locks[segment_name]
        elif segment.sync_mode == SyncMode.SEMAPHORE and segment_name in self.semaphores:
            return self.semaphores[segment_name]
        elif segment.sync_mode == SyncMode.CONDITION and segment_name in self.conditions:
            return self.conditions[segment_name]
        else:
            # Return a dummy context manager for no sync
            from contextlib import nullcontext
            return nullcontext()
    
    def _resize_segment(self, segment_name: str, new_size: int) -> bool:
        """Resize a memory segment."""
        try:
            segment = self.segments[segment_name]
            
            # Close current memory map
            if segment_name in self.memory_maps:
                self.memory_maps[segment_name].close()
                del self.memory_maps[segment_name]
            
            # Resize file
            file_obj = self.memory_files[segment_name]
            file_obj.seek(0)
            file_obj.truncate(new_size)
            
            # Create new memory map
            mmap_obj = mmap.mmap(file_obj.fileno(), new_size)
            self.memory_maps[segment_name] = mmap_obj
            
            # Update segment size
            segment.size = new_size
            
            return True
            
        except Exception:
            return False
    
    # ========== Subscription System ==========
    
    def subscribe(self, segment_name: str, callback: Callable[[Any], None]) -> bool:
        """
        Subscribe to data changes in a segment.
        
        Args:
            segment_name: Segment name to subscribe to
            callback: Callback function to call on data changes
            
        Returns:
            True if subscribed successfully
        """
        if segment_name not in self.segments:
            return False
        
        self.subscribers[segment_name].append(callback)
        return True
    
    def unsubscribe(self, segment_name: str, callback: Callable[[Any], None]) -> bool:
        """Unsubscribe from segment data changes."""
        if segment_name not in self.subscribers:
            return False
        
        try:
            self.subscribers[segment_name].remove(callback)
            return True
        except ValueError:
            return False
    
    def _notify_subscribers(self, segment_name: str, data: Any) -> None:
        """Notify all subscribers of data changes."""
        if segment_name in self.subscribers:
            for callback in self.subscribers[segment_name]:
                try:
                    callback(data)
                except Exception:
                    pass  # Ignore callback errors
    
    # ========== Utility Methods ==========
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_allocated = sum(segment.size for segment in self.segments.values())
        total_used = sum(len(self.memory_maps.get(name, b'')) for name in self.segments.keys())
        
        return {
            'total_segments': len(self.segments),
            'total_allocated_bytes': total_allocated,
            'total_used_bytes': total_used,
            'memory_blocks': len(self.memory_blocks),
            'active_subscriptions': sum(len(subs) for subs in self.subscribers.values()),
            'access_statistics': self.access_stats.copy()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        total_operations = self.access_stats['reads'] + self.access_stats['writes']
        
        metrics = {
            'total_operations': total_operations,
            'read_write_ratio': self.access_stats['reads'] / max(1, self.access_stats['writes']),
            'cache_hit_rate': self.access_stats['cache_hits'] / max(1, 
                self.access_stats['cache_hits'] + self.access_stats['cache_misses']),
            'avg_bytes_per_read': self.access_stats['total_bytes_read'] / max(1, self.access_stats['reads']),
            'avg_bytes_per_write': self.access_stats['total_bytes_written'] / max(1, self.access_stats['writes']),
            'last_operation_time': self.access_stats['last_operation_time']
        }
        
        return metrics
    
    def export_segments_info(self) -> List[Dict[str, Any]]:
        """Export information about all segments."""
        return [self.get_segment_info(name) for name in self.segments.keys()]
    
    def clear_segment_data(self, segment_name: str) -> bool:
        """Clear all data in a segment."""
        if segment_name not in self.segments:
            return False
        
        try:
            with self._acquire_sync(segment_name):
                mmap_obj = self.memory_maps[segment_name]
                mmap_obj.seek(0)
                mmap_obj.write(b'\x00' * self.segments[segment_name].size)
                mmap_obj.flush()
            
            # Remove memory blocks for this segment
            blocks_to_remove = [bid for bid in self.memory_blocks.keys() if bid.startswith(segment_name)]
            for block_id in blocks_to_remove:
                del self.memory_blocks[block_id]
            
            return True
            
        except Exception:
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CSharedMemory(segments={len(self.segments)}, "
                f"blocks={len(self.memory_blocks)}, "
                f"subscribers={sum(len(s) for s in self.subscribers.values())})")


# ========== Utility Functions ==========

def create_market_data_segment(size_mb: int = 2) -> SharedMemorySegment:
    """Create a market data segment configuration."""
    return SharedMemorySegment(
        name="market_data_realtime",
        memory_type=SharedMemoryType.MARKET_DATA,
        size=size_mb * 1024 * 1024,
        access_mode=AccessMode.READ_WRITE,
        sync_mode=SyncMode.MUTEX,
        auto_resize=True,
        write_cache=True
    )


def create_trading_signals_segment(size_kb: int = 512) -> SharedMemorySegment:
    """Create a trading signals segment configuration."""
    return SharedMemorySegment(
        name="trading_signals_live",
        memory_type=SharedMemoryType.TRADING_SIGNALS,
        size=size_kb * 1024,
        access_mode=AccessMode.READ_WRITE,
        sync_mode=SyncMode.EVENT,
        auto_resize=True,
        compression=True
    )


def create_performance_segment(size_kb: int = 256) -> SharedMemorySegment:
    """Create a performance monitoring segment configuration."""
    return SharedMemorySegment(
        name="system_performance",
        memory_type=SharedMemoryType.PERFORMANCE,
        size=size_kb * 1024,
        access_mode=AccessMode.READ_WRITE,
        sync_mode=SyncMode.NONE,
        write_cache=True
    )