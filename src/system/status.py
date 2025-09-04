"""
System Status Management for algorithmic trading.

This module contains the CStatus class which handles comprehensive
system status monitoring, health checks, performance tracking,
and real-time status reporting for trading systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import psutil
import platform
from collections import defaultdict, deque

from ..core.base import SystemProtocol


class SystemState(Enum):
    """System state enumeration."""
    UNKNOWN = "UNKNOWN"                 # Unknown state
    INITIALIZING = "INITIALIZING"       # System initializing
    IDLE = "IDLE"                       # System idle
    RUNNING = "RUNNING"                 # System running normally
    TRADING = "TRADING"                 # Active trading
    PAUSED = "PAUSED"                   # System paused
    ERROR = "ERROR"                     # System error state
    WARNING = "WARNING"                 # System warning state
    SHUTDOWN = "SHUTDOWN"               # System shutting down
    MAINTENANCE = "MAINTENANCE"         # System in maintenance mode


class ComponentStatus(Enum):
    """Component status enumeration."""
    OFFLINE = "OFFLINE"                 # Component offline
    STARTING = "STARTING"               # Component starting
    ONLINE = "ONLINE"                   # Component online and healthy
    DEGRADED = "DEGRADED"               # Component online but degraded
    ERROR = "ERROR"                     # Component in error state
    FAILED = "FAILED"                   # Component failed
    STOPPING = "STOPPING"               # Component stopping


class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "INFO"                       # Informational
    WARNING = "WARNING"                 # Warning condition
    ERROR = "ERROR"                     # Error condition
    CRITICAL = "CRITICAL"               # Critical condition
    FATAL = "FATAL"                     # Fatal condition


@dataclass
class ComponentInfo:
    """Component information and status."""
    
    name: str
    component_type: str
    status: ComponentStatus = ComponentStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    
    # Health metrics
    uptime: timedelta = timedelta()
    error_count: int = 0
    restart_count: int = 0
    
    # Configuration
    enabled: bool = True
    critical: bool = False
    auto_restart: bool = True
    
    # Metadata
    version: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        if self.status in [ComponentStatus.ONLINE, ComponentStatus.STARTING]:
            if self.last_heartbeat:
                time_since_heartbeat = datetime.now() - self.last_heartbeat
                return time_since_heartbeat < timedelta(minutes=5)
        return False
    
    def get_health_score(self) -> float:
        """Get component health score (0-100)."""
        if not self.enabled:
            return 0.0
        
        if self.status == ComponentStatus.ONLINE:
            base_score = 100.0
        elif self.status == ComponentStatus.DEGRADED:
            base_score = 70.0
        elif self.status == ComponentStatus.STARTING:
            base_score = 50.0
        else:
            base_score = 0.0
        
        # Adjust based on error count
        if self.error_count > 0:
            error_penalty = min(50.0, self.error_count * 5.0)
            base_score = max(0.0, base_score - error_penalty)
        
        # Adjust based on response time
        if self.response_time > 1.0:  # > 1 second
            response_penalty = min(20.0, (self.response_time - 1.0) * 10.0)
            base_score = max(0.0, base_score - response_penalty)
        
        return base_score


@dataclass
class SystemAlert:
    """System alert information."""
    
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolve_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    
    timestamp: datetime
    
    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_count: int = 0
    cpu_frequency: float = 0.0
    
    # Memory metrics
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0
    memory_total_gb: float = 0.0
    
    # Disk metrics
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    
    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    
    # System metrics
    uptime_seconds: float = 0.0
    load_average: float = 0.0
    process_count: int = 0
    thread_count: int = 0


class CStatus:
    """
    Comprehensive system status management.
    
    Features:
    - Real-time system monitoring
    - Component health tracking
    - Performance metrics collection
    - Alert management and notification
    - System state management
    - Heartbeat monitoring
    - Resource usage tracking
    - Automatic health checks
    - Status reporting and logging
    """
    
    def __init__(self):
        """Initialize status manager."""
        self.is_initialized = False
        
        # System state
        self.current_state = SystemState.UNKNOWN
        self.previous_state = SystemState.UNKNOWN
        self.state_change_time = datetime.now()
        
        # Components
        self.components: Dict[str, ComponentInfo] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        
        # Alerts
        self.alerts: List[SystemAlert] = []
        self.alert_counter = 0
        self.alert_subscribers: List[Callable[[SystemAlert], None]] = []
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.performance_collectors: Dict[str, Callable[[], Dict[str, Any]]] = {}
        
        # Monitoring configuration
        self.monitoring_interval = 5.0  # seconds
        self.heartbeat_timeout = 30.0   # seconds
        self.performance_retention_hours = 24
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self.status_lock = threading.RLock()
        
        # System information
        self.system_info = self._collect_system_info()
        
        # Health check callbacks
        self.health_checkers: Dict[str, Callable[[], bool]] = {}
        
        # Statistics
        self.stats = {
            'status_checks': 0,
            'alerts_generated': 0,
            'component_failures': 0,
            'state_changes': 0,
            'uptime_start': datetime.now()
        }
    
    def initialize(self, system: SystemProtocol) -> 'CStatus':
        """
        Initialize status manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.status_lock:
            # Register core components
            self._register_core_components()
            
            # Start monitoring thread
            self._start_monitoring()
            
            # Set initial state
            self.set_system_state(SystemState.INITIALIZING)
            
            self.is_initialized = True
        
        return self
    
    def reset(self, system: SystemProtocol) -> 'CStatus':
        """
        Reset status manager.
        
        Args:
            system: System protocol interface
            
        Returns:
            Self for method chaining
        """
        with self.status_lock:
            # Stop monitoring
            self.running = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            
            # Clear data
            self.components.clear()
            self.alerts.clear()
            self.performance_history.clear()
            self.health_checkers.clear()
            
            # Reset state
            self.current_state = SystemState.UNKNOWN
            self.previous_state = SystemState.UNKNOWN
        
        return self
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect basic system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'boot_time': datetime.fromtimestamp(psutil.boot_time())
        }
    
    def _register_core_components(self) -> None:
        """Register core system components."""
        core_components = [
            ComponentInfo(
                name="system_core",
                component_type="core",
                status=ComponentStatus.STARTING,
                critical=True,
                description="Core trading system"
            ),
            ComponentInfo(
                name="data_feed",
                component_type="data",
                status=ComponentStatus.OFFLINE,
                critical=True,
                description="Market data feed"
            ),
            ComponentInfo(
                name="trading_engine",
                component_type="trading",
                status=ComponentStatus.OFFLINE,
                critical=True,
                description="Trading execution engine"
            ),
            ComponentInfo(
                name="risk_manager",
                component_type="risk",
                status=ComponentStatus.OFFLINE,
                critical=True,
                description="Risk management system"
            ),
            ComponentInfo(
                name="performance_monitor",
                component_type="monitoring",
                status=ComponentStatus.STARTING,
                critical=False,
                description="Performance monitoring"
            )
        ]
        
        for component in core_components:
            self.register_component(component)
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_worker(self) -> None:
        """Background monitoring worker."""
        while self.running:
            try:
                self._perform_monitoring_cycle()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self._generate_alert(
                    AlertLevel.ERROR,
                    "monitoring_system",
                    f"Monitoring cycle failed: {str(e)}"
                )
                time.sleep(self.monitoring_interval * 2)  # Back off on error
    
    def _perform_monitoring_cycle(self) -> None:
        """Perform one monitoring cycle."""
        with self.status_lock:
            # Collect performance metrics
            self._collect_performance_metrics()
            
            # Check component health
            self._check_component_health()
            
            # Run custom health checks
            self._run_health_checks()
            
            # Update system state based on component status
            self._update_system_state()
            
            # Clean up old data
            self._cleanup_old_data()
            
            # Update statistics
            self.stats['status_checks'] += 1
    
    def _collect_performance_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                cpu_count=psutil.cpu_count(),
                cpu_frequency=psutil.cpu_freq().current if psutil.cpu_freq() else 0.0,
                memory_usage_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                uptime_seconds=(datetime.now() - self.stats['uptime_start']).total_seconds(),
                process_count=len(psutil.pids()),
                thread_count=threading.active_count()
            )
            
            # Store metrics
            self.performance_history.append(metrics)
            
            # Run custom performance collectors
            for name, collector in self.performance_collectors.items():
                try:
                    custom_metrics = collector()
                    # Store custom metrics in component info or separate storage
                except Exception:
                    pass
            
            # Check for performance alerts
            self._check_performance_alerts(metrics)
            
        except Exception as e:
            self._generate_alert(
                AlertLevel.WARNING,
                "performance_collector",
                f"Failed to collect performance metrics: {str(e)}"
            )
    
    def _check_component_health(self) -> None:
        """Check health of all registered components."""
        current_time = datetime.now()
        
        for component_name, component in self.components.items():
            if not component.enabled:
                continue
            
            # Check heartbeat timeout
            if component.last_heartbeat:
                time_since_heartbeat = current_time - component.last_heartbeat
                if time_since_heartbeat > timedelta(seconds=self.heartbeat_timeout):
                    if component.status == ComponentStatus.ONLINE:
                        component.status = ComponentStatus.DEGRADED
                        self._generate_alert(
                            AlertLevel.WARNING,
                            component_name,
                            f"Component heartbeat timeout ({time_since_heartbeat.total_seconds():.1f}s)"
                        )
            
            # Update component uptime
            if component.status in [ComponentStatus.ONLINE, ComponentStatus.DEGRADED]:
                # Component is running
                pass
    
    def _run_health_checks(self) -> None:
        """Run custom health check callbacks."""
        for component_name, health_checker in self.health_checkers.items():
            try:
                is_healthy = health_checker()
                component = self.components.get(component_name)
                
                if component:
                    if is_healthy:
                        if component.status == ComponentStatus.ERROR:
                            component.status = ComponentStatus.ONLINE
                            self._generate_alert(
                                AlertLevel.INFO,
                                component_name,
                                "Component recovered from error state"
                            )
                    else:
                        if component.status == ComponentStatus.ONLINE:
                            component.status = ComponentStatus.ERROR
                            component.error_count += 1
                            self._generate_alert(
                                AlertLevel.ERROR,
                                component_name,
                                "Component health check failed"
                            )
            except Exception as e:
                self._generate_alert(
                    AlertLevel.WARNING,
                    component_name,
                    f"Health check exception: {str(e)}"
                )
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance-based alerts."""
        # CPU usage alerts
        if metrics.cpu_usage_percent > 90.0:
            self._generate_alert(
                AlertLevel.CRITICAL,
                "system_performance",
                f"High CPU usage: {metrics.cpu_usage_percent:.1f}%"
            )
        elif metrics.cpu_usage_percent > 80.0:
            self._generate_alert(
                AlertLevel.WARNING,
                "system_performance",
                f"Elevated CPU usage: {metrics.cpu_usage_percent:.1f}%"
            )
        
        # Memory usage alerts
        if metrics.memory_usage_percent > 90.0:
            self._generate_alert(
                AlertLevel.CRITICAL,
                "system_performance",
                f"High memory usage: {metrics.memory_usage_percent:.1f}%"
            )
        elif metrics.memory_usage_percent > 80.0:
            self._generate_alert(
                AlertLevel.WARNING,
                "system_performance",
                f"Elevated memory usage: {metrics.memory_usage_percent:.1f}%"
            )
        
        # Disk usage alerts
        if metrics.disk_usage_percent > 95.0:
            self._generate_alert(
                AlertLevel.CRITICAL,
                "system_performance",
                f"Very high disk usage: {metrics.disk_usage_percent:.1f}%"
            )
        elif metrics.disk_usage_percent > 85.0:
            self._generate_alert(
                AlertLevel.WARNING,
                "system_performance",
                f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            )
    
    def _update_system_state(self) -> None:
        """Update overall system state based on component status."""
        critical_components = [c for c in self.components.values() if c.critical and c.enabled]
        
        if not critical_components:
            return
        
        # Check if any critical components are failed
        failed_critical = [c for c in critical_components if c.status == ComponentStatus.FAILED]
        if failed_critical:
            self.set_system_state(SystemState.ERROR)
            return
        
        # Check if any critical components have errors
        error_critical = [c for c in critical_components if c.status == ComponentStatus.ERROR]
        if error_critical:
            self.set_system_state(SystemState.WARNING)
            return
        
        # Check if all critical components are online
        online_critical = [c for c in critical_components if c.status == ComponentStatus.ONLINE]
        if len(online_critical) == len(critical_components):
            if self.current_state in [SystemState.INITIALIZING, SystemState.WARNING, SystemState.ERROR]:
                self.set_system_state(SystemState.RUNNING)
            return
        
        # Some critical components are starting or degraded
        starting_critical = [c for c in critical_components if c.status == ComponentStatus.STARTING]
        if starting_critical:
            if self.current_state == SystemState.UNKNOWN:
                self.set_system_state(SystemState.INITIALIZING)
    
    def _cleanup_old_data(self) -> None:
        """Clean up old performance data and resolved alerts."""
        current_time = datetime.now()
        retention_threshold = current_time - timedelta(hours=self.performance_retention_hours)
        
        # Clean up old alerts (keep resolved alerts for 24 hours)
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or (current_time - alert.timestamp) < timedelta(hours=24)
        ]
    
    # ========== Public API Methods ==========
    
    def set_system_state(self, new_state: SystemState) -> None:
        """Set system state and generate appropriate alerts."""
        with self.status_lock:
            if new_state != self.current_state:
                self.previous_state = self.current_state
                self.current_state = new_state
                self.state_change_time = datetime.now()
                self.stats['state_changes'] += 1
                
                # Generate state change alert
                alert_level = AlertLevel.INFO
                if new_state in [SystemState.ERROR, SystemState.SHUTDOWN]:
                    alert_level = AlertLevel.CRITICAL
                elif new_state in [SystemState.WARNING, SystemState.PAUSED]:
                    alert_level = AlertLevel.WARNING
                
                self._generate_alert(
                    alert_level,
                    "system_core",
                    f"System state changed from {self.previous_state.value} to {new_state.value}"
                )
    
    def get_system_state(self) -> SystemState:
        """Get current system state."""
        return self.current_state
    
    def register_component(self, component: ComponentInfo) -> bool:
        """Register a new component for monitoring."""
        with self.status_lock:
            if component.name in self.components:
                return False
            
            component.last_heartbeat = datetime.now()
            self.components[component.name] = component
            
            self._generate_alert(
                AlertLevel.INFO,
                component.name,
                f"Component registered: {component.description}"
            )
            
            return True
    
    def unregister_component(self, component_name: str) -> bool:
        """Unregister a component."""
        with self.status_lock:
            if component_name not in self.components:
                return False
            
            del self.components[component_name]
            self.health_checkers.pop(component_name, None)
            
            return True
    
    def update_component_status(self, component_name: str, status: ComponentStatus,
                              details: Optional[Dict[str, Any]] = None) -> bool:
        """Update component status."""
        with self.status_lock:
            if component_name not in self.components:
                return False
            
            component = self.components[component_name]
            old_status = component.status
            component.status = status
            component.last_heartbeat = datetime.now()
            
            # Update metrics if provided
            if details:
                component.cpu_usage = details.get('cpu_usage', component.cpu_usage)
                component.memory_usage = details.get('memory_usage', component.memory_usage)
                component.response_time = details.get('response_time', component.response_time)
            
            # Generate alert for significant status changes
            if old_status != status:
                if status in [ComponentStatus.ERROR, ComponentStatus.FAILED]:
                    alert_level = AlertLevel.ERROR
                    component.error_count += 1
                elif status == ComponentStatus.DEGRADED:
                    alert_level = AlertLevel.WARNING
                else:
                    alert_level = AlertLevel.INFO
                
                self._generate_alert(
                    alert_level,
                    component_name,
                    f"Component status changed from {old_status.value} to {status.value}"
                )
            
            return True
    
    def heartbeat(self, component_name: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Record component heartbeat."""
        with self.status_lock:
            if component_name not in self.components:
                return False
            
            component = self.components[component_name]
            component.last_heartbeat = datetime.now()
            
            # Update metrics if provided
            if metrics:
                component.cpu_usage = metrics.get('cpu_usage', component.cpu_usage)
                component.memory_usage = metrics.get('memory_usage', component.memory_usage)
                component.response_time = metrics.get('response_time', component.response_time)
            
            # Auto-promote component to online if it was starting
            if component.status == ComponentStatus.STARTING:
                component.status = ComponentStatus.ONLINE
            
            return True
    
    def _generate_alert(self, level: AlertLevel, component: str, message: str,
                       details: Optional[Dict[str, Any]] = None) -> SystemAlert:
        """Generate a system alert."""
        self.alert_counter += 1
        
        alert = SystemAlert(
            alert_id=f"ALERT_{self.alert_counter:06d}",
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        self.stats['alerts_generated'] += 1
        
        # Notify subscribers
        for subscriber in self.alert_subscribers:
            try:
                subscriber(alert)
            except Exception:
                pass
        
        return alert
    
    def subscribe_to_alerts(self, callback: Callable[[SystemAlert], None]) -> bool:
        """Subscribe to alert notifications."""
        if callback not in self.alert_subscribers:
            self.alert_subscribers.append(callback)
            return True
        return False
    
    def register_health_checker(self, component_name: str, 
                               health_checker: Callable[[], bool]) -> bool:
        """Register a custom health check callback."""
        if component_name in self.components:
            self.health_checkers[component_name] = health_checker
            return True
        return False
    
    def register_performance_collector(self, name: str,
                                     collector: Callable[[], Dict[str, Any]]) -> bool:
        """Register a custom performance metrics collector."""
        self.performance_collectors[name] = collector
        return True
    
    # ========== Information Methods ==========
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self.status_lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
            
            return {
                'system_state': self.current_state.value,
                'previous_state': self.previous_state.value,
                'state_change_time': self.state_change_time.isoformat(),
                'uptime_seconds': (datetime.now() - self.stats['uptime_start']).total_seconds(),
                'components': {
                    name: {
                        'status': comp.status.value,
                        'health_score': comp.get_health_score(),
                        'last_heartbeat': comp.last_heartbeat.isoformat() if comp.last_heartbeat else None,
                        'error_count': comp.error_count,
                        'critical': comp.critical
                    }
                    for name, comp in self.components.items()
                },
                'alerts': {
                    'total': len(self.alerts),
                    'active': len(active_alerts),
                    'critical': len(critical_alerts)
                },
                'performance': self._get_latest_performance(),
                'statistics': self.stats.copy(),
                'system_info': self.system_info
            }
    
    def get_component_status(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific component."""
        if component_name not in self.components:
            return None
        
        component = self.components[component_name]
        return {
            'name': component.name,
            'type': component.component_type,
            'status': component.status.value,
            'health_score': component.get_health_score(),
            'last_heartbeat': component.last_heartbeat.isoformat() if component.last_heartbeat else None,
            'uptime': component.uptime.total_seconds(),
            'error_count': component.error_count,
            'restart_count': component.restart_count,
            'cpu_usage': component.cpu_usage,
            'memory_usage': component.memory_usage,
            'response_time': component.response_time,
            'enabled': component.enabled,
            'critical': component.critical,
            'auto_restart': component.auto_restart,
            'version': component.version,
            'description': component.description,
            'dependencies': component.dependencies
        }
    
    def get_alerts(self, level: Optional[AlertLevel] = None,
                   component: Optional[str] = None,
                   resolved: Optional[bool] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get system alerts with optional filtering."""
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if component:
            filtered_alerts = [a for a in filtered_alerts if a.component == component]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        # Sort by timestamp (newest first) and limit
        filtered_alerts = sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'details': alert.details,
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved,
                'resolve_time': alert.resolve_time.isoformat() if alert.resolve_time else None
            }
            for alert in filtered_alerts
        ]
    
    def _get_latest_performance(self) -> Optional[Dict[str, Any]]:
        """Get latest performance metrics."""
        if not self.performance_history:
            return None
        
        metrics = self.performance_history[-1]
        return {
            'timestamp': metrics.timestamp.isoformat(),
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'memory_usage_percent': metrics.memory_usage_percent,
            'disk_usage_percent': metrics.disk_usage_percent,
            'memory_available_gb': metrics.memory_available_gb,
            'disk_free_gb': metrics.disk_free_gb,
            'uptime_seconds': metrics.uptime_seconds,
            'process_count': metrics.process_count,
            'thread_count': metrics.thread_count
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolve_time = datetime.now()
                return True
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CStatus(state={self.current_state.value}, components={len(self.components)}, "
                f"alerts={len([a for a in self.alerts if not a.resolved])})")