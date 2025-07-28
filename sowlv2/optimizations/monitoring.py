"""
Real-time monitoring dashboard for SOWLv2 pipeline performance.
Provides live performance metrics display, progress tracking, and alerting.
"""
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

import psutil
import torch

from .performance_collector import PerformanceCollector
from .resource_manager import AdvancedResourceManager, MemoryStats


@dataclass
class AlertConfig:
    """Configuration for performance alerts."""
    memory_threshold: float = 85.0  # percentage
    gpu_memory_threshold: float = 90.0  # percentage
    processing_time_threshold: float = 30.0  # seconds
    cpu_threshold: float = 95.0  # percentage
    enable_email_alerts: bool = False
    enable_console_alerts: bool = True


@dataclass
class ProgressInfo:
    """Progress tracking information."""
    operation_name: str
    current_step: int
    total_steps: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_stage: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUtilization:
    """Current resource utilization snapshot."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    gpu_utilization: float
    disk_io_read: float  # MB/s
    disk_io_write: float = 0.0  # MB/s
    network_io_sent: float = 0.0  # MB/s
    network_io_recv: float = 0.0  # MB/s
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class MonitoringDashboard:
    """Real-time performance monitoring dashboard with alerting capabilities."""

    def __init__(self, device: str = "cuda", update_interval: float = 1.0,
                 alert_config: Optional[AlertConfig] = None):
        """
        Initialize the monitoring dashboard.

        Args:
            device: Primary device to monitor
            update_interval: Update frequency in seconds
            alert_config: Alert configuration
        """
        self.device = device
        self.update_interval = update_interval
        self.alert_config = alert_config or AlertConfig()

        # Monitoring components
        self.performance_collector = PerformanceCollector(device=device)
        self.resource_manager = AdvancedResourceManager(device=device)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Data storage (keep last 1000 data points)
        self.resource_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: deque = deque(maxlen=100)

        # Progress tracking
        self.active_operations: Dict[str, ProgressInfo] = {}
        self.metrics_history: List[ResourceUtilization] = []

        # Callbacks for external integration
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.progress_callbacks: List[Callable[[str, ProgressInfo], None]] = []

        # Baseline measurements
        self._baseline_measurements = self._get_baseline_measurements()

    def _get_baseline_measurements(self) -> Dict[str, float]:
        """Get baseline system measurements for comparison."""
        baseline = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io_read': 0,
            'disk_io_write': 0,
            'network_io_sent': 0,
            'network_io_recv': 0
        }

        if torch.cuda.is_available() and self.device == "cuda":
            baseline['gpu_memory_percent'] = (
                torch.cuda.memory_allocated() /
                torch.cuda.get_device_properties(0).total_memory
            ) * 100
            baseline['gpu_utilization'] = 0  # Will be updated during monitoring

        return baseline

    def collect_resource_utilization(self) -> ResourceUtilization:
        """Collect current resource utilization."""
        # Get current disk and network IO for calculation
        current_disk_io = psutil.disk_io_counters()
        current_network_io = psutil.net_io_counters()
        
        # Use baseline as previous values for calculation
        last_disk_io = current_disk_io  # For simplicity, use current values
        last_network_io = current_network_io
        time_delta = 1.0  # 1 second interval
        
        return self._collect_resource_utilization(last_disk_io, last_network_io, time_delta)

    def update_progress(self, operation_id: str, progress: ProgressInfo):
        """Update progress for an operation."""
        self.active_operations[operation_id] = progress
        
        # Trigger progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(operation_id, progress)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def check_alerts(self, utilization: ResourceUtilization) -> List[str]:
        """Check for alerts and return list of alert messages."""
        self._check_alerts(utilization)
        # Return current alert messages
        return [alert.message for alert in self.active_alerts]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        current_utilization = self.collect_resource_utilization()
        
        return {
            'resource_utilization': current_utilization,
            'active_operations': dict(self.active_operations),
            'recent_alerts': [alert.__dict__ for alert in self.active_alerts],
            'metrics_history': self.metrics_history[-100:],  # Last 100 entries
            'is_monitoring': self.is_monitoring
        }

    def clear_completed_operations(self):
        """Clear completed operations from active tracking."""
        completed_ops = []
        for op_id, progress in self.active_operations.items():
            if progress.current_step >= progress.total_steps or progress.current_stage == "completed":
                completed_ops.append(op_id)
        
        for op_id in completed_ops:
            del self.active_operations[op_id]

    def start_monitoring(self):
        """Start real-time monitoring in a background thread."""
        if self.is_monitoring:
            print("Monitoring is already active")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        print(f"Real-time monitoring started (update interval: {self.update_interval}s)")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        print("Real-time monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        last_disk_io = psutil.disk_io_counters()
        last_network_io = psutil.net_io_counters()
        last_time = time.time()

        while self.is_monitoring:
            try:
                current_time = time.time()
                time_delta = current_time - last_time

                # Collect resource utilization
                utilization = self._collect_resource_utilization(
                    last_disk_io, last_network_io, time_delta
                )
                self.resource_history.append(utilization)

                # Check for alerts
                self._check_alerts(utilization)

                # Update progress for active operations
                self._update_operation_progress()

                # Store current measurements for next iteration
                last_disk_io = psutil.disk_io_counters()
                last_network_io = psutil.net_io_counters()
                last_time = current_time

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def _collect_resource_utilization(self, last_disk_io, last_network_io,
                                    time_delta: float) -> ResourceUtilization:
        """Collect current resource utilization metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        if last_disk_io and time_delta > 0:
            disk_read_rate = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024*1024) / time_delta
            disk_write_rate = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024*1024) / time_delta
        else:
            disk_read_rate = disk_write_rate = 0

        # Network I/O
        current_network_io = psutil.net_io_counters()
        if last_network_io and time_delta > 0:
            network_sent_rate = (current_network_io.bytes_sent - last_network_io.bytes_sent) / (1024*1024) / time_delta
            network_recv_rate = (current_network_io.bytes_recv - last_network_io.bytes_recv) / (1024*1024) / time_delta
        else:
            network_sent_rate = network_recv_rate = 0

        # GPU metrics
        gpu_memory_percent = 0
        gpu_utilization = 0

        if torch.cuda.is_available() and self.device == "cuda":
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100

            # Try to get GPU utilization if nvidia-ml-py is available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization_rates.gpu
            except ImportError:
                gpu_utilization = 0

        return ResourceUtilization(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization=gpu_utilization,
            disk_io_read=disk_read_rate,
            disk_io_write=disk_write_rate,
            network_io_sent=network_sent_rate,
            network_io_recv=network_recv_rate
        )

    def _check_alerts(self, utilization: ResourceUtilization):
        """Check for performance alerts based on current utilization."""
        alerts_to_add = []

        # Memory alert
        if utilization.memory_percent > self.alert_config.memory_threshold:
            alert = PerformanceAlert(
                alert_type="high_memory_usage",
                severity="high" if utilization.memory_percent > 95 else "medium",
                message=f"System memory usage is {utilization.memory_percent:.1f}%",
                metric_value=utilization.memory_percent,
                threshold=self.alert_config.memory_threshold
            )
            alerts_to_add.append(alert)

        # GPU memory alert
        if utilization.gpu_memory_percent > self.alert_config.gpu_memory_threshold:
            alert = PerformanceAlert(
                alert_type="high_gpu_memory_usage",
                severity="critical" if utilization.gpu_memory_percent > 98 else "high",
                message=f"GPU memory usage is {utilization.gpu_memory_percent:.1f}%",
                metric_value=utilization.gpu_memory_percent,
                threshold=self.alert_config.gpu_memory_threshold
            )
            alerts_to_add.append(alert)

        # CPU alert
        if utilization.cpu_percent > self.alert_config.cpu_threshold:
            alert = PerformanceAlert(
                alert_type="high_cpu_usage",
                severity="medium",
                message=f"CPU usage is {utilization.cpu_percent:.1f}%",
                metric_value=utilization.cpu_percent,
                threshold=self.alert_config.cpu_threshold
            )
            alerts_to_add.append(alert)

        # Add new alerts and trigger callbacks
        for alert in alerts_to_add:
            # Check if similar alert already exists
            existing_alert = next(
                (a for a in self.active_alerts
                 if a.alert_type == alert.alert_type and not a.resolved),
                None
            )

            if not existing_alert:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                self._trigger_alert(alert)

        # Resolve alerts that are no longer active
        for alert in self.active_alerts:
            if not alert.resolved:
                should_resolve = False

                if alert.alert_type == "high_memory_usage" and utilization.memory_percent < self.alert_config.memory_threshold - 5:
                    should_resolve = True
                elif alert.alert_type == "high_gpu_memory_usage" and utilization.gpu_memory_percent < self.alert_config.gpu_memory_threshold - 5:
                    should_resolve = True
                elif alert.alert_type == "high_cpu_usage" and utilization.cpu_percent < self.alert_config.cpu_threshold - 5:
                    should_resolve = True

                if should_resolve:
                    alert.resolved = True
                    if self.alert_config.enable_console_alerts:
                        print(f"✓ Alert resolved: {alert.message}")

    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger alert notifications."""
        if self.alert_config.enable_console_alerts:
            severity_icon = {
                "low": "ℹ️",
                "medium": "⚠️",
                "high": "🚨",
                "critical": "🔥"
            }.get(alert.severity, "⚠️")

            print(f"{severity_icon} ALERT [{alert.severity.upper()}]: {alert.message}")

        # Trigger registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")

    def start_operation_tracking(self, operation_name: str, total_steps: int,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking progress for a long-running operation.

        Args:
            operation_name: Name of the operation
            total_steps: Total number of steps
            metadata: Additional operation metadata

        Returns:
            str: Operation ID for progress updates
        """
        operation_id = f"{operation_name}_{int(time.time())}"

        progress_info = ProgressInfo(
            operation_name=operation_name,
            current_step=0,
            total_steps=total_steps,
            start_time=datetime.now(),
            metadata=metadata or {}
        )

        self.active_operations[operation_id] = progress_info

        print(f"📊 Started tracking: {operation_name} (0/{total_steps})")
        return operation_id

    def update_operation_progress(self, operation_id: str, current_step: int,
                                current_stage: str = ""):
        """
        Update progress for a tracked operation.

        Args:
            operation_id: Operation ID from start_operation_tracking
            current_step: Current step number
            current_stage: Current stage description
        """
        if operation_id not in self.active_operations:
            return

        progress_info = self.active_operations[operation_id]
        progress_info.current_step = current_step
        progress_info.current_stage = current_stage

        # Estimate completion time
        if current_step > 0:
            elapsed = datetime.now() - progress_info.start_time
            estimated_total = elapsed * (progress_info.total_steps / current_step)
            progress_info.estimated_completion = progress_info.start_time + estimated_total

        # Trigger progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(operation_id, progress_info)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def complete_operation_tracking(self, operation_id: str):
        """Complete tracking for an operation."""
        if operation_id in self.active_operations:
            progress_info = self.active_operations.pop(operation_id)
            elapsed = datetime.now() - progress_info.start_time

            print(f"✅ Completed: {progress_info.operation_name} "
                  f"({progress_info.total_steps}/{progress_info.total_steps}) "
                  f"in {elapsed.total_seconds():.1f}s")

    def _update_operation_progress(self):
        """Update progress display for active operations."""
        for operation_id, progress_info in self.active_operations.items():
            if progress_info.current_step > 0:
                percent = (progress_info.current_step / progress_info.total_steps) * 100
                elapsed = datetime.now() - progress_info.start_time

                # Simple progress display (could be enhanced with progress bars)
                stage_info = f" - {progress_info.current_stage}" if progress_info.current_stage else ""
                print(f"⏳ {progress_info.operation_name}: {percent:.1f}% "
                      f"({progress_info.current_step}/{progress_info.total_steps})"
                      f"{stage_info} [{elapsed.total_seconds():.1f}s]")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status and metrics."""
        current_utilization = self.resource_history[-1] if self.resource_history else None

        status = {
            'monitoring_active': self.is_monitoring,
            'update_interval': self.update_interval,
            'active_operations': len(self.active_operations),
            'active_alerts': len([a for a in self.active_alerts if not a.resolved]),
            'total_alerts': len(self.alert_history),
            'data_points_collected': len(self.resource_history)
        }

        if current_utilization:
            status['current_utilization'] = {
                'cpu_percent': current_utilization.cpu_percent,
                'memory_percent': current_utilization.memory_percent,
                'gpu_memory_percent': current_utilization.gpu_memory_percent,
                'gpu_utilization': current_utilization.gpu_utilization
            }

        return status

    def get_resource_trends(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get resource utilization trends over specified time window."""
        if not self.resource_history:
            return {}

        # Filter data within time window
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_data = [
            util for util in self.resource_history
            if util.timestamp >= cutoff_time
        ]

        if not recent_data:
            return {}

        # Calculate trends
        cpu_values = [u.cpu_percent for u in recent_data]
        memory_values = [u.memory_percent for u in recent_data]
        gpu_memory_values = [u.gpu_memory_percent for u in recent_data]

        return {
            'window_minutes': window_minutes,
            'data_points': len(recent_data),
            'cpu': {
                'current': cpu_values[-1],
                'average': sum(cpu_values) / len(cpu_values),
                'peak': max(cpu_values),
                'trend': 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing'
            },
            'memory': {
                'current': memory_values[-1],
                'average': sum(memory_values) / len(memory_values),
                'peak': max(memory_values),
                'trend': 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing'
            },
            'gpu_memory': {
                'current': gpu_memory_values[-1],
                'average': sum(gpu_memory_values) / len(gpu_memory_values),
                'peak': max(gpu_memory_values),
                'trend': 'increasing' if gpu_memory_values[-1] > gpu_memory_values[0] else 'decreasing'
            }
        }

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)

    def add_progress_callback(self, callback: Callable[[str, ProgressInfo], None]):
        """Add callback function for progress updates."""
        self.progress_callbacks.append(callback)

    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to JSON file."""
        data = {
            'resource_history': [
                {
                    'cpu_percent': u.cpu_percent,
                    'memory_percent': u.memory_percent,
                    'gpu_memory_percent': u.gpu_memory_percent,
                    'gpu_utilization': u.gpu_utilization,
                    'disk_io_read': u.disk_io_read,
                    'disk_io_write': u.disk_io_write,
                    'network_io_sent': u.network_io_sent,
                    'network_io_recv': u.network_io_recv,
                    'timestamp': u.timestamp.isoformat()
                }
                for u in self.resource_history
            ],
            'alert_history': [
                {
                    'alert_type': a.alert_type,
                    'severity': a.severity,
                    'message': a.message,
                    'metric_value': a.metric_value,
                    'threshold': a.threshold,
                    'timestamp': a.timestamp.isoformat(),
                    'resolved': a.resolved
                }
                for a in self.alert_history
            ],
            'export_timestamp': datetime.now().isoformat(),
            'monitoring_config': {
                'device': self.device,
                'update_interval': self.update_interval,
                'alert_config': {
                    'memory_threshold': self.alert_config.memory_threshold,
                    'gpu_memory_threshold': self.alert_config.gpu_memory_threshold,
                    'cpu_threshold': self.alert_config.cpu_threshold
                }
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Monitoring data exported to: {filepath}")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
