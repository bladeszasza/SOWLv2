"""
Unit tests for MonitoringDashboard.
Tests real-time monitoring, alert system, and performance tracking.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sowlv2.optimizations.monitoring import (
    MonitoringDashboard, AlertConfig, ProgressInfo, ResourceUtilization
)


class TestMonitoringDashboard:
    """Test suite for MonitoringDashboard class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        dashboard = MonitoringDashboard()
        assert dashboard.alert_config.memory_threshold == 85.0
        assert dashboard.alert_config.gpu_memory_threshold == 90.0
        assert dashboard.alert_config.processing_time_threshold == 30.0
        assert dashboard.alert_config.cpu_threshold == 95.0
        assert dashboard.alert_config.enable_console_alerts is True
        assert dashboard.is_monitoring is False
        assert len(dashboard.metrics_history) == 0

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = AlertConfig(
            memory_threshold=80.0,
            gpu_memory_threshold=85.0,
            processing_time_threshold=25.0,
            cpu_threshold=90.0,
            enable_console_alerts=False,
            enable_email_alerts=True
        )
        dashboard = MonitoringDashboard(alert_config=config)
        assert dashboard.alert_config.memory_threshold == 80.0
        assert dashboard.alert_config.gpu_memory_threshold == 85.0
        assert dashboard.alert_config.processing_time_threshold == 25.0
        assert dashboard.alert_config.cpu_threshold == 90.0
        assert dashboard.alert_config.enable_console_alerts is False
        assert dashboard.alert_config.enable_email_alerts is True

    def test_start_monitoring(self):
        """Test starting the monitoring process."""
        dashboard = MonitoringDashboard()
        with patch.object(dashboard, '_monitoring_loop') as mock_loop:
            dashboard.start_monitoring()
            assert dashboard.is_monitoring is True
            mock_loop.assert_called_once()

    def test_stop_monitoring(self):
        """Test stopping the monitoring process."""
        dashboard = MonitoringDashboard()
        dashboard.is_monitoring = True
        dashboard.stop_monitoring()
        assert dashboard.is_monitoring is False

    def test_collect_resource_utilization(self):
        """Test resource utilization collection."""
        dashboard = MonitoringDashboard()
        
        with patch('psutil.cpu_percent', return_value=45.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(percent=60.0)
                with patch('torch.cuda.is_available', return_value=False):
                    utilization = dashboard.collect_resource_utilization()

        assert isinstance(utilization, ResourceUtilization)
        assert utilization.cpu_percent == 45.0
        assert utilization.memory_percent == 60.0
        assert utilization.gpu_memory_percent == 0.0
        assert utilization.gpu_utilization == 0.0

    def test_update_progress(self):
        """Test progress tracking update."""
        dashboard = MonitoringDashboard()
        
        progress = ProgressInfo(
            operation_name="test_operation",
            current_step=5,
            total_steps=10,
            start_time=datetime.now(),
            current_stage="processing"
        )
        
        dashboard.update_progress("test_op", progress)
        
        assert "test_op" in dashboard.active_operations
        assert dashboard.active_operations["test_op"].current_step == 5
        assert dashboard.active_operations["test_op"].total_steps == 10
        assert dashboard.active_operations["test_op"].current_stage == "processing"

    def test_check_alerts_no_violations(self):
        """Test alert checking with no violations."""
        dashboard = MonitoringDashboard()
        
        utilization = ResourceUtilization(
            cpu_percent=50.0,  # Below 95% threshold
            memory_percent=60.0,  # Below 85% threshold
            gpu_memory_percent=70.0,  # Below 90% threshold
            gpu_utilization=80.0,
            disk_io_read=10.0
        )
        
        alerts = dashboard.check_alerts(utilization)
        assert len(alerts) == 0

    def test_check_alerts_memory_violation(self):
        """Test alert checking with memory violation."""
        dashboard = MonitoringDashboard()
        
        utilization = ResourceUtilization(
            cpu_percent=50.0,
            memory_percent=90.0,  # Above 85% threshold
            gpu_memory_percent=70.0,
            gpu_utilization=80.0,
            disk_io_read=10.0
        )
        
        alerts = dashboard.check_alerts(utilization)
        assert len(alerts) >= 1
        assert any("memory" in alert.lower() for alert in alerts)

    def test_get_dashboard_data(self):
        """Test getting dashboard data."""
        dashboard = MonitoringDashboard()
        
        # Add some mock progress
        progress = ProgressInfo(
            operation_name="test_operation",
            current_step=3,
            total_steps=10,
            start_time=datetime.now(),
            current_stage="processing"
        )
        dashboard.active_operations["test_op"] = progress
        
        data = dashboard.get_dashboard_data()
        
        assert isinstance(data, dict)
        assert "resource_utilization" in data
        assert "active_operations" in data
        assert "recent_alerts" in data
        assert "is_monitoring" in data
        assert len(data["active_operations"]) == 1

    def test_alert_config_dataclass(self):
        """Test AlertConfig dataclass functionality."""
        config = AlertConfig(
            memory_threshold=80.0,
            gpu_memory_threshold=85.0,
            processing_time_threshold=25.0,
            cpu_threshold=90.0,
            enable_email_alerts=True,
            enable_console_alerts=False
        )

        assert config.memory_threshold == 80.0
        assert config.gpu_memory_threshold == 85.0
        assert config.processing_time_threshold == 25.0
        assert config.cpu_threshold == 90.0
        assert config.enable_email_alerts is True
        assert config.enable_console_alerts is False

    def test_progress_info_dataclass(self):
        """Test ProgressInfo dataclass functionality."""
        start_time = datetime.now()
        estimated_completion = start_time + timedelta(minutes=10)
        
        progress = ProgressInfo(
            operation_name="test_operation",
            current_step=5,
            total_steps=10,
            start_time=start_time,
            estimated_completion=estimated_completion,
            current_stage="processing",
            metadata={"batch_size": 4}
        )

        assert progress.operation_name == "test_operation"
        assert progress.current_step == 5
        assert progress.total_steps == 10
        assert progress.start_time == start_time
        assert progress.estimated_completion == estimated_completion
        assert progress.current_stage == "processing"
        assert progress.metadata["batch_size"] == 4

    def test_resource_utilization_dataclass(self):
        """Test ResourceUtilization dataclass functionality."""
        utilization = ResourceUtilization(
            cpu_percent=75.5,
            memory_percent=68.2,
            gpu_memory_percent=82.1,
            gpu_utilization=71.5,
            disk_io_read=15.3
        )

        assert utilization.cpu_percent == 75.5
        assert utilization.memory_percent == 68.2
        assert utilization.gpu_memory_percent == 82.1
        assert utilization.gpu_utilization == 71.5
        assert utilization.disk_io_read == 15.3

    def test_clear_completed_operations(self):
        """Test clearing completed operations."""
        dashboard = MonitoringDashboard()
        
        # Add completed operation
        completed_progress = ProgressInfo(
            operation_name="completed_op",
            current_step=10,
            total_steps=10,
            start_time=datetime.now() - timedelta(minutes=5),
            current_stage="completed"
        )
        dashboard.active_operations["completed_op"] = completed_progress
        
        # Add ongoing operation
        ongoing_progress = ProgressInfo(
            operation_name="ongoing_op",
            current_step=5,
            total_steps=10,
            start_time=datetime.now(),
            current_stage="processing"
        )
        dashboard.active_operations["ongoing_op"] = ongoing_progress
        
        dashboard.clear_completed_operations()
        
        # Only ongoing operation should remain
        assert "ongoing_op" in dashboard.active_operations
        assert "completed_op" not in dashboard.active_operations