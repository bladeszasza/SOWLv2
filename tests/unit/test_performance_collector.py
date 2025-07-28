"""
Unit tests for PerformanceCollector.
Tests timing, memory monitoring, GPU utilization tracking, and model comparison.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import torch

from sowlv2.optimizations.performance_collector import (
    PerformanceCollector, PerformanceMetrics, ComparisonReport, TimingContext
)


class TestPerformanceCollector:
    """Test suite for PerformanceCollector class."""
    
    def test_init_cuda_device(self):
        """Test initialization with CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1e9):
                collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                
                assert collector.device == "cuda"
                assert collector.enable_gpu_monitoring is True
                assert collector._active_timers == {}
                assert len(collector.operation_metrics) == 0
                assert len(collector.model_metrics) == 0
                assert len(collector.performance_history) == 0
    
    def test_init_cpu_device(self):
        """Test initialization with CPU device."""
        collector = PerformanceCollector(device="cpu", enable_gpu_monitoring=False)
        
        assert collector.device == "cpu"
        assert collector.enable_gpu_monitoring is False
    
    def test_init_gpu_monitoring_disabled_when_cuda_unavailable(self):
        """Test that GPU monitoring is disabled when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
            
            assert collector.enable_gpu_monitoring is False
    
    def test_start_timing_basic(self):
        """Test basic timing start functionality."""
        collector = PerformanceCollector(device="cpu")
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(used=4e9)  # 4GB used
            
            timer_id = collector.start_timing("test_operation")
            
            assert timer_id.startswith("test_operation_")
            assert timer_id in collector._active_timers
            
            context = collector._active_timers[timer_id]
            assert isinstance(context, TimingContext)
            assert context.operation == "test_operation"
            assert context.start_time > 0
            assert context.start_memory == 4.0  # 4GB
    
    def test_start_timing_with_metadata(self):
        """Test timing start with metadata."""
        collector = PerformanceCollector(device="cpu")
        metadata = {"frame_count": 10, "batch_size": 4}
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(used=2e9)
            
            timer_id = collector.start_timing("test_operation", metadata)
            
            context = collector._active_timers[timer_id]
            assert context.metadata == metadata
    
    def test_start_timing_cuda(self):
        """Test timing start with CUDA monitoring."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.synchronize'):
                with patch('torch.cuda.memory_allocated', return_value=2e9):
                    with patch('psutil.virtual_memory') as mock_memory:
                        mock_memory.return_value = Mock(used=4e9)
                        
                        collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                        timer_id = collector.start_timing("cuda_operation")
                        
                        context = collector._active_timers[timer_id]
                        assert context.start_gpu_memory == 2.0  # 2GB GPU memory
    
    def test_end_timing_basic(self):
        """Test basic timing end functionality."""
        collector = PerformanceCollector(device="cpu")
        
        with patch('psutil.virtual_memory') as mock_memory:
            with patch('psutil.cpu_percent', return_value=50.0):
                mock_memory.return_value = Mock(used=4e9)
                
                timer_id = collector.start_timing("test_operation")
                
                # Simulate some processing time
                time.sleep(0.01)
                
                mock_memory.return_value = Mock(used=5e9)  # Memory increased
                metrics = collector.end_timing(timer_id)
                
                assert isinstance(metrics, PerformanceMetrics)
                assert metrics.processing_time > 0
                assert metrics.memory_peak_usage == 1.0  # 1GB increase
                assert metrics.cpu_utilization == 50.0
                assert timer_id not in collector._active_timers
    
    def test_end_timing_with_frame_count(self):
        """Test timing end with throughput calculation."""
        collector = PerformanceCollector(device="cpu")
        metadata = {"frame_count": 10}
        
        with patch('psutil.virtual_memory') as mock_memory:
            with patch('psutil.cpu_percent', return_value=30.0):
                mock_memory.return_value = Mock(used=4e9)
                
                timer_id = collector.start_timing("test_operation", metadata)
                time.sleep(0.02)  # 20ms processing time
                
                metrics = collector.end_timing(timer_id)
                
                assert metrics.throughput_fps > 0
                # Should be approximately 10 frames / processing_time
                assert metrics.throughput_fps > 300  # Allow for timing variations
    
    def test_end_timing_cuda(self):
        """Test timing end with CUDA monitoring."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.synchronize'):
                with patch('torch.cuda.memory_allocated', side_effect=[1e9, 1e9, 3e9]):  # Start, start, end
                    with patch('psutil.virtual_memory') as mock_memory:
                        with patch('psutil.cpu_percent', return_value=40.0):
                            mock_memory.return_value = Mock(used=4e9)
                            
                            collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                            timer_id = collector.start_timing("cuda_operation")
                            time.sleep(0.01)
                            
                            metrics = collector.end_timing(timer_id)
                            
                            assert metrics.memory_peak_usage == 2.0  # 2GB GPU memory increase
                            assert metrics.gpu_utilization >= 0
    
    def test_end_timing_invalid_timer_id(self):
        """Test ending timing with invalid timer ID."""
        collector = PerformanceCollector(device="cpu")
        
        with pytest.raises(ValueError) as exc_info:
            collector.end_timing("invalid_timer_id")
        
        assert "Timer ID invalid_timer_id not found" in str(exc_info.value)
    
    def test_record_memory_usage_cpu(self):
        """Test memory usage recording for CPU."""
        collector = PerformanceCollector(device="cpu", enable_gpu_monitoring=False)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                used=8e9,  # 8GB used
                percent=75.0,
                available=2e9  # 2GB available
            )
            
            memory_stats = collector.record_memory_usage("test_stage")
            
            assert memory_stats['stage'] == "test_stage"
            assert memory_stats['system_memory_used_gb'] == 8.0
            assert memory_stats['system_memory_percent'] == 75.0
            assert memory_stats['system_memory_available_gb'] == 2.0
            assert 'timestamp' in memory_stats
            assert 'gpu_memory_allocated_gb' not in memory_stats
    
    def test_record_memory_usage_cuda(self):
        """Test memory usage recording for CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=4e9):
                with patch('torch.cuda.memory_reserved', return_value=5e9):
                    with patch('torch.cuda.get_device_properties') as mock_props:
                        mock_props.return_value = Mock(total_memory=8e9)
                        
                        with patch('psutil.virtual_memory') as mock_memory:
                            mock_memory.return_value = Mock(used=6e9, percent=60.0, available=4e9)
                            
                            collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                            memory_stats = collector.record_memory_usage("cuda_stage")
                            
                            assert memory_stats['gpu_memory_allocated_gb'] == 4.0
                            assert memory_stats['gpu_memory_reserved_gb'] == 5.0
                            assert memory_stats['gpu_memory_total_gb'] == 8.0
                            assert memory_stats['gpu_memory_percent'] == 50.0  # 4/8 * 100
    
    def test_record_gpu_utilization_unavailable(self):
        """Test GPU utilization recording when GPU is unavailable."""
        collector = PerformanceCollector(device="cpu", enable_gpu_monitoring=False)
        
        gpu_stats = collector.record_gpu_utilization("test_stage")
        
        assert gpu_stats['stage'] == "test_stage"
        assert gpu_stats['gpu_available'] is False
        assert 'gpu_memory_allocated_gb' not in gpu_stats
    
    def test_record_gpu_utilization_available(self):
        """Test GPU utilization recording when GPU is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=3e9):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_device_props = Mock()
                    mock_device_props.total_memory = 8e9
                    mock_device_props.name = "Test GPU"
                    mock_device_props.major = 7
                    mock_device_props.minor = 5
                    mock_device_props.multi_processor_count = 80
                    mock_props.return_value = mock_device_props
                    
                    collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                    gpu_stats = collector.record_gpu_utilization("cuda_stage")
                    
                    assert gpu_stats['gpu_available'] is True
                    assert gpu_stats['memory_utilization_percent'] == 37.5  # 3/8 * 100
                    assert gpu_stats['memory_allocated_gb'] == 3.0
                    assert gpu_stats['memory_total_gb'] == 8.0
                    assert gpu_stats['device_name'] == "Test GPU"
                    assert gpu_stats['compute_capability'] == "7.5"
                    assert gpu_stats['multiprocessor_count'] == 80
    
    def test_record_gpu_utilization_with_pynvml(self):
        """Test GPU utilization recording with pynvml available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=2e9):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value = Mock(
                        total_memory=8e9, name="Test GPU", major=7, minor=5, multi_processor_count=80
                    )
                    
                    # Mock pynvml
                    mock_pynvml = Mock()
                    mock_handle = Mock()
                    mock_utilization = Mock()
                    mock_utilization.gpu = 85.0
                    mock_utilization.memory = 60.0
                    
                    mock_pynvml.nvmlInit.return_value = None
                    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
                    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_utilization
                    mock_pynvml.nvmlDeviceGetTemperature.return_value = 75
                    mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 250000  # 250W in mW
                    mock_pynvml.NVML_TEMPERATURE_GPU = 0
                    
                    with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
                        collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                        gpu_stats = collector.record_gpu_utilization("cuda_stage")
                        
                        assert gpu_stats['gpu_utilization_percent'] == 85.0
                        assert gpu_stats['memory_utilization_percent'] == 60.0
                        assert gpu_stats['temperature_celsius'] == 75
                        assert gpu_stats['power_usage_watts'] == 250.0
    
    def test_record_gpu_utilization_pynvml_unavailable(self):
        """Test GPU utilization recording when pynvml is unavailable."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=2e9):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value = Mock(
                        total_memory=8e9, name="Test GPU", major=7, minor=5, multi_processor_count=80
                    )
                    
                    collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
                    
                    # pynvml import will fail
                    with patch('builtins.__import__', side_effect=ImportError("pynvml not available")):
                        gpu_stats = collector.record_gpu_utilization("cuda_stage")
                        
                        assert gpu_stats['gpu_utilization_percent'] == 0.0
                        assert 'note' in gpu_stats
                        assert "nvidia-ml-py" in gpu_stats['note']
    
    def test_compare_models_basic(self):
        """Test basic model comparison."""
        collector = PerformanceCollector(device="cpu")
        
        sam2_metrics = PerformanceMetrics(
            processing_time=2.0,
            memory_peak_usage=4.0,
            gpu_utilization=60.0,
            throughput_fps=10.0,
            model_loading_time=1.0,
            cpu_utilization=70.0
        )
        
        edgetam_metrics = PerformanceMetrics(
            processing_time=1.0,  # 2x faster
            memory_peak_usage=2.0,  # 2x less memory
            gpu_utilization=40.0,
            throughput_fps=20.0,
            model_loading_time=0.5,
            cpu_utilization=50.0
        )
        
        comparison = collector.compare_models(sam2_metrics, edgetam_metrics)
        
        assert isinstance(comparison, ComparisonReport)
        assert comparison.sam2_metrics == sam2_metrics
        assert comparison.edgetam_metrics == edgetam_metrics
        assert comparison.speed_improvement == 50.0  # (2.0 - 1.0) / 2.0 * 100
        assert comparison.memory_savings == 50.0  # (4.0 - 2.0) / 4.0 * 100
        assert "EdgeTAM" in comparison.recommendation
    
    def test_compare_models_with_quality_scores(self):
        """Test model comparison with quality scores."""
        collector = PerformanceCollector(device="cpu")
        
        sam2_metrics = PerformanceMetrics(
            processing_time=1.5, memory_peak_usage=3.0, gpu_utilization=50.0,
            throughput_fps=15.0, model_loading_time=0.8, cpu_utilization=60.0
        )
        
        edgetam_metrics = PerformanceMetrics(
            processing_time=1.0, memory_peak_usage=2.0, gpu_utilization=40.0,
            throughput_fps=20.0, model_loading_time=0.5, cpu_utilization=45.0
        )
        
        quality_scores = {
            "iou_score": (0.85, 0.80),  # SAM2 better
            "dice_score": (0.90, 0.88)  # SAM2 slightly better
        }
        
        comparison = collector.compare_models(sam2_metrics, edgetam_metrics, quality_scores)
        
        assert comparison.quality_comparison is not None
        assert "iou_score" in comparison.quality_comparison
        assert "dice_score" in comparison.quality_comparison
        
        # EdgeTAM should have negative quality difference (worse than SAM2)
        assert comparison.quality_comparison["iou_score"] < 0
        assert comparison.quality_comparison["dice_score"] < 0
    
    def test_compare_models_zero_values(self):
        """Test model comparison with zero values."""
        collector = PerformanceCollector(device="cpu")
        
        sam2_metrics = PerformanceMetrics(
            processing_time=0.0,  # Zero time
            memory_peak_usage=0.0,  # Zero memory
            gpu_utilization=0.0, throughput_fps=0.0, model_loading_time=0.0, cpu_utilization=0.0
        )
        
        edgetam_metrics = PerformanceMetrics(
            processing_time=1.0, memory_peak_usage=2.0, gpu_utilization=40.0,
            throughput_fps=20.0, model_loading_time=0.5, cpu_utilization=45.0
        )
        
        comparison = collector.compare_models(sam2_metrics, edgetam_metrics)
        
        assert comparison.speed_improvement == 0.0  # Can't calculate with zero baseline
        assert comparison.memory_savings == 0.0
    
    def test_generate_model_recommendation_edgetam_better(self):
        """Test recommendation generation when EdgeTAM is better."""
        collector = PerformanceCollector(device="cpu")
        
        recommendation = collector._generate_model_recommendation(
            speed_improvement=25.0,  # EdgeTAM 25% faster
            memory_savings=20.0,     # EdgeTAM uses 20% less memory
            quality_comparison={"iou": -2.0}  # EdgeTAM slightly worse quality
        )
        
        assert "EdgeTAM" in recommendation
        assert "performance-critical" in recommendation
        assert "significant speed improvement" in recommendation
        assert "less memory" in recommendation
    
    def test_generate_model_recommendation_sam2_better(self):
        """Test recommendation generation when SAM2 is better."""
        collector = PerformanceCollector(device="cpu")
        
        recommendation = collector._generate_model_recommendation(
            speed_improvement=-15.0,  # SAM2 15% faster
            memory_savings=-20.0,     # SAM2 uses 20% less memory
            quality_comparison={"iou": 8.0}  # SAM2 much better quality
        )
        
        assert "SAM2" in recommendation
        assert "significantly faster" in recommendation
        assert "more memory efficient" in recommendation
        assert "better quality" in recommendation
    
    def test_get_operation_summary_success(self):
        """Test getting operation summary with data."""
        collector = PerformanceCollector(device="cpu")
        
        # Add some mock metrics
        metrics1 = PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0)
        metrics2 = PerformanceMetrics(1.5, 2.5, 35.0, 12.0, 0.6, 45.0)
        metrics3 = PerformanceMetrics(0.8, 1.8, 25.0, 15.0, 0.4, 35.0)
        
        collector.operation_metrics["test_operation"] = [metrics1, metrics2, metrics3]
        
        summary = collector.get_operation_summary("test_operation")
        
        assert summary["operation"] == "test_operation"
        assert summary["total_runs"] == 3
        assert summary["processing_time"]["mean"] == pytest.approx(1.1, rel=1e-2)  # (1.0+1.5+0.8)/3
        assert summary["processing_time"]["min"] == 0.8
        assert summary["processing_time"]["max"] == 1.5
        assert summary["memory_usage"]["mean"] == pytest.approx(2.1, rel=1e-2)  # (2.0+2.5+1.8)/3
        assert summary["throughput"]["mean"] == pytest.approx(12.33, rel=1e-2)  # (10+12+15)/3
    
    def test_get_operation_summary_no_data(self):
        """Test getting operation summary with no data."""
        collector = PerformanceCollector(device="cpu")
        
        summary = collector.get_operation_summary("nonexistent_operation")
        
        assert "error" in summary
        assert "No metrics found" in summary["error"]
    
    def test_get_operation_summary_empty_metrics(self):
        """Test getting operation summary with empty metrics list."""
        collector = PerformanceCollector(device="cpu")
        collector.operation_metrics["empty_operation"] = []
        
        summary = collector.get_operation_summary("empty_operation")
        
        assert "error" in summary
        assert "No metrics recorded" in summary["error"]
    
    def test_clear_metrics_specific_operation(self):
        """Test clearing metrics for specific operation."""
        collector = PerformanceCollector(device="cpu")
        
        # Add metrics for multiple operations
        collector.operation_metrics["op1"] = [Mock()]
        collector.operation_metrics["op2"] = [Mock()]
        collector.model_metrics["model1"] = Mock()
        
        collector.clear_metrics("op1")
        
        assert len(collector.operation_metrics["op1"]) == 0
        assert len(collector.operation_metrics["op2"]) == 1  # Should remain
        assert len(collector.model_metrics) == 1  # Should remain
    
    def test_clear_metrics_all(self):
        """Test clearing all metrics."""
        collector = PerformanceCollector(device="cpu")
        
        # Add various metrics
        collector.operation_metrics["op1"] = [Mock()]
        collector.model_metrics["model1"] = Mock()
        collector.performance_history.append({"test": "data"})
        
        collector.clear_metrics()
        
        assert len(collector.operation_metrics) == 0
        assert len(collector.model_metrics) == 0
        assert len(collector.performance_history) == 0
    
    def test_export_metrics(self):
        """Test exporting metrics."""
        with patch('torch.cuda.is_available', return_value=True):
            collector = PerformanceCollector(device="cuda", enable_gpu_monitoring=True)
        
        # Add some test data
        metrics = PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0)
        collector.operation_metrics["test_op"] = [metrics]
        collector.model_metrics["test_model"] = metrics
        collector.performance_history.append({"type": "test", "data": {"value": 123}})
        
        exported = collector.export_metrics()
        
        assert exported["device"] == "cuda"
        assert exported["gpu_monitoring_enabled"] is True
        assert "operation_metrics" in exported
        assert "model_metrics" in exported
        assert "performance_history" in exported
        assert "export_timestamp" in exported
        
        # Check operation metrics structure
        assert "test_op" in exported["operation_metrics"]
        op_metrics = exported["operation_metrics"]["test_op"][0]
        assert op_metrics["processing_time"] == 1.0
        assert op_metrics["memory_peak_usage"] == 2.0
        assert "timestamp" in op_metrics
        
        # Check model metrics structure
        assert "test_model" in exported["model_metrics"]
        model_metrics = exported["model_metrics"]["test_model"]
        assert model_metrics["processing_time"] == 1.0
        assert model_metrics["gpu_utilization"] == 30.0
    
    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass functionality."""
        metrics = PerformanceMetrics(
            processing_time=1.5,
            memory_peak_usage=3.2,
            gpu_utilization=75.0,
            throughput_fps=25.5,
            model_loading_time=0.8,
            cpu_utilization=60.0
        )
        
        assert metrics.processing_time == 1.5
        assert metrics.memory_peak_usage == 3.2
        assert metrics.gpu_utilization == 75.0
        assert metrics.throughput_fps == 25.5
        assert metrics.model_loading_time == 0.8
        assert metrics.cpu_utilization == 60.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_comparison_report_dataclass(self):
        """Test ComparisonReport dataclass functionality."""
        sam2_metrics = PerformanceMetrics(2.0, 4.0, 60.0, 10.0, 1.0, 70.0)
        edgetam_metrics = PerformanceMetrics(1.0, 2.0, 40.0, 20.0, 0.5, 50.0)
        
        report = ComparisonReport(
            sam2_metrics=sam2_metrics,
            edgetam_metrics=edgetam_metrics,
            speed_improvement=50.0,
            memory_savings=50.0,
            quality_comparison={"iou": -5.0},
            recommendation="Use EdgeTAM for speed"
        )
        
        assert report.sam2_metrics == sam2_metrics
        assert report.edgetam_metrics == edgetam_metrics
        assert report.speed_improvement == 50.0
        assert report.memory_savings == 50.0
        assert report.quality_comparison == {"iou": -5.0}
        assert report.recommendation == "Use EdgeTAM for speed"
    
    def test_timing_context_dataclass(self):
        """Test TimingContext dataclass functionality."""
        metadata = {"frame_count": 10}
        
        context = TimingContext(
            operation="test_op",
            start_time=time.time(),
            start_memory=2.0,
            start_gpu_memory=1.0,
            metadata=metadata
        )
        
        assert context.operation == "test_op"
        assert context.start_time > 0
        assert context.start_memory == 2.0
        assert context.start_gpu_memory == 1.0
        assert context.metadata == metadata