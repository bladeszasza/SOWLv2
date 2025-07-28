"""
Unit tests for IntelligentBatchOptimizer.
Tests GPU profiling, adaptive optimization, and batch processing with failure recovery.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import torch

from sowlv2.optimizations.batch_optimizer import (
    IntelligentBatchOptimizer, BatchConfig, GPUProfile, BatchPerformanceMetrics,
    OptimizationLevel
)


class TestIntelligentBatchOptimizer:
    """Test suite for IntelligentBatchOptimizer class."""
    
    def test_init_cuda_device(self):
        """Test initialization with CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9,
                    major=7,
                    minor=5,
                    multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(
                        device="cuda", 
                        optimization_level=OptimizationLevel.BALANCED
                    )
                    
                    assert optimizer.device == "cuda"
                    assert optimizer.optimization_level == OptimizationLevel.BALANCED
                    assert optimizer.gpu_profile is not None
                    assert optimizer.gpu_profile.total_memory == 8.0
                    assert optimizer.gpu_profile.available_memory == 7.0  # 8 - 1
                    assert optimizer.gpu_profile.supports_mixed_precision is True
                    assert optimizer.gpu_profile.compute_units == 80
                    assert optimizer.profiling_results == {}
                    assert optimizer.adaptive_history == []
                    assert optimizer.failure_recovery_enabled is True
    
    def test_init_cpu_device(self):
        """Test initialization with CPU device."""
        optimizer = IntelligentBatchOptimizer(device="cpu")
        
        assert optimizer.device == "cpu"
        assert optimizer.gpu_profile is None
        assert optimizer.optimization_level == OptimizationLevel.BALANCED
    
    def test_estimate_memory_bandwidth(self):
        """Test memory bandwidth estimation."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        # Test Ampere (major >= 8)
        props_ampere = Mock(major=8, minor=0)
        bandwidth = optimizer._estimate_memory_bandwidth(props_ampere)
        assert bandwidth == 900.0
        
        # Test Turing/Volta (major == 7)
        props_turing = Mock(major=7, minor=5)
        bandwidth = optimizer._estimate_memory_bandwidth(props_turing)
        assert bandwidth == 600.0
        
        # Test older architectures
        props_old = Mock(major=6, minor=1)
        bandwidth = optimizer._estimate_memory_bandwidth(props_old)
        assert bandwidth == 400.0
    
    def test_profile_gpu_memory_for_batch_size_success(self):
        """Test GPU memory profiling for successful batch sizes."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                # Mock memory_allocated to return consistent values for each call
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    def test_func(batch_size):
                        # Simulate successful processing
                        time.sleep(0.01)  # Small delay
                        return ["result"] * batch_size
                    
                    with patch('torch.cuda.empty_cache'):
                        with patch('torch.cuda.synchronize'):
                            with patch('torch.cuda.max_memory_allocated', return_value=3e9):
                                with patch('torch.cuda.reset_peak_memory_stats'):
                                    
                                    results = optimizer.profile_gpu_memory_for_batch_size(
                                        test_func, [2, 4, 8]
                                    )
                    
                    assert len(results) == 3
                    for batch_size in [2, 4, 8]:
                        assert batch_size in results
                        metrics = results[batch_size]
                        assert isinstance(metrics, BatchPerformanceMetrics)
                        assert metrics.batch_size == batch_size
                        assert metrics.processing_time > 0
                        assert metrics.memory_peak > 0
                        assert metrics.throughput > 0
                        assert metrics.success_rate == 1.0
    
    def test_profile_gpu_memory_for_batch_size_oom(self):
        """Test GPU memory profiling with out-of-memory error."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    def test_func(batch_size):
                        if batch_size > 4:
                            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                        return ["result"] * batch_size
                    
                    with patch('torch.cuda.empty_cache'):
                        with patch('torch.cuda.synchronize'):
                            with patch('torch.cuda.max_memory_allocated', return_value=3e9):
                                with patch('torch.cuda.reset_peak_memory_stats'):
                                    
                                    results = optimizer.profile_gpu_memory_for_batch_size(
                                        test_func, [2, 4, 8, 16]
                                    )
                    
                    # Should stop at batch size 8 due to OOM
                    assert 2 in results
                    assert 4 in results
                    assert 8 in results
                    assert 16 not in results
                    
                    # Check that OOM batch size has success_rate = 0
                    assert results[8].success_rate == 0.0
    
    def test_profile_gpu_memory_no_gpu(self):
        """Test GPU memory profiling without GPU."""
        optimizer = IntelligentBatchOptimizer(device="cpu")
        
        def test_func(batch_size):
            return ["result"] * batch_size
        
        results = optimizer.profile_gpu_memory_for_batch_size(test_func, [2, 4])
        
        assert results == {}
    
    def test_find_optimal_batch_size(self):
        """Test finding optimal batch size through binary search."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    def test_func(batch_size):
                        # Simulate memory usage that fails at batch_size > 8
                        if batch_size > 8:
                            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                        return ["result"] * batch_size
                    
                    with patch.object(optimizer, 'profile_gpu_memory_for_batch_size') as mock_profile:
                        # Mock successful results up to batch size 8
                        def mock_profile_func(func, batch_sizes, *args, **kwargs):
                            results = {}
                            for bs in batch_sizes:
                                if bs <= 8:
                                    results[bs] = BatchPerformanceMetrics(
                                        batch_size=bs,
                                        processing_time=0.1,
                                        memory_peak=bs * 0.5,  # Linear memory usage
                                        throughput=bs / 0.1,
                                        memory_efficiency=bs * 0.5 / 8.0,  # Memory efficiency
                                        success_rate=1.0
                                    )
                                else:
                                    results[bs] = BatchPerformanceMetrics(
                                        batch_size=bs,
                                        processing_time=0.0,
                                        memory_peak=0.0,
                                        throughput=0.0,
                                        memory_efficiency=0.0,
                                        success_rate=0.0
                                    )
                            return results
                        
                        mock_profile.side_effect = mock_profile_func
                        
                        optimal_size = optimizer.find_optimal_batch_size(
                            test_func, max_batch_size=16, target_memory_usage=0.8
                        )
                        
                        assert optimal_size == 8  # Should find batch size 8 as optimal
    
    def test_find_optimal_batch_size_no_gpu(self):
        """Test finding optimal batch size without GPU."""
        optimizer = IntelligentBatchOptimizer(device="cpu")
        
        def test_func(batch_size):
            return ["result"] * batch_size
        
        optimal_size = optimizer.find_optimal_batch_size(test_func)
        assert optimal_size == 1
    
    def test_profile_and_optimize_cuda(self):
        """Test profiling and optimization with CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(
                        device="cuda", 
                        optimization_level=OptimizationLevel.BALANCED
                    )
                    
                    config = optimizer.profile_and_optimize(
                        test_image_size=(1024, 1024),
                        num_prompts=2,
                        memory_limit=6.0
                    )
                    
                    assert isinstance(config, BatchConfig)
                    assert config.detection_batch_size >= 1
                    assert config.segmentation_batch_size >= 1
                    assert config.frame_batch_size >= 1
                    assert config.use_mixed_precision is True
                    assert config.enable_gradient_checkpointing is False  # Balanced mode
                    assert config.optimization_level == OptimizationLevel.BALANCED
                    assert config.memory_limit_gb == 6.0
    
    def test_profile_and_optimize_cpu(self):
        """Test profiling and optimization with CPU."""
        optimizer = IntelligentBatchOptimizer(device="cpu")
        
        config = optimizer.profile_and_optimize(
            test_image_size=(1024, 1024),
            num_prompts=1
        )
        
        assert config.detection_batch_size == 1
        assert config.segmentation_batch_size == 1
        assert config.frame_batch_size == 1
        assert config.use_mixed_precision is False
        assert config.enable_gradient_checkpointing is True
    
    @pytest.mark.parametrize("optimization_level,expected_target,expected_safety", [
        (OptimizationLevel.CONSERVATIVE, 0.6, 0.7),
        (OptimizationLevel.BALANCED, 0.75, 0.8),
        (OptimizationLevel.AGGRESSIVE, 0.9, 0.9)
    ])
    def test_profile_and_optimize_levels(self, optimization_level, expected_target, expected_safety):
        """Test different optimization levels."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(
                        device="cuda", 
                        optimization_level=optimization_level
                    )
                    
                    config = optimizer.profile_and_optimize(
                        test_image_size=(512, 512),
                        num_prompts=1
                    )
                    
                    assert config.optimization_level == optimization_level
                    
                    # Conservative should have smaller batch sizes
                    if optimization_level == OptimizationLevel.CONSERVATIVE:
                        assert config.detection_batch_size <= 4
                        assert config.segmentation_batch_size <= 2
                        assert config.frame_batch_size <= 8
                    elif optimization_level == OptimizationLevel.AGGRESSIVE:
                        # Aggressive can have larger batch sizes
                        assert config.enable_gradient_checkpointing is False
    
    def test_adaptive_batch_processing_success(self):
        """Test successful adaptive batch processing."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    items = list(range(10))  # 10 items to process
                    
                    def process_func(batch):
                        return [f"processed_{item}" for item in batch]
                    
                    with patch('torch.cuda.empty_cache'):
                        with patch('torch.cuda.synchronize'):
                            with patch('torch.cuda.max_memory_allocated', return_value=3e9):
                                
                                results = optimizer.adaptive_batch_processing(
                                    items, process_func, initial_batch_size=3
                                )
                    
                    assert len(results) == 10
                    assert all("processed_" in result for result in results)
    
    def test_adaptive_batch_processing_oom_recovery(self):
        """Test adaptive batch processing with OOM recovery."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    items = list(range(8))
                    
                    def process_func(batch):
                        if len(batch) > 2:  # Fail for batch sizes > 2
                            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                        return [f"processed_{item}" for item in batch]
                    
                    with patch('torch.cuda.empty_cache'):
                        with patch('torch.cuda.synchronize'):
                            with patch('torch.cuda.max_memory_allocated', return_value=3e9):
                                
                                results = optimizer.adaptive_batch_processing(
                                    items, process_func, initial_batch_size=4
                                )
                    
                    assert len(results) == 8  # All items should be processed
                    assert all("processed_" in result for result in results)
    
    def test_adaptive_batch_processing_persistent_failure(self):
        """Test adaptive batch processing with persistent failures."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    items = list(range(5))
                    
                    def failing_process_func(batch):
                        raise torch.cuda.OutOfMemoryError("Persistent failure")
                    
                    with patch('torch.cuda.empty_cache'):
                        with patch('torch.cuda.synchronize'):
                            
                            results = optimizer.adaptive_batch_processing(
                                items, failing_process_func, initial_batch_size=2, max_retries=2
                            )
                    
                    # Should skip items after max retries
                    assert len(results) == 0
    
    def test_update_adaptive_parameters(self):
        """Test updating adaptive parameters."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    optimizer._update_adaptive_parameters(
                        batch_size=4,
                        processing_time=1.5,
                        memory_used=2.0,
                        success=True
                    )
                    
                    assert len(optimizer.adaptive_history) == 1
                    metrics = optimizer.adaptive_history[0]
                    assert metrics.batch_size == 4
                    assert metrics.processing_time == 1.5
                    assert metrics.memory_peak == 2.0
                    assert metrics.success_rate == 1.0
                    assert metrics.throughput == 4 / 1.5
    
    def test_adjust_batch_size_dynamically_increase(self):
        """Test dynamic batch size increase."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    with patch('torch.cuda.memory_allocated', return_value=2e9):  # Low memory usage
                        
                        optimizer = IntelligentBatchOptimizer(
                            device="cuda", 
                            optimization_level=OptimizationLevel.BALANCED
                        )
                        
                        new_size = optimizer._adjust_batch_size_dynamically(
                            current_batch_size=4,
                            consecutive_successes=5  # Many successes
                        )
                        
                        assert new_size > 4  # Should increase
    
    def test_adjust_batch_size_dynamically_decrease(self):
        """Test dynamic batch size decrease."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=7e9):  # High memory usage
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    new_size = optimizer._adjust_batch_size_dynamically(
                        current_batch_size=8,
                        consecutive_successes=1
                    )
                    
                    assert new_size < 8  # Should decrease
    
    def test_handle_batch_failure(self):
        """Test batch failure handling."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        # Test normal failure
        new_size = optimizer._handle_batch_failure(
            current_batch_size=8,
            consecutive_failures=1,
            retry_count=1
        )
        assert new_size == 4  # Should halve
        
        # Test repeated failures
        new_size = optimizer._handle_batch_failure(
            current_batch_size=8,
            consecutive_failures=3,
            retry_count=2
        )
        assert new_size == 2  # More aggressive reduction
    
    def test_enable_mixed_precision_support(self):
        """Test mixed precision support enabling."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    with patch('torch.cuda.amp.autocast'):
                        with patch('torch.randn') as mock_randn:
                            with patch('torch.matmul') as mock_matmul:
                                mock_tensor = Mock()
                                mock_randn.return_value = mock_tensor
                                mock_matmul.return_value = mock_tensor
                                
                                result = optimizer.enable_mixed_precision_support()
                                
                                assert result is True
    
    def test_enable_mixed_precision_support_failure(self):
        """Test mixed precision support failure."""
        optimizer = IntelligentBatchOptimizer(device="cpu")  # No GPU
        
        result = optimizer.enable_mixed_precision_support()
        assert result is False
    
    def test_get_optimization_recommendations(self):
        """Test optimization recommendations."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9, major=7, minor=5, multi_processor_count=80
                )
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    
                    optimizer = IntelligentBatchOptimizer(device="cuda")
                    
                    # Add some mock history
                    for i in range(5):
                        optimizer.adaptive_history.append(
                            BatchPerformanceMetrics(
                                batch_size=4,
                                processing_time=1.0,
                                memory_peak=2.0,
                                throughput=4.0,
                                memory_efficiency=0.25,  # Low efficiency
                                success_rate=1.0
                            )
                        )
                    
                    recommendations = optimizer.get_optimization_recommendations()
                    
                    assert "current_performance" in recommendations
                    assert "recommendations" in recommendations
                    assert recommendations["current_performance"]["memory_efficiency"] == 0.25
                    assert any("increasing batch sizes" in rec for rec in recommendations["recommendations"])
    
    def test_get_optimization_recommendations_no_data(self):
        """Test optimization recommendations with no data."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        recommendations = optimizer.get_optimization_recommendations()
        
        assert recommendations["status"] == "No profiling data available"
    
    def test_reset_adaptive_history(self):
        """Test resetting adaptive history."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        # Add some data
        optimizer.adaptive_history.append(
            BatchPerformanceMetrics(4, 1.0, 2.0, 4.0, 0.5, 1.0)
        )
        optimizer.profiling_results["test"] = BatchPerformanceMetrics(4, 1.0, 2.0, 4.0, 0.5, 1.0)
        
        optimizer.reset_adaptive_history()
        
        assert len(optimizer.adaptive_history) == 0
        assert len(optimizer.profiling_results) == 0
    
    def test_set_optimization_level(self):
        """Test setting optimization level."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        optimizer.set_optimization_level(OptimizationLevel.AGGRESSIVE)
        
        assert optimizer.optimization_level == OptimizationLevel.AGGRESSIVE
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        # Add mock metrics
        metrics = [
            BatchPerformanceMetrics(2, 0.5, 1.0, 4.0, 0.2, 1.0),
            BatchPerformanceMetrics(4, 1.0, 2.0, 4.0, 0.4, 1.0),
            BatchPerformanceMetrics(8, 2.0, 4.0, 4.0, 0.8, 0.5)
        ]
        optimizer.adaptive_history = metrics
        
        summary = optimizer.get_performance_summary()
        
        assert summary["total_batches_processed"] == 3
        assert summary["average_batch_size"] == (2 + 4 + 8) / 3
        assert summary["average_throughput"] == 4.0
        assert summary["average_memory_efficiency"] == (0.2 + 0.4 + 0.8) / 3
        assert summary["overall_success_rate"] == (1.0 + 1.0 + 0.5) / 3
        assert summary["total_processing_time"] == 3.5
    
    def test_get_performance_summary_empty(self):
        """Test performance summary with no data."""
        optimizer = IntelligentBatchOptimizer(device="cuda")
        
        summary = optimizer.get_performance_summary()
        
        assert summary == {}
    
    def test_optimization_level_enum(self):
        """Test OptimizationLevel enum values."""
        assert OptimizationLevel.CONSERVATIVE.value == 1
        assert OptimizationLevel.BALANCED.value == 2
        assert OptimizationLevel.AGGRESSIVE.value == 3
    
    def test_batch_config_dataclass(self):
        """Test BatchConfig dataclass."""
        config = BatchConfig(
            detection_batch_size=4,
            segmentation_batch_size=2,
            frame_batch_size=8,
            use_mixed_precision=True,
            enable_gradient_checkpointing=False,
            optimization_level=OptimizationLevel.BALANCED,
            memory_limit_gb=6.0
        )
        
        assert config.detection_batch_size == 4
        assert config.segmentation_batch_size == 2
        assert config.frame_batch_size == 8
        assert config.use_mixed_precision is True
        assert config.enable_gradient_checkpointing is False
        assert config.optimization_level == OptimizationLevel.BALANCED
        assert config.memory_limit_gb == 6.0
    
    def test_gpu_profile_dataclass(self):
        """Test GPUProfile dataclass."""
        profile = GPUProfile(
            total_memory=8.0,
            available_memory=6.0,
            compute_capability=(7, 5),
            supports_mixed_precision=True,
            memory_bandwidth=600.0,
            compute_units=80
        )
        
        assert profile.total_memory == 8.0
        assert profile.available_memory == 6.0
        assert profile.compute_capability == (7, 5)
        assert profile.supports_mixed_precision is True
        assert profile.memory_bandwidth == 600.0
        assert profile.compute_units == 80
    
    def test_batch_performance_metrics_dataclass(self):
        """Test BatchPerformanceMetrics dataclass."""
        metrics = BatchPerformanceMetrics(
            batch_size=4,
            processing_time=1.5,
            memory_peak=2.0,
            throughput=2.67,
            memory_efficiency=0.25,
            success_rate=1.0
        )
        
        assert metrics.batch_size == 4
        assert metrics.processing_time == 1.5
        assert metrics.memory_peak == 2.0
        assert metrics.throughput == 2.67
        assert metrics.memory_efficiency == 0.25
        assert metrics.success_rate == 1.0