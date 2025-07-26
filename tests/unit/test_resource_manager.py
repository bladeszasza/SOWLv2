"""
Unit tests for AdvancedResourceManager.
Tests memory monitoring, batch optimization, streaming configuration, and device allocation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import psutil

from sowlv2.optimizations.resource_manager import (
    AdvancedResourceManager, MemoryStats, BatchConfig, StreamingConfig, 
    DeviceAllocation, ProcessingMode
)


class TestAdvancedResourceManager:
    """Test suite for AdvancedResourceManager class."""
    
    def test_init_cuda_device(self):
        """Test initialization with CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(
                    total_memory=8e9,  # 8GB
                    major=7,
                    minor=5
                )
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value = Mock(total=16e9)  # 16GB system RAM
                    
                    manager = AdvancedResourceManager(device="cuda", memory_limit=6.0)
                    
                    assert manager.device == "cuda"
                    assert manager.memory_limit == 6.0
                    assert manager.total_gpu_memory == 8.0
                    assert manager.supports_mixed_precision is True
                    assert manager.total_system_memory == 16.0
    
    def test_init_cpu_device(self):
        """Test initialization with CPU device."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(total=16e9)
            
            manager = AdvancedResourceManager(device="cpu")
            
            assert manager.device == "cpu"
            assert manager.gpu_properties is None
            assert manager.total_gpu_memory == 0
            assert manager.supports_mixed_precision is False
            assert manager.total_system_memory == 16.0
    
    def test_monitor_memory_usage_cuda(self):
        """Test memory monitoring with CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=2e9):  # 2GB allocated
                with patch('torch.cuda.memory_reserved', return_value=3e9):  # 3GB cached
                    with patch('psutil.virtual_memory') as mock_memory:
                        mock_memory.return_value = Mock(percent=60.0)
                        
                        manager = AdvancedResourceManager(device="cuda")
                        manager.total_gpu_memory = 8.0  # 8GB total
                        
                        stats = manager.monitor_memory_usage()
                        
                        assert isinstance(stats, MemoryStats)
                        assert stats.total_memory == 8.0
                        assert stats.allocated_memory == 2.0
                        assert stats.cached_memory == 3.0
                        assert stats.free_memory == 6.0
                        assert stats.utilization_percentage == 25.0  # 2/8 * 100
                        assert stats.system_memory_usage == 60.0
    
    def test_monitor_memory_usage_cpu(self):
        """Test memory monitoring with CPU device."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9,
                available=8e9,
                percent=50.0
            )
            
            manager = AdvancedResourceManager(device="cpu")
            
            stats = manager.monitor_memory_usage()
            
            assert isinstance(stats, MemoryStats)
            assert stats.total_memory == 16.0
            assert stats.allocated_memory == 0
            assert stats.cached_memory == 0
            assert stats.free_memory == 8.0
            assert stats.utilization_percentage == 50.0
            assert stats.system_memory_usage == 50.0
    
    def test_monitor_memory_usage_history(self):
        """Test memory usage history tracking."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=8e9, percent=50.0
            )
            
            manager = AdvancedResourceManager(device="cpu")
            
            # Generate multiple measurements
            for _ in range(5):
                manager.monitor_memory_usage()
            
            assert len(manager.memory_history) == 5
            
            # Test history limit
            for _ in range(100):
                manager.monitor_memory_usage()
            
            assert len(manager.memory_history) == 100
    
    def test_optimize_batch_sizes_normal_mode(self):
        """Test batch size optimization in normal mode."""
        manager = AdvancedResourceManager(device="cuda")
        manager.total_gpu_memory = 8.0
        manager.supports_mixed_precision = True
        
        config = manager.optimize_batch_sizes(
            current_usage=50.0,  # Normal usage
            image_size=(1024, 1024),
            num_prompts=2
        )
        
        assert isinstance(config, BatchConfig)
        assert config.processing_mode == ProcessingMode.NORMAL
        assert config.detection_batch_size >= 1
        assert config.segmentation_batch_size >= 1
        assert config.frame_batch_size >= 1
        assert config.use_mixed_precision is True
        assert config.enable_gradient_checkpointing is False
    
    def test_optimize_batch_sizes_memory_efficient_mode(self):
        """Test batch size optimization in memory efficient mode."""
        manager = AdvancedResourceManager(device="cuda")
        manager.total_gpu_memory = 8.0
        
        config = manager.optimize_batch_sizes(
            current_usage=75.0,  # High usage
            image_size=(1024, 1024),
            num_prompts=1
        )
        
        assert config.processing_mode == ProcessingMode.MEMORY_EFFICIENT
        assert config.detection_batch_size <= 4
        assert config.segmentation_batch_size <= 2
        assert config.frame_batch_size <= 8
        assert config.enable_gradient_checkpointing is True
    
    def test_optimize_batch_sizes_streaming_mode(self):
        """Test batch size optimization in streaming mode."""
        manager = AdvancedResourceManager(device="cuda")
        manager.total_gpu_memory = 8.0
        
        config = manager.optimize_batch_sizes(
            current_usage=85.0,  # Very high usage
            image_size=(2048, 2048),
            num_prompts=3
        )
        
        assert config.processing_mode == ProcessingMode.STREAMING
        assert config.enable_gradient_checkpointing is True
    
    def test_optimize_batch_sizes_cpu_fallback_mode(self):
        """Test batch size optimization in CPU fallback mode."""
        manager = AdvancedResourceManager(device="cuda")
        manager.total_gpu_memory = 8.0
        
        config = manager.optimize_batch_sizes(
            current_usage=95.0,  # Critical usage
            image_size=(1024, 1024),
            num_prompts=1
        )
        
        assert config.processing_mode == ProcessingMode.CPU_FALLBACK
        assert config.detection_batch_size == 1
        assert config.segmentation_batch_size == 1
        assert config.frame_batch_size == 1
        assert config.use_mixed_precision is False
        assert config.enable_gradient_checkpointing is True
    
    def test_optimize_batch_sizes_with_memory_limit(self):
        """Test batch size optimization with memory limit."""
        manager = AdvancedResourceManager(device="cuda", memory_limit=4.0)
        manager.total_gpu_memory = 8.0
        
        config = manager.optimize_batch_sizes(
            current_usage=30.0,  # Low usage but limited memory
            image_size=(1024, 1024),
            num_prompts=1
        )
        
        # Should respect memory limit
        assert config.detection_batch_size <= 8
        assert config.segmentation_batch_size <= 4
    
    def test_enable_streaming_mode(self):
        """Test streaming mode configuration."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=8e9, percent=50.0
            )
            
            manager = AdvancedResourceManager(device="cuda")
            
            config = manager.enable_streaming_mode(
                video_size=1000,
                target_memory_usage=0.7
            )
            
            assert isinstance(config, StreamingConfig)
            assert config.chunk_size > 0
            assert config.overlap_frames >= 0
            assert config.memory_threshold == 0.7
            assert config.auto_cleanup is True
    
    def test_enable_streaming_mode_large_video(self):
        """Test streaming mode for large video."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=4e9, percent=75.0
            )
            
            manager = AdvancedResourceManager(device="cuda")
            
            config = manager.enable_streaming_mode(
                video_size=10000,  # Large video
                target_memory_usage=0.6
            )
            
            assert config.chunk_size < 10000
            assert config.enable_progressive_loading is True
            assert config.overlap_frames > 0
    
    def test_cleanup_resources_cuda(self):
        """Test resource cleanup with CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                with patch('torch.cuda.synchronize') as mock_sync:
                    with patch('gc.collect') as mock_gc:
                        with patch('psutil.virtual_memory') as mock_memory:
                            mock_memory.return_value = Mock(
                                total=16e9, available=4e9, percent=90.0
                            )
                            
                            manager = AdvancedResourceManager(device="cuda")
                            manager.cleanup_threshold = 0.8
                            
                            manager.cleanup_resources()
                            
                            mock_gc.assert_called_once()
                            mock_empty_cache.assert_called_once()
                            mock_sync.assert_called_once()
    
    def test_cleanup_resources_force(self):
        """Test forced resource cleanup."""
        with patch('gc.collect') as mock_gc:
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(
                    total=16e9, available=12e9, percent=25.0
                )
                
                manager = AdvancedResourceManager(device="cpu")
                
                manager.cleanup_resources(force=True)
                
                mock_gc.assert_called_once()
    
    def test_get_optimal_device_allocation_cuda_available(self):
        """Test device allocation with CUDA available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(
                    total=16e9, available=8e9, percent=50.0
                )
                
                manager = AdvancedResourceManager(device="cuda")
                
                allocation = manager.get_optimal_device_allocation()
                
                assert isinstance(allocation, DeviceAllocation)
                assert allocation.primary_device == "cuda"
                assert allocation.fallback_device == "cpu"
                assert "owl" in allocation.model_device_mapping
                assert "sam2" in allocation.model_device_mapping
                assert "edgetam" in allocation.model_device_mapping
                assert "vjepa2" in allocation.model_device_mapping
                assert sum(allocation.memory_allocation.values()) <= 1.0
    
    def test_get_optimal_device_allocation_high_memory_usage(self):
        """Test device allocation with high memory usage."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(
                    total=16e9, available=2e9, percent=85.0
                )
                
                manager = AdvancedResourceManager(device="cuda")
                
                allocation = manager.get_optimal_device_allocation()
                
                # Should fallback to CPU for primary device
                assert allocation.primary_device == "cpu"
                assert allocation.fallback_device == "cuda"
    
    def test_get_optimal_device_allocation_cpu_only(self):
        """Test device allocation with CPU only."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(
                    total=16e9, available=8e9, percent=50.0
                )
                
                manager = AdvancedResourceManager(device="cpu")
                
                allocation = manager.get_optimal_device_allocation()
                
                assert allocation.primary_device == "cpu"
                assert allocation.fallback_device == "cpu"
                assert all(device == "cpu" for device in allocation.model_device_mapping.values())
    
    def test_get_memory_trend_empty_history(self):
        """Test memory trend analysis with empty history."""
        manager = AdvancedResourceManager(device="cpu")
        
        trend = manager.get_memory_trend()
        
        assert trend["trend"] == 0.0
        assert trend["stability"] == 1.0
        assert trend["peak_usage"] == 0.0
    
    def test_get_memory_trend_with_history(self):
        """Test memory trend analysis with history."""
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate increasing memory usage
            memory_values = [
                Mock(total=16e9, available=12e9, percent=25.0),
                Mock(total=16e9, available=10e9, percent=37.5),
                Mock(total=16e9, available=8e9, percent=50.0),
                Mock(total=16e9, available=6e9, percent=62.5),
                Mock(total=16e9, available=4e9, percent=75.0)
            ]
            
            manager = AdvancedResourceManager(device="cpu")
            
            for memory_value in memory_values:
                mock_memory.return_value = memory_value
                manager.monitor_memory_usage()
            
            trend = manager.get_memory_trend(window_size=5)
            
            assert trend["trend"] > 0  # Increasing trend
            assert trend["peak_usage"] == 75.0
            assert trend["current_usage"] == 75.0
            assert "stability" in trend
    
    def test_should_enable_streaming_high_memory_requirement(self):
        """Test streaming recommendation for high memory requirement."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=4e9, percent=75.0
            )
            
            manager = AdvancedResourceManager(device="cuda")
            
            should_stream = manager.should_enable_streaming(
                video_frames=5000,  # Large video
                frame_size=(1920, 1080)
            )
            
            assert should_stream is True
    
    def test_should_enable_streaming_high_current_usage(self):
        """Test streaming recommendation for high current memory usage."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=12e9, percent=80.0  # High usage
            )
            
            manager = AdvancedResourceManager(device="cuda")
            
            should_stream = manager.should_enable_streaming(
                video_frames=500,  # Moderate video
                frame_size=(1024, 1024)
            )
            
            assert should_stream is True
    
    def test_should_enable_streaming_long_video(self):
        """Test streaming recommendation for very long video."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=12e9, percent=25.0  # Low usage
            )
            
            manager = AdvancedResourceManager(device="cuda")
            
            should_stream = manager.should_enable_streaming(
                video_frames=2000,  # Very long video
                frame_size=(512, 512)
            )
            
            assert should_stream is True
    
    def test_should_enable_streaming_small_video(self):
        """Test streaming recommendation for small video."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=12e9, percent=25.0
            )
            
            manager = AdvancedResourceManager(device="cuda")
            
            should_stream = manager.should_enable_streaming(
                video_frames=100,  # Small video
                frame_size=(512, 512)
            )
            
            assert should_stream is False
    
    def test_monitoring_enabled_disabled(self):
        """Test disabling memory monitoring."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=16e9, available=8e9, percent=50.0
            )
            
            manager = AdvancedResourceManager(device="cpu")
            manager.monitoring_enabled = False
            
            # Monitor memory multiple times
            for _ in range(3):
                manager.monitor_memory_usage()
            
            # History should not be updated
            assert len(manager.memory_history) == 0
    
    def test_memory_stats_dataclass(self):
        """Test MemoryStats dataclass functionality."""
        stats = MemoryStats(
            total_memory=8.0,
            allocated_memory=4.0,
            cached_memory=1.0,
            free_memory=4.0,
            utilization_percentage=50.0,
            system_memory_usage=60.0
        )
        
        assert stats.total_memory == 8.0
        assert stats.allocated_memory == 4.0
        assert stats.cached_memory == 1.0
        assert stats.free_memory == 4.0
        assert stats.utilization_percentage == 50.0
        assert stats.system_memory_usage == 60.0
    
    def test_batch_config_dataclass(self):
        """Test BatchConfig dataclass functionality."""
        config = BatchConfig(
            detection_batch_size=4,
            segmentation_batch_size=2,
            frame_batch_size=8,
            use_mixed_precision=True,
            enable_gradient_checkpointing=False,
            processing_mode=ProcessingMode.NORMAL
        )
        
        assert config.detection_batch_size == 4
        assert config.segmentation_batch_size == 2
        assert config.frame_batch_size == 8
        assert config.use_mixed_precision is True
        assert config.enable_gradient_checkpointing is False
        assert config.processing_mode == ProcessingMode.NORMAL
    
    def test_streaming_config_dataclass(self):
        """Test StreamingConfig dataclass functionality."""
        config = StreamingConfig(
            chunk_size=100,
            overlap_frames=5,
            enable_progressive_loading=True,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        assert config.chunk_size == 100
        assert config.overlap_frames == 5
        assert config.enable_progressive_loading is True
        assert config.memory_threshold == 0.8
        assert config.auto_cleanup is True
    
    def test_device_allocation_dataclass(self):
        """Test DeviceAllocation dataclass functionality."""
        allocation = DeviceAllocation(
            primary_device="cuda",
            fallback_device="cpu",
            model_device_mapping={"owl": "cuda", "sam2": "cpu"},
            memory_allocation={"owl": 0.3, "sam2": 0.4}
        )
        
        assert allocation.primary_device == "cuda"
        assert allocation.fallback_device == "cpu"
        assert allocation.model_device_mapping["owl"] == "cuda"
        assert allocation.memory_allocation["owl"] == 0.3
    
    def test_processing_mode_enum(self):
        """Test ProcessingMode enum values."""
        assert ProcessingMode.NORMAL.value == "normal"
        assert ProcessingMode.MEMORY_EFFICIENT.value == "memory_efficient"
        assert ProcessingMode.STREAMING.value == "streaming"
        assert ProcessingMode.CPU_FALLBACK.value == "cpu_fallback"
    
    @pytest.mark.parametrize("device,expected_mixed_precision", [
        ("cuda", True),
        ("cpu", False)
    ])
    def test_mixed_precision_support(self, device, expected_mixed_precision):
        """Test mixed precision support detection."""
        if device == "cuda":
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value = Mock(
                        total_memory=8e9, major=7, minor=5
                    )
                    with patch('psutil.virtual_memory') as mock_memory:
                        mock_memory.return_value = Mock(total=16e9)
                        
                        manager = AdvancedResourceManager(device=device)
                        assert manager.supports_mixed_precision == expected_mixed_precision
        else:
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(total=16e9)
                
                manager = AdvancedResourceManager(device=device)
                assert manager.supports_mixed_precision == expected_mixed_precision
    
    @pytest.mark.parametrize("usage,expected_mode", [
        (50.0, ProcessingMode.NORMAL),
        (75.0, ProcessingMode.MEMORY_EFFICIENT),
        (85.0, ProcessingMode.STREAMING),
        (95.0, ProcessingMode.CPU_FALLBACK)
    ])
    def test_processing_mode_selection(self, usage, expected_mode):
        """Test processing mode selection based on memory usage."""
        manager = AdvancedResourceManager(device="cuda")
        manager.total_gpu_memory = 8.0
        
        config = manager.optimize_batch_sizes(
            current_usage=usage,
            image_size=(1024, 1024),
            num_prompts=1
        )
        
        assert config.processing_mode == expected_mode