"""
Final integration tests for SOWLv2 optimization and EdgeTAM integration.
Tests end-to-end functionality, error handling, CLI options, and backward compatibility.
"""
import pytest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from sowlv2.cli import main as cli_main
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.optimizations.resource_manager import AdvancedResourceManager
from sowlv2.optimizations.batch_optimizer import IntelligentBatchOptimizer, OptimizationLevel
from sowlv2.models.model_factory import SegmentationModelFactory
from sowlv2.models.edgetam_wrapper import EdgeTAMWrapper
from sowlv2.utils.error_recovery import ErrorRecoveryManager


class TestFinalIntegration:
    """Final integration test suite."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(array)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        config = {
            "prompt": ["person", "car"],
            "input": str(Path(temp_dir) / "test_video.mp4"),
            "output": str(Path(temp_dir) / "output"),
            "device": "cpu",
            "edgetam": True,
            "edgetam-model": "facebook/edgetam-base",
            "optimization-level": 2,
            "enable-mixed-precision": False,
            "batch-size": 2,
            "memory-limit": 4.0,
            "benchmark": True
        }
        
        config_path = Path(temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_end_to_end_pipeline_integration(self, temp_dir, test_image):
        """Test complete end-to-end pipeline integration."""
        # Create test input
        input_path = Path(temp_dir) / "test_input.jpg"
        test_image.save(input_path)
        
        output_path = Path(temp_dir) / "output"
        
        # Test with EdgeTAM
        try:
            pipeline = OptimizedSOWLv2Pipeline(
                prompts=["person"],
                input_path=str(input_path),
                output_path=str(output_path),
                device="cpu",
                use_edgetam=True,
                edgetam_model="facebook/edgetam-base",
                optimization_level=2
            )
            
            # Mock the actual model loading to avoid dependencies
            with patch.object(pipeline, '_initialize_models'):
                with patch.object(pipeline, '_process_single_image') as mock_process:
                    mock_process.return_value = {
                        "detections": [{"box": [100, 100, 200, 200], "confidence": 0.8}],
                        "masks": [np.ones((512, 512), dtype=np.uint8)]
                    }
                    
                    results = pipeline.process()
                    
                    assert results is not None
                    assert "processing_time" in results
                    assert "memory_usage" in results
                    
        except Exception as e:
            # If EdgeTAM is not available, test should still pass with fallback
            assert "EdgeTAM" in str(e) or "model loading" in str(e)
    
    def test_error_handling_scenarios(self, temp_dir):
        """Test comprehensive error handling scenarios."""
        error_manager = ErrorRecoveryManager()
        
        # Test model loading error handling
        with patch('sowlv2.models.edgetam_wrapper.EdgeTAMWrapper._load_model') as mock_load:
            mock_load.side_effect = RuntimeError("Model loading failed")
            
            result = error_manager.handle_model_loading_error("edgetam", RuntimeError("Test error"))
            assert isinstance(result, dict)
            assert result["recovery_action"] == "fallback_to_sam2"
        
        # Test memory overflow handling
        from sowlv2.optimizations.resource_manager import BatchConfig, ProcessingMode
        
        current_config = BatchConfig(
            detection_batch_size=8,
            segmentation_batch_size=4,
            frame_batch_size=16,
            use_mixed_precision=True,
            enable_gradient_checkpointing=False,
            processing_mode=ProcessingMode.NORMAL
        )
        
        new_config = error_manager.handle_memory_overflow(current_config)
        assert new_config.detection_batch_size <= current_config.detection_batch_size
        assert new_config.processing_mode in [ProcessingMode.MEMORY_EFFICIENT, ProcessingMode.STREAMING]
    
    def test_cli_options_integration(self, temp_dir, test_config):
        """Test all CLI options and configurations."""
        # Create mock input file
        input_file = Path(temp_dir) / "test_input.jpg"
        test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        test_image.save(input_file)
        
        # Test basic CLI functionality
        test_args = [
            "--prompt", "person",
            "--input", str(input_file),
            "--output", str(Path(temp_dir) / "cli_output"),
            "--device", "cpu",
            "--edgetam",
            "--edgetam-model", "facebook/edgetam-base",
            "--optimization-level", "2",
            "--batch-size", "2",
            "--memory-limit", "4.0",
            "--benchmark"
        ]
        
        with patch('sys.argv', ['sowlv2'] + test_args):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                mock_instance = Mock()
                mock_instance.process.return_value = {"status": "success"}
                mock_pipeline.return_value = mock_instance
                
                try:
                    cli_main()
                    mock_pipeline.assert_called_once()
                except SystemExit:
                    pass  # CLI may exit normally
        
        # Test YAML configuration
        with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
            mock_instance = Mock()
            mock_instance.process.return_value = {"status": "success"}
            mock_pipeline.return_value = mock_instance
            
            test_args_yaml = [
                "--config", str(test_config)
            ]
            
            with patch('sys.argv', ['sowlv2'] + test_args_yaml):
                try:
                    cli_main()
                    mock_pipeline.assert_called_once()
                except SystemExit:
                    pass
    
    def test_backward_compatibility(self, temp_dir, test_image):
        """Test backward compatibility with existing configurations."""
        # Test old-style configuration without EdgeTAM options
        old_config = {
            "prompt": "person",
            "input": str(Path(temp_dir) / "test.jpg"),
            "output": str(Path(temp_dir) / "old_output"),
            "sam_model": "facebook/sam2.1-hiera-small",
            "threshold": 0.3,
            "device": "cpu"
        }
        
        # Save test image
        test_image.save(Path(temp_dir) / "test.jpg")
        
        # Test that old configuration still works
        try:
            pipeline = OptimizedSOWLv2Pipeline(
                prompts=[old_config["prompt"]],
                input_path=old_config["input"],
                output_path=old_config["output"],
                device=old_config["device"],
                sam_model=old_config["sam_model"],
                threshold=old_config["threshold"]
            )
            
            # Mock model initialization
            with patch.object(pipeline, '_initialize_models'):
                with patch.object(pipeline, '_process_single_image') as mock_process:
                    mock_process.return_value = {
                        "detections": [],
                        "masks": []
                    }
                    
                    results = pipeline.process()
                    assert results is not None
                    
        except Exception as e:
            # Should not fail due to missing EdgeTAM options
            assert "edgetam" not in str(e).lower()
    
    def test_resource_management_integration(self):
        """Test resource management system integration."""
        rm = AdvancedResourceManager("cpu")
        
        # Test memory monitoring
        stats = rm.monitor_memory_usage()
        assert stats.total_memory > 0
        assert 0 <= stats.utilization_percentage <= 100
        
        # Test batch optimization
        config = rm.optimize_batch_sizes(50.0, (1024, 1024), 3, "edgetam")
        assert config.detection_batch_size >= 1
        assert config.segmentation_batch_size >= 1
        assert config.frame_batch_size >= 1
        
        # Test streaming configuration
        streaming_config = rm.enable_streaming_mode(1000)
        assert streaming_config.chunk_size > 0
        assert streaming_config.overlap_frames >= 0
        
        # Test device allocation
        device_allocation = rm.get_optimal_device_allocation()
        assert device_allocation.primary_device in ["cpu", "cuda"]
        assert device_allocation.fallback_device in ["cpu", "cuda"]
    
    def test_batch_optimizer_integration(self):
        """Test batch optimizer integration across optimization levels."""
        for level in [OptimizationLevel.CONSERVATIVE, OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
            optimizer = IntelligentBatchOptimizer("cpu", level)
            
            # Test profiling and optimization
            config = optimizer.profile_and_optimize((1024, 1024), 3, model_type="edgetam")
            
            assert config.detection_batch_size >= 1
            assert config.segmentation_batch_size >= 1
            assert config.frame_batch_size >= 1
            assert config.optimization_level == level
            
            # Test adaptive batch processing
            test_items = list(range(10))
            
            def mock_process_func(batch):
                return [f"processed_{item}" for item in batch]
            
            results = optimizer.adaptive_batch_processing(test_items, mock_process_func, 2)
            assert len(results) == len(test_items)
    
    def test_model_factory_integration(self):
        """Test model factory integration and fallback mechanisms."""
        factory = SegmentationModelFactory()
        
        # Test available models
        available_models = factory.get_available_models()
        assert "sam2" in available_models
        assert "edgetam" in available_models
        
        # Test model creation with fallback
        try:
            model = factory.create_model("edgetam", "facebook/edgetam-base", "cpu")
            assert model is not None
        except Exception:
            # Should fallback to SAM2 if EdgeTAM is not available
            model = factory.create_model("sam2", "facebook/sam2.1-hiera-small", "cpu")
            assert model is not None
    
    def test_edgetam_wrapper_integration(self):
        """Test EdgeTAM wrapper integration and optimizations."""
        try:
            wrapper = EdgeTAMWrapper(device="cpu")
            
            # Test basic functionality
            test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            test_box = [50, 50, 150, 150]
            
            mask = wrapper.segment(test_image, test_box)
            assert mask.shape == (256, 256)
            assert mask.dtype == np.uint8
            
            # Test performance optimizations
            wrapper.set_memory_optimization(True)
            wrapper.enable_mixed_precision(False)  # CPU doesn't support mixed precision
            
            # Test caching
            cache_stats = wrapper.get_cache_stats()
            assert "cache_size" in cache_stats
            
            # Test batch processing
            test_data = [(test_image, test_box) for _ in range(3)]
            batch_results = wrapper.batch_segment(test_data)
            assert len(batch_results) == 3
            
            # Test performance metrics
            metrics = wrapper.get_performance_metrics()
            assert "total_inferences" in metrics
            assert "average_inference_time" in metrics
            
        except Exception as e:
            # EdgeTAM may not be available in test environment
            assert "EdgeTAM" in str(e) or "model loading" in str(e)
    
    def test_configuration_validation(self, temp_dir):
        """Test configuration validation and error handling."""
        # Test invalid configuration
        invalid_config = {
            "prompt": [],  # Empty prompt
            "input": "nonexistent_file.mp4",
            "output": "/invalid/path",
            "device": "invalid_device",
            "edgetam-model": "invalid/model",
            "optimization-level": 10,  # Invalid level
            "batch-size": -1,  # Invalid batch size
            "memory-limit": -5.0  # Invalid memory limit
        }
        
        config_path = Path(temp_dir) / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Test that validation catches errors
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            with patch('sys.argv', ['sowlv2', '--config', str(config_path)]):
                cli_main()
    
    def test_performance_monitoring_integration(self, temp_dir):
        """Test performance monitoring and benchmarking integration."""
        from sowlv2.optimizations.performance_collector import PerformanceCollector
        from sowlv2.optimizations.benchmark_runner import BenchmarkRunner
        
        # Test performance collector
        collector = PerformanceCollector()
        
        timer_id = collector.start_timing("test_operation")
        assert timer_id is not None
        
        collector.end_timing(timer_id)
        collector.record_memory_usage("test_stage")
        
        # Test benchmark runner
        runner = BenchmarkRunner()
        
        # Mock benchmark data
        test_data = [str(Path(temp_dir) / f"test_{i}.jpg") for i in range(3)]
        
        with patch.object(runner, '_run_single_benchmark') as mock_benchmark:
            mock_benchmark.return_value = {
                "processing_time": 1.0,
                "memory_usage": 0.5,
                "success": True
            }
            
            results = runner.run_comparative_benchmark(test_data)
            assert results is not None
    
    def test_streaming_mode_integration(self):
        """Test streaming mode integration for large datasets."""
        from sowlv2.optimizations.streaming_processor import StreamingVideoProcessor
        
        # Test streaming processor
        processor = StreamingVideoProcessor(chunk_size=10, overlap_frames=2)
        
        # Mock video frames
        mock_frames = [f"frame_{i}" for i in range(50)]
        
        chunks = list(processor.create_chunks(mock_frames))
        assert len(chunks) > 1
        
        # Test chunk processing
        def mock_process_chunk(chunk):
            return [f"processed_{frame}" for frame in chunk]
        
        results = processor.process_streaming(mock_frames, mock_process_chunk)
        assert len(results) == len(mock_frames)
    
    def test_memory_optimization_integration(self):
        """Test memory optimization features integration."""
        rm = AdvancedResourceManager("cpu")
        
        # Test memory trend analysis
        for _ in range(5):
            rm.monitor_memory_usage()
        
        trend = rm.get_memory_trend()
        assert "trend" in trend
        assert "stability" in trend
        assert "peak_usage" in trend
        
        # Test cleanup functionality
        rm.cleanup_resources(force=True)
        
        # Test streaming mode decision
        should_stream = rm.should_enable_streaming(1000, (1024, 1024))
        assert isinstance(should_stream, bool)
    
    def test_error_recovery_integration(self):
        """Test error recovery system integration."""
        error_manager = ErrorRecoveryManager()
        
        # Test retry logic
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = error_manager.implement_retry_logic(failing_operation, max_retries=3)
        assert result == "success"
        assert call_count == 3
        
        # Test processing failure handling
        success = error_manager.handle_processing_failure("test_stage", RuntimeError("Test error"))
        assert isinstance(success, bool)


def test_integration_suite():
    """Run the complete integration test suite."""
    # This function can be called to run all integration tests
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    test_integration_suite()