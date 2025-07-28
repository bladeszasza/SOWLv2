"""
Unit tests for BenchmarkRunner.
Tests comparative benchmarking, memory profiling, and throughput measurement.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from sowlv2.optimizations.benchmark_runner import (
    BenchmarkRunner, BenchmarkConfig, BenchmarkResults, MemoryProfile, 
    ThroughputResults
)
from sowlv2.optimizations.performance_collector import PerformanceMetrics


class TestBenchmarkRunner:
    """Test suite for BenchmarkRunner class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        runner = BenchmarkRunner()
        
        assert runner.device == "cuda"
        assert os.path.exists(runner.output_dir)
        assert hasattr(runner, 'performance_collector')
        assert runner._test_images_cache == {}
    
    def test_init_custom_output_dir(self):
        """Test initialization with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_output = os.path.join(temp_dir, "custom_benchmarks")
            runner = BenchmarkRunner(device="cpu", output_dir=custom_output)
            
            assert runner.device == "cpu"
            assert runner.output_dir == custom_output
            assert os.path.exists(custom_output)
    
    def test_generate_test_data_basic(self):
        """Test basic test data generation."""
        runner = BenchmarkRunner()
        
        images = runner.generate_test_data((256, 256), count=5)
        
        assert len(images) == 5
        assert all(isinstance(img, Image.Image) for img in images)
        assert all(img.size == (256, 256) for img in images)
        assert all(img.mode == 'RGB' for img in images)
    
    def test_generate_test_data_caching(self):
        """Test that test data is cached properly."""
        runner = BenchmarkRunner()
        
        # Generate images first time
        images1 = runner.generate_test_data((128, 128), count=3)
        
        # Generate images second time (should use cache)
        images2 = runner.generate_test_data((128, 128), count=3)
        
        assert len(images1) == 3
        assert len(images2) == 3
        # Should be the same images from cache
        assert (128, 128) in runner._test_images_cache
        assert len(runner._test_images_cache[(128, 128)]) >= 3
    
    def test_generate_test_data_different_patterns(self):
        """Test that different image patterns are generated."""
        runner = BenchmarkRunner()
        
        images = runner.generate_test_data((100, 100), count=8)
        
        # Convert to numpy arrays for comparison
        arrays = [np.array(img) for img in images]
        
        # Check that images are different (not all the same)
        assert not all(np.array_equal(arrays[0], arr) for arr in arrays[1:])
        
        # Check that we get different patterns based on index % 4
        # (solid color, gradient, checkerboard, noise)
        assert len(set(arr.shape for arr in arrays)) == 1  # All same shape
        assert all(arr.shape == (100, 100, 3) for arr in arrays)  # RGB images
    
    def test_benchmark_config_defaults(self):
        """Test BenchmarkConfig default values."""
        config = BenchmarkConfig()
        
        assert config.test_iterations == 5
        assert config.warmup_iterations == 2
        assert config.batch_sizes == [1, 2, 4, 8]
        assert config.image_sizes == [(512, 512), (1024, 1024)]
        assert config.prompt_counts == [1, 3, 5]
        assert config.enable_memory_profiling is True
        assert config.enable_throughput_testing is True
        assert config.output_format == "json"
    
    def test_benchmark_config_custom(self):
        """Test BenchmarkConfig with custom values."""
        config = BenchmarkConfig(
            test_iterations=10,
            warmup_iterations=3,
            batch_sizes=[1, 4, 16],
            image_sizes=[(256, 256)],
            prompt_counts=[1, 2],
            enable_memory_profiling=False,
            enable_throughput_testing=False,
            output_format="csv"
        )
        
        assert config.test_iterations == 10
        assert config.warmup_iterations == 3
        assert config.batch_sizes == [1, 4, 16]
        assert config.image_sizes == [(256, 256)]
        assert config.prompt_counts == [1, 2]
        assert config.enable_memory_profiling is False
        assert config.enable_throughput_testing is False
        assert config.output_format == "csv"
    
    def test_run_comparative_benchmark_success(self):
        """Test successful comparative benchmark run."""
        runner = BenchmarkRunner()
        
        # Mock models
        model1 = Mock()
        model2 = Mock()
        models = {"model1": model1, "model2": model2}
        
        config = BenchmarkConfig(
            test_iterations=2,
            warmup_iterations=1,
            batch_sizes=[1, 2],
            image_sizes=[(256, 256)],
            prompt_counts=[1],
            enable_memory_profiling=False,
            enable_throughput_testing=False
        )
        
        # Mock the benchmark methods
        with patch.object(runner, '_benchmark_single_model') as mock_benchmark:
            with patch.object(runner, '_save_benchmark_results'):
                with patch.object(runner, '_generate_comparative_analysis'):
                    
                    # Mock benchmark results
                    mock_result1 = BenchmarkResults(
                        model_name="model1",
                        configuration={},
                        performance_metrics=PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0),
                        detailed_results={},
                        test_conditions={},
                        timestamp="2023-01-01 12:00:00"
                    )
                    
                    mock_result2 = BenchmarkResults(
                        model_name="model2",
                        configuration={},
                        performance_metrics=PerformanceMetrics(0.8, 1.5, 25.0, 12.0, 0.4, 35.0),
                        detailed_results={},
                        test_conditions={},
                        timestamp="2023-01-01 12:01:00"
                    )
                    
                    mock_benchmark.side_effect = [mock_result1, mock_result2]
                    
                    results = runner.run_comparative_benchmark(models, config)
        
        assert len(results) == 2
        assert "model1" in results
        assert "model2" in results
        assert results["model1"] == mock_result1
        assert results["model2"] == mock_result2
    
    def test_run_comparative_benchmark_with_error(self):
        """Test comparative benchmark with model error."""
        runner = BenchmarkRunner()
        
        model1 = Mock()
        model2 = Mock()
        models = {"model1": model1, "model2": model2}
        
        config = BenchmarkConfig(test_iterations=1, warmup_iterations=0)
        
        with patch.object(runner, '_benchmark_single_model') as mock_benchmark:
            with patch.object(runner, '_save_benchmark_results'):
                # First model succeeds, second fails
                mock_result1 = BenchmarkResults(
                    model_name="model1",
                    configuration={},
                    performance_metrics=PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0),
                    detailed_results={},
                    test_conditions={},
                    timestamp="2023-01-01 12:00:00"
                )
                
                mock_benchmark.side_effect = [mock_result1, Exception("Model2 failed")]
                
                results = runner.run_comparative_benchmark(models, config)
        
        assert len(results) == 2
        assert "model1" in results
        assert "model2" in results
        assert results["model1"] == mock_result1
        
        # Check error result
        error_result = results["model2"]
        assert error_result.model_name == "model2"
        assert "error" in error_result.configuration
        assert "Model2 failed" in error_result.configuration["error"]
    
    def test_benchmark_single_model(self):
        """Test benchmarking a single model."""
        runner = BenchmarkRunner()
        model = Mock()
        
        config = BenchmarkConfig(
            test_iterations=1,
            warmup_iterations=1,
            batch_sizes=[1],
            image_sizes=[(256, 256)],
            prompt_counts=[1],
            enable_memory_profiling=False,
            enable_throughput_testing=False
        )
        
        with patch.object(runner, '_run_single_inference'):
            with patch.object(runner, '_test_batch_size') as mock_batch_test:
                with patch.object(runner, '_test_image_size') as mock_size_test:
                    with patch.object(runner, '_test_prompt_count') as mock_prompt_test:
                        with patch.object(runner, '_get_model_configuration', return_value={}):
                            with patch.object(runner, '_calculate_aggregate_metrics') as mock_aggregate:
                                
                                # Mock test results
                                mock_metrics = PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0)
                                mock_batch_test.return_value = {'individual_runs': [mock_metrics]}
                                mock_size_test.return_value = {'individual_runs': [mock_metrics]}
                                mock_prompt_test.return_value = {'individual_runs': [mock_metrics]}
                                mock_aggregate.return_value = mock_metrics
                                
                                result = runner._benchmark_single_model("test_model", model, config)
        
        assert isinstance(result, BenchmarkResults)
        assert result.model_name == "test_model"
        assert result.performance_metrics == mock_metrics
        assert 'batch_size_tests' in result.detailed_results
        assert 'image_size_tests' in result.detailed_results
        assert 'prompt_count_tests' in result.detailed_results
    
    def test_test_batch_size(self):
        """Test batch size testing."""
        runner = BenchmarkRunner()
        model = Mock()
        config = BenchmarkConfig(test_iterations=2)
        
        with patch.object(runner, 'generate_test_data') as mock_generate:
            with patch.object(runner, '_run_batch_inference'):
                with patch.object(runner.performance_collector, 'start_timing', return_value="timer1"):
                    with patch.object(runner.performance_collector, 'end_timing') as mock_end_timing:
                        
                        # Mock test images
                        mock_images = [Mock() for _ in range(4)]
                        mock_generate.return_value = mock_images
                        
                        # Mock performance metrics
                        mock_metrics1 = PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0)
                        mock_metrics2 = PerformanceMetrics(1.1, 2.1, 32.0, 9.5, 0.5, 42.0)
                        mock_end_timing.side_effect = [mock_metrics1, mock_metrics2]
                        
                        result = runner._test_batch_size(model, batch_size=2, config=config)
        
        assert result['batch_size'] == 2
        assert len(result['individual_runs']) == 2
        assert result['individual_runs'][0] == mock_metrics1
        assert result['individual_runs'][1] == mock_metrics2
        assert result['average_metrics'] is not None
    
    def test_test_batch_size_with_error(self):
        """Test batch size testing with inference error."""
        runner = BenchmarkRunner()
        model = Mock()
        config = BenchmarkConfig(test_iterations=2)
        
        with patch.object(runner, 'generate_test_data', return_value=[Mock(), Mock()]):
            with patch.object(runner, '_run_batch_inference', side_effect=Exception("Inference failed")):
                with patch.object(runner.performance_collector, 'start_timing', return_value="timer1"):
                    
                    result = runner._test_batch_size(model, batch_size=2, config=config)
        
        assert result['batch_size'] == 2
        assert len(result['individual_runs']) == 0  # All runs failed
        assert result['average_metrics'] is None
    
    def test_profile_memory_usage(self):
        """Test memory usage profiling."""
        runner = BenchmarkRunner()
        model = Mock()
        config = BenchmarkConfig()
        
        with patch.object(runner, 'generate_test_data') as mock_generate:
            with patch.object(runner, '_run_single_inference'):
                with patch.object(runner, '_get_current_memory_usage') as mock_memory:
                    with patch.object(runner.performance_collector, 'start_timing', return_value="timer1"):
                        with patch.object(runner.performance_collector, 'end_timing'):
                            with patch('torch.cuda.is_available', return_value=True):
                                with patch('torch.cuda.empty_cache'):
                                    with patch('torch.cuda.reset_peak_memory_stats'):
                                        
                                        mock_generate.return_value = [Mock()]
                                        # Simulate memory usage pattern
                                        mock_memory.side_effect = [1.0, 1.2, 1.5, 1.3, 1.4, 1.1]
                                        
                                        profile = runner.profile_memory_usage(model, config)
        
        assert isinstance(profile, MemoryProfile)
        assert profile.peak_memory_usage > 0
        assert len(profile.memory_timeline) > 0
        assert 0 <= profile.memory_efficiency <= 100
        assert profile.fragmentation_score >= 0
        assert 'baseline' in profile.allocation_pattern
        assert 'peak' in profile.allocation_pattern
        assert 'average' in profile.allocation_pattern
    
    def test_measure_throughput(self):
        """Test throughput measurement."""
        runner = BenchmarkRunner()
        model = Mock()
        batch_sizes = [1, 2, 4]
        
        with patch.object(runner, 'generate_test_data') as mock_generate:
            with patch.object(runner, '_run_batch_inference'):
                with patch.object(runner, '_get_current_memory_usage') as mock_memory:
                    
                    mock_generate.return_value = [Mock() for _ in range(12)]  # Enough for all batches
                    # Need more memory values for multiple batch sizes and iterations
                    mock_memory.side_effect = [1.0, 1.5, 1.2, 1.6, 1.3, 1.7]
                    
                    results = runner.measure_throughput(model, batch_sizes)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ThroughputResults)
            assert result.batch_size == batch_sizes[i]
            assert result.throughput_fps >= 0
            assert result.latency_ms >= 0
            assert result.memory_usage_gb >= 0
            assert result.efficiency_score >= 0
    
    def test_measure_throughput_with_error(self):
        """Test throughput measurement with inference error."""
        runner = BenchmarkRunner()
        model = Mock()
        batch_sizes = [1, 2]
        
        with patch.object(runner, 'generate_test_data', return_value=[Mock(), Mock(), Mock(), Mock()]):
            with patch.object(runner, '_get_current_memory_usage', return_value=1.0):
                # Mock inference to succeed during warmup but fail during measurement
                call_count = 0
                def mock_inference(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    # Allow warmup calls to succeed (2 per batch size = 4 total)
                    # Then fail on measurement calls (which are inside try-except)
                    if call_count <= 4:  
                        return {"simulated": True}
                    # This will be caught by the try-except in measure_throughput
                    raise Exception("Inference failed")
                
                with patch.object(runner, '_run_batch_inference', side_effect=mock_inference):
                    results = runner.measure_throughput(model, batch_sizes)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ThroughputResults)
            assert result.throughput_fps == 0
            assert result.latency_ms == 0
            assert result.memory_usage_gb == 0
            assert result.efficiency_score == 0
    
    def test_get_current_memory_usage_cuda(self):
        """Test current memory usage with CUDA."""
        runner = BenchmarkRunner(device="cuda")
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=2e9):  # 2GB
                
                memory_usage = runner._get_current_memory_usage()
                
                assert memory_usage == 2.0
    
    def test_get_current_memory_usage_cpu(self):
        """Test current memory usage with CPU."""
        runner = BenchmarkRunner(device="cpu")
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(used=4e9)  # 4GB
            
            memory_usage = runner._get_current_memory_usage()
            
            assert memory_usage == 4.0
    
    def test_calculate_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        runner = BenchmarkRunner()
        
        metrics_list = [
            PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0),
            PerformanceMetrics(1.2, 2.2, 32.0, 12.0, 0.6, 42.0),
            PerformanceMetrics(0.8, 1.8, 28.0, 8.0, 0.4, 38.0)
        ]
        
        aggregate = runner._calculate_aggregate_metrics(metrics_list)
        
        assert isinstance(aggregate, PerformanceMetrics)
        assert aggregate.processing_time == pytest.approx(1.0, rel=1e-2)  # (1.0+1.2+0.8)/3
        assert aggregate.memory_peak_usage == pytest.approx(2.0, rel=1e-2)  # (2.0+2.2+1.8)/3
        assert aggregate.gpu_utilization == pytest.approx(30.0, rel=1e-2)  # (30+32+28)/3
        assert aggregate.throughput_fps == pytest.approx(10.0, rel=1e-2)  # (10+12+8)/3
    
    def test_calculate_aggregate_metrics_empty(self):
        """Test aggregate metrics calculation with empty list."""
        runner = BenchmarkRunner()
        
        aggregate = runner._calculate_aggregate_metrics([])
        
        assert isinstance(aggregate, PerformanceMetrics)
        assert aggregate.processing_time == 0
        assert aggregate.memory_peak_usage == 0
        assert aggregate.gpu_utilization == 0
        assert aggregate.throughput_fps == 0
    
    def test_get_model_configuration(self):
        """Test model configuration extraction."""
        runner = BenchmarkRunner()
        
        # Mock model with configuration
        model = Mock()
        model.config = {"param1": "value1", "param2": 42}
        model.model_name = "test_model"
        
        config = runner._get_model_configuration(model)
        
        assert config["model_type"] == "Mock"
        assert config["device"] == runner.device
        assert config["param1"] == "value1"
        assert config["param2"] == 42
        assert config["model_name"] == "test_model"
    
    def test_get_model_configuration_minimal(self):
        """Test model configuration extraction with minimal model."""
        runner = BenchmarkRunner()
        
        model = Mock()
        # Remove config and model_name attributes
        del model.config
        del model.model_name
        
        config = runner._get_model_configuration(model)
        
        assert config["model_type"] == "Mock"
        assert config["device"] == runner.device
        assert len(config) == 2  # Only type and device
    
    def test_save_benchmark_results_json(self):
        """Test saving benchmark results in JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(output_dir=temp_dir)
            
            results = BenchmarkResults(
                model_name="test_model",
                configuration={"param": "value"},
                performance_metrics=PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0),
                detailed_results={"test": "data"},
                test_conditions={"device": "cuda"},
                timestamp="2023-01-01 12:00:00"
            )
            
            runner._save_benchmark_results(results, "json")
            
            # Check that file was created
            files = os.listdir(temp_dir)
            json_files = [f for f in files if f.endswith('.json')]
            assert len(json_files) == 1
            
            # Check file content
            with open(os.path.join(temp_dir, json_files[0]), 'r') as f:
                data = json.load(f)
            
            assert data["model_name"] == "test_model"
            assert data["configuration"]["param"] == "value"
            assert data["performance_metrics"]["processing_time"] == 1.0
    
    def test_serialize_results(self):
        """Test benchmark results serialization."""
        runner = BenchmarkRunner()
        
        results = BenchmarkResults(
            model_name="test_model",
            configuration={"param": "value"},
            performance_metrics=PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0),
            detailed_results={"test": "data"},
            test_conditions={"device": "cuda"},
            timestamp="2023-01-01 12:00:00"
        )
        
        serialized = runner._serialize_results(results)
        
        assert isinstance(serialized, dict)
        assert serialized["model_name"] == "test_model"
        assert serialized["configuration"]["param"] == "value"
        assert serialized["performance_metrics"]["processing_time"] == 1.0
        assert serialized["performance_metrics"]["memory_peak_usage"] == 2.0
        assert serialized["detailed_results"]["test"] == "data"
        assert serialized["test_conditions"]["device"] == "cuda"
        assert serialized["timestamp"] == "2023-01-01 12:00:00"
    
    def test_generate_comparative_analysis(self):
        """Test comparative analysis generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(output_dir=temp_dir)
            
            results = {
                "model1": BenchmarkResults(
                    model_name="model1",
                    configuration={},
                    performance_metrics=PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0),
                    detailed_results={},
                    test_conditions={},
                    timestamp="2023-01-01 12:00:00"
                ),
                "model2": BenchmarkResults(
                    model_name="model2",
                    configuration={},
                    performance_metrics=PerformanceMetrics(0.8, 1.5, 25.0, 12.0, 0.4, 35.0),
                    detailed_results={},
                    test_conditions={},
                    timestamp="2023-01-01 12:01:00"
                )
            }
            
            config = BenchmarkConfig()
            
            runner._generate_comparative_analysis(results, config)
            
            # Check that analysis file was created
            analysis_file = os.path.join(temp_dir, "comparative_analysis.json")
            assert os.path.exists(analysis_file)
            
            # Check file content
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            assert "summary" in analysis
            assert "detailed_comparison" in analysis
            assert "test_configuration" in analysis
            
            # Check summary
            summary = analysis["summary"]
            assert summary["fastest_model"] == "model2"  # 0.8s < 1.0s
            assert summary["most_memory_efficient"] == "model2"  # 1.5GB < 2.0GB
            assert summary["highest_throughput"] == "model2"  # 12.0 > 10.0
    
    def test_run_single_inference_placeholder(self):
        """Test single inference placeholder method."""
        runner = BenchmarkRunner()
        model = Mock()
        image = Mock()
        prompts = ["test"]
        
        result = runner._run_single_inference(model, image, prompts)
        
        assert result == {"simulated": True}
    
    def test_run_batch_inference_placeholder(self):
        """Test batch inference placeholder method."""
        runner = BenchmarkRunner()
        model = Mock()
        images = [Mock(), Mock(), Mock()]
        prompts = ["test1", "test2", "test3"]
        
        result = runner._run_batch_inference(model, images, prompts)
        
        assert result == {"simulated": True, "batch_size": 3}
    
    def test_memory_profile_dataclass(self):
        """Test MemoryProfile dataclass functionality."""
        timeline = [(0.0, 1.0), (1.0, 1.5), (2.0, 1.2)]
        allocation_pattern = {"baseline": 1.0, "peak": 1.5, "average": 1.23}
        
        profile = MemoryProfile(
            peak_memory_usage=1.5,
            memory_timeline=timeline,
            memory_efficiency=82.5,
            fragmentation_score=0.15,
            allocation_pattern=allocation_pattern
        )
        
        assert profile.peak_memory_usage == 1.5
        assert profile.memory_timeline == timeline
        assert profile.memory_efficiency == 82.5
        assert profile.fragmentation_score == 0.15
        assert profile.allocation_pattern == allocation_pattern
    
    def test_throughput_results_dataclass(self):
        """Test ThroughputResults dataclass functionality."""
        result = ThroughputResults(
            batch_size=4,
            throughput_fps=25.5,
            latency_ms=40.0,
            memory_usage_gb=2.1,
            efficiency_score=12.14
        )
        
        assert result.batch_size == 4
        assert result.throughput_fps == 25.5
        assert result.latency_ms == 40.0
        assert result.memory_usage_gb == 2.1
        assert result.efficiency_score == 12.14
    
    def test_benchmark_results_dataclass(self):
        """Test BenchmarkResults dataclass functionality."""
        metrics = PerformanceMetrics(1.0, 2.0, 30.0, 10.0, 0.5, 40.0)
        
        results = BenchmarkResults(
            model_name="test_model",
            configuration={"param": "value"},
            performance_metrics=metrics,
            detailed_results={"test": "data"},
            test_conditions={"device": "cuda"},
            timestamp="2023-01-01 12:00:00"
        )
        
        assert results.model_name == "test_model"
        assert results.configuration == {"param": "value"}
        assert results.performance_metrics == metrics
        assert results.detailed_results == {"test": "data"}
        assert results.test_conditions == {"device": "cuda"}
        assert results.timestamp == "2023-01-01 12:00:00"