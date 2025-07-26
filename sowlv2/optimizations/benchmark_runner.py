"""
Benchmark runner for comparative performance analysis between models and configurations.
Provides automated testing and detailed performance profiling capabilities.
"""
import os
import time
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image

from .performance_collector import PerformanceCollector, PerformanceMetrics, ComparisonReport


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    test_iterations: int = 5
    warmup_iterations: int = 2
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    image_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(512, 512), (1024, 1024)])
    prompt_counts: List[int] = field(default_factory=lambda: [1, 3, 5])
    enable_memory_profiling: bool = True
    enable_throughput_testing: bool = True
    output_format: str = "json"  # json, csv, html


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""
    model_name: str
    configuration: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    detailed_results: Dict[str, Any]
    test_conditions: Dict[str, Any]
    timestamp: str


@dataclass
class MemoryProfile:
    """Detailed memory usage profile."""
    peak_memory_usage: float  # GB
    memory_timeline: List[Tuple[float, float]]  # (timestamp, memory_usage)
    memory_efficiency: float  # percentage
    fragmentation_score: float
    allocation_pattern: Dict[str, float]


@dataclass
class ThroughputResults:
    """Throughput analysis results."""
    batch_size: int
    throughput_fps: float
    latency_ms: float
    memory_usage_gb: float
    efficiency_score: float


class BenchmarkRunner:
    """Comprehensive benchmark runner for model and configuration comparison."""
    
    def __init__(self, device: str = "cuda", output_dir: Optional[str] = None):
        """
        Initialize the benchmark runner.
        
        Args:
            device: Device to run benchmarks on
            output_dir: Directory to save benchmark results
        """
        self.device = device
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="sowlv2_benchmarks_")
        self.performance_collector = PerformanceCollector(device=device)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test data cache
        self._test_images_cache: Dict[Tuple[int, int], List[Image.Image]] = {}
        
    def generate_test_data(self, image_size: Tuple[int, int], 
                          count: int = 10) -> List[Image.Image]:
        """
        Generate synthetic test images for benchmarking.
        
        Args:
            image_size: Size of images to generate (width, height)
            count: Number of images to generate
            
        Returns:
            List of PIL Images
        """
        cache_key = (image_size[0], image_size[1])
        
        if cache_key in self._test_images_cache:
            cached_images = self._test_images_cache[cache_key]
            if len(cached_images) >= count:
                return cached_images[:count]
                
        # Generate new test images
        images = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(count):
            # Create varied synthetic images
            if i % 4 == 0:
                # Solid color with noise
                base_color = np.random.randint(0, 256, 3)
                image_array = np.full((*image_size[::-1], 3), base_color, dtype=np.uint8)
                noise = np.random.randint(-30, 30, image_array.shape)
                image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            elif i % 4 == 1:
                # Gradient pattern
                x = np.linspace(0, 255, image_size[0])
                y = np.linspace(0, 255, image_size[1])
                xx, yy = np.meshgrid(x, y)
                image_array = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.uint8)
            elif i % 4 == 2:
                # Checkerboard pattern
                checker_size = max(32, min(image_size) // 16)
                pattern = np.indices(image_size[::-1]) // checker_size
                checker = (pattern[0] + pattern[1]) % 2
                image_array = np.stack([checker * 255] * 3, axis=-1).astype(np.uint8)
            else:
                # Random noise
                image_array = np.random.randint(0, 256, (*image_size[::-1], 3), dtype=np.uint8)
                
            images.append(Image.fromarray(image_array))
            
        # Cache the generated images
        self._test_images_cache[cache_key] = images
        return images
        
    def run_comparative_benchmark(self, models: Dict[str, Any], 
                                config: BenchmarkConfig = None) -> Dict[str, BenchmarkResults]:
        """
        Run comparative benchmark between multiple models.
        
        Args:
            models: Dictionary of model_name -> model_instance
            config: Benchmark configuration
            
        Returns:
            Dictionary of model_name -> BenchmarkResults
        """
        if config is None:
            config = BenchmarkConfig()
            
        results = {}
        
        print(f"Starting comparative benchmark with {len(models)} models...")
        print(f"Output directory: {self.output_dir}")
        
        for model_name, model in models.items():
            print(f"\nBenchmarking {model_name}...")
            
            try:
                # Run benchmark for this model
                model_results = self._benchmark_single_model(
                    model_name, model, config
                )
                results[model_name] = model_results
                
                # Save individual results
                self._save_benchmark_results(model_results, config.output_format)
                
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                # Create error result
                results[model_name] = BenchmarkResults(
                    model_name=model_name,
                    configuration={"error": str(e)},
                    performance_metrics=PerformanceMetrics(
                        processing_time=0, memory_peak_usage=0, gpu_utilization=0,
                        throughput_fps=0, model_loading_time=0, cpu_utilization=0
                    ),
                    detailed_results={"error": str(e)},
                    test_conditions={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
        # Generate comparative analysis
        if len(results) >= 2:
            self._generate_comparative_analysis(results, config)
            
        return results
        
    def _benchmark_single_model(self, model_name: str, model: Any, 
                               config: BenchmarkConfig) -> BenchmarkResults:
        """Benchmark a single model with various configurations."""
        detailed_results = {
            'batch_size_tests': [],
            'image_size_tests': [],
            'prompt_count_tests': [],
            'memory_profiles': [],
            'throughput_tests': []
        }
        
        # Warmup runs
        print(f"  Running {config.warmup_iterations} warmup iterations...")
        test_images = self.generate_test_data((1024, 1024), 5)
        for _ in range(config.warmup_iterations):
            self._run_single_inference(model, test_images[0], ["test prompt"])
            
        # Main benchmark runs
        all_metrics = []
        
        # Test different batch sizes
        for batch_size in config.batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            batch_metrics = self._test_batch_size(model, batch_size, config)
            detailed_results['batch_size_tests'].append(batch_metrics)
            all_metrics.extend(batch_metrics['individual_runs'])
            
        # Test different image sizes
        for image_size in config.image_sizes:
            print(f"  Testing image size: {image_size}")
            size_metrics = self._test_image_size(model, image_size, config)
            detailed_results['image_size_tests'].append(size_metrics)
            all_metrics.extend(size_metrics['individual_runs'])
            
        # Test different prompt counts
        for prompt_count in config.prompt_counts:
            print(f"  Testing prompt count: {prompt_count}")
            prompt_metrics = self._test_prompt_count(model, prompt_count, config)
            detailed_results['prompt_count_tests'].append(prompt_metrics)
            all_metrics.extend(prompt_metrics['individual_runs'])
            
        # Memory profiling
        if config.enable_memory_profiling:
            print("  Running memory profiling...")
            memory_profile = self.profile_memory_usage(model, config)
            detailed_results['memory_profiles'].append(memory_profile)
            
        # Throughput testing
        if config.enable_throughput_testing:
            print("  Running throughput tests...")
            throughput_results = self.measure_throughput(model, config.batch_sizes)
            detailed_results['throughput_tests'] = throughput_results
            
        # Calculate aggregate metrics
        if all_metrics:
            aggregate_metrics = self._calculate_aggregate_metrics(all_metrics)
        else:
            aggregate_metrics = PerformanceMetrics(
                processing_time=0, memory_peak_usage=0, gpu_utilization=0,
                throughput_fps=0, model_loading_time=0, cpu_utilization=0
            )
            
        return BenchmarkResults(
            model_name=model_name,
            configuration=self._get_model_configuration(model),
            performance_metrics=aggregate_metrics,
            detailed_results=detailed_results,
            test_conditions={
                'device': self.device,
                'batch_sizes': config.batch_sizes,
                'image_sizes': config.image_sizes,
                'prompt_counts': config.prompt_counts,
                'test_iterations': config.test_iterations
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    def _test_batch_size(self, model: Any, batch_size: int, 
                        config: BenchmarkConfig) -> Dict[str, Any]:
        """Test model performance with specific batch size."""
        test_images = self.generate_test_data((1024, 1024), batch_size * 2)
        prompts = ["test object"] * batch_size
        
        individual_runs = []
        for i in range(config.test_iterations):
            batch_images = test_images[i:i+batch_size] if i+batch_size <= len(test_images) else test_images[:batch_size]
            
            timer_id = self.performance_collector.start_timing(
                f"batch_size_{batch_size}",
                metadata={'batch_size': batch_size, 'frame_count': len(batch_images)}
            )
            
            try:
                # Run inference
                results = self._run_batch_inference(model, batch_images, prompts)
                metrics = self.performance_collector.end_timing(timer_id)
                individual_runs.append(metrics)
                
            except Exception as e:
                print(f"    Error in batch size test: {e}")
                continue
                
        return {
            'batch_size': batch_size,
            'individual_runs': individual_runs,
            'average_metrics': self._calculate_aggregate_metrics(individual_runs) if individual_runs else None
        }
        
    def _test_image_size(self, model: Any, image_size: Tuple[int, int], 
                        config: BenchmarkConfig) -> Dict[str, Any]:
        """Test model performance with specific image size."""
        test_images = self.generate_test_data(image_size, config.test_iterations)
        
        individual_runs = []
        for i, image in enumerate(test_images):
            timer_id = self.performance_collector.start_timing(
                f"image_size_{image_size[0]}x{image_size[1]}",
                metadata={'image_size': image_size, 'frame_count': 1}
            )
            
            try:
                result = self._run_single_inference(model, image, ["test object"])
                metrics = self.performance_collector.end_timing(timer_id)
                individual_runs.append(metrics)
                
            except Exception as e:
                print(f"    Error in image size test: {e}")
                continue
                
        return {
            'image_size': image_size,
            'individual_runs': individual_runs,
            'average_metrics': self._calculate_aggregate_metrics(individual_runs) if individual_runs else None
        }
        
    def _test_prompt_count(self, model: Any, prompt_count: int, 
                          config: BenchmarkConfig) -> Dict[str, Any]:
        """Test model performance with specific number of prompts."""
        test_images = self.generate_test_data((1024, 1024), config.test_iterations)
        prompts = [f"test object {i+1}" for i in range(prompt_count)]
        
        individual_runs = []
        for image in test_images:
            timer_id = self.performance_collector.start_timing(
                f"prompt_count_{prompt_count}",
                metadata={'prompt_count': prompt_count, 'frame_count': 1}
            )
            
            try:
                result = self._run_single_inference(model, image, prompts)
                metrics = self.performance_collector.end_timing(timer_id)
                individual_runs.append(metrics)
                
            except Exception as e:
                print(f"    Error in prompt count test: {e}")
                continue
                
        return {
            'prompt_count': prompt_count,
            'individual_runs': individual_runs,
            'average_metrics': self._calculate_aggregate_metrics(individual_runs) if individual_runs else None
        }
        
    def profile_memory_usage(self, model: Any, config: BenchmarkConfig) -> MemoryProfile:
        """
        Profile detailed memory usage during model execution.
        
        Args:
            model: Model to profile
            config: Benchmark configuration
            
        Returns:
            MemoryProfile: Detailed memory usage analysis
        """
        memory_timeline = []
        peak_memory = 0.0
        
        # Clear memory before profiling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        test_image = self.generate_test_data((1024, 1024), 1)[0]
        
        # Profile memory during inference
        start_time = time.time()
        
        try:
            # Record baseline
            baseline_memory = self._get_current_memory_usage()
            memory_timeline.append((0.0, baseline_memory))
            
            # Run inference with memory tracking
            timer_id = self.performance_collector.start_timing("memory_profiling")
            
            # Multiple inference runs to capture memory patterns
            for i in range(5):
                current_time = time.time() - start_time
                
                # Record memory before inference
                pre_memory = self._get_current_memory_usage()
                memory_timeline.append((current_time, pre_memory))
                
                # Run inference
                result = self._run_single_inference(model, test_image, ["test object"])
                
                # Record memory after inference
                post_memory = self._get_current_memory_usage()
                memory_timeline.append((current_time + 0.1, post_memory))
                peak_memory = max(peak_memory, post_memory)
                
            metrics = self.performance_collector.end_timing(timer_id)
            
        except Exception as e:
            print(f"Error during memory profiling: {e}")
            
        # Calculate memory efficiency and fragmentation
        if memory_timeline:
            memory_values = [mem for _, mem in memory_timeline]
            memory_efficiency = (np.mean(memory_values) / peak_memory) * 100 if peak_memory > 0 else 0
            fragmentation_score = np.std(memory_values) / np.mean(memory_values) if np.mean(memory_values) > 0 else 0
        else:
            memory_efficiency = 0
            fragmentation_score = 0
            
        return MemoryProfile(
            peak_memory_usage=peak_memory,
            memory_timeline=memory_timeline,
            memory_efficiency=memory_efficiency,
            fragmentation_score=fragmentation_score,
            allocation_pattern={
                'baseline': baseline_memory if 'baseline_memory' in locals() else 0,
                'peak': peak_memory,
                'average': np.mean([mem for _, mem in memory_timeline]) if memory_timeline else 0
            }
        )
        
    def measure_throughput(self, model: Any, batch_sizes: List[int]) -> List[ThroughputResults]:
        """
        Measure processing throughput for different batch sizes.
        
        Args:
            model: Model to test
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of ThroughputResults
        """
        results = []
        
        for batch_size in batch_sizes:
            print(f"    Measuring throughput for batch size {batch_size}")
            
            # Generate test data
            test_images = self.generate_test_data((1024, 1024), batch_size * 3)
            prompts = ["throughput test"] * batch_size
            
            # Warmup
            for _ in range(2):
                self._run_batch_inference(model, test_images[:batch_size], prompts)
                
            # Measure throughput
            start_time = time.time()
            start_memory = self._get_current_memory_usage()
            
            total_frames = 0
            iterations = 10
            
            try:
                for i in range(iterations):
                    batch_start = (i * batch_size) % len(test_images)
                    batch_end = min(batch_start + batch_size, len(test_images))
                    batch_images = test_images[batch_start:batch_end]
                    
                    self._run_batch_inference(model, batch_images, prompts[:len(batch_images)])
                    total_frames += len(batch_images)
                    
                end_time = time.time()
                end_memory = self._get_current_memory_usage()
                
                # Calculate metrics
                total_time = end_time - start_time
                throughput_fps = total_frames / total_time if total_time > 0 else 0
                latency_ms = (total_time / iterations) * 1000
                memory_usage_gb = end_memory - start_memory
                
                # Efficiency score (frames per second per GB of memory)
                efficiency_score = throughput_fps / max(memory_usage_gb, 0.1)
                
                results.append(ThroughputResults(
                    batch_size=batch_size,
                    throughput_fps=throughput_fps,
                    latency_ms=latency_ms,
                    memory_usage_gb=memory_usage_gb,
                    efficiency_score=efficiency_score
                ))
                
            except Exception as e:
                print(f"      Error measuring throughput: {e}")
                results.append(ThroughputResults(
                    batch_size=batch_size,
                    throughput_fps=0,
                    latency_ms=0,
                    memory_usage_gb=0,
                    efficiency_score=0
                ))
                
        return results
        
    def _run_single_inference(self, model: Any, image: Image.Image, 
                             prompts: List[str]) -> Any:
        """Run single inference - to be implemented based on model interface."""
        # This is a placeholder - actual implementation depends on model interface
        # For now, simulate processing time
        time.sleep(0.01)  # Simulate processing
        return {"simulated": True}
        
    def _run_batch_inference(self, model: Any, images: List[Image.Image], 
                            prompts: List[str]) -> Any:
        """Run batch inference - to be implemented based on model interface."""
        # This is a placeholder - actual implementation depends on model interface
        # For now, simulate processing time proportional to batch size
        time.sleep(0.01 * len(images))
        return {"simulated": True, "batch_size": len(images)}
        
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available() and self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        else:
            import psutil
            return psutil.virtual_memory().used / 1e9
            
    def _calculate_aggregate_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate aggregate metrics from a list of individual metrics."""
        if not metrics_list:
            return PerformanceMetrics(
                processing_time=0, memory_peak_usage=0, gpu_utilization=0,
                throughput_fps=0, model_loading_time=0, cpu_utilization=0
            )
            
        return PerformanceMetrics(
            processing_time=np.mean([m.processing_time for m in metrics_list]),
            memory_peak_usage=np.mean([m.memory_peak_usage for m in metrics_list]),
            gpu_utilization=np.mean([m.gpu_utilization for m in metrics_list]),
            throughput_fps=np.mean([m.throughput_fps for m in metrics_list if m.throughput_fps > 0]),
            model_loading_time=np.mean([m.model_loading_time for m in metrics_list]),
            cpu_utilization=np.mean([m.cpu_utilization for m in metrics_list])
        )
        
    def _get_model_configuration(self, model: Any) -> Dict[str, Any]:
        """Extract model configuration information."""
        config = {
            'model_type': type(model).__name__,
            'device': self.device
        }
        
        # Try to extract additional configuration if available
        if hasattr(model, 'config'):
            config.update(model.config)
        if hasattr(model, 'model_name'):
            config['model_name'] = model.model_name
            
        return config
        
    def _save_benchmark_results(self, results: BenchmarkResults, output_format: str):
        """Save benchmark results to file."""
        filename = f"benchmark_{results.model_name}_{results.timestamp.replace(':', '-').replace(' ', '_')}"
        
        if output_format == "json":
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(self._serialize_results(results), f, indent=2)
        elif output_format == "csv":
            # Implement CSV export if needed
            pass
            
        print(f"  Results saved to: {filepath}")
        
    def _serialize_results(self, results: BenchmarkResults) -> Dict[str, Any]:
        """Serialize benchmark results for JSON export."""
        return {
            'model_name': results.model_name,
            'configuration': results.configuration,
            'performance_metrics': {
                'processing_time': results.performance_metrics.processing_time,
                'memory_peak_usage': results.performance_metrics.memory_peak_usage,
                'gpu_utilization': results.performance_metrics.gpu_utilization,
                'throughput_fps': results.performance_metrics.throughput_fps,
                'model_loading_time': results.performance_metrics.model_loading_time,
                'cpu_utilization': results.performance_metrics.cpu_utilization
            },
            'detailed_results': results.detailed_results,
            'test_conditions': results.test_conditions,
            'timestamp': results.timestamp
        }
        
    def _generate_comparative_analysis(self, results: Dict[str, BenchmarkResults], 
                                     config: BenchmarkConfig):
        """Generate comparative analysis report."""
        analysis_file = os.path.join(self.output_dir, "comparative_analysis.json")
        
        # Extract key metrics for comparison
        comparison_data = {}
        for model_name, result in results.items():
            comparison_data[model_name] = {
                'processing_time': result.performance_metrics.processing_time,
                'memory_usage': result.performance_metrics.memory_peak_usage,
                'throughput': result.performance_metrics.throughput_fps,
                'gpu_utilization': result.performance_metrics.gpu_utilization
            }
            
        # Find best performing model for each metric
        best_speed = min(comparison_data.items(), key=lambda x: x[1]['processing_time'])
        best_memory = min(comparison_data.items(), key=lambda x: x[1]['memory_usage'])
        best_throughput = max(comparison_data.items(), key=lambda x: x[1]['throughput'])
        
        analysis = {
            'summary': {
                'fastest_model': best_speed[0],
                'most_memory_efficient': best_memory[0],
                'highest_throughput': best_throughput[0]
            },
            'detailed_comparison': comparison_data,
            'test_configuration': {
                'batch_sizes': config.batch_sizes,
                'image_sizes': config.image_sizes,
                'prompt_counts': config.prompt_counts,
                'iterations': config.test_iterations
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        print(f"\nComparative analysis saved to: {analysis_file}")
        print(f"Summary:")
        print(f"  Fastest model: {best_speed[0]} ({best_speed[1]['processing_time']:.3f}s)")
        print(f"  Most memory efficient: {best_memory[0]} ({best_memory[1]['memory_usage']:.2f}GB)")
        print(f"  Highest throughput: {best_throughput[0]} ({best_throughput[1]['throughput']:.1f} FPS)")