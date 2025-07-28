"""
Enhanced intelligent batch processing with adaptive optimization and GPU profiling.
"""
import time
import gc
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import torch


class OptimizationLevel(Enum):
    """Optimization levels for batch processing."""
    CONSERVATIVE = 1
    BALANCED = 2
    AGGRESSIVE = 3


@dataclass
class BatchConfig:
    """Enhanced batch configuration with adaptive features."""
    detection_batch_size: int
    segmentation_batch_size: int
    frame_batch_size: int
    use_mixed_precision: bool
    enable_gradient_checkpointing: bool
    optimization_level: OptimizationLevel
    memory_limit_gb: Optional[float] = None


@dataclass
class GPUProfile:
    """GPU performance profile for batch optimization."""
    total_memory: float  # GB
    available_memory: float  # GB
    compute_capability: Tuple[int, int]
    supports_mixed_precision: bool
    memory_bandwidth: float  # GB/s (estimated)
    compute_units: int


@dataclass
class BatchPerformanceMetrics:
    """Performance metrics for batch processing."""
    batch_size: int
    processing_time: float
    memory_peak: float
    throughput: float  # items/second
    memory_efficiency: float  # 0-1
    success_rate: float  # 0-1


class IntelligentBatchOptimizer:
    """Enhanced batch optimizer with GPU profiling and adaptive optimization."""

    def __init__(self, device: str = "cuda", optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.device = device
        self.optimization_level = optimization_level
        self.profiling_results: Dict[str, BatchPerformanceMetrics] = {}
        self.gpu_profile: Optional[GPUProfile] = None
        self.adaptive_history: List[BatchPerformanceMetrics] = []
        self.failure_recovery_enabled = True

        # Initialize GPU profiling
        self._initialize_gpu_profile()

    def _initialize_gpu_profile(self):
        """Initialize GPU profiling information."""
        if self.device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)

            self.gpu_profile = GPUProfile(
                total_memory=props.total_memory / 1e9,
                available_memory=(props.total_memory - torch.cuda.memory_allocated()) / 1e9,
                compute_capability=(props.major, props.minor),
                supports_mixed_precision=props.major >= 7,
                memory_bandwidth=self._estimate_memory_bandwidth(props),
                compute_units=props.multi_processor_count
            )
        else:
            self.gpu_profile = None

    def _estimate_memory_bandwidth(self, props) -> float:
        """Estimate memory bandwidth based on GPU properties."""
        # Rough estimates based on common GPU architectures
        if props.major >= 8:  # Ampere and newer
            return 900.0  # GB/s
        elif props.major == 7:  # Turing/Volta
            return 600.0
        else:  # Older architectures
            return 400.0

    def profile_gpu_memory_for_batch_size(self,
                                        test_func: Callable,
                                        batch_sizes: List[int],
                                        *args, **kwargs) -> Dict[int, BatchPerformanceMetrics]:
        """
        Profile GPU memory usage for different batch sizes.

        Args:
            test_func: Function to test with different batch sizes
            batch_sizes: List of batch sizes to test
            *args, **kwargs: Arguments for test function

        Returns:
            Dictionary mapping batch size to performance metrics
        """
        if not self.gpu_profile:
            return {}

        results = {}

        for batch_size in batch_sizes:
            try:
                # Clear cache before testing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Measure initial memory
                initial_memory = torch.cuda.memory_allocated()
                start_time = time.time()

                # Run test function
                success = True
                try:
                    test_func(batch_size, *args, **kwargs)
                except torch.cuda.OutOfMemoryError:
                    success = False
                except Exception as e:
                    print(f"Error testing batch size {batch_size}: {e}")
                    success = False

                # Measure final memory and time
                torch.cuda.synchronize()
                end_time = time.time()
                peak_memory = torch.cuda.max_memory_allocated()

                # Calculate metrics
                processing_time = end_time - start_time
                memory_used = (peak_memory - initial_memory) / 1e9  # GB
                throughput = batch_size / processing_time if processing_time > 0 else 0
                memory_efficiency = memory_used / self.gpu_profile.total_memory

                results[batch_size] = BatchPerformanceMetrics(
                    batch_size=batch_size,
                    processing_time=processing_time,
                    memory_peak=memory_used,
                    throughput=throughput,
                    memory_efficiency=memory_efficiency,
                    success_rate=1.0 if success else 0.0
                )

                # Reset peak memory counter
                torch.cuda.reset_peak_memory_stats()

                if not success:
                    break  # Stop testing larger batch sizes

            except Exception as e:
                print(f"Failed to profile batch size {batch_size}: {e}")
                continue

        return results

    def find_optimal_batch_size(self,
                              test_func: Callable,
                              max_batch_size: int = 32,
                              target_memory_usage: float = 0.8,
                              *args, **kwargs) -> int:
        """
        Find optimal batch size through binary search and profiling.

        Args:
            test_func: Function to test batch processing
            max_batch_size: Maximum batch size to test
            target_memory_usage: Target memory utilization (0-1)
            *args, **kwargs: Arguments for test function

        Returns:
            Optimal batch size
        """
        if not self.gpu_profile:
            return 1

        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal_batch_size = 1

        while low <= high:
            mid = (low + high) // 2

            # Test this batch size
            profile_results = self.profile_gpu_memory_for_batch_size(
                test_func, [mid], *args, **kwargs
            )

            if mid in profile_results and profile_results[mid].success_rate > 0:
                metrics = profile_results[mid]

                if metrics.memory_efficiency <= target_memory_usage:
                    optimal_batch_size = mid
                    low = mid + 1  # Try larger batch size
                else:
                    high = mid - 1  # Try smaller batch size
            else:
                high = mid - 1  # Batch size too large

        return optimal_batch_size

    def profile_and_optimize(self,
                           test_image_size: Tuple[int, int],
                           num_prompts: int,
                           memory_limit: Optional[float] = None,
                           model_type: str = "sam2") -> BatchConfig:
        """Enhanced profiling with adaptive optimization and performance tuning."""
        if self.device == "cpu" or not self.gpu_profile:
            return BatchConfig(
                detection_batch_size=1,
                segmentation_batch_size=1,
                frame_batch_size=1,
                use_mixed_precision=False,
                enable_gradient_checkpointing=True,
                optimization_level=self.optimization_level
            )

        # Use provided memory limit or calculate from available memory with safety margin
        available_memory = memory_limit or (self.gpu_profile.available_memory * 0.9)

        # Enhanced target memory usage with dynamic adjustment
        target_configs = {
            OptimizationLevel.CONSERVATIVE: {"target": 0.6, "safety": 0.8},
            OptimizationLevel.BALANCED: {"target": 0.75, "safety": 0.85},
            OptimizationLevel.AGGRESSIVE: {"target": 0.9, "safety": 0.95}
        }

        config = target_configs[self.optimization_level]
        target_memory_usage = config["target"]
        memory_safety_factor = config["safety"]

        # Calculate memory requirements with model-specific optimizations
        pixels_per_image = test_image_size[0] * test_image_size[1]
        base_memory_per_image = pixels_per_image * 4 * 3 / 1e9  # RGB float32

        # Model-specific memory optimizations
        model_optimizations = {
            "sam2": {"detection_factor": 1.0, "segmentation_factor": 1.0, "base_overhead": 3.0},
            "edgetam": {"detection_factor": 0.7, "segmentation_factor": 0.6, "base_overhead": 2.0},
            "owl": {"detection_factor": 1.2, "segmentation_factor": 1.0, "base_overhead": 3.5}
        }

        model_opt = model_optimizations.get(model_type, model_optimizations["sam2"])

        # Enhanced memory estimation with GPU architecture considerations
        if self.gpu_profile.compute_capability[0] >= 8:  # Ampere and newer
            memory_efficiency_factor = 1.2
        elif self.gpu_profile.compute_capability[0] >= 7:  # Turing/Volta
            memory_efficiency_factor = 1.1
        else:
            memory_efficiency_factor = 1.0

        # Adaptive memory allocation based on image size
        if pixels_per_image > 2048 * 2048:  # Very large images
            memory_allocation = {"detection": 0.25, "segmentation": 0.5, "frame": 0.25}
        elif pixels_per_image > 1024 * 1024:  # Large images
            memory_allocation = {"detection": 0.3, "segmentation": 0.45, "frame": 0.25}
        else:  # Normal/small images
            memory_allocation = {"detection": 0.35, "segmentation": 0.4, "frame": 0.25}

        # Calculate optimized batch sizes
        effective_memory = available_memory * target_memory_usage * memory_safety_factor * memory_efficiency_factor

        # Detection batch size with model optimization
        detection_memory_per_batch = (model_opt["base_overhead"] + base_memory_per_image * num_prompts) * model_opt["detection_factor"]
        detection_batch_size = max(1, int(
            (effective_memory * memory_allocation["detection"]) / detection_memory_per_batch
        ))

        # Segmentation batch size with model optimization
        segmentation_memory_per_image = (4.5 + base_memory_per_image * 2) * model_opt["segmentation_factor"]
        segmentation_batch_size = max(1, int(
            (effective_memory * memory_allocation["segmentation"]) / segmentation_memory_per_image
        ))

        # Frame processing batch size with temporal optimization
        frame_memory_per_batch = base_memory_per_image * 16
        frame_batch_size = max(1, int(
            (effective_memory * memory_allocation["frame"]) / frame_memory_per_batch
        ))

        # Apply intelligent constraints based on GPU capabilities
        gpu_memory_gb = self.gpu_profile.total_memory
        compute_units = self.gpu_profile.compute_units

        # Scale limits based on GPU power
        gpu_scale_factor = min(2.0, max(0.5, gpu_memory_gb / 8.0))  # Scale based on 8GB baseline
        compute_scale_factor = min(1.5, max(0.7, compute_units / 80))  # Scale based on typical GPU

        combined_scale = (gpu_scale_factor + compute_scale_factor) / 2

        # Enhanced optimization level constraints with GPU scaling
        base_limits = {
            OptimizationLevel.CONSERVATIVE: {"detection": 4, "segmentation": 2, "frame": 8},
            OptimizationLevel.BALANCED: {"detection": 8, "segmentation": 4, "frame": 16},
            OptimizationLevel.AGGRESSIVE: {"detection": 16, "segmentation": 8, "frame": 32}
        }

        limits = base_limits[self.optimization_level]
        scaled_limits = {k: max(1, int(v * combined_scale)) for k, v in limits.items()}

        # Apply final constraints
        detection_batch_size = min(detection_batch_size, scaled_limits["detection"])
        segmentation_batch_size = min(segmentation_batch_size, scaled_limits["segmentation"])
        frame_batch_size = min(frame_batch_size, scaled_limits["frame"])

        # Ensure minimum performance thresholds
        detection_batch_size = max(1, detection_batch_size)
        segmentation_batch_size = max(1, segmentation_batch_size)
        frame_batch_size = max(1, frame_batch_size)

        return BatchConfig(
            detection_batch_size=detection_batch_size,
            segmentation_batch_size=segmentation_batch_size,
            frame_batch_size=frame_batch_size,
            use_mixed_precision=self.gpu_profile.supports_mixed_precision and self.optimization_level != OptimizationLevel.CONSERVATIVE,
            enable_gradient_checkpointing=self.optimization_level == OptimizationLevel.CONSERVATIVE,
            optimization_level=self.optimization_level,
            memory_limit_gb=memory_limit
        )

    def adaptive_batch_processing(self,
                                items: List[Any],
                                process_func: Callable,
                                initial_batch_size: int,
                                max_retries: int = 3,
                                *args, **kwargs) -> List[Any]:
        """Enhanced adaptive batch processing with failure recovery."""
        results = []
        current_batch_size = initial_batch_size
        i = 0
        consecutive_successes = 0
        consecutive_failures = 0

        while i < len(items):
            batch_end = min(i + current_batch_size, len(items))
            batch = items[i:batch_end]
            retry_count = 0
            batch_processed = False

            while retry_count < max_retries and not batch_processed:
                try:
                    # Clear cache and synchronize
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Measure performance
                    start_time = time.time()
                    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                    # Process batch
                    batch_results = process_func(batch, *args, **kwargs)
                    results.extend(batch_results)

                    # Record performance metrics
                    processing_time = time.time() - start_time
                    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    memory_used = (peak_memory - initial_memory) / 1e9  # GB

                    # Update adaptive parameters
                    self._update_adaptive_parameters(
                        current_batch_size, processing_time, memory_used, True
                    )

                    consecutive_successes += 1
                    consecutive_failures = 0
                    batch_processed = True

                    # Dynamically adjust batch size based on performance
                    current_batch_size = self._adjust_batch_size_dynamically(
                        current_batch_size, consecutive_successes
                    )

                    i = batch_end

                except torch.cuda.OutOfMemoryError as e:
                    consecutive_failures += 1
                    consecutive_successes = 0
                    retry_count += 1

                    # Implement failure recovery with size reduction
                    new_batch_size = self._handle_batch_failure(
                        current_batch_size, consecutive_failures, retry_count
                    )

                    if new_batch_size < current_batch_size:
                        current_batch_size = new_batch_size
                        print(f"Reduced batch size to {current_batch_size} after OOM (attempt {retry_count})")

                    # If single item still fails after retries, skip it
                    if current_batch_size == 1 and retry_count >= max_retries:
                        print(f"Skipping item {i} after {max_retries} failed attempts")
                        i += 1
                        batch_processed = True

                except Exception as e:
                    print(f"Unexpected error in batch processing: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Skipping batch starting at {i} after {max_retries} failed attempts")
                        i = batch_end
                        batch_processed = True

        return results

    def _update_adaptive_parameters(self, batch_size: int, processing_time: float,
                                  memory_used: float, success: bool):
        """Update adaptive parameters based on processing results."""
        metrics = BatchPerformanceMetrics(
            batch_size=batch_size,
            processing_time=processing_time,
            memory_peak=memory_used,
            throughput=batch_size / processing_time if processing_time > 0 else 0,
            memory_efficiency=memory_used / self.gpu_profile.total_memory if self.gpu_profile else 0,
            success_rate=1.0 if success else 0.0
        )

        self.adaptive_history.append(metrics)

        # Keep only recent history
        if len(self.adaptive_history) > 100:
            self.adaptive_history.pop(0)

    def _adjust_batch_size_dynamically(self, current_batch_size: int,
                                     consecutive_successes: int) -> int:
        """Dynamically adjust batch size based on recent performance."""
        if not self.gpu_profile:
            return current_batch_size

        # Check current memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        else:
            memory_usage = 0.5  # Conservative estimate for CPU

        # Increase batch size if memory usage is low and we've had consecutive successes
        if consecutive_successes >= 3 and memory_usage < 0.6:
            max_increase = {
                OptimizationLevel.CONSERVATIVE: 1,
                OptimizationLevel.BALANCED: 2,
                OptimizationLevel.AGGRESSIVE: 4
            }[self.optimization_level]

            return min(current_batch_size + 1, current_batch_size + max_increase)

        # Decrease batch size if memory usage is high
        elif memory_usage > 0.8:
            return max(1, current_batch_size - 1)

        return current_batch_size

    def _handle_batch_failure(self, current_batch_size: int,
                            consecutive_failures: int, retry_count: int) -> int:
        """Handle batch processing failure with intelligent size reduction."""
        if not self.failure_recovery_enabled:
            return max(1, current_batch_size // 2)

        # More aggressive reduction for repeated failures
        if consecutive_failures > 2:
            reduction_factor = 4
        elif retry_count > 1:
            reduction_factor = 3
        else:
            reduction_factor = 2

        new_batch_size = max(1, current_batch_size // reduction_factor)

        # Record failure for future optimization
        self._update_adaptive_parameters(current_batch_size, 0.0, 0.0, False)

        return new_batch_size

    def enable_mixed_precision_support(self) -> bool:
        """
        Enable mixed precision support if available.

        Returns:
            bool: True if mixed precision is enabled
        """
        if self.gpu_profile and self.gpu_profile.supports_mixed_precision:
            try:
                # Test mixed precision capability
                with torch.cuda.amp.autocast():
                    test_tensor = torch.randn(10, 10, device=self.device)
                    _ = torch.matmul(test_tensor, test_tensor)
                return True
            except Exception as e:
                print(f"Mixed precision not available: {e}")
                return False
        return False

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on profiling history."""
        if not self.adaptive_history:
            return {"status": "No profiling data available"}

        # Analyze recent performance
        recent_metrics = self.adaptive_history[-10:]  # Last 10 batches

        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_memory_efficiency = sum(m.memory_efficiency for m in recent_metrics) / len(recent_metrics)
        success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)

        recommendations = {
            "current_performance": {
                "average_throughput": avg_throughput,
                "memory_efficiency": avg_memory_efficiency,
                "success_rate": success_rate
            },
            "recommendations": []
        }

        # Generate recommendations
        if avg_memory_efficiency < 0.5:
            recommendations["recommendations"].append(
                "Consider increasing batch sizes - memory is underutilized"
            )
        elif avg_memory_efficiency > 0.9:
            recommendations["recommendations"].append(
                "Consider reducing batch sizes - high memory pressure detected"
            )

        if success_rate < 0.9:
            recommendations["recommendations"].append(
                "Enable gradient checkpointing to reduce memory usage"
            )

        if self.gpu_profile and self.gpu_profile.supports_mixed_precision:
            recommendations["recommendations"].append(
                "Enable mixed precision training for better performance"
            )

        return recommendations

    def reset_adaptive_history(self):
        """Reset adaptive learning history."""
        self.adaptive_history.clear()
        self.profiling_results.clear()

    def set_optimization_level(self, level: OptimizationLevel):
        """Change optimization level."""
        self.optimization_level = level
        print(f"Optimization level set to: {level.name}")

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics."""
        if not self.adaptive_history:
            return {}

        metrics = self.adaptive_history
        return {
            "total_batches_processed": len(metrics),
            "average_batch_size": sum(m.batch_size for m in metrics) / len(metrics),
            "average_throughput": sum(m.throughput for m in metrics) / len(metrics),
            "average_memory_efficiency": sum(m.memory_efficiency for m in metrics) / len(metrics),
            "overall_success_rate": sum(m.success_rate for m in metrics) / len(metrics),
            "total_processing_time": sum(m.processing_time for m in metrics)
        }
