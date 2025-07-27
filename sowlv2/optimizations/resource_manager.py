"""
Advanced resource management system for SOWLv2 pipeline.
Provides comprehensive memory monitoring, batch optimization, and streaming capabilities.
"""
import gc
import psutil
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import torch


class ProcessingMode(Enum):
    """Processing mode based on resource availability."""
    NORMAL = "normal"
    MEMORY_EFFICIENT = "memory_efficient"
    STREAMING = "streaming"
    CPU_FALLBACK = "cpu_fallback"


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: float  # GB
    allocated_memory: float  # GB
    cached_memory: float  # GB
    free_memory: float  # GB
    utilization_percentage: float
    system_memory_usage: float  # System RAM usage percentage


@dataclass
class BatchConfig:
    """Dynamic batch configuration based on available resources."""
    detection_batch_size: int
    segmentation_batch_size: int
    frame_batch_size: int
    use_mixed_precision: bool
    enable_gradient_checkpointing: bool
    processing_mode: ProcessingMode


@dataclass
class StreamingConfig:
    """Configuration for streaming video processing."""
    chunk_size: int
    overlap_frames: int
    enable_progressive_loading: bool
    memory_threshold: float
    auto_cleanup: bool


@dataclass
class DeviceAllocation:
    """Device allocation strategy."""
    primary_device: str
    fallback_device: str
    model_device_mapping: Dict[str, str]
    memory_allocation: Dict[str, float]


class AdvancedResourceManager:
    """Advanced resource management with real-time monitoring and optimization."""

    def __init__(self, device: str = "cuda", memory_limit: Optional[float] = None):
        """
        Initialize the advanced resource manager.

        Args:
            device: Primary device to use ('cuda' or 'cpu')
            memory_limit: Optional memory limit in GB
        """
        self.device = device
        self.memory_limit = memory_limit
        self.monitoring_enabled = True
        self.cleanup_threshold = 0.85  # 85% memory usage triggers cleanup
        self.streaming_threshold = 0.9  # 90% memory usage triggers streaming mode

        # Performance tracking
        self.memory_history: List[MemoryStats] = []
        self.performance_metrics: Dict[str, float] = {}

        # Initialize device capabilities
        self._initialize_device_capabilities()

    def _initialize_device_capabilities(self):
        """Initialize device capabilities and constraints."""
        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_properties = torch.cuda.get_device_properties(0)
            self.total_gpu_memory = self.gpu_properties.total_memory / 1e9  # GB
            self.supports_mixed_precision = self.gpu_properties.major >= 7
        else:
            self.gpu_properties = None
            self.total_gpu_memory = 0
            self.supports_mixed_precision = False

        # System memory
        self.total_system_memory = psutil.virtual_memory().total / 1e9  # GB

    def monitor_memory_usage(self) -> MemoryStats:
        """
        Monitor real-time memory usage across GPU and system.

        Returns:
            MemoryStats: Current memory usage statistics
        """
        if self.device == "cuda" and torch.cuda.is_available():
            # GPU memory
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = self.total_gpu_memory
            free = total - allocated
            utilization = (allocated / total) * 100 if total > 0 else 0
        else:
            # CPU mode - monitor system memory
            allocated = 0
            cached = 0
            total = self.total_system_memory
            free = psutil.virtual_memory().available / 1e9
            utilization = psutil.virtual_memory().percent

        # System memory usage
        system_memory = psutil.virtual_memory().percent

        stats = MemoryStats(
            total_memory=total,
            allocated_memory=allocated,
            cached_memory=cached,
            free_memory=free,
            utilization_percentage=utilization,
            system_memory_usage=system_memory
        )

        # Store in history for trend analysis
        if self.monitoring_enabled:
            self.memory_history.append(stats)
            # Keep only last 100 measurements
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)

        return stats

    def optimize_batch_sizes(self, current_usage: float,
                           image_size: Tuple[int, int] = (1024, 1024),
                           num_prompts: int = 1,
                           model_type: str = "sam2") -> BatchConfig:
        """
        Dynamically optimize batch sizes with enhanced algorithms and model-specific tuning.

        Args:
            current_usage: Current memory utilization percentage
            image_size: Input image dimensions
            num_prompts: Number of detection prompts
            model_type: Type of model being used (sam2, edgetam, etc.)

        Returns:
            BatchConfig: Optimized batch configuration
        """
        # Enhanced processing mode determination with hysteresis
        mode = self._determine_processing_mode_with_hysteresis(current_usage)

        # Calculate base memory requirements with model-specific factors
        pixels = image_size[0] * image_size[1]
        base_memory_per_image = pixels * 4 * 3 / 1e9  # RGB float32 in GB

        # Model-specific memory multipliers
        model_memory_factors = {
            "sam2": {"detection": 1.0, "segmentation": 1.0},
            "edgetam": {"detection": 0.7, "segmentation": 0.6},  # EdgeTAM is more efficient
            "owl": {"detection": 1.2, "segmentation": 1.0}
        }

        model_factor = model_memory_factors.get(model_type, {"detection": 1.0, "segmentation": 1.0})

        # CPU fallback configuration
        if mode == ProcessingMode.CPU_FALLBACK:
            return BatchConfig(
                detection_batch_size=1,
                segmentation_batch_size=1,
                frame_batch_size=1,
                use_mixed_precision=False,
                enable_gradient_checkpointing=True,
                processing_mode=mode
            )

        # Calculate available memory with safety margin
        safety_margins = {
            ProcessingMode.NORMAL: 0.1,
            ProcessingMode.MEMORY_EFFICIENT: 0.2,
            ProcessingMode.STREAMING: 0.3
        }

        safety_margin = safety_margins.get(mode, 0.1)
        available_memory = self.total_gpu_memory * (1 - current_usage / 100) * (1 - safety_margin)

        if self.memory_limit:
            available_memory = min(available_memory, self.memory_limit * (1 - safety_margin))

        # Enhanced memory allocation strategy with adaptive factors
        memory_allocation = self._get_adaptive_memory_allocation(mode, current_usage)

        # Calculate optimal batch sizes with model-specific adjustments
        detection_memory_per_batch = (2.0 + base_memory_per_image * num_prompts) * model_factor["detection"]
        detection_batch_size = max(1, int(
            (available_memory * memory_allocation["detection"]) / detection_memory_per_batch
        ))

        segmentation_memory_per_image = (4.0 + base_memory_per_image * 2) * model_factor["segmentation"]
        segmentation_batch_size = max(1, int(
            (available_memory * memory_allocation["segmentation"]) / segmentation_memory_per_image
        ))

        frame_memory_per_batch = base_memory_per_image * 16
        frame_batch_size = max(1, int(
            (available_memory * memory_allocation["frame"]) / frame_memory_per_batch
        ))

        # Apply intelligent caps with performance considerations
        caps = self._get_performance_aware_caps(mode, image_size, model_type)

        detection_batch_size = min(detection_batch_size, caps["detection"])
        segmentation_batch_size = min(segmentation_batch_size, caps["segmentation"])
        frame_batch_size = min(frame_batch_size, caps["frame"])

        # Ensure minimum viable batch sizes
        detection_batch_size = max(1, detection_batch_size)
        segmentation_batch_size = max(1, segmentation_batch_size)
        frame_batch_size = max(1, frame_batch_size)

        return BatchConfig(
            detection_batch_size=detection_batch_size,
            segmentation_batch_size=segmentation_batch_size,
            frame_batch_size=frame_batch_size,
            use_mixed_precision=self.supports_mixed_precision and mode != ProcessingMode.CPU_FALLBACK,
            enable_gradient_checkpointing=mode in [ProcessingMode.MEMORY_EFFICIENT, ProcessingMode.STREAMING],
            processing_mode=mode
        )

    def _determine_processing_mode_with_hysteresis(self, current_usage: float) -> ProcessingMode:
        """Determine processing mode with hysteresis to prevent oscillation."""
        # Get previous mode if available
        previous_mode = getattr(self, '_previous_mode', ProcessingMode.NORMAL)

        # Define thresholds with hysteresis
        if previous_mode == ProcessingMode.NORMAL:
            cpu_threshold, streaming_threshold, efficient_threshold = 92, 82, 72
        elif previous_mode == ProcessingMode.MEMORY_EFFICIENT:
            cpu_threshold, streaming_threshold, efficient_threshold = 90, 80, 65
        elif previous_mode == ProcessingMode.STREAMING:
            cpu_threshold, streaming_threshold, efficient_threshold = 88, 75, 70
        else:  # CPU_FALLBACK
            cpu_threshold, streaming_threshold, efficient_threshold = 85, 78, 68

        # Determine new mode
        if current_usage > cpu_threshold:
            mode = ProcessingMode.CPU_FALLBACK
        elif current_usage > streaming_threshold:
            mode = ProcessingMode.STREAMING
        elif current_usage > efficient_threshold:
            mode = ProcessingMode.MEMORY_EFFICIENT
        else:
            mode = ProcessingMode.NORMAL

        self._previous_mode = mode
        return mode

    def _get_adaptive_memory_allocation(self, mode: ProcessingMode, current_usage: float) -> Dict[str, float]:
        """Get adaptive memory allocation factors based on mode and usage."""
        base_allocations = {
            ProcessingMode.NORMAL: {"detection": 0.35, "segmentation": 0.45, "frame": 0.2},
            ProcessingMode.MEMORY_EFFICIENT: {"detection": 0.25, "segmentation": 0.35, "frame": 0.15},
            ProcessingMode.STREAMING: {"detection": 0.2, "segmentation": 0.3, "frame": 0.1}
        }

        allocation = base_allocations.get(mode, base_allocations[ProcessingMode.NORMAL])

        # Adjust based on current usage (more conservative as usage increases)
        usage_factor = max(0.5, 1.0 - (current_usage - 50) / 100)

        return {k: v * usage_factor for k, v in allocation.items()}

    def _get_performance_aware_caps(self, mode: ProcessingMode, image_size: Tuple[int, int],
                                  model_type: str) -> Dict[str, int]:
        """Get performance-aware batch size caps."""
        # Base caps by mode
        base_caps = {
            ProcessingMode.NORMAL: {"detection": 12, "segmentation": 6, "frame": 24},
            ProcessingMode.MEMORY_EFFICIENT: {"detection": 6, "segmentation": 3, "frame": 12},
            ProcessingMode.STREAMING: {"detection": 4, "segmentation": 2, "frame": 8}
        }

        caps = base_caps.get(mode, base_caps[ProcessingMode.NORMAL])

        # Adjust for image size (larger images need smaller batches)
        pixels = image_size[0] * image_size[1]
        if pixels > 2048 * 2048:  # Very large images
            size_factor = 0.5
        elif pixels > 1024 * 1024:  # Large images
            size_factor = 0.7
        else:  # Normal/small images
            size_factor = 1.0

        # Adjust for model type
        model_factors = {
            "sam2": 1.0,
            "edgetam": 1.4,  # EdgeTAM can handle larger batches
            "owl": 0.8
        }

        model_factor = model_factors.get(model_type, 1.0)

        # Apply adjustments
        final_factor = size_factor * model_factor

        return {k: max(1, int(v * final_factor)) for k, v in caps.items()}

    def enable_streaming_mode(self, video_size: int,
                            target_memory_usage: float = 0.7) -> StreamingConfig:
        """
        Configure streaming mode for large video processing.

        Args:
            video_size: Total number of frames in video
            target_memory_usage: Target memory utilization percentage

        Returns:
            StreamingConfig: Streaming configuration
        """
        current_stats = self.monitor_memory_usage()
        available_memory = current_stats.free_memory

        # Estimate memory per frame (conservative estimate)
        memory_per_frame = 0.1  # GB per frame

        # Calculate optimal chunk size
        max_frames_in_memory = int((available_memory * target_memory_usage) / memory_per_frame)
        chunk_size = min(max_frames_in_memory, max(10, video_size // 10))

        # Overlap for temporal consistency
        overlap_frames = min(5, chunk_size // 4)

        # Enable progressive loading for very large videos
        enable_progressive = video_size > chunk_size * 2

        return StreamingConfig(
            chunk_size=chunk_size,
            overlap_frames=overlap_frames,
            enable_progressive_loading=enable_progressive,
            memory_threshold=target_memory_usage,
            auto_cleanup=True
        )

    def cleanup_resources(self, force: bool = False):
        """
        Clean up resources and free memory.

        Args:
            force: Force cleanup regardless of current usage
        """
        current_stats = self.monitor_memory_usage()

        if force or current_stats.utilization_percentage > self.cleanup_threshold * 100:
            # Clear Python garbage
            gc.collect()

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Log cleanup action
            print(f"Resource cleanup performed. Memory usage: {current_stats.utilization_percentage:.1f}%")

    def get_optimal_device_allocation(self) -> DeviceAllocation:
        """
        Determine optimal device allocation strategy.

        Returns:
            DeviceAllocation: Device allocation configuration
        """
        current_stats = self.monitor_memory_usage()

        # Primary device selection
        if self.device == "cuda" and torch.cuda.is_available():
            if current_stats.utilization_percentage < 80:
                primary_device = "cuda"
                fallback_device = "cpu"
            else:
                primary_device = "cpu"
                fallback_device = "cuda"
        else:
            primary_device = "cpu"
            fallback_device = "cpu"

        # Model-specific device mapping
        model_device_mapping = {
            "owl": primary_device,
            "sam2": primary_device if current_stats.utilization_percentage < 70 else fallback_device,
            "edgetam": primary_device,
            "vjepa2": fallback_device if current_stats.utilization_percentage > 60 else primary_device
        }

        # Memory allocation per model (percentage of available memory)
        if primary_device == "cuda":
            memory_allocation = {
                "owl": 0.3,
                "sam2": 0.4,
                "edgetam": 0.35,
                "vjepa2": 0.2
            }
        else:
            memory_allocation = {
                "owl": 0.25,
                "sam2": 0.3,
                "edgetam": 0.25,
                "vjepa2": 0.2
            }

        return DeviceAllocation(
            primary_device=primary_device,
            fallback_device=fallback_device,
            model_device_mapping=model_device_mapping,
            memory_allocation=memory_allocation
        )

    def get_memory_trend(self, window_size: int = 10) -> Dict[str, float]:
        """
        Analyze memory usage trends.

        Args:
            window_size: Number of recent measurements to analyze

        Returns:
            Dict containing trend analysis
        """
        if len(self.memory_history) < 2:
            return {"trend": 0.0, "stability": 1.0, "peak_usage": 0.0}

        recent_history = self.memory_history[-window_size:]

        # Calculate trend (positive = increasing usage)
        if len(recent_history) >= 2:
            trend = (recent_history[-1].utilization_percentage -
                    recent_history[0].utilization_percentage) / len(recent_history)
        else:
            trend = 0.0

        # Calculate stability (lower = more stable)
        utilizations = [stat.utilization_percentage for stat in recent_history]
        if len(utilizations) > 1:
            stability = sum(abs(utilizations[i] - utilizations[i-1])
                          for i in range(1, len(utilizations))) / (len(utilizations) - 1)
        else:
            stability = 0.0

        # Peak usage
        peak_usage = max(stat.utilization_percentage for stat in recent_history)

        return {
            "trend": trend,
            "stability": stability,
            "peak_usage": peak_usage,
            "current_usage": recent_history[-1].utilization_percentage
        }

    def should_enable_streaming(self, video_frames: int,
                              frame_size: Tuple[int, int] = (1024, 1024)) -> bool:
        """
        Determine if streaming mode should be enabled for a video.

        Args:
            video_frames: Number of frames in video
            frame_size: Frame dimensions

        Returns:
            bool: True if streaming should be enabled
        """
        current_stats = self.monitor_memory_usage()

        # Estimate memory needed for full video processing
        pixels_per_frame = frame_size[0] * frame_size[1]
        memory_per_frame = pixels_per_frame * 4 * 3 / 1e9  # RGB float32
        estimated_memory = video_frames * memory_per_frame * 2  # 2x for processing overhead

        # Enable streaming if:
        # 1. Estimated memory exceeds available memory
        # 2. Current memory usage is already high
        # 3. Video is very long (>1000 frames)
        return (estimated_memory > current_stats.free_memory * 0.8 or
                current_stats.utilization_percentage > 70 or
                video_frames > 1000)
