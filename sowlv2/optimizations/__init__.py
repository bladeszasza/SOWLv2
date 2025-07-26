"""Optimization modules for SOWLv2 pipeline."""

from .parallel_processor import (
    ParallelConfig,
    ParallelDetectionProcessor,
    ParallelSegmentationProcessor,
    ParallelIOProcessor,
    ParallelFrameProcessor,
    BatchDetectionResult
)

from .gpu_optimizations import (
    GPUOptimizer,
    StreamedProcessing,
    TensorRTOptimizer
)

from .optimized_pipeline import (
    OptimizedSOWLv2Pipeline,
    ModelOptimizations,
    CachedModelWrapper
)

from .vjepa2_optimization import (
    VJepa2VideoOptimizer,
    create_vjepa2_optimizer
)

from .temporal_detection import (
    TemporalDetection,
    TrackedObject,
    compute_iou,
    merge_temporal_detections,
    select_key_frames_for_detection
)

from .model_cache import IntelligentModelCache

from .batch_optimizer import (
    BatchConfig,
    IntelligentBatchOptimizer
)

from .performance_collector import (
    PerformanceCollector,
    PerformanceMetrics,
    ComparisonReport,
    TimingContext
)

from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResults,
    MemoryProfile,
    ThroughputResults
)

from .monitoring import (
    MonitoringDashboard,
    AlertConfig,
    ProgressInfo,
    ResourceUtilization,
    PerformanceAlert
)

__all__ = [
    # Parallel processing
    'ParallelConfig',
    'ParallelDetectionProcessor',
    'ParallelSegmentationProcessor',
    'ParallelIOProcessor',
    'ParallelFrameProcessor',
    'BatchDetectionResult',

    # GPU optimizations
    'GPUOptimizer',
    'StreamedProcessing',
    'TensorRTOptimizer',

    # Optimized pipeline
    'OptimizedSOWLv2Pipeline',
    'ModelOptimizations',
    'CachedModelWrapper',

    # V-JEPA 2 optimization
    'VJepa2VideoOptimizer',
    'create_vjepa2_optimizer',

    # Temporal detection
    'TemporalDetection',
    'TrackedObject',
    'compute_iou',
    'merge_temporal_detections',
    'select_key_frames_for_detection',

    # Model management
    'IntelligentModelCache',
    'BatchConfig',
    'IntelligentBatchOptimizer',

    # Performance monitoring
    'PerformanceCollector',
    'PerformanceMetrics',
    'ComparisonReport',
    'TimingContext',

    # Benchmarking
    'BenchmarkRunner',
    'BenchmarkConfig',
    'BenchmarkResults',
    'MemoryProfile',
    'ThroughputResults',

    # Real-time monitoring
    'MonitoringDashboard',
    'AlertConfig',
    'ProgressInfo',
    'ResourceUtilization',
    'PerformanceAlert',
]
