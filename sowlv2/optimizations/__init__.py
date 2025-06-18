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

    # Pipeline
    'OptimizedSOWLv2Pipeline',
    'ModelOptimizations',
    'CachedModelWrapper',

    # V-JEPA 2 optimization
    'VJepa2VideoOptimizer',
    'create_vjepa2_optimizer'
]
