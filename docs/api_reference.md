# SOWLv2 API Reference

## Overview

This document provides comprehensive API documentation for SOWLv2 with EdgeTAM integration and optimization features. The API is organized into several key modules for different functionality areas.

## Table of Contents

- [Model Management](#model-management)
- [EdgeTAM Integration](#edgetam-integration)
- [Resource Management](#resource-management)
- [Performance Optimization](#performance-optimization)
- [V-JEPA2 Enhancement](#v-jepa2-enhancement)
- [Monitoring and Benchmarking](#monitoring-and-benchmarking)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

## Model Management

### SegmentationModelFactory

Factory class for creating and managing segmentation models.

```python
from sowlv2.models.model_factory import SegmentationModelFactory

# Create EdgeTAM model
model = SegmentationModelFactory.create_model(
    model_type="edgetam",
    model_name="facebook/edgetam-base",
    device="cuda",
    enable_fallback=True
)

# Create SAM2 model
model = SegmentationModelFactory.create_model(
    model_type="sam2",
    model_name="facebook/sam2.1-hiera-small",
    device="cuda"
)
```

#### Methods

##### `create_model(model_type, model_name, device="cpu", enable_fallback=True)`

Creates a segmentation model instance with automatic fallback support.

**Parameters:**
- `model_type` (str): Type of model ("sam2" or "edgetam")
- `model_name` (str): Specific model name/identifier
- `device` (str): Device to run model on ('cuda' or 'cpu')
- `enable_fallback` (bool): Enable automatic fallback on failure

**Returns:**
- Model instance (EdgeTAMWrapper or SAM2Wrapper)

**Raises:**
- `ModelLoadingError`: If model fails to load and fallback is disabled
- `UnsupportedModelError`: If model type is not supported

##### `get_available_models()`

Returns dictionary of available models by type.

**Returns:**
- `Dict[str, List[str]]`: Dictionary mapping model types to available models

```python
models = SegmentationModelFactory.get_available_models()
# Returns: {
#     "edgetam": ["facebook/edgetam-small", "facebook/edgetam-base"],
#     "sam2": ["facebook/sam2.1-hiera-tiny", "facebook/sam2.1-hiera-small"]
# }
```

##### `validate_model_compatibility(model_type, model_name, device)`

Validates model compatibility with current system.

**Parameters:**
- `model_type` (str): Model type to validate
- `model_name` (str): Model name to validate
- `device` (str): Target device

**Returns:**
- `bool`: True if compatible, False otherwise

## EdgeTAM Integration

### EdgeTAMWrapper

Wrapper class providing SAM2-compatible interface for EdgeTAM models.

```python
from sowlv2.models.edgetam_wrapper import EdgeTAMWrapper

# Initialize EdgeTAM
edgetam = EdgeTAMWrapper(
    model_name="facebook/edgetam-base",
    device="cuda"
)

# Single image segmentation
mask = edgetam.segment(image, box_xyxy=[100, 100, 200, 200])

# Video tracking
state = edgetam.init_state("frames_directory/")
edgetam.add_new_box(state, frame_idx=0, box=[100, 100, 200, 200], obj_idx=1)
for frame_idx, masks in edgetam.propagate_in_video(state):
    print(f"Frame {frame_idx}: {len(masks)} objects tracked")
```

#### Methods

##### `__init__(model_name="facebook/edgetam-base", device="cpu")`

Initialize EdgeTAM wrapper.

**Parameters:**
- `model_name` (str): EdgeTAM model identifier
- `device` (str): Device to run model on

##### `segment(pil_image, box_xyxy)`

Perform single-image segmentation.

**Parameters:**
- `pil_image` (PIL.Image): Input image
- `box_xyxy` (List[float]): Bounding box coordinates [x1, y1, x2, y2]

**Returns:**
- `np.ndarray`: Binary segmentation mask

##### `init_state(frames_dir)`

Initialize video tracking state.

**Parameters:**
- `frames_dir` (str): Directory containing video frames

**Returns:**
- Video tracking state object

##### `add_new_box(state, frame_idx, box, obj_idx)`

Add new object to track in video.

**Parameters:**
- `state`: Video tracking state
- `frame_idx` (int): Frame index to add object
- `box` (List[float]): Bounding box coordinates
- `obj_idx` (int): Object identifier

##### `propagate_in_video(state)`

Propagate object tracking through video frames.

**Parameters:**
- `state`: Video tracking state

**Returns:**
- `Iterator`: Iterator yielding (frame_idx, masks) tuples

##### `get_performance_metrics()`

Get performance metrics for the model.

**Returns:**
- `Dict[str, float]`: Performance metrics including timing and memory usage

## Resource Management

### AdvancedResourceManager

Comprehensive resource management for memory, GPU, and processing optimization.

```python
from sowlv2.optimizations.resource_manager import AdvancedResourceManager

# Initialize resource manager
resource_manager = AdvancedResourceManager(
    device="cuda",
    memory_limit=8.0  # 8GB limit
)

# Monitor memory usage
memory_stats = resource_manager.monitor_memory_usage()
print(f"GPU Memory: {memory_stats.utilization_percentage:.1f}%")

# Optimize batch sizes
batch_config = resource_manager.optimize_batch_sizes(current_usage=0.7)
print(f"Recommended batch size: {batch_config.detection_batch_size}")

# Enable streaming for large videos
streaming_config = resource_manager.enable_streaming_mode(video_size=1000)
```

#### Data Classes

##### `MemoryStats`

Memory usage statistics.

**Attributes:**
- `total_memory` (float): Total GPU memory in GB
- `allocated_memory` (float): Currently allocated memory in GB
- `cached_memory` (float): Cached memory in GB
- `free_memory` (float): Free memory in GB
- `utilization_percentage` (float): Memory utilization percentage
- `system_memory_usage` (float): System RAM usage percentage

##### `BatchConfig`

Dynamic batch configuration.

**Attributes:**
- `detection_batch_size` (int): Batch size for detection
- `segmentation_batch_size` (int): Batch size for segmentation
- `frame_batch_size` (int): Batch size for frame processing
- `use_mixed_precision` (bool): Whether to use mixed precision
- `enable_gradient_checkpointing` (bool): Whether to use gradient checkpointing
- `processing_mode` (ProcessingMode): Current processing mode

##### `StreamingConfig`

Streaming processing configuration.

**Attributes:**
- `chunk_size` (int): Number of frames per chunk
- `overlap_frames` (int): Overlap between chunks
- `enable_progressive_loading` (bool): Whether to load frames progressively

#### Methods

##### `monitor_memory_usage()`

Monitor current memory usage across GPU and system.

**Returns:**
- `MemoryStats`: Current memory statistics

##### `optimize_batch_sizes(current_usage)`

Optimize batch sizes based on current memory usage.

**Parameters:**
- `current_usage` (float): Current memory utilization (0.0-1.0)

**Returns:**
- `BatchConfig`: Optimized batch configuration

##### `enable_streaming_mode(video_size)`

Configure streaming mode for large video processing.

**Parameters:**
- `video_size` (int): Video size in frames

**Returns:**
- `StreamingConfig`: Streaming configuration

##### `cleanup_resources(force=False)`

Clean up GPU memory and cached resources.

**Parameters:**
- `force` (bool): Force aggressive cleanup

##### `get_optimal_device_allocation()`

Get optimal device allocation for multi-device systems.

**Returns:**
- `DeviceAllocation`: Optimal device allocation configuration

## Performance Optimization

### IntelligentBatchOptimizer

Advanced batch processing optimization with adaptive sizing.

```python
from sowlv2.optimizations.batch_optimizer import IntelligentBatchOptimizer

# Initialize optimizer
optimizer = IntelligentBatchOptimizer(
    device="cuda",
    initial_batch_size=16,
    memory_limit=8.0
)

# Optimize batch processing
optimized_batches = optimizer.optimize_batch_processing(
    frames=video_frames,
    prompts=["person", "car"]
)

# Get performance metrics
metrics = optimizer.get_optimization_metrics()
```

#### Methods

##### `optimize_batch_processing(frames, prompts)`

Optimize batch processing for given frames and prompts.

**Parameters:**
- `frames` (List): List of video frames
- `prompts` (List[str]): Detection prompts

**Returns:**
- `List[BatchResult]`: Optimized batch results

##### `adaptive_batch_sizing(current_memory_usage)`

Dynamically adjust batch size based on memory usage.

**Parameters:**
- `current_memory_usage` (float): Current memory utilization

**Returns:**
- `int`: Optimal batch size

##### `enable_mixed_precision_optimization()`

Enable mixed precision optimization for compatible hardware.

**Returns:**
- `bool`: True if successfully enabled

### StreamingVideoProcessor

Streaming processor for large video files.

```python
from sowlv2.optimizations.streaming_processor import StreamingVideoProcessor

# Initialize streaming processor
processor = StreamingVideoProcessor(
    chunk_size=100,
    overlap_frames=5,
    memory_limit=6.0
)

# Process video in chunks
for chunk_result in processor.process_video_stream(
    video_path="large_video.mp4",
    prompts=["person"]
):
    print(f"Processed chunk {chunk_result.chunk_id}")
```

#### Methods

##### `process_video_stream(video_path, prompts)`

Process video in streaming chunks.

**Parameters:**
- `video_path` (str): Path to video file
- `prompts` (List[str]): Detection prompts

**Returns:**
- `Iterator[ChunkResult]`: Iterator of chunk processing results

##### `configure_streaming_parameters(video_info)`

Configure streaming parameters based on video characteristics.

**Parameters:**
- `video_info` (Dict): Video metadata

**Returns:**
- `StreamingConfig`: Optimized streaming configuration

## V-JEPA2 Enhancement

### VJepa2VideoOptimizer

Enhanced V-JEPA2 optimization with motion-aware frame selection.

```python
from sowlv2.optimizations.vjepa2_optimization import VJepa2VideoOptimizer

# Initialize V-JEPA2 optimizer
optimizer = VJepa2VideoOptimizer(
    model_name="facebook/vjepa2-base",
    device="cuda"
)

# Get importance scores for frames
importance_scores = optimizer.get_motion_aware_importance_scores(
    frames=video_frames,
    content_type="dynamic"
)

# Select optimal frames
selected_frames = optimizer.select_keyframes_with_temporal_diversity(
    frames=video_frames,
    importance_scores=importance_scores,
    max_frames=100
)
```

#### Methods

##### `get_motion_aware_importance_scores(frames, content_type="mixed")`

Calculate motion-aware importance scores for video frames.

**Parameters:**
- `frames` (List[PIL.Image]): Video frames
- `content_type` (str): Content type ("static", "dynamic", "mixed")

**Returns:**
- `List[float]`: Importance scores for each frame

##### `select_keyframes_with_temporal_diversity(frames, importance_scores, max_frames)`

Select keyframes with temporal diversity consideration.

**Parameters:**
- `frames` (List[PIL.Image]): Video frames
- `importance_scores` (List[float]): Frame importance scores
- `max_frames` (int): Maximum number of frames to select

**Returns:**
- `List[int]`: Indices of selected frames

##### `optimize_for_content_type(content_analysis)`

Optimize parameters based on content analysis.

**Parameters:**
- `content_analysis` (ContentAnalysis): Video content analysis results

**Returns:**
- `OptimizationConfig`: Content-specific optimization configuration

### ContentAnalyzer

Video content analysis for adaptive optimization.

```python
from sowlv2.optimizations.content_analyzer import ContentAnalyzer

# Initialize content analyzer
analyzer = ContentAnalyzer()

# Analyze video content
content_analysis = analyzer.analyze_video_content(video_frames)
print(f"Content type: {content_analysis.content_type}")
print(f"Motion level: {content_analysis.motion_level}")

# Get optimization recommendations
recommendations = analyzer.get_optimization_recommendations(content_analysis)
```

#### Methods

##### `analyze_video_content(frames)`

Analyze video content characteristics.

**Parameters:**
- `frames` (List[PIL.Image]): Video frames to analyze

**Returns:**
- `ContentAnalysis`: Content analysis results

##### `get_optimization_recommendations(content_analysis)`

Get optimization recommendations based on content analysis.

**Parameters:**
- `content_analysis` (ContentAnalysis): Content analysis results

**Returns:**
- `Dict[str, Any]`: Optimization recommendations

## Monitoring and Benchmarking

### PerformanceCollector

Comprehensive performance metrics collection.

```python
from sowlv2.optimizations.performance_collector import PerformanceCollector

# Initialize collector
collector = PerformanceCollector()

# Start timing operation
timer_id = collector.start_timing("detection")

# ... perform detection ...

# End timing
collector.end_timing(timer_id)

# Record memory usage
collector.record_memory_usage("post_detection")

# Generate performance report
report = collector.generate_report()
```

#### Methods

##### `start_timing(operation)`

Start timing an operation.

**Parameters:**
- `operation` (str): Operation name

**Returns:**
- `str`: Timer ID for ending the timing

##### `end_timing(timer_id)`

End timing for an operation.

**Parameters:**
- `timer_id` (str): Timer ID from start_timing

##### `record_memory_usage(stage)`

Record memory usage at a specific stage.

**Parameters:**
- `stage` (str): Processing stage name

##### `record_gpu_utilization(stage)`

Record GPU utilization at a specific stage.

**Parameters:**
- `stage` (str): Processing stage name

##### `compare_models(sam2_metrics, edgetam_metrics)`

Compare performance metrics between models.

**Parameters:**
- `sam2_metrics` (Dict): SAM2 performance metrics
- `edgetam_metrics` (Dict): EdgeTAM performance metrics

**Returns:**
- `ComparisonReport`: Detailed comparison report

##### `generate_report()`

Generate comprehensive performance report.

**Returns:**
- `PerformanceReport`: Complete performance report

### BenchmarkRunner

Automated benchmarking system.

```python
from sowlv2.optimizations.benchmark_runner import BenchmarkRunner

# Initialize benchmark runner
runner = BenchmarkRunner()

# Run comparative benchmark
results = runner.run_comparative_benchmark(
    test_data=["video1.mp4", "video2.mp4"],
    models=["edgetam-base", "sam2-small"]
)

# Profile memory usage
memory_profile = runner.profile_memory_usage(pipeline_config)

# Measure throughput
throughput_results = runner.measure_throughput(batch_sizes=[1, 4, 8, 16])
```

#### Methods

##### `run_comparative_benchmark(test_data, models=None)`

Run comparative benchmark across different models.

**Parameters:**
- `test_data` (List[str]): List of test video paths
- `models` (List[str], optional): Models to benchmark

**Returns:**
- `BenchmarkResults`: Comprehensive benchmark results

##### `profile_memory_usage(pipeline_config)`

Profile memory usage for a pipeline configuration.

**Parameters:**
- `pipeline_config` (PipelineConfig): Pipeline configuration

**Returns:**
- `MemoryProfile`: Detailed memory usage profile

##### `measure_throughput(batch_sizes)`

Measure processing throughput for different batch sizes.

**Parameters:**
- `batch_sizes` (List[int]): Batch sizes to test

**Returns:**
- `ThroughputResults`: Throughput measurement results

## Error Handling

### ErrorRecoveryManager

Comprehensive error recovery and fallback system.

```python
from sowlv2.utils.error_recovery import ErrorRecoveryManager

# Initialize error recovery
recovery_manager = ErrorRecoveryManager()

# Handle model loading error with fallback
fallback_model = recovery_manager.handle_model_loading_error(
    model_name="facebook/edgetam-base",
    error=loading_exception
)

# Handle memory overflow
new_config = recovery_manager.handle_memory_overflow(current_config)

# Implement retry logic
result = recovery_manager.implement_retry_logic(
    operation=lambda: process_frame(frame),
    max_retries=3
)
```

#### Methods

##### `handle_model_loading_error(model_name, error)`

Handle model loading errors with automatic fallback.

**Parameters:**
- `model_name` (str): Name of failed model
- `error` (Exception): Loading error

**Returns:**
- `str`: Fallback model name

##### `handle_memory_overflow(current_config)`

Handle memory overflow by adjusting configuration.

**Parameters:**
- `current_config` (BatchConfig): Current batch configuration

**Returns:**
- `BatchConfig`: Adjusted configuration

##### `handle_processing_failure(stage, error)`

Handle processing failures with recovery strategies.

**Parameters:**
- `stage` (str): Processing stage that failed
- `error` (Exception): Processing error

**Returns:**
- `bool`: True if recovery successful

##### `implement_retry_logic(operation, max_retries=3)`

Implement retry logic with exponential backoff.

**Parameters:**
- `operation` (Callable): Operation to retry
- `max_retries` (int): Maximum retry attempts

**Returns:**
- Result of successful operation

## Configuration

### Configuration Classes

Data classes for various configuration options.

#### `EdgeTAMConfig`

EdgeTAM-specific configuration.

```python
from sowlv2.data.config import EdgeTAMConfig

config = EdgeTAMConfig(
    model_name="facebook/edgetam-base",
    enable_video_tracking=True,
    optimization_level=2,
    memory_efficient_mode=False
)
```

**Attributes:**
- `model_name` (str): EdgeTAM model name
- `enable_video_tracking` (bool): Enable video tracking mode
- `optimization_level` (int): Optimization level (0-3)
- `memory_efficient_mode` (bool): Enable memory-efficient processing

#### `OptimizationConfig`

General optimization configuration.

```python
from sowlv2.data.config import OptimizationConfig

config = OptimizationConfig(
    enable_mixed_precision=True,
    use_gradient_checkpointing=False,
    streaming_chunk_size=100,
    memory_limit_gb=8.0,
    optimization_level=2
)
```

**Attributes:**
- `enable_mixed_precision` (bool): Use FP16 precision
- `use_gradient_checkpointing` (bool): Enable gradient checkpointing
- `streaming_chunk_size` (int): Frames per streaming chunk
- `memory_limit_gb` (float): GPU memory limit
- `optimization_level` (int): Global optimization level

#### `BenchmarkConfig`

Benchmarking configuration.

```python
from sowlv2.data.config import BenchmarkConfig

config = BenchmarkConfig(
    enable_benchmarking=True,
    collect_memory_stats=True,
    compare_models=True,
    output_format="html"
)
```

**Attributes:**
- `enable_benchmarking` (bool): Enable benchmarking
- `collect_memory_stats` (bool): Collect memory statistics
- `compare_models` (bool): Compare different models
- `output_format` (str): Output format ("json", "html", "csv")

## Usage Examples

### Basic EdgeTAM Usage

```python
from sowlv2.models.model_factory import SegmentationModelFactory
from PIL import Image

# Create EdgeTAM model
model = SegmentationModelFactory.create_model(
    model_type="edgetam",
    model_name="facebook/edgetam-base",
    device="cuda"
)

# Load image and perform segmentation
image = Image.open("test_image.jpg")
mask = model.segment(image, box_xyxy=[100, 100, 300, 300])
```

### Advanced Pipeline with Optimization

```python
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.data.config import OptimizationConfig, EdgeTAMConfig

# Configure optimization
opt_config = OptimizationConfig(
    enable_mixed_precision=True,
    streaming_chunk_size=100,
    memory_limit_gb=8.0,
    optimization_level=2
)

# Configure EdgeTAM
edgetam_config = EdgeTAMConfig(
    model_name="facebook/edgetam-base",
    optimization_level=2
)

# Initialize optimized pipeline
pipeline = OptimizedSOWLv2Pipeline(
    optimization_config=opt_config,
    edgetam_config=edgetam_config
)

# Process video
results = pipeline.process_video(
    video_path="input_video.mp4",
    prompts=["person", "car"],
    output_dir="output/"
)
```

### Performance Monitoring

```python
from sowlv2.optimizations.performance_collector import PerformanceCollector
from sowlv2.optimizations.benchmark_runner import BenchmarkRunner

# Initialize monitoring
collector = PerformanceCollector()
benchmark_runner = BenchmarkRunner()

# Run benchmark with monitoring
with collector.monitor_operation("full_pipeline"):
    results = benchmark_runner.run_comparative_benchmark(
        test_data=["test_video.mp4"],
        models=["edgetam-base", "sam2-small"]
    )

# Generate report
report = collector.generate_report()
print(f"Total processing time: {report.total_time:.2f}s")
print(f"Peak memory usage: {report.peak_memory:.1f}GB")
```

## Error Handling Examples

### Automatic Fallback

```python
from sowlv2.models.model_factory import SegmentationModelFactory

try:
    # Try to create EdgeTAM model
    model = SegmentationModelFactory.create_model(
        model_type="edgetam",
        model_name="facebook/edgetam-base",
        device="cuda",
        enable_fallback=True  # Enable automatic fallback
    )
except Exception as e:
    print(f"Model creation failed: {e}")
    # Fallback will be handled automatically
```

### Manual Error Recovery

```python
from sowlv2.utils.error_recovery import ErrorRecoveryManager

recovery_manager = ErrorRecoveryManager()

def process_with_recovery(frames, prompts):
    try:
        return process_frames(frames, prompts)
    except MemoryError as e:
        # Handle memory overflow
        new_config = recovery_manager.handle_memory_overflow(current_config)
        return process_frames(frames, prompts, config=new_config)
    except Exception as e:
        # Generic retry logic
        return recovery_manager.implement_retry_logic(
            operation=lambda: process_frames(frames, prompts),
            max_retries=3
        )
```

For more examples and detailed usage patterns, see the [User Documentation](edgetam_integration.md) and [Performance Tuning Guide](performance_tuning.md).