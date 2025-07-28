# Optimization Configuration Guide

## Overview

SOWLv2 provides comprehensive optimization capabilities to maximize performance across different hardware configurations and use cases. This guide covers all optimization settings and how to configure them effectively.

## Configuration Methods

### 1. CLI Arguments

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --optimization-level 2 \
  --memory-limit 8.0 \
  --enable-mixed-precision \
  --streaming-chunk-size 150
```

### 2. YAML Configuration

```yaml
optimization:
  level: 2
  memory_limit_gb: 8.0
  enable_mixed_precision: true
  streaming_chunk_size: 150
  gpu_batching: true
  model_caching: true
```

### 3. Environment Variables

```bash
export SOWLV2_OPTIMIZATION_LEVEL=2
export SOWLV2_MEMORY_LIMIT=8.0
export SOWLV2_MIXED_PRECISION=true
```

## Optimization Levels

### Level 0: Disabled
- No optimizations applied
- Maximum compatibility
- Slowest performance

```yaml
optimization:
  level: 0
```

### Level 1: Basic (Default)
- Intelligent batching
- Basic memory management
- Model caching enabled

```yaml
optimization:
  level: 1
  batch_optimization: true
  model_caching: true
  memory_monitoring: true
```

### Level 2: Aggressive
- Mixed precision processing
- Advanced memory optimization
- Streaming processing for large videos

```yaml
optimization:
  level: 2
  enable_mixed_precision: true
  advanced_memory_management: true
  streaming_processing: true
  gpu_memory_optimization: true
```

### Level 3: Maximum
- All optimizations enabled
- Parallel processing
- Advanced GPU utilization

```yaml
optimization:
  level: 3
  enable_mixed_precision: true
  parallel_processing: true
  advanced_gpu_batching: true
  streaming_processing: true
  model_preloading: true
```

## Memory Management

### Memory Limit Configuration

Set memory limits to prevent system overload:

```yaml
optimization:
  memory_limit_gb: 8.0  # Total GPU memory limit
  memory_threshold: 0.8  # Trigger optimization at 80% usage
  enable_memory_monitoring: true
```

### Streaming Processing

For large videos that exceed memory capacity:

```yaml
optimization:
  streaming_processing: true
  streaming_chunk_size: 100  # Frames per chunk
  chunk_overlap: 5  # Frames overlap between chunks
  progressive_loading: true
```

### Memory Optimization Strategies

```yaml
optimization:
  memory_strategies:
    - "gradient_checkpointing"
    - "model_offloading"
    - "intermediate_cleanup"
    - "garbage_collection"
```

## GPU Optimization

### Batch Processing

Configure intelligent batching:

```yaml
optimization:
  batch_processing:
    enable: true
    adaptive_batch_size: true
    max_batch_size: 32
    min_batch_size: 1
    memory_based_adjustment: true
```

### Mixed Precision

Enable mixed precision for faster processing:

```yaml
optimization:
  mixed_precision:
    enable: true
    autocast: true
    grad_scaler: true
    loss_scaling: "dynamic"
```

### GPU Memory Management

```yaml
optimization:
  gpu_management:
    memory_fraction: 0.9  # Use 90% of GPU memory
    allow_growth: true
    memory_pool_size: "auto"
    enable_memory_defragmentation: true
```

## Model Optimization

### Model Caching

Configure intelligent model caching:

```yaml
optimization:
  model_caching:
    enable: true
    cache_size_gb: 4.0
    eviction_policy: "lru"  # Least Recently Used
    preload_models: ["owl", "sam2"]
    cache_persistence: true
```

### Model Loading

```yaml
optimization:
  model_loading:
    parallel_loading: true
    lazy_loading: true
    model_quantization: false
    model_pruning: false
```

## V-JEPA2 Optimization

### Frame Selection

```yaml
vjepa2:
  optimization:
    importance_threshold: 0.7
    max_frames: 100
    temporal_diversity: true
    motion_aware_scoring: true
    adaptive_frame_spacing: true
```

### Content Analysis

```yaml
vjepa2:
  content_analysis:
    enable: true
    content_type_detection: true
    adaptive_parameters: true
    similarity_threshold: 0.8
```

## Parallel Processing

### Multi-Threading

```yaml
optimization:
  parallel_processing:
    enable: true
    num_workers: 4  # CPU threads
    gpu_parallel: true
    async_processing: true
```

### Batch Parallelization

```yaml
optimization:
  batch_parallel:
    enable: true
    parallel_prompts: true
    parallel_frames: true
    synchronization_points: ["detection", "segmentation"]
```

## Hardware-Specific Configurations

### High-End GPU (RTX 4090, A100)

```yaml
optimization:
  level: 3
  memory_limit_gb: 20.0
  enable_mixed_precision: true
  batch_processing:
    max_batch_size: 64
  streaming_chunk_size: 200
```

### Mid-Range GPU (RTX 3070, RTX 4060)

```yaml
optimization:
  level: 2
  memory_limit_gb: 8.0
  enable_mixed_precision: true
  batch_processing:
    max_batch_size: 16
  streaming_chunk_size: 100
```

### Low-End GPU (GTX 1660, RTX 3050)

```yaml
optimization:
  level: 1
  memory_limit_gb: 4.0
  enable_mixed_precision: false
  batch_processing:
    max_batch_size: 4
  streaming_chunk_size: 50
  fallback_to_cpu: true
```

### CPU-Only Processing

```yaml
optimization:
  level: 1
  device: "cpu"
  cpu_optimization: true
  num_workers: 8
  memory_limit_gb: 16.0
```

## Performance Monitoring

### Enable Monitoring

```yaml
monitoring:
  enable: true
  real_time_metrics: true
  performance_logging: true
  resource_tracking: true
```

### Metrics Collection

```yaml
monitoring:
  metrics:
    - "processing_time"
    - "memory_usage"
    - "gpu_utilization"
    - "throughput"
    - "model_loading_time"
```

## Benchmarking Configuration

### Basic Benchmarking

```yaml
benchmark:
  enable: true
  output_format: "json"
  detailed_metrics: true
  compare_models: false
```

### Comprehensive Benchmarking

```yaml
benchmark:
  enable: true
  output_format: "html"
  detailed_metrics: true
  compare_models: true
  test_configurations:
    - optimization_level: 1
    - optimization_level: 2
    - optimization_level: 3
  performance_history: true
```

## Use Case Configurations

### Real-Time Processing

```yaml
optimization:
  level: 3
  enable_mixed_precision: true
  streaming_processing: true
  streaming_chunk_size: 30  # 1 second at 30fps
  model_preloading: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
```

### High-Quality Processing

```yaml
optimization:
  level: 2
  enable_mixed_precision: false
  batch_processing:
    max_batch_size: 8
    
segmentation:
  model_type: "sam2"
  model_name: "facebook/sam2-hiera-large"
```

### Memory-Constrained Processing

```yaml
optimization:
  level: 1
  memory_limit_gb: 4.0
  streaming_processing: true
  streaming_chunk_size: 25
  fallback_to_cpu: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
```

### Batch Video Processing

```yaml
optimization:
  level: 2
  batch_processing:
    enable: true
    parallel_videos: 2
    shared_model_cache: true
  model_caching:
    cache_size_gb: 8.0
    preload_models: ["owl", "edgetam", "vjepa2"]
```

## Advanced Configuration

### Custom Optimization Profiles

```yaml
optimization_profiles:
  speed_focused:
    level: 3
    enable_mixed_precision: true
    model_type: "edgetam"
    model_name: "facebook/edgetam-small"
    
  quality_focused:
    level: 2
    enable_mixed_precision: false
    model_type: "sam2"
    model_name: "facebook/sam2-hiera-large"
    
  balanced:
    level: 2
    enable_mixed_precision: true
    model_type: "edgetam"
    model_name: "facebook/edgetam-base"
```

### Dynamic Configuration

```yaml
optimization:
  dynamic_adjustment: true
  auto_optimization: true
  performance_targets:
    min_fps: 10
    max_memory_usage: 0.8
    target_quality: 0.9
```

## Troubleshooting Optimization Issues

### Common Problems

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable streaming processing
   - Lower optimization level

2. **Slow Processing**
   - Increase optimization level
   - Enable mixed precision
   - Use EdgeTAM instead of SAM2

3. **Quality Issues**
   - Disable mixed precision
   - Use higher quality models
   - Adjust V-JEPA2 thresholds

### Debug Configuration

```yaml
debug:
  enable: true
  log_level: "DEBUG"
  performance_profiling: true
  memory_tracking: true
  optimization_logging: true
```

For more troubleshooting help, see the [Troubleshooting Guide](troubleshooting.md).