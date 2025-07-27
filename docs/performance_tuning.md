# Performance Tuning Guide

## Overview

This guide provides detailed strategies for optimizing SOWLv2 performance across different hardware configurations, use cases, and quality requirements. Follow these recommendations to achieve optimal processing speed and resource utilization.

## Performance Analysis

### Benchmarking Your System

Before tuning, establish baseline performance:

```bash
# Run comprehensive benchmark
python -m sowlv2.cli --input test_video.mp4 --prompts "person,car" \
  --benchmark --benchmark-output benchmark_results.json

# Compare models
python -m sowlv2.cli --input test_video.mp4 --prompts "person" \
  --benchmark --compare-models --benchmark-output model_comparison.html
```

### Understanding Performance Metrics

Key metrics to monitor:
- **Processing Time**: Total time per frame/video
- **Memory Usage**: Peak GPU/CPU memory consumption
- **GPU Utilization**: Percentage of GPU compute used
- **Throughput**: Frames processed per second
- **Model Loading Time**: Time to initialize models

## Hardware-Specific Tuning

### High-End GPU Systems (RTX 4090, A100, H100)

**Recommended Configuration:**
```yaml
optimization:
  level: 3
  memory_limit_gb: 20.0
  enable_mixed_precision: true
  
batch_processing:
  max_batch_size: 64
  adaptive_batch_size: true
  
streaming:
  chunk_size: 300
  parallel_chunks: 2
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-base"
```

**Expected Performance:**
- 15-25 FPS on 1080p video
- 8-15 FPS on 4K video
- Memory usage: 12-18GB

### Mid-Range GPU Systems (RTX 3070, RTX 4060, RTX 3080)

**Recommended Configuration:**
```yaml
optimization:
  level: 2
  memory_limit_gb: 8.0
  enable_mixed_precision: true
  
batch_processing:
  max_batch_size: 32
  adaptive_batch_size: true
  
streaming:
  chunk_size: 150
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-base"
```

**Expected Performance:**
- 8-15 FPS on 1080p video
- 4-8 FPS on 4K video
- Memory usage: 6-8GB

### Entry-Level GPU Systems (GTX 1660, RTX 3050, RTX 4050)

**Recommended Configuration:**
```yaml
optimization:
  level: 1
  memory_limit_gb: 4.0
  enable_mixed_precision: false
  
batch_processing:
  max_batch_size: 8
  adaptive_batch_size: true
  
streaming:
  chunk_size: 75
  enable_cpu_fallback: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
```

**Expected Performance:**
- 3-8 FPS on 1080p video
- 1-3 FPS on 4K video
- Memory usage: 3-4GB

### CPU-Only Systems

**Recommended Configuration:**
```yaml
optimization:
  level: 1
  device: "cpu"
  num_workers: 8
  memory_limit_gb: 16.0
  
batch_processing:
  max_batch_size: 4
  cpu_optimization: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
```

**Expected Performance:**
- 0.5-2 FPS on 1080p video
- Processing time: 5-20x slower than GPU

## Model Selection for Performance

### Speed vs Quality Trade-offs

| Model | Speed | Quality | Memory | Best Use Case |
|-------|-------|---------|--------|---------------|
| EdgeTAM-small | 4.0x | 90% | Low | Real-time, mobile |
| EdgeTAM-base | 2.5x | 95% | Medium | General purpose |
| EdgeTAM-large | 1.8x | 98% | High | Quality-focused |
| SAM2-tiny | 1.2x | 92% | Low | Compatibility |
| SAM2-small | 1.0x | 96% | Medium | Baseline |
| SAM2-base | 0.8x | 98% | High | High quality |
| SAM2-large | 0.6x | 100% | Very High | Maximum quality |

### Model Selection Strategy

```python
# Performance-focused selection
def select_model_for_performance(gpu_memory_gb, target_fps):
    if gpu_memory_gb < 4:
        return "edgetam-small"
    elif gpu_memory_gb < 8:
        return "edgetam-base" if target_fps > 10 else "sam2-small"
    else:
        return "edgetam-base" if target_fps > 15 else "sam2-base"
```

## Memory Optimization Strategies

### 1. Streaming Processing

Enable for videos > 500MB or when memory usage > 80%:

```yaml
optimization:
  streaming_processing: true
  streaming_chunk_size: 100  # Adjust based on available memory
  chunk_overlap: 5
  progressive_loading: true
```

**Chunk Size Guidelines:**
- 4GB GPU: 50-75 frames
- 8GB GPU: 100-150 frames
- 16GB+ GPU: 200-300 frames

### 2. Model Caching Optimization

```yaml
optimization:
  model_caching:
    cache_size_gb: 4.0  # 50% of GPU memory
    eviction_policy: "lru"
    preload_priority: ["owl", "edgetam", "vjepa2"]
    cache_warmup: true
```

### 3. Memory Monitoring and Adjustment

```yaml
optimization:
  memory_monitoring:
    enable: true
    check_interval: 10  # seconds
    auto_adjustment: true
    emergency_cleanup: true
    memory_threshold: 0.85
```

## Batch Processing Optimization

### Adaptive Batch Sizing

```yaml
batch_processing:
  adaptive_batch_size: true
  initial_batch_size: 16
  max_batch_size: 64
  min_batch_size: 1
  adjustment_factor: 0.8  # Reduce by 20% on OOM
  memory_safety_margin: 0.1  # Keep 10% memory free
```

### Batch Size Guidelines

| GPU Memory | Recommended Batch Size | Max Batch Size |
|------------|----------------------|----------------|
| 4GB | 4-8 | 16 |
| 8GB | 8-16 | 32 |
| 12GB | 16-24 | 48 |
| 16GB+ | 24-32 | 64 |

## V-JEPA2 Optimization

### Frame Selection Tuning

```yaml
vjepa2:
  optimization:
    importance_threshold: 0.7  # Higher = fewer frames
    max_frames: 100  # Limit total frames processed
    temporal_diversity: true
    motion_aware_scoring: true
    adaptive_frame_spacing: true
    
  content_analysis:
    enable: true
    fast_motion_threshold: 0.8
    static_content_threshold: 0.3
    similarity_threshold: 0.85
```

### Content-Aware Optimization

```yaml
vjepa2:
  content_profiles:
    static_video:  # Security cameras, presentations
      importance_threshold: 0.8
      max_frames: 50
      frame_spacing: 30
      
    dynamic_video:  # Sports, action scenes
      importance_threshold: 0.6
      max_frames: 150
      frame_spacing: 5
      
    mixed_content:  # General videos
      importance_threshold: 0.7
      max_frames: 100
      adaptive_spacing: true
```

## Parallel Processing Optimization

### Multi-GPU Configuration

```yaml
optimization:
  multi_gpu:
    enable: true
    gpu_ids: [0, 1]
    load_balancing: "dynamic"
    model_replication: true
    
  parallel_processing:
    parallel_prompts: true
    parallel_frames: true
    synchronization_strategy: "async"
```

### CPU Parallelization

```yaml
optimization:
  cpu_parallel:
    num_workers: 8  # Number of CPU cores
    thread_pool_size: 16
    async_io: true
    prefetch_frames: 10
```

## Real-Time Processing Optimization

### Low-Latency Configuration

```yaml
optimization:
  real_time:
    enable: true
    max_latency_ms: 100
    frame_dropping: true
    priority_scheduling: true
    
  streaming:
    chunk_size: 1  # Process frame by frame
    buffer_size: 3
    prefetch_enabled: false
    
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
  optimization_level: 3
```

### Frame Rate Optimization

```python
# Target frame rate configuration
target_fps_configs = {
    30: {  # Real-time processing
        "model": "edgetam-small",
        "batch_size": 1,
        "optimization_level": 3,
        "mixed_precision": True
    },
    15: {  # Near real-time
        "model": "edgetam-base",
        "batch_size": 2,
        "optimization_level": 2,
        "mixed_precision": True
    },
    5: {   # High quality
        "model": "sam2-base",
        "batch_size": 4,
        "optimization_level": 1,
        "mixed_precision": False
    }
}
```

## Quality vs Performance Tuning

### Quality-Focused Configuration

```yaml
optimization:
  level: 1
  enable_mixed_precision: false
  quality_preservation: true
  
segmentation:
  model_type: "sam2"
  model_name: "facebook/sam2-hiera-large"
  
vjepa2:
  importance_threshold: 0.5  # Process more frames
  max_frames: 200
```

### Speed-Focused Configuration

```yaml
optimization:
  level: 3
  enable_mixed_precision: true
  aggressive_optimization: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
  
vjepa2:
  importance_threshold: 0.8  # Process fewer frames
  max_frames: 50
```

### Balanced Configuration

```yaml
optimization:
  level: 2
  enable_mixed_precision: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-base"
  
vjepa2:
  importance_threshold: 0.7
  max_frames: 100
  adaptive_parameters: true
```

## Performance Monitoring and Tuning

### Real-Time Monitoring

```yaml
monitoring:
  enable: true
  real_time_display: true
  metrics_interval: 5  # seconds
  alert_thresholds:
    memory_usage: 0.9
    processing_time: 2.0  # seconds per frame
    gpu_utilization: 0.95
```

### Performance Profiling

```bash
# Profile specific operations
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --profile --profile-output profile_results.json

# Memory profiling
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --memory-profile --memory-profile-output memory_profile.html
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### 1. Slow Processing Speed

**Symptoms:**
- Low FPS (< 1 FPS on modern GPU)
- High processing time per frame

**Solutions:**
```yaml
# Try these optimizations in order
optimization:
  level: 3
  enable_mixed_precision: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
  
vjepa2:
  importance_threshold: 0.8
```

#### 2. High Memory Usage

**Symptoms:**
- Out of memory errors
- System freezing
- Slow performance due to memory swapping

**Solutions:**
```yaml
optimization:
  memory_limit_gb: 6.0  # Set below total GPU memory
  streaming_processing: true
  streaming_chunk_size: 50
  
batch_processing:
  max_batch_size: 8
  adaptive_batch_size: true
```

#### 3. Low GPU Utilization

**Symptoms:**
- GPU usage < 70%
- CPU bottleneck
- Slow data loading

**Solutions:**
```yaml
optimization:
  parallel_processing: true
  prefetch_frames: 20
  async_processing: true
  
batch_processing:
  max_batch_size: 32  # Increase batch size
```

### Performance Debugging

```bash
# Enable detailed logging
export SOWLV2_LOG_LEVEL=DEBUG
export SOWLV2_PROFILE_MEMORY=true
export SOWLV2_PROFILE_GPU=true

python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --debug --performance-log performance.log
```

## Best Practices Summary

### 1. Hardware Assessment
- Benchmark your system first
- Identify memory and compute limitations
- Choose appropriate model and settings

### 2. Model Selection
- Use EdgeTAM for speed-critical applications
- Use SAM2 for quality-critical applications
- Consider model size vs available memory

### 3. Memory Management
- Enable streaming for large videos
- Use adaptive batch sizing
- Monitor memory usage continuously

### 4. Optimization Strategy
- Start with level 2 optimization
- Enable mixed precision on modern GPUs
- Use V-JEPA2 for intelligent frame selection

### 5. Monitoring and Tuning
- Monitor performance metrics
- Adjust settings based on actual usage
- Profile regularly to identify bottlenecks

For specific issues, consult the [Troubleshooting Guide](troubleshooting.md).