# EdgeTAM Integration Guide

## Overview

EdgeTAM (Edge-optimized Tracking Any Model) is a faster alternative to SAM2 for segmentation tasks in SOWLv2. This guide covers how to use EdgeTAM for improved processing speed while maintaining good segmentation quality.

## Quick Start

### Basic Usage

To use EdgeTAM instead of SAM2, simply add the `--edgetam` flag:

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person,car" --edgetam
```

### Specifying EdgeTAM Model

You can specify a specific EdgeTAM model variant:

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person,car" --edgetam --edgetam-model facebook/edgetam-base
```

Available EdgeTAM models:
- `facebook/edgetam-base` (default) - Balanced speed and accuracy
- `facebook/edgetam-small` - Fastest processing, lower accuracy
- `facebook/edgetam-large` - Higher accuracy, slower processing

## Configuration Options

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--edgetam` | Enable EdgeTAM segmentation | False |
| `--edgetam-model` | Specify EdgeTAM model variant | facebook/edgetam-base |
| `--edgetam-optimization-level` | Optimization level (1-3) | 1 |

### YAML Configuration

```yaml
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-base"
  optimization_level: 2
  enable_video_tracking: true
  memory_efficient_mode: false
```

## Performance Comparison

### Speed vs Accuracy Trade-offs

| Model | Relative Speed | Relative Accuracy | Best Use Case |
|-------|----------------|-------------------|---------------|
| SAM2 | 1.0x | 100% | High accuracy requirements |
| EdgeTAM-base | 2.5x | 95% | Balanced performance |
| EdgeTAM-small | 4.0x | 90% | Real-time processing |
| EdgeTAM-large | 1.8x | 98% | Quality-focused fast processing |

### Benchmarking

To compare EdgeTAM and SAM2 performance:

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" --benchmark --compare-models
```

This will output detailed performance metrics for both models.

## Video Processing Features

### Single-Frame Mode

For image processing or frame-by-frame video analysis:

```bash
python -m sowlv2.cli --input image.jpg --prompts "object" --edgetam
```

### Video Tracking Mode

EdgeTAM supports efficient video tracking:

```yaml
segmentation:
  model_type: "edgetam"
  enable_video_tracking: true
  tracking_config:
    temporal_consistency: true
    object_persistence: true
```

## Optimization Levels

EdgeTAM supports three optimization levels:

### Level 1 (Default)
- Basic optimizations enabled
- Good balance of speed and memory usage
- Suitable for most use cases

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" --edgetam --edgetam-optimization-level 1
```

### Level 2 (Aggressive)
- Advanced memory optimizations
- Mixed precision processing
- Higher speed, slightly more memory usage

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" --edgetam --edgetam-optimization-level 2
```

### Level 3 (Maximum)
- All optimizations enabled
- Streaming processing for large videos
- Maximum speed, requires more GPU memory

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" --edgetam --edgetam-optimization-level 3
```

## Memory Management

### Automatic Memory Optimization

EdgeTAM automatically manages memory usage:

```yaml
optimization:
  memory_limit_gb: 8.0
  enable_streaming: true
  streaming_chunk_size: 100
```

### Memory-Constrained Environments

For systems with limited GPU memory:

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" --edgetam --memory-limit 4.0 --optimization-level 1
```

## Error Handling and Fallbacks

### Automatic Fallback

If EdgeTAM fails to load, SOWLv2 automatically falls back to SAM2:

```
[WARNING] EdgeTAM model failed to load: CUDA out of memory
[INFO] Falling back to SAM2 for segmentation
```

### Manual Fallback Configuration

```yaml
segmentation:
  model_type: "edgetam"
  fallback_model: "sam2"
  fallback_on_error: true
```

## Integration with V-JEPA2

EdgeTAM works seamlessly with V-JEPA2 optimization:

```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" --edgetam --vjepa2 --vjepa2-importance-threshold 0.7
```

This combination provides:
- Intelligent frame selection via V-JEPA2
- Fast segmentation via EdgeTAM
- Optimal processing speed with maintained quality

## Best Practices

### 1. Model Selection
- Use `edgetam-small` for real-time applications
- Use `edgetam-base` for general-purpose processing
- Use `edgetam-large` when accuracy is critical but speed is still important

### 2. Memory Management
- Set appropriate memory limits for your hardware
- Enable streaming mode for large videos
- Use optimization level 2 for most scenarios

### 3. Quality vs Speed
- Combine with V-JEPA2 for intelligent frame selection
- Use benchmarking to find optimal settings for your use case
- Consider SAM2 fallback for critical accuracy requirements

### 4. Batch Processing
- Process similar videos together for better efficiency
- Use consistent optimization settings across batches
- Monitor memory usage during batch processing

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for common EdgeTAM issues and solutions.