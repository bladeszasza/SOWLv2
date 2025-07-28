# Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when using SOWLv2 with EdgeTAM integration and optimization features. Issues are organized by category with step-by-step solutions.

## Quick Diagnostic Commands

### System Information
```bash
# Check GPU information
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else 'No GPU')"

# Check SOWLv2 installation
python -m sowlv2.cli --version

# Test basic functionality
python -m sowlv2.cli --test-installation
```

### Performance Diagnostics
```bash
# Run system benchmark
python -m sowlv2.cli --benchmark-system --output system_benchmark.json

# Test model loading
python -m sowlv2.cli --test-models --output model_test.json
```

## Installation Issues

### Issue: EdgeTAM Model Download Fails

**Symptoms:**
```
Error: Failed to download EdgeTAM model
ConnectionError: Unable to connect to model repository
```

**Solutions:**

1. **Check Internet Connection:**
```bash
# Test connectivity
curl -I https://huggingface.co/facebook/edgetam-base

# Use proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

2. **Manual Model Download:**
```bash
# Download manually
git lfs install
git clone https://huggingface.co/facebook/edgetam-base ~/.cache/huggingface/transformers/

# Set local path
python -m sowlv2.cli --edgetam --edgetam-model ~/.cache/huggingface/transformers/edgetam-base
```

3. **Use Offline Mode:**
```yaml
segmentation:
  model_type: "edgetam"
  offline_mode: true
  model_path: "/path/to/local/model"
```

### Issue: CUDA Out of Memory During Installation

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB
```

**Solutions:**

1. **Clear GPU Memory:**
```bash
# Kill GPU processes
nvidia-smi --gpu-reset

# Clear PyTorch cache
python -c "import torch; torch.cuda.empty_cache()"
```

2. **Install with Memory Limit:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python -m pip install sowlv2
```

## Model Loading Issues

### Issue: EdgeTAM Model Fails to Load

**Symptoms:**
```
[ERROR] EdgeTAM model failed to initialize
[WARNING] Falling back to SAM2
```

**Solutions:**

1. **Check Model Compatibility:**
```python
# Test model loading
from sowlv2.models.edgetam_wrapper import EdgeTAMWrapper
try:
    model = EdgeTAMWrapper("facebook/edgetam-base", "cuda")
    print("EdgeTAM loaded successfully")
except Exception as e:
    print(f"EdgeTAM loading failed: {e}")
```

2. **Use Different Model Variant:**
```bash
# Try smaller model
python -m sowlv2.cli --edgetam --edgetam-model facebook/edgetam-small

# Try CPU version
python -m sowlv2.cli --edgetam --device cpu
```

3. **Check Dependencies:**
```bash
pip install transformers>=4.30.0 torch>=2.0.0 torchvision>=0.15.0
```

### Issue: SAM2 Fallback Not Working

**Symptoms:**
```
[ERROR] Both EdgeTAM and SAM2 failed to load
RuntimeError: No segmentation model available
```

**Solutions:**

1. **Verify SAM2 Installation:**
```bash
# Test SAM2 loading
python -c "from sowlv2.models.sam2_wrapper import SAM2Wrapper; print('SAM2 available')"
```

2. **Reinstall Models:**
```bash
pip uninstall sowlv2
pip install sowlv2 --no-cache-dir
```

3. **Manual Fallback Configuration:**
```yaml
segmentation:
  model_type: "sam2"
  fallback_enabled: true
  fallback_model: "facebook/sam2-hiera-tiny"
```

## Memory Issues

### Issue: CUDA Out of Memory During Processing

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB (GPU 0; X.XXGiB total capacity)
```

**Solutions:**

1. **Enable Memory Management:**
```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --memory-limit 6.0 \
  --optimization-level 1 \
  --streaming-chunk-size 50
```

2. **Reduce Batch Size:**
```yaml
optimization:
  batch_processing:
    max_batch_size: 4
    adaptive_batch_size: true
    memory_safety_margin: 0.2
```

3. **Enable Streaming Processing:**
```yaml
optimization:
  streaming_processing: true
  streaming_chunk_size: 25
  progressive_loading: true
  memory_monitoring: true
```

### Issue: System Freezing During Large Video Processing

**Symptoms:**
- System becomes unresponsive
- High memory usage (>90% RAM)
- Swap file usage increases dramatically

**Solutions:**

1. **Enable System Resource Limits:**
```bash
# Set memory limit
ulimit -v 16777216  # 16GB virtual memory limit

# Use systemd-run for resource control
systemd-run --scope -p MemoryMax=8G python -m sowlv2.cli --input large_video.mp4
```

2. **Configure Streaming Mode:**
```yaml
optimization:
  streaming_processing: true
  streaming_chunk_size: 30
  memory_limit_gb: 8.0
  enable_cpu_fallback: true
```

## Performance Issues

### Issue: Very Slow Processing Speed

**Symptoms:**
- Processing speed < 0.5 FPS
- High CPU usage, low GPU usage
- Long model loading times

**Solutions:**

1. **Check GPU Utilization:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check if GPU is being used
python -c "import torch; print(f'Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')"
```

2. **Optimize Configuration:**
```yaml
optimization:
  level: 3
  enable_mixed_precision: true
  parallel_processing: true
  
segmentation:
  model_type: "edgetam"
  model_name: "facebook/edgetam-small"
```

3. **Enable Prefetching:**
```yaml
optimization:
  prefetch_frames: 10
  async_processing: true
  parallel_data_loading: true
```

### Issue: High Memory Usage with Low Performance

**Symptoms:**
- Memory usage > 80% but slow processing
- Frequent garbage collection
- Memory fragmentation warnings

**Solutions:**

1. **Optimize Memory Usage:**
```yaml
optimization:
  memory_optimization:
    enable_garbage_collection: true
    memory_defragmentation: true
    intermediate_cleanup: true
```

2. **Adjust Model Caching:**
```yaml
optimization:
  model_caching:
    cache_size_gb: 2.0  # Reduce cache size
    aggressive_eviction: true
    preload_models: []  # Disable preloading
```

## V-JEPA2 Issues

### Issue: V-JEPA2 Model Not Loading

**Symptoms:**
```
[ERROR] V-JEPA2 model failed to load
[WARNING] Falling back to uniform frame sampling
```

**Solutions:**

1. **Check V-JEPA2 Installation:**
```bash
# Verify installation
python -c "from sowlv2.optimizations.vjepa2_optimization import VJepa2VideoOptimizer; print('V-JEPA2 available')"
```

2. **Download V-JEPA2 Model Manually:**
```bash
# Download model
huggingface-cli download facebook/vjepa2-base --local-dir ~/.cache/vjepa2/
```

3. **Use Alternative Frame Selection:**
```yaml
vjepa2:
  enable: false
  fallback_method: "uniform_sampling"
  uniform_interval: 30  # Every 30 frames
```

### Issue: Poor Frame Selection Quality

**Symptoms:**
- Important scenes are skipped
- Too many similar frames selected
- Processing time not reduced

**Solutions:**

1. **Adjust Importance Threshold:**
```yaml
vjepa2:
  importance_threshold: 0.6  # Lower = more frames
  temporal_diversity: true
  motion_aware_scoring: true
```

2. **Tune Content Analysis:**
```yaml
vjepa2:
  content_analysis:
    fast_motion_threshold: 0.7
    static_content_threshold: 0.4
    similarity_threshold: 0.8
```

## CLI and Configuration Issues

### Issue: Configuration File Not Found

**Symptoms:**
```
[ERROR] Configuration file not found: config.yaml
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**

1. **Create Default Configuration:**
```bash
# Generate default config
python -m sowlv2.cli --generate-config config.yaml

# Use built-in config
python -m sowlv2.cli --input video.mp4 --prompts "person" --use-default-config
```

2. **Specify Full Path:**
```bash
python -m sowlv2.cli --config /full/path/to/config.yaml
```

### Issue: Invalid CLI Arguments

**Symptoms:**
```
error: unrecognized arguments: --edgetam-model
usage: cli.py [-h] --input INPUT --prompts PROMPTS
```

**Solutions:**

1. **Check SOWLv2 Version:**
```bash
python -m sowlv2.cli --version
pip install --upgrade sowlv2
```

2. **Use Correct Argument Format:**
```bash
# Correct format
python -m sowlv2.cli --input video.mp4 --prompts "person,car" --edgetam

# Check available arguments
python -m sowlv2.cli --help
```

## Error Recovery Issues

### Issue: Processing Stops on Single Frame Error

**Symptoms:**
```
[ERROR] Frame 150 processing failed
ProcessingError: Segmentation failed for frame
[INFO] Processing stopped
```

**Solutions:**

1. **Enable Error Recovery:**
```yaml
error_handling:
  continue_on_error: true
  max_consecutive_errors: 5
  error_recovery_strategy: "skip_frame"
```

2. **Configure Retry Logic:**
```yaml
error_handling:
  retry_attempts: 3
  retry_delay: 1.0
  exponential_backoff: true
```

### Issue: No Fallback When EdgeTAM Fails

**Symptoms:**
```
[ERROR] EdgeTAM processing failed
[ERROR] No fallback model configured
```

**Solutions:**

1. **Enable Automatic Fallback:**
```yaml
segmentation:
  model_type: "edgetam"
  fallback_enabled: true
  fallback_model: "sam2"
  fallback_on_error: true
```

2. **Configure Fallback Chain:**
```yaml
segmentation:
  fallback_chain:
    - "edgetam-base"
    - "edgetam-small"
    - "sam2-tiny"
    - "cpu_fallback"
```

## Debugging and Logging

### Enable Debug Mode

```bash
# Enable detailed logging
export SOWLV2_LOG_LEVEL=DEBUG
export SOWLV2_DEBUG=true

python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --debug --log-file debug.log
```

### Performance Debugging

```bash
# Enable performance profiling
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --profile --profile-output profile.json \
  --memory-profile --memory-profile-output memory.html
```

### Error Context Collection

```yaml
debug:
  enable: true
  collect_system_info: true
  collect_model_info: true
  collect_performance_context: true
  save_error_frames: true
  error_report_path: "error_reports/"
```

## Frequently Asked Questions

### Q: Which model should I use for real-time processing?

**A:** Use EdgeTAM-small with optimization level 3:
```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --edgetam --edgetam-model facebook/edgetam-small \
  --optimization-level 3 --enable-mixed-precision
```

### Q: How do I process videos larger than my GPU memory?

**A:** Enable streaming processing:
```yaml
optimization:
  streaming_processing: true
  streaming_chunk_size: 50  # Adjust based on GPU memory
  memory_limit_gb: 6.0
```

### Q: Why is EdgeTAM slower than expected?

**A:** Check these common issues:
1. Mixed precision not enabled
2. Batch size too small
3. CPU bottleneck in data loading
4. Insufficient GPU memory causing swapping

### Q: How do I improve segmentation quality?

**A:** Use higher quality models and settings:
```yaml
segmentation:
  model_type: "sam2"
  model_name: "facebook/sam2-hiera-large"
  
optimization:
  level: 1
  enable_mixed_precision: false
```

### Q: Can I use SOWLv2 without GPU?

**A:** Yes, but performance will be significantly slower:
```bash
python -m sowlv2.cli --input video.mp4 --prompts "person" \
  --device cpu --optimization-level 1
```

### Q: How do I benchmark different configurations?

**A:** Use the built-in benchmarking:
```bash
python -m sowlv2.cli --input test_video.mp4 --prompts "person" \
  --benchmark --compare-models --benchmark-output results.html
```

## Getting Help

### Collect System Information

```bash
# Generate system report
python -m sowlv2.cli --system-info --output system_info.json

# Test installation
python -m sowlv2.cli --test-installation --verbose
```

### Report Issues

When reporting issues, include:
1. System information output
2. Complete error messages
3. Configuration file used
4. Steps to reproduce
5. Expected vs actual behavior

### Community Resources

- GitHub Issues: [SOWLv2 Issues](https://github.com/your-repo/sowlv2/issues)
- Documentation: [SOWLv2 Docs](https://sowlv2.readthedocs.io)
- Examples: [SOWLv2 Examples](https://github.com/your-repo/sowlv2/tree/main/examples)

For additional help, consult the [Performance Tuning Guide](performance_tuning.md) and [Optimization Configuration Guide](optimization_configuration.md).