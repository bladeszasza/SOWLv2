"""
Command Line Interface for SOWLv2.

This script provides a CLI to detect and segment objects in images,
folders of frames, or video files using a text prompt.
It leverages the optimized SOWLv2Pipeline by default for faster processing.
"""
import argparse
import os
import sys
import yaml
from sowlv2.data.config import PipelineBaseData, PipelineConfig, OptimizationConfig, BenchmarkConfig
from sowlv2.optimizations import OptimizedSOWLv2Pipeline, ParallelConfig, create_vjepa2_optimizer
from sowlv2.utils.frame_utils import VALID_EXTS, VALID_VIDEO_EXTS
from sowlv2.utils.pipeline_utils import CPU, CUDA
from sowlv2.utils.error_recovery import ModelFallbackManager

def migrate_legacy_config(config_dict):
    """Migrate legacy configuration keys to new format for backward compatibility."""
    migrations = {
        # Legacy key -> new key mappings
        'use_edgetam': 'edgetam',
        'edgetam_model': 'edgetam-model',
        'edgetam_optimization_level': 'edgetam-optimization-level',
        'optimization_level': 'optimization-level',
        'optimization_preset': 'optimization-preset',
        'memory_limit': 'memory-limit',
        'streaming_chunk_size': 'streaming-chunk-size',
        'enable_mixed_precision': 'enable-mixed-precision',
        'disable_gpu_batching': 'disable-gpu-batching',
        'enable_model_caching': 'enable-model-caching',
        'cache_size_limit': 'cache-size-limit',
        'enable_streaming_mode': 'enable-streaming-mode',
        'benchmark_output': 'benchmark-output',
        'compare_models': 'compare-models',
        'benchmark_iterations': 'benchmark-iterations',
        'collect_memory_stats': 'collect-memory-stats',
        'collect_gpu_stats': 'collect-gpu-stats',
        'benchmark_test_data': 'benchmark-test-data',
        'performance_profile': 'performance-profile',
        'export_metrics': 'export-metrics',
        'enable_vjepa2': 'enable-vjepa2',
        'vjepa2_frames_per_clip': 'vjepa2-frames-per-clip',
        'use_temporal_detection': 'use-temporal-detection',
        'temporal_detection_frames': 'temporal-detection-frames',
        'temporal_merge_threshold': 'temporal-merge-threshold',
        'max_workers': 'max-workers',
        'batch_size': 'batch-size',
        'owl_model': 'owl-model',
        'sam_model': 'sam-model'
    }

    migrated_config = {}
    migration_warnings = []

    for key, value in config_dict.items():
        if key in migrations:
            new_key = migrations[key]
            migrated_config[new_key] = value
            migration_warnings.append(f"Migrated legacy key '{key}' to '{new_key}'")
        else:
            migrated_config[key] = value

    if migration_warnings:
        print("Configuration migration warnings:")
        for warning in migration_warnings:
            print(f"  MIGRATION: {warning}")
        print("Consider updating your configuration file to use the new key names.")
        print()

    return migrated_config

def validate_configuration(args):
    """Validate configuration parameters and provide helpful error messages."""
    errors = []
    warnings = []

    # Validate optimization level
    if not (0 <= args.optimization_level <= 3):
        errors.append(f"optimization-level must be between 0 and 3, got {args.optimization_level}")

    # Validate EdgeTAM optimization level
    if not (0 <= args.edgetam_optimization_level <= 3):
        errors.append(f"edgetam-optimization-level must be between 0 and 3, got {args.edgetam_optimization_level}")

    # Validate memory limit
    if args.memory_limit is not None and args.memory_limit <= 0:
        errors.append(f"memory-limit must be positive, got {args.memory_limit}")

    # Validate streaming chunk size
    if args.streaming_chunk_size <= 0:
        errors.append(f"streaming-chunk-size must be positive, got {args.streaming_chunk_size}")

    # Validate batch size
    if args.batch_size <= 0:
        errors.append(f"batch-size must be positive, got {args.batch_size}")

    # Validate cache size limit
    if args.cache_size_limit <= 0:
        errors.append(f"cache-size-limit must be positive, got {args.cache_size_limit}")

    # Validate benchmark iterations
    if args.benchmark_iterations <= 0:
        errors.append(f"benchmark-iterations must be positive, got {args.benchmark_iterations}")

    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        errors.append(f"threshold must be between 0.0 and 1.0, got {args.threshold}")

    # Validate fps
    if args.fps <= 0:
        errors.append(f"fps must be positive, got {args.fps}")

    # Validate temporal detection frames
    if args.temporal_detection_frames <= 0:
        errors.append(f"temporal-detection-frames must be positive, got {args.temporal_detection_frames}")

    # Validate temporal merge threshold
    if not (0.0 <= args.temporal_merge_threshold <= 1.0):
        errors.append(f"temporal-merge-threshold must be between 0.0 and 1.0, got {args.temporal_merge_threshold}")

    # Validate V-JEPA2 frames per clip
    if args.vjepa2_frames_per_clip <= 0:
        errors.append(f"vjepa2-frames-per-clip must be positive, got {args.vjepa2_frames_per_clip}")

    # Check for conflicting options
    if args.disable_gpu_batching and args.batch_size > 1:
        warnings.append("GPU batching is disabled but batch-size > 1. Batch size will be ignored.")

    if args.edgetam and args.compare_models:
        warnings.append("EdgeTAM is selected but model comparison is enabled. Both models will be tested.")

    if args.enable_streaming_mode and args.streaming_chunk_size > 500:
        warnings.append("Large streaming chunk size may reduce memory benefits of streaming mode.")

    if args.enable_mixed_precision and args.device == "cpu":
        warnings.append("Mixed precision is enabled but device is CPU. Mixed precision will be ignored.")

    # Check file paths
    if args.input and not os.path.exists(args.input):
        errors.append(f"Input path does not exist: {args.input}")

    if args.benchmark_test_data and not os.path.exists(args.benchmark_test_data):
        errors.append(f"Benchmark test data path does not exist: {args.benchmark_test_data}")

    # Validate benchmark output format
    if args.benchmark_output:
        valid_extensions = ['.json', '.csv', '.html']
        ext = os.path.splitext(args.benchmark_output)[1].lower()
        if ext not in valid_extensions:
            warnings.append(f"Benchmark output extension '{ext}' may not be supported. "
                          f"Recommended: {valid_extensions}")

    return errors, warnings

def apply_optimization_preset(args):
    """Apply optimization preset configurations."""
    preset = args.optimization_preset

    if preset == "speed":
        # Prioritize speed
        args.optimization_level = max(args.optimization_level, 2)
        args.enable_mixed_precision = True
        args.streaming_chunk_size = min(args.streaming_chunk_size, 50)
        args.enable_model_caching = True
        if args.edgetam is None:
            args.edgetam = True  # Prefer EdgeTAM for speed

    elif preset == "quality":
        # Prioritize quality
        args.optimization_level = min(args.optimization_level, 1)
        args.enable_mixed_precision = False
        args.streaming_chunk_size = max(args.streaming_chunk_size, 200)
        args.edgetam_optimization_level = min(args.edgetam_optimization_level, 1)

    elif preset == "memory":
        # Minimize memory usage
        args.enable_streaming_mode = True
        args.streaming_chunk_size = min(args.streaming_chunk_size, 25)
        args.cache_size_limit = min(args.cache_size_limit, 2.0)
        args.enable_mixed_precision = True
        args.batch_size = min(args.batch_size, 2)

    elif preset == "balanced":
        # Default balanced settings - no changes needed
        pass

    return args

def print_benchmark_help():
    """Print detailed benchmarking help and examples."""
    help_text = """
Benchmarking and Performance Monitoring Help
============================================

Basic Benchmarking:
  --benchmark: Enable comprehensive performance monitoring
  --benchmark-output: Save results to file (JSON, CSV, or HTML)
  --benchmark-iterations: Run multiple iterations for accuracy

Model Comparison:
  --compare-models: Compare SAM2 vs EdgeTAM performance
  Automatically runs the same input with both models

Performance Monitoring:
  --collect-memory-stats: Monitor memory usage (default: enabled)
  --collect-gpu-stats: Monitor GPU utilization (default: enabled)
  --performance-profile: Detailed line-by-line profiling

Export Options:
  --export-metrics: Choose export format (json, csv, html, all)
  Supports multiple output formats for different use cases

Test Data:
  --benchmark-test-data: Use specific test dataset
  Can be directory of test files or JSON configuration

Examples:
  # Basic benchmarking
  python -m sowlv2.cli --prompt "car" --input video.mp4 --benchmark

  # Compare models with detailed output
  python -m sowlv2.cli --prompt "person" --input frames/ \\
    --compare-models --benchmark-output results.html

  # Comprehensive benchmarking with multiple iterations
  python -m sowlv2.cli --prompt "animal" --input test_video.mp4 \\
    --benchmark --benchmark-iterations 5 --performance-profile \\
    --export-metrics all --benchmark-output benchmark_results

  # Test dataset benchmarking
  python -m sowlv2.cli --benchmark --benchmark-test-data test_dataset/ \\
    --compare-models --benchmark-output comparison_report.json

Benchmark Output Includes:
  - Processing time per stage
  - Memory usage statistics
  - GPU utilization metrics
  - Throughput measurements
  - Model comparison results
  - Performance recommendations

Supported Output Formats:
  - JSON: Machine-readable results for analysis
  - CSV: Tabular data for spreadsheet analysis
  - HTML: Interactive reports with charts and graphs
"""
    print(help_text)

def print_optimization_help():
    """Print detailed optimization help and examples."""
    help_text = """
Optimization Options Help
=========================

Global Optimization Levels:
  - Level 0: No optimization, maximum quality
  - Level 1: Basic optimization, balanced performance (default)
  - Level 2: Aggressive optimization, prioritize speed
  - Level 3: Maximum optimization, fastest processing

Optimization Presets:
  - speed: Prioritize processing speed over quality
  - balanced: Balance speed and quality (default)
  - quality: Prioritize output quality over speed
  - memory: Minimize memory usage for resource-constrained systems

Memory Management:
  --memory-limit: Set GPU memory limit in GB
  --streaming-chunk-size: Process videos in chunks to save memory
  --enable-streaming-mode: Force streaming for all videos

Performance Options:
  --enable-mixed-precision: Use FP16 for faster inference
  --disable-gpu-batching: Disable batching (for debugging)
  --enable-model-caching: Cache models for faster switching
  --cache-size-limit: Limit model cache size

Examples:
  # Speed-optimized processing
  python -m sowlv2.cli --prompt "car" --input video.mp4 \\
    --optimization-preset speed --enable-mixed-precision

  # Memory-constrained processing
  python -m sowlv2.cli --prompt "person" --input large_video.mp4 \\
    --optimization-preset memory --memory-limit 4.0

  # Quality-focused processing
  python -m sowlv2.cli --prompt "animal" --input frames/ \\
    --optimization-preset quality --optimization-level 0

  # Custom optimization
  python -m sowlv2.cli --prompt "object" --input video.mp4 \\
    --optimization-level 2 --streaming-chunk-size 50 \\
    --enable-mixed-precision --cache-size-limit 6.0
"""
    print(help_text)

def print_edgetam_help():
    """Print detailed EdgeTAM help and examples."""
    help_text = """
EdgeTAM Integration Help
========================

EdgeTAM (Edge-optimized Tracking Any Model) provides faster segmentation with minimal quality loss.

Available Models:
  - facebook/edgetam-base: Balanced speed and quality (recommended)
  - facebook/edgetam-small: Fastest inference, lower quality
  - facebook/edgetam-large: Higher quality, slower than base

Optimization Levels:
  - Level 0: No optimization, maximum quality
  - Level 1: Basic optimization, balanced speed/quality (default)
  - Level 2: Aggressive optimization, prioritize speed
  - Level 3: Maximum optimization, fastest inference

Examples:
  # Basic EdgeTAM usage
  python -m sowlv2.cli --prompt "cat" --input video.mp4 --edgetam

  # Use specific EdgeTAM model with optimization
  python -m sowlv2.cli --prompt "dog" --input frames/ --edgetam \\
    --edgetam-model facebook/edgetam-large --edgetam-optimization-level 2

  # EdgeTAM with YAML configuration
  python -m sowlv2.cli --config edgetam_config.yaml

Configuration File Example (edgetam_config.yaml):
  prompt: "person"
  input: "video.mp4"
  edgetam: true
  edgetam-model: "facebook/edgetam-base"
  edgetam-optimization-level: 1

Performance Comparison:
  - EdgeTAM is typically 2-3x faster than SAM2
  - Quality difference is usually minimal for most use cases
  - Automatic fallback to SAM2 if EdgeTAM fails to load
  - Best for real-time processing and resource-constrained environments

Troubleshooting:
  - If EdgeTAM fails to load, check CUDA/PyTorch installation
  - Use --device cpu if GPU memory is insufficient
  - Lower optimization levels if quality is important
  - Check available models with validation warnings
"""
    print(help_text)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOWLv2: Detect and segment objects in images/frames/video with a text prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --edgetam-help for detailed EdgeTAM documentation and examples."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        nargs='+', # Allows one or more arguments for prompt
        help="Text prompt(s) for object detection (e.g. 'cat' or 'lizard' 'dog' 'a red bicycle')"
    )
    parser.add_argument(
        "--input", type=str, required=False,
        help="Path to input (image file, directory of frames, or video file)"
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Directory to save output masks and overlays"
    )
    parser.add_argument(
        "--owl-model", type=str, default="google/owlv2-base-patch16-ensemble",
        help="OWLv2 model (HuggingFace name)"
    )
    parser.add_argument(
        "--sam-model", type=str, default="facebook/sam2.1-hiera-small",
        help="SAM2 model (HuggingFace name)"
    )
    parser.add_argument(
        "--edgetam", action="store_true",
        help="Use EdgeTAM instead of SAM2 for faster segmentation. "
             "EdgeTAM provides significantly faster inference with minimal quality loss. "
             "Automatically falls back to SAM2 if EdgeTAM fails to load."
    )
    parser.add_argument(
        "--edgetam-model", type=str, default="facebook/edgetam-base",
        help="EdgeTAM model name (default: facebook/edgetam-base). "
             "Available models: facebook/edgetam-base, facebook/edgetam-large. "
             "Larger models provide better quality at the cost of speed."
    )
    parser.add_argument(
        "--edgetam-optimization-level", type=int, default=1, choices=[0, 1, 2, 3],
        help="EdgeTAM optimization level (0-3, default: 1). "
             "0: No optimization, maximum quality. "
             "1: Basic optimization, balanced speed/quality. "
             "2: Aggressive optimization, prioritize speed. "
             "3: Maximum optimization, fastest inference."
    )
    parser.add_argument(
        "--edgetam-help", action="store_true",
        help="Show detailed EdgeTAM help, examples, and configuration options"
    )
    parser.add_argument(
        "--optimization-help", action="store_true",
        help="Show detailed optimization help, presets, and configuration options"
    )

    # Benchmarking and performance monitoring options
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Enable comprehensive benchmarking and performance monitoring. "
             "Collects detailed timing, memory usage, and throughput metrics."
    )
    parser.add_argument(
        "--benchmark-output", type=str, default=None,
        help="Output file for benchmark results (supports .json, .csv, .html formats). "
             "If not specified, results are printed to console."
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Compare performance between SAM2 and EdgeTAM models. "
             "Runs the same input with both models and generates comparison report."
    )
    parser.add_argument(
        "--benchmark-iterations", type=int, default=1,
        help="Number of benchmark iterations to run for averaging results (default: 1). "
             "Higher values provide more accurate performance measurements."
    )
    parser.add_argument(
        "--collect-memory-stats", action="store_true", default=True,
        help="Collect detailed memory usage statistics during processing (default: enabled). "
             "Includes GPU memory, system memory, and model memory usage."
    )
    parser.add_argument(
        "--collect-gpu-stats", action="store_true", default=True,
        help="Collect GPU utilization and performance statistics (default: enabled). "
             "Requires NVIDIA GPU and nvidia-ml-py package."
    )
    parser.add_argument(
        "--benchmark-test-data", type=str, default=None,
        help="Path to test dataset for benchmarking. If not specified, uses provided input. "
             "Can be a directory of test images/videos or a JSON file with test configurations."
    )
    parser.add_argument(
        "--performance-profile", action="store_true",
        help="Enable detailed performance profiling with line-by-line timing. "
             "Useful for identifying specific bottlenecks in the pipeline."
    )
    parser.add_argument(
        "--export-metrics", type=str, choices=["json", "csv", "html", "all"], default="json",
        help="Format for exporting performance metrics (default: json). "
             "'all' exports in all supported formats."
    )
    parser.add_argument(
        "--benchmark-help", action="store_true",
        help="Show detailed benchmarking help, options, and examples"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, # Default from README
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Sampling rate (frames per second) for video"
    )
    parser.add_argument(
        "--device", type=str, default=CUDA,
        help="PyTorch device (cpu or cuda). Default uses GPU if available."
    )
    parser.add_argument(
        "--no-merged", dest="merged", action="store_false",
        help="Disables merged mode (enabled by default)."
    )
    parser.add_argument(
        "--no-binary", dest="binary", action="store_false",
        help="Disables binary processing (enabled by default)."
    )
    parser.add_argument(
        "--no-overlay", dest="overlay", action="store_false",
        help="Disables overlay functionality (enabled by default)."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional)"
    )
    # Optimization options
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Maximum number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for GPU processing (default: 4)"
    )
    parser.add_argument(
        "--disable-gpu-batching", action="store_true",
        help="Disable GPU batching optimization"
    )
    parser.add_argument(
        "--enable-vjepa2", action="store_true",
        help="Enable V-JEPA 2 video optimization (experimental)"
    )
    parser.add_argument(
        "--vjepa2-frames-per-clip", type=int, default=16,
        help="Number of frames per clip for V-JEPA 2 processing"
    )
    parser.add_argument(
        "--temporal-detection-frames", type=int, default=5,
        help="Number of temporally important frames to run detection on (default: 5)"
    )
    parser.add_argument(
        "--temporal-merge-threshold", type=float, default=0.7,
        help="IoU threshold for merging same objects across frames (default: 0.7)"
    )
    parser.add_argument(
        "--use-temporal-detection", action="store_true",
        help="Enable temporal detection across multiple frames (requires V-JEPA 2)"
    )

    # Advanced optimization options
    parser.add_argument(
        "--optimization-level", type=int, default=1, choices=[0, 1, 2, 3],
        help="Global optimization level (0-3, default: 1). "
             "0: No optimization, maximum quality. "
             "1: Basic optimization, balanced performance. "
             "2: Aggressive optimization, prioritize speed. "
             "3: Maximum optimization, fastest processing."
    )
    parser.add_argument(
        "--memory-limit", type=float, default=None,
        help="Memory limit in GB for GPU processing (default: auto-detect). "
             "Automatically adjusts batch sizes and enables streaming for large videos."
    )
    parser.add_argument(
        "--streaming-chunk-size", type=int, default=100,
        help="Chunk size for streaming video processing (default: 100 frames). "
             "Smaller values use less memory but may be slower."
    )
    parser.add_argument(
        "--enable-mixed-precision", action="store_true",
        help="Enable mixed precision (FP16) processing for faster inference on compatible GPUs. "
             "Reduces memory usage and increases speed with minimal quality impact."
    )

    parser.add_argument(
        "--enable-model-caching", action="store_true", default=True,
        help="Enable intelligent model caching with LRU eviction (default: enabled). "
             "Keeps frequently used models in memory for faster switching."
    )
    parser.add_argument(
        "--cache-size-limit", type=float, default=4.0,
        help="Model cache size limit in GB (default: 4.0). "
             "Controls how many models can be cached simultaneously."
    )
    parser.add_argument(
        "--enable-streaming-mode", action="store_true",
        help="Force enable streaming mode for all videos. "
             "Useful for processing very large videos or when memory is limited."
    )
    parser.add_argument(
        "--optimization-preset", type=str, choices=["speed", "balanced", "quality", "memory"],
        default="balanced",
        help="Optimization preset (default: balanced). "
             "speed: Prioritize processing speed. "
             "balanced: Balance speed and quality. "
             "quality: Prioritize output quality. "
             "memory: Minimize memory usage."
    )
    args = parser.parse_args()

    # Handle help requests early, before validation
    if hasattr(args, 'edgetam_help') and args.edgetam_help:
        print_edgetam_help()
        sys.exit(0)

    if hasattr(args, 'optimization_help') and args.optimization_help:
        print_optimization_help()
        sys.exit(0)

    if hasattr(args, 'benchmark_help') and args.benchmark_help:
        print_benchmark_help()
        sys.exit(0)

    # If config file is provided, override defaults
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as config_file:
                config_from_file = yaml.safe_load(config_file)
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML in configuration file: {e}")
            sys.exit(1)

        # Apply configuration migration for backward compatibility
        config_from_file = migrate_legacy_config(config_from_file)

        # Override args with config values if not explicitly provided
        for key, value in config_from_file.items():
            # Convert hyphenated keys to underscore format for argparse compatibility
            attr_key = key.replace('-', '_')

            # For 'prompt', CLI takes precedence if provided. Otherwise, use config.
            if key == "prompt":
                if getattr(args, attr_key) is None and args.config and key in config_from_file:
                    # Ensure prompt from config is a list
                    setattr(args, attr_key, value if isinstance(value, list) else [value])
            elif hasattr(args, attr_key) and (getattr(args, attr_key) is None or (
                args.config and key in config_from_file and
                getattr(args, attr_key) == parser.get_default(attr_key))):
                # Allow config to override if arg is None or if it's still the default CLI value
                setattr(args, attr_key, value)


    # Validate required fields
    if args.prompt is None or args.input is None:
        print("Error: --prompt and --input are required arguments or must be in the config file.")
        parser.print_help()
        sys.exit(1)

    # Ensure args.prompt is a list, even if only one prompt came from config (and not CLI)
    # If from CLI with nargs='+', it's already a list.
    if args.prompt and not isinstance(args.prompt, list):
        args.prompt = [args.prompt]

    # Apply optimization preset
    args = apply_optimization_preset(args)

    # Validate configuration
    errors, warnings = validate_configuration(args)

    # Handle validation errors
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  ERROR: {error}")
        print("\nPlease fix the above errors and try again.")
        sys.exit(1)

    # Handle validation warnings
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
        print()

    return args

def main():
    """Main function to run the SOWLv2 pipeline from CLI."""
    args = parse_args()


    # Determine input type
    input_path = args.input
    output_path = args.output

    # args.prompt is now a list of strings from nargs='+' or after config parsing.
    # OWLV2Wrapper.detect expects Union[str, List[str]].
    # If list has one item, pass it as str. Otherwise, pass the list.
    prompt_input = args.prompt[0] if len(args.prompt) == 1 else args.prompt

    # Determine device
    device_choice = args.device
    if device_choice == CUDA and hasattr(__import__("torch"), 'cuda') and \
       __import__("torch").cuda.is_available():
        device = CUDA
    else:
        device = CPU
        if device_choice == CUDA:
            print("CUDA selected, but not available. Falling back to CPU.")

    # PipelineConfig options from CLI/config, with defaults

    pipeline_config = PipelineConfig(merged=args.merged,
                                    binary=args.binary,
                                    overlay=args.overlay)

    # Create optimization configuration
    optimization_config = OptimizationConfig(
        optimization_level=args.optimization_level,
        memory_limit=args.memory_limit,
        streaming_chunk_size=args.streaming_chunk_size,
        enable_mixed_precision=args.enable_mixed_precision,
        disable_gpu_batching=args.disable_gpu_batching,
        enable_model_caching=args.enable_model_caching,
        cache_size_limit=args.cache_size_limit,
        enable_streaming_mode=args.enable_streaming_mode,
        optimization_preset=args.optimization_preset
    )

    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        enable_benchmarking=args.benchmark,
        benchmark_output=args.benchmark_output,
        compare_models=args.compare_models,
        benchmark_iterations=args.benchmark_iterations,
        collect_memory_stats=args.collect_memory_stats,
        collect_gpu_stats=args.collect_gpu_stats,
        benchmark_test_data=args.benchmark_test_data,
        performance_profile=args.performance_profile,
        export_metrics=args.export_metrics
    )

    config = PipelineBaseData(
        owl_model=args.owl_model,
        sam_model=args.sam_model,
        threshold=args.threshold,
        fps=args.fps,
        device=device,
        pipeline_config=pipeline_config,
        use_edgetam=args.edgetam,
        edgetam_model=args.edgetam_model,
        edgetam_optimization_level=args.edgetam_optimization_level,
        optimization_config=optimization_config,
        benchmark_config=benchmark_config
    )

    # Use optimized pipeline exclusively
    print("Using optimized SOWLv2 pipeline...")

    # Display optimization settings
    print(f"Optimization preset: {args.optimization_preset}")
    print(f"Optimization level: {args.optimization_level}")
    if args.memory_limit:
        print(f"Memory limit: {args.memory_limit} GB")
    if args.enable_mixed_precision:
        print("Mixed precision (FP16) enabled")
    if args.enable_streaming_mode:
        print(f"Streaming mode enabled (chunk size: {args.streaming_chunk_size})")
    if args.enable_model_caching:
        print(f"Model caching enabled (cache limit: {args.cache_size_limit} GB)")

    # Display benchmarking settings
    if args.benchmark:
        print("Benchmarking enabled - collecting performance metrics")
        if args.benchmark_iterations > 1:
            print(f"Running {args.benchmark_iterations} iterations for accuracy")
        if args.compare_models:
            print("Model comparison enabled - will test both SAM2 and EdgeTAM")
        if args.performance_profile:
            print("Detailed performance profiling enabled")
        if args.benchmark_output:
            print(f"Benchmark results will be saved to: {args.benchmark_output}")
        else:
            print("Benchmark results will be displayed in console")

    # Display segmentation model choice and validate
    if args.edgetam:
        optimization_levels = {
            0: "No optimization (maximum quality)",
            1: "Basic optimization (balanced speed/quality)",
            2: "Aggressive optimization (prioritize speed)",
            3: "Maximum optimization (fastest inference)"
        }

        print(f"Using EdgeTAM model: {args.edgetam_model} for faster segmentation")
        print(f"EdgeTAM optimization level: {args.edgetam_optimization_level} - "
              f"{optimization_levels[args.edgetam_optimization_level]}")

        # Validate EdgeTAM configuration
        from sowlv2.models.model_factory import SegmentationModelFactory
        validation_result = SegmentationModelFactory.validate_model_compatibility(
            "edgetam", args.edgetam_model, device
        )

        if not validation_result["is_valid"]:
            print("WARNING: EdgeTAM configuration validation failed:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")

            if validation_result["recommendations"]:
                print("Recommendations:")
                for rec in validation_result["recommendations"]:
                    print(f"  - {rec}")

            print("Will attempt to use EdgeTAM with automatic fallback to SAM2 if needed.")
        else:
            # Show model info for successful validation
            model_info = SegmentationModelFactory.get_model_info("edgetam", args.edgetam_model)
            if model_info.get("performance_characteristics"):
                perf = model_info["performance_characteristics"]
                print(f"EdgeTAM characteristics: Accuracy={perf.get('accuracy', 'unknown')}, "
                      f"Speed={perf.get('speed', 'unknown')}, "
                      f"Memory={perf.get('memory_usage', 'unknown')}")

        # Log model selection
        ModelFallbackManager.log_model_selection_event(
            "edgetam", args.edgetam_model, was_fallback=False
        )
    else:
        print(f"Using SAM2 model: {args.sam_model} for segmentation")

        # Validate SAM2 configuration
        from sowlv2.models.model_factory import SegmentationModelFactory
        validation_result = SegmentationModelFactory.validate_model_compatibility(
            "sam2", args.sam_model, device
        )

        if validation_result["is_valid"]:
            model_info = SegmentationModelFactory.get_model_info("sam2", args.sam_model)
            if model_info.get("performance_characteristics"):
                perf = model_info["performance_characteristics"]
                print(f"SAM2 characteristics: Accuracy={perf.get('accuracy', 'unknown')}, "
                      f"Speed={perf.get('speed', 'unknown')}, "
                      f"Memory={perf.get('memory_usage', 'unknown')}")

        ModelFallbackManager.log_model_selection_event(
            "sam2", args.sam_model, was_fallback=False
        )

    # Configure parallel processing
    parallel_config = ParallelConfig(
        max_workers=args.max_workers,
        detection_batch_size=args.batch_size,
        segmentation_batch_size=2,
        io_batch_size=8
    )
    pipeline = OptimizedSOWLv2Pipeline(config, parallel_config)
    # Configure V-JEPA 2 if enabled
    if args.enable_vjepa2:
        print("Enabling V-JEPA 2 video optimization...")
        vjepa2_optimizer = create_vjepa2_optimizer(
            config,
            enable_vjepa2=True
        )
        if vjepa2_optimizer:
            print("V-JEPA 2 optimization ready!")
            # Store optimizer reference for potential use in video processing
            pipeline.vjepa2_optimizer = vjepa2_optimizer

            # Set temporal detection parameters
            if args.use_temporal_detection:
                pipeline.use_temporal_detection = True
                pipeline.temporal_detection_frames = args.temporal_detection_frames
                pipeline.temporal_merge_threshold = args.temporal_merge_threshold
                print(f"Temporal detection enabled with {args.temporal_detection_frames} "
                      f"key frames")
        else:
            print("V-JEPA 2 optimization not available, continuing without it.")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"Processing with prompt(s): {prompt_input}")
    # Process input
    if os.path.isdir(input_path):
        pipeline.process_frames(input_path, prompt_input, output_path)
    elif os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VALID_EXTS:
            pipeline.process_image(input_path, prompt_input, output_path)
        elif ext in VALID_VIDEO_EXTS:
            pipeline.process_video(input_path, prompt_input, output_path)
        else:
            print(f"Unsupported file extension: {ext}")
            sys.exit(1)
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
