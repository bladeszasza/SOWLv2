#!/usr/bin/env python3
"""
Example script demonstrating optimized SOWLv2 pipeline for faster inference.
Addresses GitHub Issue #19: Decrease inference time
"""
import argparse
import time
import torch

from sowlv2.optimizations import (
    OptimizedSOWLv2Pipeline,
    ParallelConfig,
    GPUOptimizer
)
from sowlv2.data.config import PipelineBaseData, PipelineConfig


def benchmark_inference(pipeline, image_path, prompts, output_dir, runs=3):
    """Benchmark inference time over multiple runs."""
    times = []

    for i in range(runs):
        start = time.time()
        pipeline.process_image(image_path, prompts, f"{output_dir}/run_{i}")
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.2f}s")
    print(f"FPS (single image): {1/avg_time:.2f}")

    return avg_time


def main():
    parser = argparse.ArgumentParser(
        description="Optimized SOWLv2 inference example"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input image or video"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="+",
        help="Text prompts for detection (multiple allowed)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_optimized",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPU processing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with standard pipeline"
    )

    args = parser.parse_args()

    # Configure parallel processing
    parallel_config = ParallelConfig(
        max_workers=args.workers,
        batch_size=args.batch_size,
        use_gpu_batching=(args.device == "cuda"),
        thread_pool_size=16
    )

    # Configure pipeline
    pipeline_config = PipelineBaseData(
        owl_model="google/owlv2-base-patch16-ensemble",
        sam_model="facebook/sam2.1-hiera-small",
        threshold=0.1,
        fps=24,
        device=args.device,
        pipeline_config=PipelineConfig(
            binary=True,
            overlay=True,
            merged=True
        )
    )

    print("🚀 Initializing Optimized SOWLv2 Pipeline")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Prompts: {args.prompt}")
    print("-" * 50)

    # Initialize optimized pipeline
    optimized_pipeline = OptimizedSOWLv2Pipeline(
        pipeline_config,
        parallel_config
    )

    if args.benchmark:
        print("\n📊 Running Benchmark Mode")
        avg_time = benchmark_inference(
            optimized_pipeline,
            args.input,
            args.prompt,
            args.output,
            runs=3
        )

        if args.compare:
            print("\n📊 Comparing with Standard Pipeline")
            # Import here to avoid circular imports and only when needed
            from sowlv2.pipeline import SOWLv2Pipeline  # pylint: disable=import-outside-toplevel

            standard_pipeline = SOWLv2Pipeline(pipeline_config)
            standard_time = benchmark_inference(
                standard_pipeline,
                args.input,
                args.prompt,
                args.output + "_standard",
                runs=3
            )

            speedup = standard_time / avg_time
            print(f"\n🎯 Speedup: {speedup:.2f}x faster!")
            print(f"Standard: {standard_time:.2f}s")
            print(f"Optimized: {avg_time:.2f}s")
    else:
        # Single run
        print("\n🔍 Processing image...")
        start = time.time()
        optimized_pipeline.process_image(
            args.input,
            args.prompt,
            args.output
        )
        elapsed = time.time() - start

        print("\n✅ Processing complete!")
        print(f"Time: {elapsed:.2f}s")
        print(f"Output saved to: {args.output}")

    # Show GPU memory usage if using CUDA
    if args.device == "cuda":
        memory_stats = GPUOptimizer.profile_gpu_memory()
        print("\n💾 GPU Memory Usage:")
        print(f"  Allocated: {memory_stats['allocated']:.2f} GB")
        print(f"  Reserved: {memory_stats['reserved']:.2f} GB")
        print(f"  Free: {memory_stats['free']:.2f} GB")


if __name__ == "__main__":
    main()
