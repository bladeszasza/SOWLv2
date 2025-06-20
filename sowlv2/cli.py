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
from sowlv2.data.config import PipelineBaseData, PipelineConfig
from sowlv2.optimizations import OptimizedSOWLv2Pipeline, ParallelConfig, create_vjepa2_optimizer
from sowlv2.utils.frame_utils import VALID_EXTS, VALID_VIDEO_EXTS
from sowlv2.utils.pipeline_utils import CPU, CUDA

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOWLv2: Detect and segment objects in images/frames/video with a text prompt."
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
    args = parser.parse_args()
    # If config file is provided, override defaults
    if args.config:
        with open(args.config, "r", encoding="utf-8") as config_file:
            config_from_file = yaml.safe_load(config_file)
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

    config = PipelineBaseData(
        owl_model=args.owl_model,
        sam_model=args.sam_model,
        threshold=args.threshold,
        fps=args.fps,
        device=device,
        pipeline_config=pipeline_config
    )

    # Use optimized pipeline exclusively
    print("Using optimized SOWLv2 pipeline...")
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
