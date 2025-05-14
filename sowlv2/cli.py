"""
Command Line Interface for SOWLv2.

This script provides a CLI to detect and segment objects in images,
folders of frames, or video files using a text prompt.
It leverages the SOWLv2Pipeline for processing.
"""
import argparse
import os
import sys
import yaml
from sowlv2.data.config import PipelineBaseData
from sowlv2.pipeline import SOWLv2Pipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOWLv2: Detect and segment objects in images/frames/video with a text prompt."
    )
    args = parser.parse_args()
    # If config file is provided, override defaults
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Override args with config values if not explicitly provided
        for key, value in config.items():
            if getattr(args, key) is None or key in ("prompt", "input", "output",
                                                     "owl_model", "sam_model",
                                                     "threshold", "fps", "device",
                                                     "owl_skip_frames"):
                # Allow config to override if arg is None or if it's a configurable param
                if getattr(args, key) is None or (args.config and key in config):
                    setattr(args, key, value)

    # Validate required fields
    if args.prompt is None or args.input is None:
        print("Error: --prompt and --input are required arguments or must be in the config file.")
        parser.print_help()
        sys.exit(1)
    return args

def main():
    """Main function to run the SOWLv2 pipeline from CLI."""
    args = parse_args()
    # Determine input type
    input_path = args.input
    output_path = args.output
    prompt = args.prompt
    owl_model = args.owl_model
    sam_model = args.sam_model
    threshold = args.threshold
    fps = args.fps
    owl_skip_frames = args.owl_skip_frames
    # Determine device
    device_choice = args.device
    if device_choice == "cuda" and hasattr(__import__("torch"), 'cuda') and \
       __import__("torch").cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if device_choice == "cuda":
            print("CUDA selected, but not available. Falling back to CPU.")

    config = PipelineBaseData(
                owl_model=owl_model,
                sam_model=sam_model,
                threshold=threshold,
                fps=fps,
                device=device,
                owl_skip_frames=owl_skip_frames
            )
    pipeline = SOWLv2Pipeline(config=config)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    # Process input
    if os.path.isdir(input_path):
        pipeline.process_frames(input_path, prompt, output_path)
    elif os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            pipeline.process_image(input_path, prompt, output_path)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            pipeline.process_video(input_path, prompt, output_path)
        else:
            print(f"Unsupported file extension: {ext}")
            sys.exit(1)
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
