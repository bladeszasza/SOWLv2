import argparse
import os
import sys
import yaml
from sowlv2.pipeline import SOWLv2Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="SOWLv2: Detect and segment objects in images/frames/video with a text prompt.")
    parser.add_argument("--prompt", type=str, required=False, help="Text prompt for object detection (e.g. 'cat')")
    parser.add_argument("--input", type=str, required=False, help="Path to input (image file, directory of frames, or video file)")
    parser.add_argument("--output", type=str, default="output", help="Directory to save output masks and overlays")
    parser.add_argument("--owl-model", type=str, default="google/owlv2-base-patch16-ensemble", help="OWLv2 model (HuggingFace name)")
    parser.add_argument("--sam-model", type=str, default="facebook/sam2.1-hiera-small", help="SAM2 model (HuggingFace name)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--fps", type=int, default=24, help="Sampling rate (frames per second) for video")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device (cpu or cuda). Default uses GPU if available.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional)")
    args = parser.parse_args()
    # If config file is provided, override defaults
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override args with config values if not explicitly provided
        for key, value in config.items():
            if getattr(args, key) is None or key == "prompt":
                setattr(args, key, value)
    # Validate required fields
    if args.prompt is None or args.input is None:
        print("Error: --prompt and --input are required arguments.")
        parser.print_help()
        sys.exit(1)
    return args

def main():
    args = parse_args()
    # Determine input type
    input_path = args.input
    output_path = args.output
    prompt = args.prompt
    owl_model = args.owl_model
    sam_model = args.sam_model
    threshold = args.threshold
    fps = args.fps
    # Determine device
    device = args.device if args.device else ("cuda" if (hasattr(__import__("torch"), 'cuda') and __import__("torch").cuda.is_available()) else "cpu")

    pipeline = SOWLv2Pipeline(
        owl_model=owl_model, sam_model=sam_model,
        threshold=threshold, fps=fps, device=device
    )
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
