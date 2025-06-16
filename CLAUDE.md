# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOWLv2 is a command-line tool and Python library for text-prompted object segmentation that combines Google's OWLv2 (open-vocabulary object detector) with Meta's SAM 2 (Segment Anything Model V2) for precise pixel-level segmentation. The tool processes images, video frames, or videos based on natural language prompts.

## Architecture

### Core Pipeline Flow
1. **Input Processing**: Images/videos → Frame extraction  
2. **Detection**: OWLv2 finds objects matching text prompts → Bounding boxes
3. **Segmentation**: SAM 2 creates precise masks from bounding boxes  
4. **Output Generation**: Binary masks + visual overlays + videos

### Key Components
- `cli.py`: Command-line interface entry point
- `pipeline.py`: Main orchestration pipeline
- `image_pipeline.py`: Single image processing
- `video_pipeline.py`: Video processing with temporal tracking
- `models/owl.py`: OWLv2 wrapper for object detection
- `models/sam2_wrapper.py`: SAM 2 wrapper for segmentation
- `utils/`: File system, frame, pipeline, and video utilities

## Development Commands

### Installation
```bash
# Development install
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Code Quality
```bash
# Run linting (matches CI)
pylint $(git ls-files '*.py')
```

### Testing
- Primary testing through Google Colab notebook demonstrations
- GitHub Actions CI runs pylint on Python 3.10-3.13

### CLI Usage
```bash
# Basic usage
sowlv2-detect --prompt "cat" --input image.jpg --output results/

# Multiple objects
sowlv2-detect --prompt "cat" "dog" --input video.mp4 --output results/

# With config file
sowlv2-detect --config config.yaml
```

## Configuration

- YAML configuration files supported (see `config/config_example.yaml`)
- Key parameters: `prompt`, `owl_model`, `sam_model`, `threshold`, `fps`, `device`
- Command-line arguments override config file values

## Output Structure

```
output_dir/
├── binary/                  # Binary mask images/videos
│   ├── merged/             # Merged binary masks (all objects combined)
│   │   ├── 000001_merged_mask.png
│   │   ├── 000002_merged_mask.png
│   │   └── ...
│   └── frames/             # Individual binary masks per object
│       ├── 000001_obj1_cat_mask.png
│       ├── 000001_obj2_dog_mask.png
│       └── ...
├── overlay/                # RGB overlay images/videos
│   ├── merged/            # Merged overlays (all objects combined)
│   │   ├── 000001_merged_overlay.png
│   │   ├── 000002_merged_overlay.png
│   │   └── ...
│   └── frames/            # Individual overlays per object
│       ├── 000001_obj1_cat_overlay.png
│       ├── 000001_obj2_dog_overlay.png
│       └── ...
└── video/                 # Generated videos (for video input)
    ├── binary/            # Binary mask videos
    │   ├── merged_mask.mp4
    │   ├── obj1_cat_mask.mp4
    │   └── obj2_dog_mask.mp4
    └── overlay/           # Overlay videos
        ├── merged_overlay.mp4
        ├── obj1_cat_overlay.mp4
        └── obj2_dog_overlay.mp4
```

### File Naming Convention:
- **Individual files**: `{frame_num}_obj{obj_id}_{prompt}_mask.png` / `{frame_num}_obj{obj_id}_{prompt}_overlay.png`
- **Merged files**: `{frame_num}_merged_mask.png` / `{frame_num}_merged_overlay.png`
- **Videos**: `obj{obj_id}_{prompt}_mask.mp4` / `merged_mask.mp4`

## Dependencies

- Core ML: `torch>=1.13.0`, `transformers>=4.32.1`, `sam2>=1.1.0`
- Image/Video: `opencv-python>=4.5.5.64`, `Pillow>=9.0.0`
- Utilities: `pyyaml>=6.0`, `huggingface_hub>=0.15.0`
- GPU/CPU auto-detection with CUDA support

## Entry Points

- CLI tool: `sowlv2-detect` (defined in pyproject.toml)
- Python API: Import `sowlv2` modules directly