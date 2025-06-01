"""
Pipeline utilities for SOWLv2.
"""
import os
from typing import Dict, List, Tuple, Union
import numpy as np
import cv2
import torch
from PIL import Image

from sowlv2.data.config import PipelineConfig
from sowlv2.utils.path_config import FilePattern

# Disable no-member for cv2 (OpenCV) for the whole file
# pylint: disable=no-member

# Default color palette for object visualization
DEFAULT_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (0, 128, 255), (128, 0, 255), (192, 192, 192), (255, 215, 0),
    (138, 43, 226), (75, 0, 130), (240, 128, 128), (32, 178, 170)
]

CUDA = "cuda"
CPU = "cpu"

def validate_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and convert mask to binary format."""
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        try:
            mask = np.array(mask)
        except (TypeError, ValueError):
            return None
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask

def create_overlay(
    pil_image: Image.Image,
    bool_mask_np: np.ndarray,
    color: Tuple[int, int, int]
) -> Image.Image:
    """Blend a colored mask with a PIL image and return the overlay as a PIL.Image.

    Args:
        pil_image: Input PIL image
        bool_mask_np: Boolean mask for overlay
        color: RGB color tuple for the overlay

    Returns:
        PIL Image with overlay applied
    """
    bool_mask_np = validate_mask(bool_mask_np)

    overlay_pil = pil_image.copy()
    overlay_np = np.array(overlay_pil)
    color_np = np.array(color, dtype=np.uint8)

    if overlay_np.ndim == 2:
        overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_GRAY2RGB)
    elif overlay_np.ndim == 3 and overlay_np.shape[2] == 4:
        overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_RGBA2RGB)
    elif overlay_np.ndim == 3 and overlay_np.shape[2] != 3:
        raise ValueError(f"Invalid image shape {overlay_np.shape}")

    overlay_np[bool_mask_np] = (
        0.5 * overlay_np[bool_mask_np] + 0.5 * color_np
    ).astype(np.uint8)
    return Image.fromarray(overlay_np)

def create_merged_binary_mask(
    items: List[np.ndarray],
    output_dir: str,
    base_name: str,
    pipeline_config: PipelineConfig
) -> None:
    """Create and save a merged binary mask from multiple items.

    Args:
        items: List of binary masks to merge
        output_dir: Directory to save the merged mask
        base_name: Base name for the output file
        pipeline_config: Pipeline configuration
    """
    if not pipeline_config.binary or not items:
        return

    try:
        merged_mask = np.zeros_like(items[0], dtype=np.uint8)
        for mask in items:
            bool_mask = validate_mask(mask)
            merged_mask = np.logical_or(merged_mask, bool_mask).astype(np.uint8)

        merged_mask_pil = Image.fromarray(merged_mask * 255).convert("L")
        merged_mask_file = os.path.join(
            output_dir,
            FilePattern.MERGED_MASK.format(frame_num=base_name)
        )
        os.makedirs(os.path.dirname(merged_mask_file), exist_ok=True)
        merged_mask_pil.save(merged_mask_file)
        print(f"Saved merged binary mask: {merged_mask_file}")
    except (IOError, OSError) as e:
        print(f"Error saving merged binary mask: {e}")

def create_merged_overlay(
    original_pil_image: Image.Image,
    items: List[Tuple[np.ndarray, Tuple[int, int, int]]],
    output_dir: str,
    base_name: str,
    pipeline_config: PipelineConfig
) -> None:
    """Create and save a merged overlay from multiple items.

    Args:
        original_pil_image: Original PIL image
        items: List of (mask, color) tuples
        output_dir: Directory to save the merged overlay
        base_name: Base name for the output file
        pipeline_config: Pipeline configuration
    """
    if not pipeline_config.overlay or not items:
        return

    try:
        merged_overlay_pil = original_pil_image.copy()
        for mask, color in items:
            bool_mask = validate_mask(mask)
            merged_overlay_pil = create_overlay(merged_overlay_pil, bool_mask, color)

        merged_overlay_file = os.path.join(
            output_dir,
            FilePattern.MERGED_OVERLAY.format(frame_num=base_name)
        )
        os.makedirs(os.path.dirname(merged_overlay_file), exist_ok=True)
        merged_overlay_pil.save(merged_overlay_file)
        print(f"Saved merged overlay: {merged_overlay_file}")
    except (IOError, OSError) as e:
        print(f"Error saving merged overlay: {e}")

def get_prompt_color(
    core_prompt: str,
    prompt_color_map: Dict[str, Tuple[int, int, int]],
    palette: List[Tuple[int, int, int]],
    next_color_idx: int
) -> Tuple[Tuple[int, int, int], int]:
    """Get a consistent color for a prompt, updating the color map and index.

    Args:
        core_prompt: The prompt to get a color for
        prompt_color_map: Current mapping of prompts to colors
        palette: List of available colors
        next_color_idx: Current index in the palette

    Returns:
        Tuple of (color, updated_next_color_idx)
    """
    if core_prompt in prompt_color_map:
        return prompt_color_map[core_prompt], next_color_idx

    color = palette[next_color_idx % len(palette)]
    prompt_color_map[core_prompt] = color
    return color, next_color_idx + 1
