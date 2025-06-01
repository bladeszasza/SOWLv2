"""
Frame processing module for SOWLv2 pipeline.
Handles processing of individual frames, including mask propagation and output generation.
"""
from typing import List, Dict, Tuple
import os

from sowlv2.data.config import (
    PropagatedFrameOutput, PipelineConfig,
    MergedOverlayItem, SingleDetectionInput
)
from sowlv2.utils.filesystem_utils import create_output_directories
from sowlv2.image_pipeline import (
    process_single_detection_for_image,
    create_and_save_merged_overlay
)

def is_image_file(filename: str) -> bool:
    """
    Check if the filename has a valid image extension.
    Args:
        filename (str): The filename to check.
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_exts

def process_propagated_frame(
    frame_output_data: PropagatedFrameOutput,
    pipeline_config: PipelineConfig,
    prompt_color_map: Dict[str, Tuple[int, int, int]],
    next_color_idx: int = 0
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """Process a single propagated frame with SAM tracking.

    Args:
        frame_output_data: Data for the current frame
        pipeline_config: Pipeline configuration
        prompt_color_map: Map of prompts to colors
        palette: Color palette for object visualization
        next_color_idx: Next color index to use

    Returns:
        Tuple of (updated prompt_color_map, updated next_color_idx)
    """
    # Create output directories for this frame
    create_output_directories(frame_output_data.output_dir)

    # Process each object in the frame
    items_for_merged_overlay: List[MergedOverlayItem] = []
    current_prompt_color_map = prompt_color_map.copy()
    current_next_color_idx = next_color_idx

    for obj_idx, (sam_obj_id, mask_logits) in enumerate(zip(
        frame_output_data.sam_obj_ids_tensor,
        frame_output_data.mask_logits_tensor
    )):
        # Get detection details for this object
        det_details = next(
            (d for d in frame_output_data.detection_details_map if d['sam_id'] == sam_obj_id),
            None
        )
        if not det_details:
            print(f"Warning: No detection details found for SAM object ID {sam_obj_id}")
            continue

        # Create input for single detection processing
        single_detection_input = SingleDetectionInput(
            pil_image=frame_output_data.current_pil_img,
            detection_detail={
                'core_prompt': det_details['core_prompt'],
                'color': det_details['color'],
                'mask_logits': mask_logits
            },
            obj_idx=obj_idx + 1,
            base_name=f"{frame_output_data.frame_num:06d}",
            output_dir=frame_output_data.output_dir
        )

        # Process the detection using image pipeline
        try:
            merged_item = process_single_detection_for_image(
                single_detection_input,
                pipeline_config
            )
            if merged_item:
                items_for_merged_overlay.append(merged_item)
        except (IOError, OSError) as e:
            print(f"Error {obj_idx + 1} in frame {frame_output_data.frame_num}: {e}")
            continue

    # Create merged outputs if requested
    if items_for_merged_overlay and pipeline_config.merged:
        try:
            create_and_save_merged_overlay(
                items_for_merged_overlay,
                frame_output_data.current_pil_img,
                frame_output_data.output_dir,
                frame_output_data.frame_num
            )
        except (IOError, OSError) as e:
            print(f"Error creating merged overlay for frame {frame_output_data.frame_num}: {e}")

    return current_prompt_color_map, current_next_color_idx
