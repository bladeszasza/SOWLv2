"""
Image processing module for SOWLv2 pipeline.
"""
import os
from typing import List, Optional

from PIL import Image

from sowlv2.data.config import (
    PipelineConfig, SingleDetectionInput,
    MergedOverlayItem
)
from sowlv2.utils.filesystem_utils import create_output_directories
from sowlv2.utils.pipeline_utils import (
    validate_mask,
    create_merged_binary_mask, create_merged_overlay,
    create_overlay
)
from sowlv2.utils.path_config import (
    FilePattern, DirectoryStructure
)

def process_single_detection_for_image(
    single_detection_input: SingleDetectionInput,
    pipeline_config: PipelineConfig  # pylint: disable=unused-argument
) -> Optional[MergedOverlayItem]:
    """Process a single detection for an image.

    Args:
        single_detection_input: Input data for single detection
        pipeline_config: Pipeline configuration

    Returns:
        MergedOverlayItem if successful, None otherwise
    """
    # Create output directories
    create_output_directories(single_detection_input.output_dir)

    # Validate and process mask
    try:
        # Get mask from detection detail
        mask = single_detection_input.detection_detail.get('mask')
        if mask is None:
            print(f"Warning: No mask found for object {single_detection_input.obj_idx}")
            return None

        mask = validate_mask(mask)
        if mask is None:
            print(f"Warning: Invalid mask for object {single_detection_input.obj_idx}")
            return None

        # Get prompt and slugify for filename
        prompt = single_detection_input.detection_detail.get(
            'core_prompt', f'object_{single_detection_input.obj_idx}')
        prompt_slug = prompt.replace(' ', '_')
        base_name = single_detection_input.base_name
        base_name_slug = base_name.replace(' ', '_')
        idx = single_detection_input.obj_idx

        # Create merged item for overlay
        merged_item = MergedOverlayItem(
                mask=mask,
                color=single_detection_input.detection_detail.get('color', (255, 0, 0)),
                label=prompt
            )

        # Always save binary mask in temp directory
        # Selective copying will be handled by output copying logic
        binary_path = os.path.join(
            single_detection_input.output_dir,
            DirectoryStructure.BINARY,
            DirectoryStructure.FRAMES,
            FilePattern.INDIVIDUAL_MASK.format(
                frame_num=base_name_slug,
                obj_id=idx,
                prompt=prompt_slug
            )
        )
        Image.fromarray(mask).save(binary_path)

        # Always create and save overlay in temp directory
        # Selective copying will be handled by output copying logic
        overlay_path = os.path.join(
            single_detection_input.output_dir,
            DirectoryStructure.OVERLAY,
            DirectoryStructure.FRAMES,
            FilePattern.INDIVIDUAL_OVERLAY.format(
                frame_num=base_name_slug,
                obj_id=idx,
                prompt=prompt_slug
            )
        )
        create_overlay(
            single_detection_input.pil_image,
            mask,
            single_detection_input.detection_detail.get('color', (255, 0, 0))
        ).save(overlay_path)

        return merged_item

    except (IOError, OSError) as e:
        print(f"Error processing detection {idx}: {e}")
        return None

def create_and_save_merged_overlay(
    items_for_merged_overlay: List[MergedOverlayItem],
    original_image: Image.Image,
    output_dir: str,
    frame_num: int
) -> None:
    """Create and save merged overlay from multiple items.

    Args:
        items_for_merged_overlay: List of items to merge
        original_image: Original PIL image
        output_dir: Output directory
        frame_num: Frame number
    """
    try:
        # Use the original image name for merged outputs
        if hasattr(original_image, 'filename') and original_image.filename:
            orig_name = os.path.splitext(os.path.basename(original_image.filename))[0]
        else:
            orig_name = str(frame_num) if isinstance(frame_num, int) else str(frame_num)
        frame_num_str = FilePattern.FRAME_NUM_FORMAT.format(int(orig_name))

        # Create merged binary mask
        binary_merged_dir = os.path.join(
            output_dir,
            DirectoryStructure.BINARY,
            DirectoryStructure.MERGED
        )
        os.makedirs(binary_merged_dir, exist_ok=True)
        create_merged_binary_mask(
            [item.mask for item in items_for_merged_overlay],
            binary_merged_dir,
            frame_num_str,
            PipelineConfig(binary=True, overlay=True, merged=True)
        )

        # Create merged overlay
        overlay_merged_dir = os.path.join(
            output_dir,
            DirectoryStructure.OVERLAY,
            DirectoryStructure.MERGED
        )
        os.makedirs(overlay_merged_dir, exist_ok=True)
        create_merged_overlay(
            original_image,
            [(item.mask, item.color) for item in items_for_merged_overlay],
            overlay_merged_dir,
            frame_num_str,
            PipelineConfig(binary=True, overlay=True, merged=True)
        )

    except (IOError, OSError) as e:
        print(f"Error creating merged outputs for frame {frame_num}: {e}")
