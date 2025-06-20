"""
Video processing module for SOWLv2 pipeline.
"""
import os
import shutil
import subprocess
import tempfile
from typing import Any, Union, List, Dict, Tuple
from dataclasses import dataclass
from PIL import Image

from sowlv2.utils import video_utils
from sowlv2.models import OWLV2Wrapper, SAM2Wrapper
from sowlv2.data.config import (
    VideoProcessContext, PipelineConfig,
    VideoDirectories, TempBinaryPaths, TempOverlayPaths, TempVideoOutputPaths,
    PropagatedFrameOutput
)
from sowlv2.utils.filesystem_utils import create_output_directories
from sowlv2.utils import frame_utils
from sowlv2.utils.pipeline_utils import (
    get_prompt_color
)

# Constants moved here as they are primarily used in video context
_FIRST_FRAME = "000001.jpg"
_FIRST_FRAME_IDX = 0

@dataclass
class VideoTrackingConfig:
    """Configuration for video tracking initialization."""
    prompt: Union[str, List[str]]
    threshold: float
    prompt_color_map: Dict[str, Tuple[int, int, int]]
    palette: List[Tuple[int, int, int]]
    next_color_idx: int
    fps: int

@dataclass
class VideoProcessingConfig:
    """Configuration for video processing."""
    pipeline_config: PipelineConfig
    prompt_color_map: Dict[str, Tuple[int, int, int]]
    next_color_idx: int
    fps: int

def create_temp_directories_for_video(base_temp_dir: str) -> VideoDirectories:
    """
    Create temporary directory structure for video processing within a given base temp dir.
    """
    dirs = create_output_directories(base_temp_dir, include_video=True)

    return VideoDirectories(
        temp_dir=base_temp_dir,
        binary=TempBinaryPaths(path=dirs["binary"], merged_path=dirs["binary_merged"]),
        overlay=TempOverlayPaths(path=dirs["overlay"], merged_path=dirs["overlay_merged"]),
        video=TempVideoOutputPaths(
            path=dirs["video"],
            binary_path=dirs["video_binary"],
            overlay_path=dirs["video_overlay"]
        )
    )

def initialize_video_tracking(
    first_pil_img: Image.Image,
    config: VideoTrackingConfig,
    sam_state: Any,
    owl_model: OWLV2Wrapper,
    sam_model: SAM2Wrapper,
) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Tuple[int, int, int]], int]:
    """Initialize video tracking with configuration."""
    detection_details_for_video: List[Dict[str, Any]] = []
    detections_owl = owl_model.detect(
        image=first_pil_img, prompt=config.prompt, threshold=config.threshold
    )

    if not detections_owl:
        return [], sam_state, config.prompt_color_map, config.next_color_idx

    sam_obj_id_counter = 1
    current_prompt_color_map = config.prompt_color_map.copy()
    current_next_color_idx = config.next_color_idx

    for det in detections_owl:
        core_prompt = det["core_prompt"]
        object_color, current_next_color_idx = get_prompt_color(
            core_prompt, current_prompt_color_map, config.palette, current_next_color_idx
        )
        current_prompt_color_map[core_prompt] = object_color

        box = det["box"]
        detection_details_for_video.append({
            'sam_id': sam_obj_id_counter,
            'core_prompt': core_prompt,
            'color': object_color
        })
        sam_model.add_new_box(
            state=sam_state, frame_idx=_FIRST_FRAME_IDX,
            box=box, obj_idx=sam_obj_id_counter
        )
        sam_obj_id_counter += 1
    return detection_details_for_video, sam_state, current_prompt_color_map, current_next_color_idx

def prepare_video_context(
    video_path: str,
    config: VideoTrackingConfig,
    owl_model: OWLV2Wrapper,
    sam_model: SAM2Wrapper,
) -> Tuple[VideoProcessContext | None, Dict[str, Tuple[int, int, int]], int]:
    """Prepare video context with configuration."""
    tmp_frames_dir = tempfile.mkdtemp(prefix="sowlv2_extracted_frames_")
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-r", str(config.fps),
             os.path.join(tmp_frames_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
            check=True,
            timeout=300
        )
        initial_sam_state = sam_model.init_state(tmp_frames_dir)
        first_img_path = os.path.join(tmp_frames_dir, _FIRST_FRAME)

        if not os.path.exists(first_img_path):
            print(f"First frame {_FIRST_FRAME} not found in {tmp_frames_dir}.")
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
            return None, config.prompt_color_map, config.next_color_idx

        first_pil_img = Image.open(first_img_path).convert("RGB")
        details, updated_state, updated_map, updated_idx = initialize_video_tracking(
            first_pil_img, config, initial_sam_state, owl_model, sam_model
        )

        if not details:
            print("No objects for tracking initialized in the first frame.")
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
            return None, updated_map, updated_idx

        video_ctx = VideoProcessContext(
            tmp_frames_dir=tmp_frames_dir,
            initial_sam_state=initial_sam_state,
            first_img_path=first_img_path,
            first_pil_img=first_pil_img,
            detection_details_for_video=details,
            updated_sam_state=updated_state
        )
        return video_ctx, updated_map, updated_idx
    except (subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            OSError,
            ValueError,
            RuntimeError) as e:
        print(f"Error during video preparation for {video_path}: {e}")
        if os.path.exists(tmp_frames_dir):
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return None, config.prompt_color_map, config.next_color_idx

def process_all_video_frames(
    video_ctx: VideoProcessContext,
    sam_model: SAM2Wrapper,
    config: VideoProcessingConfig,
    temp_output_dir: str,
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """Process all video frames with configuration."""
    for fidx, sam_obj_ids_tensor, mask_logits_tensor in sam_model.propagate_in_video(
        video_ctx.updated_sam_state):
        current_frame_num = fidx + 1
        frame_file_path = os.path.join(
            video_ctx.tmp_frames_dir, f"{current_frame_num:06d}.jpg")
        if not os.path.exists(frame_file_path):
            print(f"Warning: Frame {frame_file_path} not found. Skipping frame.")
            continue
        try:
            current_pil_img = Image.open(frame_file_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Error opening frame {frame_file_path}: {e}. Skipping.")
            continue

        frame_output_data = PropagatedFrameOutput(
            current_pil_img=current_pil_img,
            frame_num=current_frame_num,
            sam_obj_ids_tensor=sam_obj_ids_tensor,
            mask_logits_tensor=mask_logits_tensor,
            detection_details_map=video_ctx.detection_details_for_video,
            output_dir=temp_output_dir
        )
        config.prompt_color_map, config.next_color_idx = frame_utils.process_propagated_frame(
            frame_output_data,
            config.pipeline_config,
            config.prompt_color_map,
            config.next_color_idx
        )
    return config.prompt_color_map, config.next_color_idx

def run_video_processing_steps(
    video_ctx: VideoProcessContext,
    sam_model: SAM2Wrapper,
    video_temp_dirs: VideoDirectories,
    config: VideoProcessingConfig,
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """Run video processing steps with configuration."""
    config.prompt_color_map, config.next_color_idx = process_all_video_frames(
        video_ctx, sam_model, config, video_temp_dirs.temp_dir
    )

    # Always generate all video types in temp directory
    # Selective copying will be handled by move_video_outputs_to_final_dir
    video_utils.generate_videos(
            temp_dir=video_temp_dirs.temp_dir,
            fps=config.fps,
            binary=True,
            overlay=True,
            merged=True,
            prompt_details=video_ctx.detection_details_for_video
        )

    return config.prompt_color_map, config.next_color_idx

def move_video_outputs_to_final_dir(
    temp_dirs: VideoDirectories,
    final_output_dir: str,
    pipeline_config: PipelineConfig
):
    """Move video outputs to final directory."""
    # Create the base video directory first if any video output is needed
    if pipeline_config.binary or pipeline_config.overlay:
        os.makedirs(os.path.join(final_output_dir, "video"), exist_ok=True)
    final_dirs = {
        "binary_frames": os.path.join(final_output_dir, "binary", "frames"),
        "binary_merged": os.path.join(final_output_dir, "binary", "merged"),
        "overlay_frames": os.path.join(final_output_dir, "overlay", "frames"),
        "overlay_merged": os.path.join(final_output_dir, "overlay", "merged"),
        "video_overlay": os.path.join(final_output_dir, "video", "overlay"),
        "video_binary": os.path.join(final_output_dir, "video", "binary")
    }

    def _robust_move_and_mkdir(src_folder, dest_folder):
        if not os.path.exists(src_folder) or not os.path.isdir(src_folder):
            return
        os.makedirs(dest_folder, exist_ok=True)
        for item_name in os.listdir(src_folder):
            s_item = os.path.join(src_folder, item_name)
            d_item = os.path.join(dest_folder, item_name)
            try:
                shutil.move(s_item, d_item)
            except (IOError, OSError) as e:
                print(f"Error moving {s_item} to {d_item}: {e}.")
        if os.path.exists(src_folder) and not os.listdir(src_folder):
            try:
                os.rmdir(src_folder)
            except OSError as e:
                print(f"Could not remove empty source directory {src_folder}: {e}")

    # Move frame outputs based on flags
    if pipeline_config.binary:
        _robust_move_and_mkdir(
            os.path.join(temp_dirs.binary.path, "frames"),
            final_dirs["binary_frames"]
        )
        if pipeline_config.merged:
            _robust_move_and_mkdir(
                temp_dirs.binary.merged_path,
                final_dirs["binary_merged"]
            )

    if pipeline_config.overlay:
        _robust_move_and_mkdir(
            os.path.join(temp_dirs.overlay.path, "frames"),
            final_dirs["overlay_frames"]
        )
        if pipeline_config.merged:
            _robust_move_and_mkdir(
                temp_dirs.overlay.merged_path,
                final_dirs["overlay_merged"]
            )

    # CRITICAL FIX: Always move video outputs when they exist, regardless of binary/overlay flags
    # When both --no-binary and --no-overlay are used, we still generate video outputs
    # Check if video directories exist in temp before moving
    if os.path.exists(temp_dirs.video.binary_path) and os.listdir(temp_dirs.video.binary_path):
        _robust_move_and_mkdir(temp_dirs.video.binary_path, final_dirs["video_binary"])
    if os.path.exists(temp_dirs.video.overlay_path) and os.listdir(temp_dirs.video.overlay_path):
        _robust_move_and_mkdir(temp_dirs.video.overlay_path, final_dirs["video_overlay"])

    # Clean up temporary directory
    base_video_temp_dir = temp_dirs.temp_dir
    if os.path.exists(base_video_temp_dir) and not os.listdir(base_video_temp_dir):
        try:
            shutil.rmtree(base_video_temp_dir)
        except OSError as e:
            print(f"Could not remove base temp directory {base_video_temp_dir}: {e}")
    elif os.path.exists(base_video_temp_dir):
        pass

    print(f"✅ Video outputs moved to {final_output_dir}")
