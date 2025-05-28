"""
Video processing module for SOWLv2 pipeline.
"""
import os
import shutil
import subprocess
import tempfile
from typing import Any, Union, List, Dict, Tuple
import glob
from PIL import Image
# import torch # Not directly used here, but by SAM/frame_pipeline

from sowlv2 import video_utils # For generate_videos and potentially ffmpeg calls
from sowlv2.owl import OWLV2Wrapper # For OWLV2 detection in first frame
from sowlv2.sam2_wrapper import SAM2Wrapper # For SAM state and propagation
from sowlv2.data.config import (
    VideoProcessContext, PipelineConfig, # VideoProcessOptions removed, use PipelineConfig
    VideoDirectories, TempBinaryPaths, TempOverlayPaths, TempVideoOutputPaths,
    PropagatedFrameOutput
)
from sowlv2.pipeline_utils import (
    get_prompt_color,
    create_output_directories
)
from sowlv2.frame_pipeline import process_propagated_frame # For processing individual frames

# Constants moved here as they are primarily used in video context
_FIRST_FRAME = "000001.jpg"
_FIRST_FRAME_IDX = 0

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
    prompt: Union[str, List[str]],
    sam_state: Any,
    owl_model: OWLV2Wrapper,
    sam_model: SAM2Wrapper,
    threshold: float,
    prompt_color_map: Dict[str, Tuple[int, int, int]],
    palette: List[Tuple[int, int, int]], # palette is DEFAULT_PALETTE from pipeline_utils
    next_color_idx: int
) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Tuple[int, int, int]], int]:
    """
    Detect objects in first frame, assign colors, and initialize SAM tracking.
    Returns updated detection details, SAM state, prompt_color_map, and next_color_idx.
    """
    detection_details_for_video: List[Dict[str, Any]] = []
    detections_owl = owl_model.detect(
        image=first_pil_img, prompt=prompt, threshold=threshold
    )

    if not detections_owl:
        return [], sam_state, prompt_color_map, next_color_idx

    sam_obj_id_counter = 1
    # Work on copies to avoid modifying the caller's map directly if it's not intended
    current_prompt_color_map = prompt_color_map.copy()
    current_next_color_idx = next_color_idx

    for det in detections_owl:
        core_prompt = det["core_prompt"]
        object_color, current_next_color_idx = get_prompt_color(
            core_prompt, current_prompt_color_map, palette, current_next_color_idx
        )
        # Ensure the map passed to get_prompt_color is updated with the new color
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
    prompt: Union[str, List[str]],
    owl_model: OWLV2Wrapper,
    sam_model: SAM2Wrapper,
    fps: int,
    threshold: float,
    prompt_color_map_initial: Dict[str, Tuple[int, int, int]],
    palette: List[Tuple[int, int, int]], # This will be DEFAULT_PALETTE
    next_color_idx_initial: int
) -> Tuple[VideoProcessContext | None, Dict[str, Tuple[int, int, int]], int]:
    """
    Extract frames, initialize SAM state, and prepare detection context.
    Returns VideoProcessContext, updated prompt_color_map, and next_color_idx,
    or (None, map, idx) if failed.
    """
    tmp_frames_dir = tempfile.mkdtemp(prefix="sowlv2_extracted_frames_")
    try:
        # Reverted to direct subprocess call for ffmpeg
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-r", str(fps),
             os.path.join(tmp_frames_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
            check=True,
            timeout=300
        )
        initial_sam_state = sam_model.init_state(tmp_frames_dir)
        first_img_path = os.path.join(tmp_frames_dir, _FIRST_FRAME)

        if not os.path.exists(first_img_path):
            print(f"First frame {_FIRST_FRAME} not found in {tmp_frames_dir}.")
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
            return None, prompt_color_map_initial, next_color_idx_initial

        first_pil_img = Image.open(first_img_path).convert("RGB")
        details, updated_state, updated_map, updated_idx = initialize_video_tracking(
            first_pil_img, prompt, initial_sam_state, owl_model, sam_model, threshold,
            prompt_color_map_initial, palette, next_color_idx_initial
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
    except subprocess.TimeoutExpired:
        print(f"Frame extraction (ffmpeg) timed out for video {video_path}.")
        shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return None, prompt_color_map_initial, next_color_idx_initial
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed for {video_path}: {e}. stderr: {e.stderr}")
        shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return None, prompt_color_map_initial, next_color_idx_initial
    except OSError as e:
        print(f"OS error during video preparation for {video_path}: {e}")
        shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return None, prompt_color_map_initial, next_color_idx_initial
    except (ValueError, RuntimeError) as e:
        print(f"Error during video preparation for {video_path}: {e}")
        if os.path.exists(tmp_frames_dir):
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        return None, prompt_color_map_initial, next_color_idx_initial

def process_all_video_frames(
    video_ctx: VideoProcessContext,
    sam_model: SAM2Wrapper,
    temp_output_dir: str,
    pipeline_config: PipelineConfig,
    prompt_color_map: Dict[str, Tuple[int, int, int]],
    next_color_idx: int
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """
    Process all frames in the video using SAM propagation and frame_pipeline.
    Returns updated prompt_color_map and next_color_idx.
    """
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
        prompt_color_map, next_color_idx = process_propagated_frame(
            frame_output_data, pipeline_config, prompt_color_map, next_color_idx
        )
    return prompt_color_map, next_color_idx

def run_video_processing_steps(
    video_ctx: VideoProcessContext,
    sam_model: SAM2Wrapper,
    video_temp_dirs: VideoDirectories, # Renamed for clarity from dirs
    pipeline_config: PipelineConfig,
    prompt_color_map: Dict[str, Tuple[int, int, int]],
    next_color_idx: int,
    fps: int
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """
    Orchestrates processing of all frames and generation of output videos.
    Returns updated prompt_color_map and next_color_idx.
    """
    prompt_color_map, next_color_idx = process_all_video_frames(
        video_ctx, sam_model, video_temp_dirs.temp_dir,
        pipeline_config, prompt_color_map, next_color_idx
    )

    # Conditional video generation based on pipeline_config
    generate_bin_vid = pipeline_config.binary and pipeline_config.merged
    generate_overlay_vid = pipeline_config.overlay and pipeline_config.merged

    if generate_bin_vid or generate_overlay_vid:
        video_utils.generate_videos(
            temp_dir=video_temp_dirs.temp_dir,
            fps=fps,
            binary=generate_bin_vid,
            overlay=generate_overlay_vid
        )
    else:
        print("Skipping temporary merged video generation as per pipeline_config.")

    return prompt_color_map, next_color_idx

def move_video_outputs_to_final_dir(
    temp_dirs: VideoDirectories,
    final_output_dir: str,
    pipeline_config: PipelineConfig
):
    """
    Move requested outputs from temp video directory to final output directory.
    """
    final_binary_frames_dir = os.path.join(final_output_dir, "binary", "frames")
    final_binary_merged_dir = os.path.join(final_output_dir, "binary", "merged")
    final_overlay_frames_dir = os.path.join(final_output_dir, "overlay", "frames")
    final_overlay_merged_dir = os.path.join(final_output_dir, "overlay", "merged")
    final_video_overlay_dir = os.path.join(final_output_dir, "video", "overlay")
    final_video_binary_dir = os.path.join(final_output_dir, "video", "binary")

    def _robust_move_and_mkdir(src_folder, dest_folder):
        if not os.path.exists(src_folder) or not os.path.isdir(src_folder):
            return
        os.makedirs(dest_folder, exist_ok=True) # Ensure destination base exists
        # Move individual items to avoid issues with shutil.move on existing dest_folder
        for item_name in os.listdir(src_folder):
            s_item = os.path.join(src_folder, item_name)
            d_item = os.path.join(dest_folder, item_name)
            try:
                shutil.move(s_item, d_item)
            except (IOError, OSError) as e:
                print(f"Error moving {s_item} to {d_item}: {e}.")
        # Clean up src_folder if empty
        if os.path.exists(src_folder) and not os.listdir(src_folder):
            try:
                os.rmdir(src_folder)
            except OSError as e:
                print(f"Could not remove empty source directory {src_folder}: {e}")

    # Move per-frame outputs
    if pipeline_config.binary:
        src_f_bin = os.path.join(temp_dirs.binary.path, "frames")
        _robust_move_and_mkdir(src_f_bin, final_binary_frames_dir)
        if pipeline_config.merged:
            src_m_bin = temp_dirs.binary.merged_path
            _robust_move_and_mkdir(src_m_bin, final_binary_merged_dir)

    if pipeline_config.overlay:
        src_f_ovr = os.path.join(temp_dirs.overlay.path, "frames")
        _robust_move_and_mkdir(src_f_ovr, final_overlay_frames_dir)
        if pipeline_config.merged:
            src_m_ovr = temp_dirs.overlay.merged_path
            _robust_move_and_mkdir(src_m_ovr, final_overlay_merged_dir)

    # Debug prints for merged image folders (used by video_utils.generate_videos)
    print("[DEBUG] Contents of binary/merged in temp (used for video gen):",
          glob.glob(os.path.join(
              temp_dirs.binary.merged_path, "*") if temp_dirs.binary.merged_path else "[]"))
    print("[DEBUG] Contents of overlay/merged in temp (used for video gen):",
          glob.glob(os.path.join(
              temp_dirs.overlay.merged_path, "*") if temp_dirs.overlay.merged_path else "[]"))

    # Move generated videos
    _robust_move_and_mkdir(temp_dirs.video.overlay_path, final_video_overlay_dir)
    _robust_move_and_mkdir(temp_dirs.video.binary_path, final_video_binary_dir)

    # Clean up the overall temp_dir for this video run if it exists and is empty
    # (excluding the ffmpeg extracted frames dir, which is handled by the caller)
    base_video_temp_dir = temp_dirs.temp_dir
    if os.path.exists(base_video_temp_dir) and not os.listdir(base_video_temp_dir):
        try:
            shutil.rmtree(base_video_temp_dir)
            print(f"Cleaned up base temp directory for video run: {base_video_temp_dir}")
        except OSError as e:
            print(f"Could not remove base temp directory {base_video_temp_dir}: {e}")
    elif os.path.exists(base_video_temp_dir):
        pass

    print(f"✅ Video outputs potentially moved to {final_output_dir}")
