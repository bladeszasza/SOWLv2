"""
Utilities for video processing, such as converting image frames to video
and generating per-object mask/overlay videos.
"""
import os
import glob
import re
from typing import List, Dict
import cv2  # pylint: disable=import-error


# Disable no-member for cv2 (OpenCV) for the whole file
# pylint: disable=no-member


def images_to_video(image_paths: List[str], output_path: str, fps: int):
    """
    Convert a list of image paths to a video file.
    """
    if not image_paths:
        print(f"No images found for video generation: {output_path}")
        return

    # Read first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Failed to read first image: {image_paths[0]}")
        return

    height, width = first_image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Failed to read frame: {image_path}")

    out.release()
    print(f"Video saved to: {output_path}")

def _parse_mask_filename(fname):
    """
    Parse a mask filename to extract sam_id_token and core_prompt_slug.
    Returns (sam_id_token, core_prompt_slug) or (None, None) if not matched.
    """
    # Example: 000001_obj1_dog_mask.png
    match = re.match(r"^\d+_(obj\d+)_([a-zA-Z0-9_]+)_mask\.png$", fname)
    if match:
        return match.group(1), match.group(2)
    # Fallback: 000001_obj1_mask.png (no prompt)
    match_simple = re.match(r"^\d+_(obj\d+)_mask\.png$", fname)
    if match_simple:
        return match_simple.group(1), None
    return None, None

def _collect_unique_tracked_objects(mask_files):
    """
    Collect unique (sam_id_token, core_prompt_slug) pairs from mask filenames.
    Returns a dict with keys as (sam_id_token, core_prompt_slug).
    """
    unique_tracked_objects = {}
    for f_path in mask_files:
        fname = os.path.basename(f_path)
        sam_id_token, core_prompt_slug = _parse_mask_filename(fname)
        if sam_id_token is not None:
            key = (sam_id_token, core_prompt_slug)
            if key not in unique_tracked_objects:
                unique_tracked_objects[key] = {
                    "sam_id_token": sam_id_token,
                    "core_prompt_slug": core_prompt_slug
                }
        else:
            print(f"Warning: Filename {fname} did not match expected pattern.")
    return unique_tracked_objects

def _get_obj_files(temp_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Get all mask and overlay files for each object from the temp directory.
    Returns a dictionary mapping object IDs to their mask and overlay files.
    """
    mask_files = {}
    binary_dir = os.path.join(temp_dir, "binary")
    overlay_dir = os.path.join(temp_dir, "overlay")

    # Get all mask files
    mask_pattern = os.path.join(binary_dir, "*_mask.png")
    for mask_file in sorted(glob.glob(mask_pattern)):
        # Extract object ID from filename
        filename = os.path.basename(mask_file)
        match = re.match(r"(\d+)_obj(\d+)_(.*?)_mask\.png", filename)
        if match:
            frame_num, obj_id, prompt = match.groups()
            if obj_id not in mask_files:
                mask_files[obj_id] = {"mask": [], "overlay": []}
            mask_files[obj_id]["mask"].append(mask_file)

            # Get corresponding overlay file
            overlay_file = os.path.join(
                overlay_dir,
                f"{frame_num}_obj{obj_id}_{prompt}_overlay.png"
            )
            if os.path.exists(overlay_file):
                mask_files[obj_id]["overlay"].append(overlay_file)

    return mask_files

def generate_videos(
    temp_dir: str,
    fps: int,
    binary: bool = True,
    overlay: bool = True
):
    """
    Generate videos from processed frames in the temp directory.
    Creates videos for individual objects and merged outputs if available.
    """
    # Get all mask files for each object
    mask_files = _get_obj_files(temp_dir)
    if not mask_files:
        print("No mask files found for video generation.")
        return

    # Create video directories
    video_dirs = _create_video_directories(temp_dir)

    # Generate per-object videos
    _generate_per_object_videos(mask_files, video_dirs, binary, overlay, fps)

    # Generate merged videos if available
    _generate_merged_videos(temp_dir, video_dirs, binary, overlay, fps)

def _create_video_directories(temp_dir: str) -> Dict[str, str]:
    """Create and return video output directories."""
    video_binary_dir = os.path.join(temp_dir, "video", "binary")
    video_overlay_dir = os.path.join(temp_dir, "video", "overlay")
    os.makedirs(video_binary_dir, exist_ok=True)
    os.makedirs(video_overlay_dir, exist_ok=True)
    return {
        "binary": video_binary_dir,
        "overlay": video_overlay_dir
    }

def _generate_per_object_videos(
    mask_files: Dict[str, Dict[str, List[str]]],
    video_dirs: Dict[str, str],
    binary: bool,
    overlay: bool,
    fps: int
):
    """Generate videos for individual objects."""
    for obj_id, files in mask_files.items():
        if binary:
            mask_video_path = os.path.join(video_dirs["binary"], f"{obj_id}_mask.mp4")
            images_to_video(files["mask"], mask_video_path, fps)
            print(f"Generated binary mask video: {mask_video_path}")

        if overlay:
            overlay_video_path = os.path.join(video_dirs["overlay"], f"{obj_id}_overlay.mp4")
            images_to_video(files["overlay"], overlay_video_path, fps)
            print(f"Generated overlay video: {overlay_video_path}")

def _generate_merged_videos(
    temp_dir: str,
    video_dirs: Dict[str, str],
    binary: bool,
    overlay: bool,
    fps: int
):
    """Generate merged videos if available."""
    merged_binary_dir = os.path.join(temp_dir, "binary", "merged")
    merged_overlay_dir = os.path.join(temp_dir, "overlay", "merged")

    if binary and os.path.exists(merged_binary_dir):
        merged_mask_files = sorted(glob.glob(os.path.join(merged_binary_dir, "*_merged_mask.png")))
        if merged_mask_files:
            merged_mask_video = os.path.join(video_dirs["binary"], "merged_mask.mp4")
            images_to_video(merged_mask_files, merged_mask_video, fps)
            print(f"Generated merged binary mask video: {merged_mask_video}")

    if overlay and os.path.exists(merged_overlay_dir):
        merged_overlay_files = sorted(glob.glob(
            os.path.join(merged_overlay_dir, "*_merged_overlay.png")))
        if merged_overlay_files:
            merged_overlay_video = os.path.join(video_dirs["overlay"], "merged_overlay.mp4")
            images_to_video(merged_overlay_files, merged_overlay_video, fps)
            print(f"Generated merged overlay video: {merged_overlay_video}")
