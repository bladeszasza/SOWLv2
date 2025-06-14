"""
Utilities for video processing, such as converting image frames to video
and generating per-object mask/overlay videos.
"""
import os
import glob
import re
from typing import List, Dict
import cv2  # pylint: disable=import-error

from sowlv2.utils.path_config import (
    FilePattern, DirectoryStructure, FilePatternMatcher
)

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


def _get_obj_files(temp_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Get all mask and overlay files for each object from the temp directory.
    Returns a dictionary mapping object IDs to their mask and overlay files.
    """
    mask_files = {}

    # Get merged files
    binary_merged_dir = os.path.join(
        temp_dir, DirectoryStructure.BINARY, DirectoryStructure.MERGED)
    overlay_merged_dir = os.path.join(
        temp_dir, DirectoryStructure.OVERLAY, DirectoryStructure.MERGED)

    # Get merged mask files
    if os.path.exists(binary_merged_dir):
        mask_pattern = os.path.join(binary_merged_dir, "*_merged_mask.png")
        for mask_file in sorted(
                glob.glob(mask_pattern), key=FilePatternMatcher.natural_sort_key):
            filename = os.path.basename(mask_file)
            match = re.match(FilePatternMatcher.get_merged_mask_pattern(), filename)
            if match:
                frame_num = match.group(1)
                obj_id = "merged"
                if obj_id not in mask_files:
                    mask_files[obj_id] = {"mask": [], "overlay": []}
                mask_files[obj_id]["mask"].append(mask_file)

                # Get corresponding overlay file
                overlay_file = os.path.join(
                    overlay_merged_dir,
                    FilePattern.MERGED_OVERLAY.format(frame_num=frame_num)
                )
                if os.path.exists(overlay_file):
                    mask_files[obj_id]["overlay"].append(overlay_file)

    # Get individual object files
    binary_frames_dir = os.path.join(
        temp_dir, DirectoryStructure.BINARY, DirectoryStructure.FRAMES)
    overlay_frames_dir = os.path.join(
        temp_dir, DirectoryStructure.OVERLAY, DirectoryStructure.FRAMES)

    if os.path.exists(binary_frames_dir):
        # Get individual mask files
        mask_pattern = os.path.join(binary_frames_dir, "*_obj*_*_mask.png")
        for mask_file in sorted(
                glob.glob(mask_pattern), key=FilePatternMatcher.natural_sort_key):
            filename = os.path.basename(mask_file)
            match = re.match(FilePatternMatcher.get_individual_mask_pattern(), filename)
            if match:
                frame_num = match.group(1)
                obj_num = match.group(2)
                prompt = match.group(3)
                obj_id = f"obj{obj_num}"

                if obj_id not in mask_files:
                    mask_files[obj_id] = {"mask": [], "overlay": []}
                mask_files[obj_id]["mask"].append(mask_file)

                # Get corresponding overlay file
                overlay_file = os.path.join(
                    overlay_frames_dir,
                    FilePattern.INDIVIDUAL_OVERLAY.format(
                        frame_num=frame_num, obj_id=obj_num, prompt=prompt
                    )
                )
                if os.path.exists(overlay_file):
                    mask_files[obj_id]["overlay"].append(overlay_file)

    return mask_files

def generate_videos(
    temp_dir: str,
    fps: int,
    binary: bool = True,
    overlay: bool = True,
    merged: bool = True
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

    # Generate videos for each object (including individual objects and merged)
    for obj_id, files in mask_files.items():
        if obj_id == "merged":
            # Only generate merged videos if merged flag is True
            if merged:
                _generate_videos_for_object(
                    obj_id, files, video_dirs, binary, overlay, fps)
        else:
            # Always generate individual object videos (controlled by binary/overlay flags)
            _generate_videos_for_object(
                obj_id, files, video_dirs, binary, overlay, fps)

def _create_video_directories(temp_dir: str) -> Dict[str, str]:
    """Create and return video output directories."""
    dirs = DirectoryStructure.get_directory_map(temp_dir, include_video=True)
    return {
        "binary": dirs[f"{DirectoryStructure.VIDEO}_binary"],
        "overlay": dirs[f"{DirectoryStructure.VIDEO}_overlay"]
    }

def _generate_videos_for_object(
    obj_id: str,
    files: Dict[str, List[str]],
    video_dirs: Dict[str, str],
    binary: bool,
    overlay: bool,
    fps: int
):
    """Generate videos for a specific object (individual or merged)."""
    # Generate binary mask video
    if binary and files.get("mask"):
        if obj_id == "merged":
            video_filename = FilePattern.VIDEO_MERGED_MASK
        else:
            video_filename = FilePattern.VIDEO_MASK.format(obj_id=obj_id)

        mask_video_path = os.path.join(video_dirs["binary"], video_filename)
        images_to_video(files["mask"], mask_video_path, fps)
        print(f"Generated binary mask video: {mask_video_path}")

    # Generate overlay video
    if overlay and files.get("overlay"):
        if obj_id == "merged":
            video_filename = FilePattern.VIDEO_MERGED_OVERLAY
        else:
            video_filename = FilePattern.VIDEO_OVERLAY.format(obj_id=obj_id)

        overlay_video_path = os.path.join(video_dirs["overlay"], video_filename)
        images_to_video(files["overlay"], overlay_video_path, fps)
        print(f"Generated overlay video: {overlay_video_path}")
