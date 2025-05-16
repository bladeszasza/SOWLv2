"""
Utilities for video processing, such as converting image frames to video
and generating per-object mask/overlay videos.
"""
import os
from glob import glob
import re
import cv2  # pylint: disable=import-error
from PIL import Image
import numpy as np


# Disable no-member for cv2 (OpenCV) for the whole file
# pylint: disable=no-member


def images_to_video(image_files, video_path, fps=30):
    """
    Convert a list of image files to a video file.
    Args:
        image_files (list): List of image file paths.
        video_path (str): Output video file path.
        fps (int): Frames per second for the output video.
    """
    if not image_files:
        print(f"No frames for video {video_path}, skipping.")
        return

    try:
        first_frame_pil = Image.open(image_files[0])
        first_frame = np.array(first_frame_pil)
    except FileNotFoundError:
        print(f"Error: First frame {image_files[0]} not found for video {video_path}. Skipping.")
        return
    except Exception as e: # pylint: disable=broad-except
        print(f"Error opening first frame {image_files[0]}: {e}. Skipping video {video_path}.")
        return


    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_file in image_files:
        try:
            frame_pil = Image.open(img_file)
            frame = np.array(frame_pil)
            if len(frame.shape) == 2: # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4: # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3: # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # PIL is RGB, OpenCV is BGR
            else:
                print(f"Warning: Frame {img_file} has unexpected shape {frame.shape}. Skipping.")
                continue
            video_writer.write(frame)
        except FileNotFoundError:
            print(f"Warning: Frame {img_file} not found during video creation. Skipping frame.")
            continue
        except Exception as e: # pylint: disable=broad-except
            print(f"Error processing frame {img_file}: {e}. Skipping frame.")
            continue


    video_writer.release()
    print(f"Saved video {video_path}")

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

def _get_obj_files(output_dir, sam_id_token, core_prompt_slug):
    """
    Get sorted mask and overlay files for a given object.
    """
    if core_prompt_slug:
        mask_pattern = os.path.join(
            output_dir, f"*_{sam_id_token}_{core_prompt_slug}_mask.png")
        overlay_pattern = os.path.join(
            output_dir, f"*_{sam_id_token}_{core_prompt_slug}_overlay.png")
        video_prefix = f"{sam_id_token}_{core_prompt_slug}"
    else:
        mask_pattern = os.path.join(output_dir, f"*_{sam_id_token}_mask.png")
        overlay_pattern = os.path.join(output_dir, f"*_{sam_id_token}_overlay.png")
        video_prefix = sam_id_token
    mask_files = sorted(glob(mask_pattern))
    overlay_files = sorted(glob(overlay_pattern))
    return mask_files, overlay_files, video_prefix

def generate_per_object_videos(output_dir, fps=30):
    """
    Generate per-object videos from mask and overlay images.
    Each object (identified by sam_id and core_prompt) will have its own
    video for masks and overlays.
    """
    all_mask_files_pattern = os.path.join(output_dir, "*_mask.png")
    all_mask_files = sorted(glob(all_mask_files_pattern))

    if not all_mask_files:
        print(f"No mask files found in {output_dir} matching pattern.")
        return

    unique_tracked_objects = _collect_unique_tracked_objects(all_mask_files)
    if not unique_tracked_objects:
        print(f"No objects successfully parsed from filenames in {output_dir}.")
        return

    for key in sorted(unique_tracked_objects.keys()):
        obj_info = unique_tracked_objects[key]
        sam_id_token = obj_info["sam_id_token"]
        core_prompt_slug = obj_info["core_prompt_slug"]

        mask_files, overlay_files, video_file_prefix = _get_obj_files(
            output_dir, sam_id_token, core_prompt_slug
        )

        mask_video_path = os.path.join(output_dir, f"{video_file_prefix}_mask_video.mp4")
        overlay_video_path = os.path.join(output_dir, f"{video_file_prefix}_overlay_video.mp4")

        images_to_video(mask_files, mask_video_path, fps)
        images_to_video(overlay_files, overlay_video_path, fps)
