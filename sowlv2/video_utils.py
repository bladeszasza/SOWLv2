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

    # Store unique combinations of (sam_id_token, core_prompt_slug)
    # sam_id_token is like "obj1", core_prompt_slug is like "dog" or "a_red_bicycle"
    unique_tracked_objects = {}

    # Regex to parse filenames like "000001_obj1_dog_mask.png"
    # or "000001_obj1_a_red_bicycle_mask.png"
    # Group 1: (obj\d+) - e.g., "obj1"
    # Group 2: ([^_]+(?:_[^_]+)*) - e.g., "dog" or "a_red_bicycle"
    filename_parser = re.compile(r"^\d+_(obj\d+)_([^_]+(?:_[^_]+)*)_mask\.png$")

    for f_path in all_mask_files:
        fname = os.path.basename(f_path)
        match = filename_parser.match(fname)
        if match:
            sam_id_token = match.group(1)  # e.g., "obj1"
            core_prompt_slug = match.group(2) # e.g., "dog" or "a_red_bicycle"

            # Use a tuple to uniquely identify the tracked object
            object_key = (sam_id_token, core_prompt_slug)
            if object_key not in unique_tracked_objects:
                unique_tracked_objects[object_key] = {
                    "sam_id_token": sam_id_token,
                    "core_prompt_slug": core_prompt_slug
                }
        else:
            print(f"Warning: Filename {fname} did not match expected pattern.")
            continue

    if not unique_tracked_objects:
        print(f"No objects successfully parsed from filenames in {output_dir}.")
        return

    for key in sorted(list(unique_tracked_objects.keys())): # Sort for deterministic order
        obj_info = unique_tracked_objects[key]
        sam_id_token = obj_info["sam_id_token"]
        core_prompt_slug = obj_info["core_prompt_slug"]

        # Construct glob patterns that include both sam_id_token and core_prompt_slug
        obj_mask_files_pattern = os.path.join(
            output_dir, f"*_{sam_id_token}_{core_prompt_slug}_mask.png")
        obj_overlay_files_pattern = os.path.join(
            output_dir, f"*_{sam_id_token}_{core_prompt_slug}_overlay.png")

        obj_mask_files = sorted(glob(obj_mask_files_pattern))
        obj_overlay_files = sorted(glob(obj_overlay_files_pattern))

        # Create a descriptive prefix for the video files
        video_file_prefix = f"{sam_id_token}_{core_prompt_slug}"

        mask_video_path = os.path.join(output_dir, f"{video_file_prefix}_mask_video.mp4")
        overlay_video_path = os.path.join(output_dir, f"{video_file_prefix}_overlay_video.mp4")

        images_to_video(obj_mask_files, mask_video_path, fps)
        images_to_video(obj_overlay_files, overlay_video_path, fps)
