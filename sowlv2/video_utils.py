"""
Utilities for video processing, such as converting image frames to video
and generating per-object mask/overlay videos.
"""

import os
from glob import glob
import shutil # For managing temporary directories
import tempfile
import logging
import cv2  # pylint: disable=import-error
from PIL import Image
import numpy as np

# Disable no-member for cv2 (OpenCV) for the whole file
# pylint: disable=no-member

# Consistent color palette for objects
# Simple palette: R, G, B, Yellow, Cyan, Magenta, etc.
# Ensure it's accessible or can be passed to where it's needed.
DEFAULT_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (0, 128, 255), (128, 0, 255),
]

def get_object_color(obj_id_str, color_map, palette=None):
    """Assigns a consistent color to an object ID."""
    if palette is None:
        palette = DEFAULT_PALETTE
    # obj_id_str is like "obj1", "obj2", etc.
    try:
        # Extract numeric part for consistent indexing if obj_id_str is "objNUMBER"
        numeric_id = int(obj_id_str.replace("obj", ""))
        if numeric_id not in color_map:
            color_map[numeric_id] = palette[len(color_map) % len(palette)]
        return color_map[numeric_id]
    except ValueError: # Fallback for non-standard obj_id formats
        if obj_id_str not in color_map:
            color_map[obj_id_str] = palette[len(color_map) % len(palette)]
        return color_map[obj_id_str]





def images_to_video(image_files, video_path, fps=30, width=None, height=None):
    """
    Convert a list of image files to a video file.
    Args:
        image_files (list): List of image file paths, sorted.
        video_path (str): Output video file path.
        fps (int): Frames per second for the output video.
        width (int, optional): Video width. If None, taken from first frame.
        height (int, optional): Video height. If None, taken from first frame.
    """
    if not image_files:
        logging.warning("No frames for video %s, skipping.", video_path)
        return

    if width is None or height is None:
        try:
            first_frame_img = Image.open(image_files[0])
            if width is None:
                width = first_frame_img.width
            if height is None:
                height = first_frame_img.height
        except (OSError, ValueError) as e:
            logging.error(
                "Could not read first frame %s to get dimensions: %s",
                image_files[0], e)
            return


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
    logging.info("Saved video %s", video_path)

def generate_per_object_videos(output_dir, fps=30):
    """
    Generate per-object videos from mask and overlay images.
    Each object will have its own video for masks and overlays.
    """

    # obj_ids are extracted from filenames like "000001_obj1_mask.png"
    all_pngs = glob(os.path.join(output_dir, "*_obj*_*.png"))
    if not all_pngs:
        logging.info(
            "No per-object PNG files found in %s to generate videos.", output_dir)
        return

    objects = set()
    for f in all_pngs:
        try:
            # Assuming format like <frame>_obj<ID>_mask.png or <frame>_obj<ID>_overlay.png
            obj_part = os.path.basename(f).split('_')[1] # Should be "objX"
            if obj_part.startswith("obj"):
                objects.add(obj_part)
        except IndexError:
            logging.warning("Could not parse object ID from filename %s", f)
            continue

    if not objects:
        logging.info("No object IDs parsed from PNG files in %s.", output_dir)
        return

    for obj_str_id in sorted(list(objects)): # e.g., "obj1", "obj10"
        obj_mask_files = sorted(glob(os.path.join(output_dir, f"*_{obj_str_id}_mask.png")))
        obj_overlay_files = sorted(glob(os.path.join(output_dir, f"*_{obj_str_id}_overlay.png")))

        if obj_mask_files:
            mask_video_path = os.path.join(output_dir, f"{obj_str_id}_mask_video.mp4")
            images_to_video(obj_mask_files, mask_video_path, fps)
        if obj_overlay_files:
            overlay_video_path = os.path.join(output_dir, f"{obj_str_id}_overlay_video.mp4")
            images_to_video(obj_overlay_files, overlay_video_path, fps)

def generate_combined_mask_video(output_dir, video_path, fps, frame_count, img_height, img_width):
    """
    Generates a single video showing the union of all object masks per frame.
    """
    logging.info("Generating combined mask video: %s", video_path)
    temp_combined_mask_frames_dir = tempfile.mkdtemp(prefix="sowlv2_merged_mask_")
    combined_mask_frame_files = []

    for frame_num in range(1, frame_count + 1):
        base_frame_name = f"{frame_num:06d}"
        per_object_mask_files_for_frame = sorted(glob
                                                 (os.path.join(output_dir,
                                                    f"{base_frame_name}_obj*_mask.png")))

        combined_mask_np = np.zeros((img_height, img_width), dtype=np.uint8)

        if per_object_mask_files_for_frame:
            for mask_file in per_object_mask_files_for_frame:
                try:
                    mask_img_np = np.array(Image.open(mask_file).convert("L")) # L for grayscale
                    combined_mask_np = np.logical_or(combined_mask_np,
                                            (mask_img_np > 128)).astype(np.uint8)
                except (OSError, ValueError) as e:
                    logging.error("Error processing mask file %s for combined mask video: %s",
                                  mask_file, e)
                    continue

        combined_mask_pil = Image.fromarray(combined_mask_np * 255)
        out_frame_path = os.path.join(temp_combined_mask_frames_dir, f"{base_frame_name}.png")
        combined_mask_pil.save(out_frame_path)
        combined_mask_frame_files.append(out_frame_path)

    if combined_mask_frame_files:
        images_to_video(combined_mask_frame_files,
                        video_path, fps, width=img_width,
                        height=img_height)
    else:
        logging.warning("No combined mask frames generated for %s", video_path)

    if os.path.exists(temp_combined_mask_frames_dir):
        shutil.rmtree(temp_combined_mask_frames_dir)

def generate_combined_overlay_video(output_dir, original_frames_dir, video_path, fps, frame_count):
    """
    Generates a single video showing the original frames with all object masks overlaid,
    each object having a distinct color.
    """
    logging.info("Generating combined overlay video: %s", video_path)
    temp_combined_overlay_frames_dir = tempfile.mkdtemp(prefix="sowlv2_merged_overlay_")
    combined_overlay_frame_files = []

    # Determine all unique object string IDs (e.g., "obj1", "obj2") from mask filenames
    all_mask_files = glob(os.path.join(output_dir, "*_obj*_mask.png"))
    unique_obj_str_ids = sorted(list(set(
        os.path.basename(f).split('_')[1]
        for f in all_mask_files if f.split('_')[1].startswith("obj")
    )))

    # Create a color map for these string IDs
    color_map_for_obj_str = {obj_str_id: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
                             for i, obj_str_id in enumerate(unique_obj_str_ids)}

    img_width, img_height = None, None

    for frame_num in range(1, frame_count + 1):
        base_frame_name = f"{frame_num:06d}"
        original_frame_path = os.path.join(original_frames_dir,
                                           f"{base_frame_name}.jpg")

        if not os.path.exists(original_frame_path):
            logging.warning("Original frame %s not found for combined overlay. Skipping.",
                            original_frame_path)
            # Create a blank frame of the correct size if dimensions are known, otherwise skip
            if img_width and img_height:
                blank_frame_np = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                Image.fromarray(blank_frame_np).save(
                    os.path.join(temp_combined_overlay_frames_dir, f"{base_frame_name}.png"))
                combined_overlay_frame_files.append(
                    os.path.join(temp_combined_overlay_frames_dir, f"{base_frame_name}.png"))
            continue

        try:
            original_pil_img = Image.open(original_frame_path).convert("RGB")
            if img_width is None: # Get dimensions from first successfully loaded frame
                img_width, img_height = original_pil_img.width, original_pil_img.height

            combined_overlay_np = np.array(original_pil_img)

            per_object_mask_files_for_frame = sorted(glob(os.path.join(output_dir,
                                            f"{base_frame_name}_obj*_mask.png")))

            if per_object_mask_files_for_frame:
                for mask_file in per_object_mask_files_for_frame:
                    try:
                        obj_str_id_part = os.path.basename(mask_file).split('_')[1]
                        if not obj_str_id_part.startswith("obj"):
                            continue

                        color = color_map_for_obj_str.get(obj_str_id_part,
                                                          (100, 100, 100)) # Default gray

                        mask_img_np = np.array(Image.open(mask_file).
                                               convert("L")) > 128 # Boolean mask

                        # Apply blending
                        for c in range(3): # For R, G, B channels
                            channel_overlay = combined_overlay_np[:,:,c]
                            channel_color = color[c]
                            channel_overlay[mask_img_np] = (0.5 * channel_overlay[mask_img_np]
                                                             + 0.5 * channel_color).astype(np.uint8)
                            combined_overlay_np[:,:,c] = channel_overlay

                    except (OSError, ValueError) as e:
                        logging.error(
                            "Error processing mask file %s for combined overlay: %s",
                                       mask_file, e)
                        continue

            out_frame_path = os.path.join(
                temp_combined_overlay_frames_dir, f"{base_frame_name}.png")
            Image.fromarray(combined_overlay_np).save(out_frame_path)
            combined_overlay_frame_files.append(out_frame_path)
        except (OSError, ValueError) as e:
            logging.error(
                "Error processing original frame %s for combined overlay: %s",
                original_frame_path, e)
            # Create a blank frame if processing fails but we have dimensions
            if img_width and img_height:
                blank_frame_np = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                Image.fromarray(blank_frame_np).save(os.path.join(
                    temp_combined_overlay_frames_dir, f"{base_frame_name}.png"))
                combined_overlay_frame_files.append(os.path.join(
                    temp_combined_overlay_frames_dir, f"{base_frame_name}.png"))


    if combined_overlay_frame_files:
        images_to_video(combined_overlay_frame_files,
                        video_path,
                        fps,
                        width=img_width,
                        height=img_height)
    else:
        logging.warning("No combined overlay frames generated for %s", video_path)

    if os.path.exists(temp_combined_overlay_frames_dir):
        shutil.rmtree(temp_combined_overlay_frames_dir)
