"""
Utilities for video processing, such as converting image frames to video
and generating per-object mask/overlay videos.
"""
import os
from glob import glob
import cv2
from PIL import Image
import numpy as np


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
    Each object will have its own video for masks and overlays.
    """
    mask_pattern = os.path.join(output_dir, "*_obj*_mask.png")
    overlay_pattern = os.path.join(output_dir, "*_obj*_overlay.png") # pylint: disable=unused-variable

    mask_files = sorted(glob(mask_pattern))
    # overlay_files = sorted(glob(overlay_pattern)) # This variable is unused.

    objects = set()
    for f in mask_files:
        try:
            # Assuming filename format like '000001_obj1_mask.png'
            obj_id = os.path.basename(f).split('_')[1] # Extracts 'obj1'
            objects.add(obj_id)
        except IndexError:
            print(f"Warning: Could not parse object ID from filename {f}. Skipping.")
            continue


    for obj in sorted(list(objects)): # Convert set to sorted list for deterministic order
        obj_mask_files = sorted(glob(os.path.join(output_dir, f"*_{obj}_mask.png")))
        obj_overlay_files = sorted(glob(os.path.join(output_dir, f"*_{obj}_overlay.png")))

        mask_video_path = os.path.join(output_dir, f"{obj}_mask_video.mp4")
        overlay_video_path = os.path.join(output_dir, f"{obj}_overlay_video.mp4")

        images_to_video(obj_mask_files, mask_video_path, fps)
        images_to_video(obj_overlay_files, overlay_video_path, fps)