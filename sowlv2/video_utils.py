# video_utils.py

import os
import cv2
from PIL import Image
import numpy as np
from glob import glob

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

    first_frame = np.array(Image.open(image_files[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_file in image_files:
        frame = np.array(Image.open(img_file))
        if len(frame.shape) == 2:  
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved video {video_path}")

def generate_per_object_videos(output_dir, fps=30):
    """
    Generate per-object videos from mask and overlay images.
    Each object will have its own video for masks and overlays.
    """
    mask_pattern = os.path.join(output_dir, "*_obj*_mask.png")
    overlay_pattern = os.path.join(output_dir, "*_obj*_overlay.png")

    mask_files = sorted(glob(mask_pattern))
    overlay_files = sorted(glob(overlay_pattern))

    objects = set()
    for f in mask_files:
        obj_id = os.path.basename(f).split('_')[1]
        objects.add(obj_id)

    for obj in sorted(objects):
        obj_mask_files = sorted(glob(os.path.join(output_dir, f"*_{obj}_mask.png")))
        obj_overlay_files = sorted(glob(os.path.join(output_dir, f"*_{obj}_overlay.png")))

        mask_video_path = os.path.join(output_dir, f"{obj}_mask_video.mp4")
        overlay_video_path = os.path.join(output_dir, f"{obj}_overlay_video.mp4")

        images_to_video(obj_mask_files, mask_video_path, fps)
        images_to_video(obj_overlay_files, overlay_video_path, fps)
