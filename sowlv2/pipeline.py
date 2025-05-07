import os
import cv2
import numpy as np
from PIL import Image
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper

class SOWLv2Pipeline:
    def __init__(self, owl_model, sam_model, threshold=0.1, fps=24, device="cpu"):
        self.owl = OWLV2Wrapper(model_name=owl_model, device=device)
        self.sam = SAM2Wrapper(model_name=sam_model, device=device)
        self.threshold = threshold
        self.fps = fps

    def process_image(self, image_path, prompt, output_dir):
        """Process a single image file."""
        image = Image.open(image_path).convert("RGB")
        detections = self.owl.detect(image, prompt, self.threshold)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if not detections:
            print(f"No objects detected for prompt '{prompt}' in image '{image_path}'.")
        for idx, det in enumerate(detections):
            box = det["box"]  # [x1, y1, x2, y2]
            # Run SAM segmentation on the detected box
            mask = self.sam.segment(image, box)
            if mask is None:
                continue
            # Save binary mask
            mask_img = Image.fromarray(mask * 255).convert("L")
            mask_file = os.path.join(output_dir, f"{base_name}_object{idx}_mask.png")
            mask_img.save(mask_file)
            # Create and save overlay image
            overlay = self._create_overlay(image, mask)
            overlay_file = os.path.join(output_dir, f"{base_name}_object{idx}_overlay.png")
            overlay.save(overlay_file)

    def process_frames(self, folder_path, prompt, output_dir):
        """Process a folder of images (frames)."""
        files = sorted(os.listdir(folder_path))
        for fname in files:
            infile = os.path.join(folder_path, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                continue
            self.process_image(infile, prompt, output_dir)

    def process_video(self, video_path, prompt, output_dir):
        """Process a video file by sampling frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(orig_fps / self.fps)) if self.fps > 0 else 1
        count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_interval > 0 and (count % frame_interval == 0):
                # Process this frame
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detections = self.owl.detect(img, prompt, self.threshold)
                base_name = f"{video_name}_frame{frame_idx:06d}"
                if not detections:
                    print(f"No objects detected for prompt '{prompt}' in frame {frame_idx}.")
                for idx, det in enumerate(detections):
                    box = det["box"]
                    mask = self.sam.segment(img, box)
                    if mask is None:
                        continue
                    mask_img = Image.fromarray(mask * 255).convert("L")
                    mask_file = os.path.join(output_dir, f"{base_name}_object{idx}_mask.png")
                    mask_img.save(mask_file)
                    overlay = self._create_overlay(img, mask)
                    overlay_file = os.path.join(output_dir, f"{base_name}_object{idx}_overlay.png")
                    overlay.save(overlay_file)
                frame_idx += 1
            count += 1
        cap.release()

    def _create_overlay(self, image, mask):
        """Return an overlay image by blending a red mask with the original image."""
        image_np = np.array(image).astype(np.uint8)
        mask_color = np.zeros_like(image_np)
        mask_color[..., 0] = 255  # red color for mask
        # Create a 3-channel boolean mask
        mask_bool = np.stack([mask == 1]*3, axis=-1)
        # Blend original and mask color
        overlay_np = image_np.copy()
        overlay_np[mask_bool] = (0.5 * overlay_np[mask_bool] + 0.5 * mask_color[mask_bool]).astype(np.uint8)
        return Image.fromarray(overlay_np)
