"""
Core pipeline for SOWLv2, combining OWLv2 for detection
and SAM2 for segmentation.
"""
import os
import subprocess
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
from sowlv2 import video_utils
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper

_FIRST_FRAME = "000001.jpg"
_FIRST_FRAME_IDX = 0

class SOWLv2Pipeline:
    """
    SOWLv2 pipeline for object detection and segmentation in images and videos.
    """
    def __init__(self, owl_model, sam_model, threshold=0.4, fps=24, device="cuda"):
        """
        Initialize the SOWLv2 pipeline.

        Args:
            owl_model (str): Name of the OWLv2 model.
            sam_model (str): Name of the SAM2 model.
            threshold (float): Detection confidence threshold.
            fps (int): Frames per second for video processing.
            device (str): PyTorch device ('cuda' or 'cpu').
        """
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

    def process_video(self, video_path, prompt, output_dir): # pylint: disable=too-many-locals
        """Process a single video file with SAM 2."""
        tmp = tempfile.mkdtemp(prefix="sowlv2_frames_")
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-r", str(self.fps),
                 os.path.join(tmp, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
                check=True
            )

            state = self.sam.init_state(tmp)

            first_img_path = os.path.join(tmp, _FIRST_FRAME)
            if not os.path.exists(first_img_path):
                print(f"First frame {_FIRST_FRAME} not found in {tmp}. "
                      "Video might be too short or ffmpeg failed.")
                return

            first_img = Image.open(first_img_path).convert("RGB")
            detections = self.owl.detect(first_img, prompt, self.threshold)

            if not detections:
                print(f"No '{prompt}' objects in first frame — aborting.")
                return

            frame_idx_init = _FIRST_FRAME_IDX
            obj_id_counter = 1

            for detection in detections:
                boxes = detection["box"]
                # Ensure boxes is a list of lists/tuples for consistency
                if not isinstance(boxes[0], (list, tuple)):
                    boxes = [boxes]

                for box in boxes:
                    # box_array = np.array(box, dtype=np.float32)
                    _, _, _ = self.sam.add_new_box( # frame_idx, obj_ids returned not used here
                        state=state,
                        frame_idx=frame_idx_init,
                        box=box,
                        obj_idx=obj_id_counter
                    )
                    obj_id_counter += 1

            for fidx, obj_ids, mask_logits in self.sam.propagate_in_video(state):
                current_frame_idx = fidx + 1
                frame_file = os.path.join(tmp, f"{current_frame_idx:06d}.jpg")
                if not os.path.exists(frame_file):
                    print(f"Warning: Frame {frame_file} not found. Skipping.")
                    continue
                img = Image.open(frame_file).convert("RGB")
                self._video_save_masks_and_overlays(
                    img, current_frame_idx, obj_ids, mask_logits, output_dir
                )
            print(f"✅ Video segmentation finished; results in {output_dir}")
            video_utils.generate_per_object_videos(output_dir, fps=self.fps)

        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"✅ Video generation finished; results in {output_dir}")

    def _save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        """
        Write <frame>_obj<id>_mask.png and _overlay.png files.
        Args:
            pil_img (PIL.Image): The original image.
            frame_idx (int): The frame index.
            obj_ids (list): List of object IDs.
            masks (list/np.ndarray): List of masks corresponding to obj_ids.
            out_dir (str): Output directory.
        """
        base = f"{frame_idx:06d}"
        for obj_id, mask_data in zip(obj_ids, masks):
            mask_bin = ((mask_data > 0.5).astype(np.uint8)) * 255
            Image.fromarray(mask_bin).save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            )
            overlay = self._create_overlay(pil_img, mask_data > 0.5)
            overlay.save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            )

    def _video_save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        # pylint: disable=too-many-locals
        """
        Process and store masks and overlays for video generation.
        Args:
            pil_img (PIL.Image): The original image for the current frame.
            frame_idx (int): The index of the current frame.
            obj_ids (list/torch.Tensor): Tensor or list of object IDs for the masks.
            masks (torch.Tensor): Tensor containing mask logits or probabilities.
            out_dir (str): Directory to save the mask and overlay images.
        Returns:
            tuple: (list_of_mask_pil_images, list_of_overlay_pil_images)
        """
        base = f"{frame_idx:06d}"
        mask_frames_pil = []
        overlay_frames_pil = []

        # Ensure obj_ids is a list or 1D tensor
        if isinstance(obj_ids, torch.Tensor):
            obj_ids_list = obj_ids.cpu().tolist()
        else: # Assuming it's already a list-like structure
            obj_ids_list = list(obj_ids)


        for i, obj_id in enumerate(obj_ids_list):
            # Assuming masks tensor is (num_objects, height, width) or (num_objects, 1, H, W)
            mask = masks[i]
            mask_bin = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255
            mask_bin = np.squeeze(mask_bin)  # Removes dimensions of size 1

            if mask_bin.ndim != 2:
                raise ValueError(
                    f"mask_bin has unexpected number of dimensions: {mask_bin.ndim}"
                )

            mask_pil = Image.fromarray(mask_bin).convert("L")

            mask_path = os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            mask_pil.save(mask_path)

            # Create overlay using the mask
            # before converting to binary (for smoother edges if needed)
            # Or use the binary mask if that's the desired visual
            overlay_mask_np = (mask > 0.5).cpu().numpy()
            overlay = self._create_overlay(pil_img, overlay_mask_np)
            overlay_path = os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            overlay.save(overlay_path)

            mask_frames_pil.append(mask_pil)
            overlay_frames_pil.append(overlay)
        return mask_frames_pil, overlay_frames_pil


    def _create_overlay(self, image, mask):
        """
        Blend a red mask with the input PIL.Image and return the overlay as a PIL.Image.
        Works whether `mask` is NumPy or a PyTorch tensor and regardless of
        leading singleton dimensions.
        """
        # 1. Make sure we have a NumPy binary mask on CPU and squeeze extra dims
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()          # move off GPU
        mask_squeezed = np.squeeze(mask)                         # drop dimensions of size 1
        if mask_squeezed.ndim != 2:
            raise ValueError(f"Expected 2-D mask, got shape {mask_squeezed.shape}")

        # 2. Prepare image & output
        image_np   = np.asarray(image, dtype=np.uint8)
        overlay_np = image_np.copy()

        # 3. Boolean index with a 2-D mask – NumPy broadcasts the channel dim
        red = np.array([255, 0, 0], dtype=np.uint8)
        # Ensure mask is boolean for indexing
        bool_mask = mask_squeezed > 0 if mask_squeezed.dtype != bool else mask_squeezed

        # Apply color blending
        # Ensure overlay_np has 3 channels if it's RGB
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            overlay_np[bool_mask] = (
                0.5 * overlay_np[bool_mask] + 0.5 * red
            ).astype(np.uint8)
        elif image_np.ndim == 2: # Grayscale image, make it RGB to add red mask
            overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_GRAY2RGB)
            overlay_np[bool_mask] = (
                0.5 * overlay_np[bool_mask] + 0.5 * red
            ).astype(np.uint8)
        else: # For other cases,e.g. RGBA, handle appropriately or raise error
            print("Warning: Overlay for images not in RGB/Grayscale might not be accurate.")
            # Simple red overlay for non-RGB images (first 3 channels)
            if overlay_np.shape[2] > 3: # e.g. RGBA
                overlay_np[bool_mask, :3] = (
                    0.5 * overlay_np[bool_mask, :3] + 0.5 * red
                ).astype(np.uint8)
            # else, if not 2D or 3D, it is an issue already caught or to be handled.

        return Image.fromarray(overlay_np)
