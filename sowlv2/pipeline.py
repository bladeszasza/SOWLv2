import os
import subprocess
import shutil
import tempfile
import numpy as np
from PIL import Image
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper

_FIRST_FRAME = "000001.jpg"
_FIRST_FRAME_IDX = 0

class SOWLv2Pipeline:
    
    def __init__(self, owl_model, sam_model, threshold=0.4, fps=24, device="cuda"):
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
        """Process a single video file with SAM 2."""
        tmp = tempfile.mkdtemp(prefix="sowlv2_frames_")
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-r", str(self.fps),
             os.path.join(tmp, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
            check=True
        )

        state = self.sam.init_state(tmp)           

        first_img = Image.open(os.path.join(tmp, _FIRST_FRAME)).convert("RGB")
        detections = self.owl.detect(first_img, prompt, self.threshold)

        if not detections:
            print(f"No '{prompt}' objects in first frame — aborting.")
            shutil.rmtree(tmp, ignore_errors=True)
            return
            
        frame_idx = _FIRST_FRAME_IDX
        obj_id_counter = 1 
        
        for detection in detections:
            boxes = detection["box"] if isinstance(detection["box"][0], (list, tuple)) else [detection["box"]]
            for box in boxes:
                box_array = np.array(box, dtype=np.float32)
                frame_idx, obj_ids, masks = self.sam.add_new_box(
                    state=state,
                    frame_idx=frame_idx,
                    box=box_array,
                    obj_idx=obj_id_counter
                )
                obj_id_counter += 1
    # for obj_id, mask_logit in zip(obj_ids, mask_logits):
    #     mask = (mask_logit > 0.0).cpu().numpy().astype(np.uint8)
        for fidx, obj_ids, mask_logits in self.sam.propagate_in_video(state):
            frame_idx = fidx+1
            img = Image.open(os.path.join(tmp, f"{frame_idx:06d}.jpg")).convert("RGB")
            self._video_save_masks_and_overlays(img, frame_idx, obj_ids, mask_logits, output_dir)

        shutil.rmtree(tmp, ignore_errors=True)
        print(f"✅ Video segmentation finished; results in {output_dir}")

    def _save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        """Write <frame>_obj<id>_mask.png and _overlay.png files."""
        base = f"{frame_idx:06d}"
        for obj_id, mask in zip(obj_ids, masks):
            mask_bin = ((mask > 0.5).astype(np.uint8)) * 255
            Image.fromarray(mask_bin).save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            )
            overlay = self._create_overlay(pil_img, mask > 0.5)
            overlay.save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            )
            
    def _video_save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        """Process and store masks and overlays for video generation."""
        base = f"{frame_idx:06d}"
        mask_frames = []
        overlay_frames = []
    
        for obj_id, mask in zip(obj_ids, masks):
            # Convert mask to binary
            mask_bin = ((mask > 0.5).cpu().numpy().astype(np.uint8)) * 255
            mask_bin = np.squeeze(mask_bin)  # Removes dimensions of size 1

            # Ensure mask_bin is 2D
            if mask_bin.ndim != 2:
                raise ValueError(f"mask_bin has unexpected number of dimensions: {mask_bin.ndim}")

            mask_pil = Image.fromarray(mask_bin)
    
            # Save individual mask image
            mask_path = os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            mask_pil.save(mask_path)
    
            # Create and save overlay image
            overlay = self._create_overlay(pil_img, mask > 0.5)
            overlay_path = os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            overlay.save(overlay_path)
    
            # Store frames for video
            mask_frames.append(mask_pil)
            overlay_frames.append(overlay)
        return mask_frames, overlay_frames

            
    def _create_overlay(self, image, mask):
        """Return an overlay image by blending a red mask with the original image."""
        image_np = np.array(image).astype(np.uint8)
    
        # Ensure mask is on CPU and convert to NumPy
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
    
        mask_color = np.zeros_like(image_np)
        mask_color[..., 0] = 255  # Red color for mask
    
        # Create a 3-channel boolean mask
        mask_bool = np.stack([mask == 1] * 3, axis=-1)
    
        # Blend original and mask color
        overlay_np = image_np.copy()
        overlay_np[mask_bool] = (
            0.5 * overlay_np[mask_bool] + 0.5 * mask_color[mask_bool]
        ).astype(np.uint8)
    
        return Image.fromarray(overlay_np)

    
