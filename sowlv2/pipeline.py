import os
import subprocess
import shutil
import tempfile
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

    def process_video_by_frames(self, video_path, prompt, output_dir):
        """
        Fast video processing:
        1) Extract frames via FFmpeg into a temp folder at self.fps.
        2) Delegate to process_frames for OWLv2+SAM2 segmentation.
        """

        # 1️⃣ Dump frames with FFmpeg
        tmpdir = tempfile.mkdtemp(prefix="sowlv2_frames_")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-r", str(self.fps),
            os.path.join(tmpdir, "%06d.png"),
            "-hide_banner", "-loglevel", "error"
        ]
        print(f"🔨 Extracting frames @ {self.fps} FPS → {tmpdir}")
        subprocess.run(cmd, check=True)

        # 2️⃣ Run OWLv2 + SAM2 on all extracted frames
        print("🔍 Running OWLv2 + SAM2 on extracted frames …")
        self.process_frames(tmpdir, prompt, output_dir)

        # 3️⃣ Clean up
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"✅ Finished video segmentation; results in {output_dir}")


    def process_video(self, video_path, prompt, output_dir):
        """
        Segment an entire video using one pass of SAM-2 VideoPredictor.
        Steps:
          1. Extract frames via ffmpeg at self.fps to a temp dir
          2. Seed SAM-2 with OWLv2 boxes on the first frame
          3. Propagate masks through the whole clip
          4. Save binary masks + overlays
        """
        tmp = tempfile.mkdtemp(prefix="sowlv2_frames_")
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-r", str(self.fps),
             os.path.join(tmp, "%06d.png"), "-hide_banner", "-loglevel", "error"],
            check=True
        )

        vp = self.sam.video_predictor()               
        state = vp.init_state(tmp)                    # load all frames

        first_img = Image.open(os.path.join(tmp, "000000.png")).convert("RGB")
        dets = self.owl.detect(first_img, prompt, self.threshold)
        boxes = [d["box"] for d in dets]
        if not boxes:
            print(f"No '{prompt}' found in the first frame; aborting video run.")
            shutil.rmtree(tmp, ignore_errors=True)
            return

        f0, obj_ids, masks = vp.add_new_points_or_box(state, boxes=boxes)
        self._save_masks_and_overlays(first_img, f0, obj_ids, masks, output_dir)

        for fidx, obj_ids, masks in vp.propagate_in_video(state):
            img = Image.open(os.path.join(tmp, f"{fidx:06d}.png")).convert("RGB")
            self._save_masks_and_overlays(img, fidx, obj_ids, masks, output_dir)

        shutil.rmtree(tmp, ignore_errors=True)
        print(f"✅ Video segmentation done → {output_dir}")

    # helper: save masks + overlays
    def _save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        base = f"{frame_idx:06d}"
        for obj_id, mask in zip(obj_ids, masks):
            bin_mask = (mask > 0.5).astype(np.uint8) * 255
            Image.fromarray(bin_mask).save(os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png"))
            self._create_overlay(pil_img, mask > 0.5).save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            )

