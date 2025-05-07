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
        Fast video segmentation via SAM 2’s VideoPredictor:
          1) Extract frames (@ self.fps) to a temp folder
          2) Init the SAM 2 video predictor on that folder
          3) Add OWLv2 boxes as prompts on frame 0
          4) Propagate masks through all frames in one go
        """
        import subprocess, tempfile, shutil, torch, os, numpy as np
        from sam2.build_sam import build_sam2_video_predictor

        # 1️⃣ dump frames
        tmp = tempfile.mkdtemp(prefix="sowlv2_vid_")
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-r", str(self.fps),
            os.path.join(tmp, "%06d.png"),
            "-hide_banner", "-loglevel", "error"
        ], check=True)
        print(f"🔨 frames → {tmp}")

        # 2️⃣ build SAM2 video predictor
        predictor = build_sam2_video_predictor(
            "sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
            "/content/sam2.1_hiera_s.pt",
            device=self.device
        )
        print("🤖 SAM2 VideoPredictor ready")

        # 3️⃣ initialize state on the frame folder
        with torch.inference_mode(), torch.autocast(self.device, torch.bfloat16):
            state = predictor.init_state(tmp)

            # detect on first frame only
            first_img = Image.open(os.path.join(tmp, "000000.png")).convert("RGB")
            dets = self.owl.detect(first_img, prompt, self.threshold)
            boxes = [np.array(d["box"], dtype=float) for d in dets]
            # feed boxes as new prompts
            frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                state, boxes=boxes
            )
            # save the masks for frame0
            for obj_id, mask in zip(obj_ids, masks):
                out_mask = (mask>0.5).astype(np.uint8)*255
                Image.fromarray(out_mask).save(
                  os.path.join(output_dir, f"{frame_idx:06d}_obj{obj_id}_mask.png"))
                overlay = self._create_overlay(first_img, mask>0.5)
                overlay.save(
                  os.path.join(output_dir, f"{frame_idx:06d}_obj{obj_id}_overlay.png"))

            # 4️⃣ propagate through remaining frames
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                img = Image.open(os.path.join(tmp, f"{frame_idx:06d}.png")).convert("RGB")
                for obj_id, mask in zip(obj_ids, masks):
                    out_mask = (mask>0.5).astype(np.uint8)*255
                    Image.fromarray(out_mask).save(
                      os.path.join(output_dir, f"{frame_idx:06d}_obj{obj_id}_mask.png"))
                    overlay = self._create_overlay(img, mask>0.5)
                    overlay.save(
                      os.path.join(output_dir, f"{frame_idx:06d}_obj{obj_id}_overlay.png"))

        # cleanup
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"✅ video masks & overlays saved to {output_dir}")


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
