"""
Core pipeline for SOWLv2, combining OWLv2 for detection
and SAM2 for segmentation.
"""
import os
import subprocess
import shutil
import tempfile
from typing import Union, List, Dict, Tuple
import cv2  # pylint: disable=import-error
import numpy as np
from PIL import Image
import torch
from sowlv2 import video_utils
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper
from sowlv2.data.config import PipelineBaseData, MergedOverlayItem, DetectionResult


# Disable no-member for cv2 (OpenCV) for the whole file
# pylint: disable=no-member

_FIRST_FRAME = "000001.jpg"
_FIRST_FRAME_IDX = 0

DEFAULT_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (0, 128, 255), (128, 0, 255), (192, 192, 192), (255, 215, 0),
    (138, 43, 226),(75, 0, 130), (240, 128, 128), (32, 178, 170)
]


class SOWLv2Pipeline:
    """
    SOWLv2 pipeline for object detection and segmentation in images and videos.

    This class integrates OWLv2 for open-vocabulary object detection and SAM2 for
    segmentation, supporting both images and videos. It assigns unique colors to
    each detected label for clear visualization in overlays.
    """
    def __init__(self, config: PipelineBaseData = None):
        """
        Initialize the SOWLv2 pipeline.

        Args:
            config (PipelineBaseData, optional): Configuration dataclass for the pipeline.
                If None, uses default values.
        """
        if config is None:
            config = PipelineBaseData(
                owl_model="google/owlv2-base-patch16-ensemble",
                sam_model="facebook/sam2.1-hiera-small",
                threshold=0.1,
                fps=24,
                device="cuda"
            )
        self.config = config
        self.owl = OWLV2Wrapper(model_name=self.config.owl_model, device=self.config.device)
        self.sam = SAM2Wrapper(model_name=self.config.sam_model, device=self.config.device)
        self.palette = DEFAULT_PALETTE
        self.prompt_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.next_color_idx = 0

    def _get_color_for_prompt(self, core_prompt: str) -> Tuple[int, int, int]:
        """
        Assigns a consistent color to a core prompt term.

        Args:
            core_prompt (str): The label or prompt for which to get a color.

        Returns:
            Tuple[int, int, int]: RGB color tuple.
        """
        if core_prompt not in self.prompt_color_map:
            color = self.palette[self.next_color_idx % len(self.palette)]
            self.prompt_color_map[core_prompt] = color
            self.next_color_idx += 1
        return self.prompt_color_map[core_prompt]

    def process_image(self, image_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Process a single image file: detect objects, segment them, and save masks/overlays.

        Args:
            image_path (str): Path to the input image.
            prompt (Union[str, List[str]]): Text prompt(s) for object detection.
            output_dir (str): Directory to save output masks and overlays.
        """
        pil_image = Image.open(image_path).convert("RGB")
        detections = self.owl.detect(
            image=pil_image,
            prompt=prompt,
            threshold=self.config.threshold)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if not detections:
            print(f"No objects detected for prompt(s) '{prompt}' in image '{image_path}'.")
            return

        merged_overlay_image = pil_image.copy()
        processed_detections_for_merge: List[MergedOverlayItem] = []
        detection_results: List[DetectionResult] = []

        for idx, det in enumerate(detections):
            box = det["box"]
            core_prompt = det["core_prompt"]
            object_color = self._get_color_for_prompt(core_prompt)

            mask_np = self.sam.segment(pil_image, box)
            if mask_np is None:
                print(f"Warning: SAM2 failed to segment object {idx} ({core_prompt}). Skipping.")
                continue

            mask_img_pil = Image.fromarray(mask_np * 255).convert("L")
            mask_file = os.path.join(
                output_dir, f"{base_name}_{core_prompt.replace(' ','_')}_{idx}_mask.png")
            mask_img_pil.save(mask_file)

            individual_overlay_pil = self._create_overlay(
                pil_image, mask_np > 0, color=object_color)
            overlay_file = os.path.join(
                output_dir, f"{base_name}_{core_prompt.replace(' ','_')}_{idx}_overlay.png")
            individual_overlay_pil.save(overlay_file)

            detection_results.append(
                DetectionResult(
                    box=box,
                    core_prompt=core_prompt,
                    object_color=object_color,
                    mask_np=mask_np,
                    mask_img_pil=mask_img_pil,
                    mask_file=mask_file,
                    individual_overlay_pil=individual_overlay_pil,
                    overlay_file=overlay_file
                )
            )

            processed_detections_for_merge.append(
                MergedOverlayItem(mask=mask_np > 0, color=object_color, label=core_prompt)
            )

        # Create and save the merged overlay image
        if processed_detections_for_merge:
            for item in processed_detections_for_merge:
                image_np = np.array(merged_overlay_image).copy()
                bool_mask_np = item.mask
                color_np = np.array(item.color, dtype=np.uint8)

                if image_np.ndim == 2:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                elif image_np.ndim == 3 and image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

                colored_mask_layer = np.zeros_like(image_np)
                colored_mask_layer[bool_mask_np] = color_np

                image_np[bool_mask_np] = (
                    0.5 * image_np[bool_mask_np] + 0.5 * color_np).astype(np.uint8)
                merged_overlay_image = Image.fromarray(image_np)

            merged_overlay_file = os.path.join(output_dir, f"{base_name}_merged_overlay.png")
            merged_overlay_image.save(merged_overlay_file)
            print(f"Saved merged overlay: {merged_overlay_file}")


    def process_video(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Process a single video file: detect and segment objects in the first frame,
        propagate masks through the video, and save per-frame/per-object masks and overlays.

        Args:
            video_path (str): Path to the input video file.
            prompt (Union[str, List[str]]): Text prompt(s) for object detection.
            output_dir (str): Directory to save output masks and overlays.
        """
        tmp_frames_dir = tempfile.mkdtemp(prefix="sowlv2_frames_")
        detection_details_for_video = [] # To store color and core_prompt per SAM object ID

        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-r", str(self.config.fps),
                 os.path.join(tmp_frames_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
                check=True
            )

            state = self.sam.init_state(tmp_frames_dir)
            first_img_path = os.path.join(tmp_frames_dir, _FIRST_FRAME)
            if not os.path.exists(first_img_path):
                print(f"First frame {_FIRST_FRAME} not found. Video might be too short.")
                return

            first_pil_img = Image.open(first_img_path).convert("RGB")
            detections_owl = self.owl.detect(
                image=first_pil_img, prompt=prompt, threshold=self.config.threshold
            )

            if not detections_owl:
                print(
                    f"No objects for prompt(s) '{prompt}' — aborting video processing.")
                return

            sam_obj_id_counter = 1
            for det in detections_owl:
                core_prompt = det["core_prompt"]
                object_color = self._get_color_for_prompt(core_prompt)
                box = det["box"]

                # Store mapping from SAM ID to color and prompt
                detection_details_for_video.append({
                    'sam_id': sam_obj_id_counter,
                    'core_prompt': core_prompt,
                    'color': object_color
                })

                self.sam.add_new_box(
                    state=state, frame_idx=_FIRST_FRAME_IDX,
                    box=box, obj_idx=sam_obj_id_counter
                )
                sam_obj_id_counter += 1

            for fidx, sam_obj_ids_tensor, mask_logits_tensor in self.sam.propagate_in_video(state):
                current_frame_num = fidx + 1
                frame_file_path = os.path.join(tmp_frames_dir, f"{current_frame_num:06d}.jpg")
                if not os.path.exists(frame_file_path):
                    print(f"Warning: Frame {frame_file_path} not found. Skipping.")
                    continue

                current_pil_img = Image.open(frame_file_path).convert("RGB")
                if isinstance(sam_obj_ids_tensor, torch.Tensor):
                    sam_obj_ids_list = sam_obj_ids_tensor.cpu().numpy().tolist()
                else:
                    sam_obj_ids_list = list(sam_obj_ids_tensor)

                for i, sam_id in enumerate(sam_obj_ids_list):
                    # Find the details (color, core_prompt) for this sam_id
                    detail = next(
                        (d for d in detection_details_for_video if d['sam_id'] == sam_id), None)
                    if not detail:
                        print(f"No SAM object ID {sam_id}. Using default color.")
                        object_color = self.palette[0] # Default to first palette color
                        core_prompt_str = f"unknown{sam_id}"
                    else:
                        object_color = detail['color']
                        core_prompt_str = detail['core_prompt']
                    # This is a single mask for the current sam_id
                    mask_for_obj = mask_logits_tensor[i]
                    mask_binary_np = (mask_for_obj > 0.5).cpu().numpy().astype(np.uint8)
                    mask_bool_np = np.squeeze(mask_binary_np > 0)

                    # Save individual binary mask
                    mask_pil_img = Image.fromarray(np.squeeze(mask_binary_np) * 255).convert("L")
                    mask_filename = (
                        f"{current_frame_num:06d}_obj{sam_id}_"
                        f"{core_prompt_str.replace(' ','_')}_mask.png"
                    )
                    mask_pil_img.save(os.path.join(output_dir, mask_filename))

                    # Create and save individual colored overlay
                    overlay_pil_img = self._create_overlay(
                        current_pil_img, mask_bool_np, color=object_color
                    )
                    overlay_filename = (
                        f"{current_frame_num:06d}_obj{sam_id}_"
                        f"{core_prompt_str.replace(' ','_')}_overlay.png"
                    )
                    overlay_pil_img.save(os.path.join(output_dir, overlay_filename))

            print(f"✅ Video frame segmentation finished; results in {output_dir}")
            # This will now use colored overlays
            video_utils.generate_per_object_videos(output_dir, fps=self.config.fps)
        finally:
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
        print(f"✅ Video generation finished; results in {output_dir}")


    def _create_overlay(self,
                        pil_image: Image.Image,
                        bool_mask_np: np.ndarray,
                        color: Tuple[int, int, int]):
        """
        Blend a colored mask with the input PIL.Image and return the overlay as a PIL.Image.

        Args:
            pil_image (Image.Image): Original image.
            bool_mask_np (np.ndarray): Boolean NumPy array for the mask (True where object is).
            color (Tuple[int,int,int]): RGB tuple for the mask color.

        Returns:
            Image.Image: The overlay image with the colored mask blended in.
        """
        if bool_mask_np.ndim != 2:
            raise ValueError(f"Expected 2-D boolean mask, got shape {bool_mask_np.shape}")

        overlay_pil = pil_image.copy() # Work on a copy
        overlay_np = np.array(overlay_pil)
        color_np = np.array(color, dtype=np.uint8)

        # Ensure overlay_np is 3-channel RGB for color application
        if overlay_np.ndim == 2: # Grayscale
            overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_GRAY2RGB)
        elif overlay_np.ndim == 3 and overlay_np.shape[2] == 4: # RGBA
            overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_RGBA2RGB)
        elif overlay_np.ndim == 3 and overlay_np.shape[2] != 3: # Other non-RGB 3-channel
            print(f"Overlay for image with shape {overlay_np.shape} might not be accurate.")
            return pil_image # Or handle as error

        # Apply color blending only where the mask is True
        overlay_np[bool_mask_np] = (
            0.5 * overlay_np[bool_mask_np] + 0.5 * color_np
        ).astype(np.uint8)

        return Image.fromarray(overlay_np)
