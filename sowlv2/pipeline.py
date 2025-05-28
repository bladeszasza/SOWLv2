"""
Core pipeline for SOWLv2: text-prompted detection (OWLv2) and segmentation (SAM2) for images/videos.
"""
import os
import shutil
import tempfile
from typing import Union, List, Dict, Tuple
import numpy as np
import cv2  # pylint: disable=import-error
from PIL import Image

from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper
from sowlv2.data.config import (
    PipelineBaseData,
    MergedOverlayItem,
    SingleDetectionInput,
    PipelineConfig,
    SaveOutputsConfig
)
from sowlv2.video_pipeline import (
    VideoTrackingConfig,
    VideoProcessingConfig
)
from sowlv2.pipeline_utils import (
    DEFAULT_PALETTE, get_prompt_color, create_overlay,
    create_output_directories
)
from sowlv2.image_pipeline import (
    process_single_detection_for_image,
    create_and_save_merged_overlay
)
from sowlv2.video_pipeline import (
    prepare_video_context,
    create_temp_directories_for_video,
    run_video_processing_steps,
    move_video_outputs_to_final_dir
)

# Disable no-member for cv2 (OpenCV) for the whole file
# pylint: disable=no-member

_FIRST_FRAME = "000001.jpg" # Retained for reference, though primarily used in video_pipeline
_FIRST_FRAME_IDX = 0     # Retained for reference

# DEFAULT_PALETTE = [
#     (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
#     (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
#     (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
#     (128, 255, 0), (0, 128, 255), (128, 0, 255), (192, 192, 192), (255, 215, 0),
#     (138, 43, 226),(75, 0, 130), (240, 128, 128), (32, 178, 170)
# ]


class SOWLv2Pipeline:
    """
    Main SOWLv2 pipeline orchestrator for images and videos.
    """
    def __init__(self, config: PipelineBaseData = None):
        """
        Initialize the pipeline with model/config parameters.
        Args:
            config (PipelineBaseData, optional): Pipeline configuration. Uses defaults if None.
        """
        if config is None:
            config = PipelineBaseData(
                owl_model="google/owlv2-base-patch16-ensemble",
                sam_model="facebook/sam2.1-hiera-small",
                threshold=0.1,
                fps=24,
                device="cuda",
                pipeline_config=PipelineConfig(merged=True,
                                               binary=True,
                                               overlay=True)
            )
        self.config = config
        self.owl = OWLV2Wrapper(model_name=self.config.owl_model, device=self.config.device)
        self.sam = SAM2Wrapper(model_name=self.config.sam_model, device=self.config.device)
        self.palette = DEFAULT_PALETTE
        self.prompt_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.next_color_idx = 0

    def _get_color_for_prompt(self, core_prompt: str) -> Tuple[int, int, int]:
        """
        Assign a consistent RGB color to a prompt/label.

        Args:
            core_prompt (str): The label or prompt for which to get a color.

        Returns:
            Tuple[int, int, int]: RGB color tuple.
        """
        color, self.next_color_idx = get_prompt_color(
            core_prompt, self.prompt_color_map, self.palette, self.next_color_idx
        )
        return color

    def _process_single_detection_for_image(
        self,
        single_detection: SingleDetectionInput
    ) -> MergedOverlayItem | None:
        """
        Segment and save mask/overlay for one detection. Return info for merged overlay.

        Args:
            single_detection (SingleDetectionInput): Dataclass containing all relevant info.
        """
        box = single_detection.detection_detail["box"]
        core_prompt = single_detection.detection_detail["core_prompt"]
        object_color = self._get_color_for_prompt(core_prompt)

        mask_np = self.sam.segment(single_detection.pil_image, box)
        if mask_np is None:
            print(f"SAM2 failed to segment object {single_detection.obj_idx} ({core_prompt}).")
            return None

        # Prepare config for saving outputs
        save_config = SaveOutputsConfig(
            output_dir=single_detection.output_dir,
            base_name=single_detection.base_name,
            core_prompt_slug=core_prompt.replace(' ','_'),
            obj_idx=single_detection.obj_idx,
            mask_np=mask_np,
            pil_image=single_detection.pil_image,
            object_color=object_color
        )

        # Save outputs
        self._save_detection_outputs(save_config)

        return MergedOverlayItem(mask=mask_np > 0, color=object_color, label=core_prompt)

    def _save_detection_outputs(self, config: SaveOutputsConfig):
        """Helper function to save binary mask and overlay for a single detection."""
        dirs = create_output_directories(config.output_dir)

        # Save binary mask if enabled
        if self.config.pipeline_config.binary:
            mask_img_pil = Image.fromarray(config.mask_np * 255).convert("L")
            mask_file = os.path.join(
                dirs["binary"],
                f"{config.base_name}_{config.core_prompt_slug}_"
                f"{config.obj_idx}_mask.png"
            )
            mask_img_pil.save(mask_file)

        # Save overlay if enabled
        if self.config.pipeline_config.overlay:
            individual_overlay_pil = create_overlay(
                config.pil_image, config.mask_np > 0, color=config.object_color)
            overlay_file = os.path.join(
                dirs["overlay"],
                f"{config.base_name}_{config.core_prompt_slug}_"
                f"{config.obj_idx}_overlay.png"
            )
            individual_overlay_pil.save(overlay_file)

    def _create_and_save_merged_overlay(
        self,
        original_pil_image: Image.Image,
        items_for_merge: List[MergedOverlayItem],
        base_name: str,
        output_dir: str
    ):
        """
        Create and save a merged overlay image with all detected object masks.
        """
        if not self.config.pipeline_config.merged or not items_for_merge:
            return

        # Create merged subdirectories
        binary_merged_dir = os.path.join(output_dir, "binary", "merged")
        overlay_merged_dir = os.path.join(output_dir, "overlay", "merged")
        os.makedirs(binary_merged_dir, exist_ok=True)
        os.makedirs(overlay_merged_dir, exist_ok=True)

        # Create and save merged binary mask
        self._create_merged_binary_mask(items_for_merge, binary_merged_dir, base_name)

        # Create and save merged overlay
        self._create_merged_overlay_image(
            original_pil_image, items_for_merge, overlay_merged_dir, base_name
        )

    def _create_merged_binary_mask(
        self,
        items_for_merge: List[MergedOverlayItem],
        binary_merged_dir: str,
        base_name: str
    ):
        """Create and save a merged binary mask from multiple items."""
        if not items_for_merge:
            return

        # Initialize with the shape of the first mask but ensure uint8 type
        merged_mask = np.zeros_like(items_for_merge[0].mask, dtype=np.uint8)
        for item in items_for_merge:
            # Ensure mask is boolean before logical operation
            bool_mask = item.mask.astype(bool)
            merged_mask = np.logical_or(merged_mask, bool_mask).astype(np.uint8)

        # Convert to PIL image and save
        merged_mask_pil = Image.fromarray(merged_mask * 255).convert("L")
        merged_mask_file = os.path.join(binary_merged_dir, f"{base_name}_merged_mask.png")
        merged_mask_pil.save(merged_mask_file)
        print(f"Saved merged binary mask: {merged_mask_file}")

    def _create_merged_overlay_image(
        self,
        original_pil_image: Image.Image,
        items_for_merge: List[MergedOverlayItem],
        overlay_merged_dir: str,
        base_name: str
    ):
        """Create and save a merged overlay image from multiple items."""
        merged_overlay_pil = original_pil_image.copy()
        for item in items_for_merge:
            merged_overlay_pil = self._blend_overlay_item(merged_overlay_pil, item)

        merged_overlay_file = os.path.join(overlay_merged_dir, f"{base_name}_merged_overlay.png")
        merged_overlay_pil.save(merged_overlay_file)
        print(f"Saved merged overlay: {merged_overlay_file}")

    def _blend_overlay_item(self, overlay_pil: Image.Image, item: MergedOverlayItem) -> Image.Image:
        """Blend a single overlay item into the existing overlay image."""
        current_image_np = np.array(overlay_pil)
        color_np = np.array(item.color, dtype=np.uint8)

        if current_image_np.ndim == 2:
            current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_GRAY2RGB)
        elif current_image_np.ndim == 3 and current_image_np.shape[2] == 4:
            current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_RGBA2RGB)

        current_image_np[item.mask] = (
            0.5 * current_image_np[item.mask] + 0.5 * color_np
        ).astype(np.uint8)
        return Image.fromarray(current_image_np)

    def process_image(self, image_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Detect, segment, and save masks/overlays for a single image.
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

        items_for_merged_overlay: List[MergedOverlayItem] = []

        for idx, det_detail in enumerate(detections):
            # Get mask from SAM
            box = det_detail["box"]
            mask = self.sam.segment(pil_image, box)
            if mask is None:
                print(f"SAM2 failed to segment object {idx} ({det_detail['core_prompt']}).")
                continue

            # Update detection detail with mask
            det_detail['mask'] = mask
            det_detail['color'] = self._get_color_for_prompt(det_detail['core_prompt'])

            single_detection_input = SingleDetectionInput(
                pil_image=pil_image,
                detection_detail=det_detail,
                obj_idx=idx,
                base_name=base_name,
                output_dir=output_dir
            )
            merged_item = process_single_detection_for_image(
                single_detection_input,
                self.config.pipeline_config
            )
            if merged_item:
                items_for_merged_overlay.append(merged_item)

        create_and_save_merged_overlay(
            items_for_merged_overlay,
            pil_image,
            output_dir,
            int(base_name) if base_name.isdigit() else 0
        )

    def process_frames(self, folder_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Process all images in a folder as frames.
        """
        files = sorted(os.listdir(folder_path))
        for fname in files:
            infile = os.path.join(folder_path, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                continue
            self.process_image(infile, prompt, output_dir)

    def process_video(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Detect and segment objects in a video, propagate masks,
        and save per-frame/per-object results.
        """
        specific_prompt_color_map: Dict[str, Tuple[int, int, int]] = {}
        specific_next_color_idx = 0

        ctx, specific_prompt_color_map, specific_next_color_idx = prepare_video_context(
            video_path,
            VideoTrackingConfig(
                prompt=prompt,
                threshold=self.config.threshold,
                prompt_color_map=specific_prompt_color_map,
                palette=self.palette,
                next_color_idx=specific_next_color_idx
            ),
            self.owl,
            self.sam
        )

        if ctx is None:
            print("Video preparation failed. Aborting video processing.")
            return

        with tempfile.TemporaryDirectory(
            prefix="sowlv2_video_processing_") as temp_dir_for_video_run:
            video_temp_dirs = create_temp_directories_for_video(temp_dir_for_video_run)
            try:
                specific_prompt_color_map, specific_next_color_idx = run_video_processing_steps(
                    ctx,
                    self.sam,
                    video_temp_dirs,
                    VideoProcessingConfig(
                        pipeline_config=self.config.pipeline_config,
                        prompt_color_map=specific_prompt_color_map,
                        next_color_idx=specific_next_color_idx,
                        fps=self.config.fps
                    )
                )
                move_video_outputs_to_final_dir(
                    video_temp_dirs,
                    output_dir,
                    self.config.pipeline_config
                )
            finally:
                if ctx.tmp_frames_dir and os.path.exists(ctx.tmp_frames_dir):
                    shutil.rmtree(ctx.tmp_frames_dir, ignore_errors=True)

        print(f"✅ Video processing finished for {video_path}; results in {output_dir}")
