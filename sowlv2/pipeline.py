"""
Core pipeline for SOWLv2: text-prompted detection (OWLv2) and segmentation (SAM2) for images/videos.
"""
import os
import shutil
import tempfile
from typing import Union, List, Dict, Tuple
from PIL import Image

from sowlv2.models import OWLV2Wrapper, SAM2Wrapper
from sowlv2.data.config import (
    PipelineBaseData,
    MergedOverlayItem,
    SingleDetectionInput,
    PipelineConfig
)
from sowlv2.video_pipeline import (
    VideoTrackingConfig,
    VideoProcessingConfig
)
from sowlv2.utils.pipeline_utils import (
    DEFAULT_PALETTE, get_prompt_color, CUDA
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
from sowlv2.utils.frame_utils import is_valid_image_extension
from sowlv2.utils.filesystem_utils import remove_empty_folders


_FIRST_FRAME = "000001.jpg" # Retained for reference, though primarily used in video_pipeline
_FIRST_FRAME_IDX = 0     # Retained for reference

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
                device=CUDA,
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

    def _filter_outputs_by_flags(self, output_dir: str):
        """Filter outputs based on pipeline configuration flags."""
        import shutil

        # If binary is disabled, remove binary directory
        if not self.config.pipeline_config.binary:
            binary_dir = os.path.join(output_dir, "binary")
            if os.path.exists(binary_dir):
                shutil.rmtree(binary_dir)

        # If overlay is disabled, remove overlay directory
        if not self.config.pipeline_config.overlay:
            overlay_dir = os.path.join(output_dir, "overlay")
            if os.path.exists(overlay_dir):
                shutil.rmtree(overlay_dir)

        # If merged is disabled, remove merged subdirectories
        if not self.config.pipeline_config.merged:
            for base_type in ["binary", "overlay"]:
                merged_dir = os.path.join(output_dir, base_type, "merged")
                if os.path.exists(merged_dir):
                    shutil.rmtree(merged_dir)

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

        # Always create merged overlay, selective copying is handled by flag logic
        create_and_save_merged_overlay(
            items_for_merged_overlay,
            pil_image,
            output_dir,
            int(base_name) if base_name.isdigit() else 0
        )

        # Apply output filtering based on flags
        self._filter_outputs_by_flags(output_dir)
        remove_empty_folders(output_dir)

    def process_frames(self, folder_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Process all images in a folder as frames.
        """
        files = sorted(os.listdir(folder_path))
        for fname in files:
            infile = os.path.join(folder_path, fname)
            root, ext = map(str.lower, os.path.splitext(fname))
            if not is_valid_image_extension(ext):
                continue
            self.process_image(infile, prompt, output_dir+"/"+root)

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
                next_color_idx=specific_next_color_idx,
                fps=self.config.fps
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
