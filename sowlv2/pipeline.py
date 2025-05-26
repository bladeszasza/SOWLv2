"""
Core pipeline for SOWLv2: text-prompted detection (OWLv2) and segmentation (SAM2) for images/videos.
"""
import os
import subprocess
import shutil
import tempfile
from typing import Any, Union, List, Dict, Tuple
import glob
import cv2  # pylint: disable=import-error
import numpy as np
from PIL import Image
import torch
from sowlv2 import video_utils
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper
from sowlv2.data.config import (
    PipelineBaseData, MergedOverlayItem, PropagatedFrameOutput,
    SingleDetectionInput, VideoProcessContext, PipelineConfig,
    VideoProcessOptions, VideoDirectories, MergedFrameItems, ObjectMaskProcessData,
    TempBinaryPaths, TempOverlayPaths, TempVideoOutputPaths, ObjectContext,
    SaveOutputsConfig, ProcessSingleObjectConfig
)

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
    Integrates OWLv2 for detection and SAM2 for segmentation. Assigns unique colors to each label.
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
        if core_prompt not in self.prompt_color_map:
            color = self.palette[self.next_color_idx % len(self.palette)]
            self.prompt_color_map[core_prompt] = color
            self.next_color_idx += 1
        return self.prompt_color_map[core_prompt]

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
        binary_dir = os.path.join(config.output_dir, "binary")
        overlay_dir = os.path.join(config.output_dir, "overlay")
        os.makedirs(binary_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        # Save binary mask if enabled
        if self.config.pipeline_config.binary:
            mask_img_pil = Image.fromarray(config.mask_np * 255).convert("L")
            mask_file = os.path.join(
                binary_dir,
                f"{config.base_name}_{config.core_prompt_slug}_"
                f"{config.obj_idx}_mask.png"
            )
            mask_img_pil.save(mask_file)

        # Save overlay if enabled
        if self.config.pipeline_config.overlay:
            individual_overlay_pil = self._create_overlay(
                config.pil_image, config.mask_np > 0, color=config.object_color)
            overlay_file = os.path.join(
                overlay_dir,
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
            single_detection = SingleDetectionInput(
                pil_image=pil_image,
                detection_detail=det_detail,
                obj_idx=idx,
                base_name=base_name,
                output_dir=output_dir
            )
            merged_item = self._process_single_detection_for_image(single_detection)
            if merged_item:
                items_for_merged_overlay.append(merged_item)

        # Only create merged overlay if enabled
        self._create_and_save_merged_overlay(
            original_pil_image=pil_image,
            items_for_merge=items_for_merged_overlay,
            base_name=base_name,
            output_dir=output_dir
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
        video_ctx = self._prepare_video_context(video_path, prompt)
        if video_ctx is None:
            print("Video preparation failed. Aborting.")
            return

        # Create temp directory structure
        with tempfile.TemporaryDirectory(prefix="sowlv2_") as temp_dir:
            dirs = self._create_temp_directories(temp_dir)
            process_options = VideoProcessOptions(
                binary=self.config.pipeline_config.binary,
                overlay=self.config.pipeline_config.overlay,
                merged=self.config.pipeline_config.merged,
                fps=self.config.fps
            )

            try:
                self._process_video_frames(video_ctx, dirs, process_options)
                self._move_outputs_to_final_dir(dirs, output_dir, process_options)
            finally:
                shutil.rmtree(video_ctx.tmp_frames_dir, ignore_errors=True)

        print(f"✅ Video processing finished; results in {output_dir}")

    def _create_temp_directories(self, temp_dir: str) -> VideoDirectories:
        """
        Create temporary directory structure for video processing.
        """
        temp_binary_path = os.path.join(temp_dir, "binary")
        temp_binary_merged_path = os.path.join(temp_binary_path, "merged")
        temp_overlay_path = os.path.join(temp_dir, "overlay")
        temp_overlay_merged_path = os.path.join(temp_overlay_path, "merged")
        temp_video_path = os.path.join(temp_dir, "video")
        temp_video_binary_path = os.path.join(temp_video_path, "binary")
        temp_video_overlay_path = os.path.join(temp_video_path, "overlay")

        # Create all temp directories
        for dir_path in [temp_binary_path, temp_binary_merged_path, temp_overlay_path,
                       temp_overlay_merged_path, temp_video_path, temp_video_binary_path,
                       temp_video_overlay_path]:
            os.makedirs(dir_path, exist_ok=True)

        return VideoDirectories(
            temp_dir=temp_dir,
            binary=TempBinaryPaths(path=temp_binary_path, merged_path=temp_binary_merged_path),
            overlay=TempOverlayPaths(path=temp_overlay_path, merged_path=temp_overlay_merged_path),
            video=TempVideoOutputPaths(path=temp_video_path, binary_path=temp_video_binary_path,
                                     overlay_path=temp_video_overlay_path)
        )

    def _process_video_frames(self, video_ctx: VideoProcessContext,
                              dirs: VideoDirectories,
                              options: VideoProcessOptions):
        """
        Process all frames in the video context and always generate merged videos.
        """
        self._process_all_frames(video_ctx, dirs)
        print("✅ Video frame segmentation finished")

        # Always generate merged videos (binary and overlay)
        video_utils.generate_videos(
            temp_dir=dirs.temp_dir,
            fps=options.fps,
            binary=True,
            overlay=True
        )

    def _process_all_frames(self, video_ctx: VideoProcessContext, dirs: VideoDirectories):
        """
        Process all frames in the video context.
        """
        for fidx, sam_obj_ids_tensor, mask_logits_tensor in self.sam.propagate_in_video(
            video_ctx.updated_sam_state):
            current_frame_num = fidx + 1
            frame_file_path = os.path.join(
                video_ctx.tmp_frames_dir, f"{current_frame_num:06d}.jpg")
            if not os.path.exists(frame_file_path):
                print(f"Warning: Frame {frame_file_path} not found. Skipping.")
                continue
            current_pil_img = Image.open(frame_file_path).convert("RGB")

            frame_output = PropagatedFrameOutput(
                current_pil_img=current_pil_img,
                frame_num=current_frame_num,
                sam_obj_ids_tensor=sam_obj_ids_tensor,
                mask_logits_tensor=mask_logits_tensor,
                detection_details_map=video_ctx.detection_details_for_video,
                output_dir=dirs.temp_dir
            )
            self._process_propagated_frame_output(frame_output)

    def _move_outputs_to_final_dir(self, dirs: VideoDirectories,
                                   output_dir: str,
                                   options: VideoProcessOptions):
        """
        Move requested outputs from temp directory to final output directory.
        Always copies merged videos to output directory.
        Avoid nested folders by not pre-creating destination for frames/merged.
        Print debug info for merged output contents before video generation.
        """

        # Define final output directories
        final_binary_dir = os.path.join(output_dir, "binary")
        final_overlay_dir = os.path.join(output_dir, "overlay")
        final_video_overlay = os.path.join(output_dir, "video", "overlay")
        final_video_binary = os.path.join(output_dir, "video", "binary")

        # Only move requested per-frame outputs to final directory (no pre-mkdir)
        if options.binary:
            src_frames = os.path.join(dirs.binary.path, "frames")
            if os.path.exists(src_frames):
                shutil.move(src_frames, final_binary_dir)
            if options.merged:
                src_merged = os.path.join(dirs.binary.path, "merged")
                if os.path.exists(src_merged):
                    shutil.move(src_merged, final_binary_dir)

        if options.overlay:
            src_frames = os.path.join(dirs.overlay.path, "frames")
            if os.path.exists(src_frames):
                shutil.move(src_frames, final_overlay_dir)
            if options.merged:
                src_merged = os.path.join(dirs.overlay.path, "merged")
                if os.path.exists(src_merged):
                    shutil.move(src_merged, final_overlay_dir)

        # Debug print: show contents of merged folders before video generation
        print("[DEBUG] Contents of binary/merged in temp:",
              glob.glob(os.path.join(dirs.binary.path, "merged", "*")))
        print("[DEBUG] Contents of overlay/merged in temp:",
              glob.glob(os.path.join(dirs.overlay.path, "merged", "*")))

        # Always move merged videos to output directory
        if os.path.exists(dirs.video.overlay_path):
            os.makedirs(final_video_overlay, exist_ok=True)
            for f in os.listdir(dirs.video.overlay_path):
                shutil.move(os.path.join(dirs.video.overlay_path, f), final_video_overlay)
        if os.path.exists(dirs.video.binary_path):
            os.makedirs(final_video_binary, exist_ok=True)
            for f in os.listdir(dirs.video.binary_path):
                shutil.move(os.path.join(dirs.video.binary_path, f), final_video_binary)

    def _process_propagated_frame_output(
        self,
        frame_output: PropagatedFrameOutput
    ):
        """
        Save masks/overlays for a single frame from SAM's propagation output.
        Always saves images to temp directory, cleanup will be handled later.
        """
        sam_obj_ids_list = self._get_sam_obj_ids_list(frame_output.sam_obj_ids_tensor)
        dirs_frame_output = self._create_frame_output_directories(frame_output.output_dir)
        merged_items = MergedFrameItems(binary_items=[], overlay_items=[])

        # Process each object in the frame
        for i, sam_id in enumerate(sam_obj_ids_list):
            config = ProcessSingleObjectConfig(
                frame_output=frame_output,
                sam_id=sam_id,
                obj_idx=i,
                dirs_frame_output=dirs_frame_output,
                merged_items=merged_items
            )
            self._process_single_object(config)

        # Create and save merged outputs if enabled
        if self.config.pipeline_config.merged:
            self._create_merged_outputs(
                merged_items,
                frame_output.current_pil_img,
                frame_output.frame_num,
                dirs_frame_output
            )

    def _get_sam_obj_ids_list(self, sam_obj_ids_tensor: torch.Tensor) -> List[int]:
        """
        Convert SAM object IDs tensor to a list.
        """
        if isinstance(sam_obj_ids_tensor, torch.Tensor):
            return sam_obj_ids_tensor.cpu().numpy().tolist()
        return list(sam_obj_ids_tensor)

    def _process_single_object(
        self,
        config: ProcessSingleObjectConfig
    ):
        """
        Process a single object in the frame.
        """
        object_color, core_prompt_str = self._get_object_details(
            config.sam_id, config.frame_output.detection_details_map, config.frame_output.frame_num
        )

        obj_ctx = ObjectContext(
            sam_id=config.sam_id,
            core_prompt_str=core_prompt_str,
            object_color=object_color
        )

        process_data = ObjectMaskProcessData(
            mask_for_obj=config.frame_output.mask_logits_tensor[config.obj_idx],
            obj_ctx=obj_ctx,
            frame_num=config.frame_output.frame_num,
            pil_image=config.frame_output.current_pil_img,
            dirs=config.dirs_frame_output,
            merged_items=config.merged_items
        )
        _ = self._process_object_mask(process_data)

    def _create_frame_output_directories(self, output_dir: str) -> Dict[str, str]:
        """
        Create output subdirectories for frame processing.
        """
        binary_frames_dir = os.path.join(output_dir, "binary", "frames")
        binary_merged_dir = os.path.join(output_dir, "binary", "merged")
        overlay_frames_dir = os.path.join(output_dir, "overlay", "frames")
        overlay_merged_dir = os.path.join(output_dir, "overlay", "merged")

        os.makedirs(binary_frames_dir, exist_ok=True)
        os.makedirs(binary_merged_dir, exist_ok=True)
        os.makedirs(overlay_frames_dir, exist_ok=True)
        os.makedirs(overlay_merged_dir, exist_ok=True)

        return {
            "binary_frames": binary_frames_dir,
            "binary_merged": binary_merged_dir,
            "overlay_frames": overlay_frames_dir,
            "overlay_merged": overlay_merged_dir
        }

    def _get_object_details(self,
                            sam_id: int,
                            detection_details_map: List[Dict[str, Any]],
                            frame_num: int) -> Tuple[Tuple[int, int, int], str]:
        """
        Get details for an object from the detection details map.
        
        Args:
            sam_id (int): The SAM object ID to look up
            detection_details_map (List[Dict[str, Any]]): List of detection details
            frame_num (int): Current frame number for error reporting
            
        Returns:
            Tuple[Tuple[int, int, int], str]: Color tuple and core prompt string
        """
        detail = next(
            (d for d in detection_details_map if d['sam_id'] == sam_id), None
        )

        if not detail:
            print(f"No details found for {sam_id} in frame {frame_num}.")
            object_color = self.palette[0]
            core_prompt_str = f"unknown{sam_id}"
        else:
            object_color = detail['color']
            core_prompt_str = detail['core_prompt']

        return object_color, core_prompt_str

    def _process_object_mask(
        self,
        data: ObjectMaskProcessData
    ) -> np.ndarray:
        """
        Process a single object mask and save binary/overlay outputs.
        """
        mask_binary_np = (data.mask_for_obj > 0.5).cpu().numpy().astype(np.uint8)
        mask_bool_np = np.squeeze(mask_binary_np > 0)

        # Save individual binary mask
        mask_pil_img = Image.fromarray(np.squeeze(mask_binary_np) * 255).convert("L")
        mask_filename = (
            f"{data.frame_num:06d}_obj{data.obj_ctx.sam_id}_"
            f"{data.obj_ctx.core_prompt_str.replace(' ','_')}_mask.png"
        )
        mask_pil_img.save(os.path.join(data.dirs["binary_frames"], mask_filename))
        data.merged_items.binary_items.append(mask_bool_np)

        # Save individual overlay
        overlay_pil_img = self._create_overlay(
            data.pil_image, mask_bool_np, color=data.obj_ctx.object_color
        )
        overlay_filename = (
            f"{data.frame_num:06d}_obj{data.obj_ctx.sam_id}_"
            f"{data.obj_ctx.core_prompt_str.replace(' ','_')}_overlay.png"
        )
        overlay_pil_img.save(os.path.join(data.dirs["overlay_frames"], overlay_filename))
        data.merged_items.overlay_items.append((mask_bool_np, data.obj_ctx.object_color))

        return mask_bool_np

    def _create_merged_outputs(
        self,
        merged_items: MergedFrameItems,
        current_pil_img: Image.Image,
        frame_num: int,
        dirs: Dict[str, str]
    ):
        """
        Create and save merged binary mask and overlay outputs.
        """
        # Create merged binary mask
        if merged_items.binary_items and self.config.pipeline_config.binary:
            # Initialize with the shape of the first mask but ensure uint8 type
            merged_mask = np.zeros_like(merged_items.binary_items[0], dtype=np.uint8)
            for mask in merged_items.binary_items:
                # Ensure mask is boolean before logical operation
                bool_mask = mask.astype(bool)
                merged_mask = np.logical_or(merged_mask, bool_mask).astype(np.uint8)

            merged_mask_pil = Image.fromarray(merged_mask * 255).convert("L")
            merged_mask_file = os.path.join(
                dirs["binary_merged"],
                f"{frame_num:06d}_merged_mask.png"
            )
            merged_mask_pil.save(merged_mask_file)
            print(f"Saved merged binary mask for frame {frame_num}")

        # Create merged overlay
        if merged_items.overlay_items and self.config.pipeline_config.overlay:
            merged_overlay_pil = current_pil_img.copy()
            for mask, color in merged_items.overlay_items:
                current_image_np = np.array(merged_overlay_pil)
                color_np = np.array(color, dtype=np.uint8)
                if current_image_np.ndim == 2:
                    current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_GRAY2RGB)
                elif current_image_np.ndim == 3 and current_image_np.shape[2] == 4:
                    current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_RGBA2RGB)
                # Ensure mask is boolean before indexing
                bool_mask = mask.astype(bool)
                current_image_np[bool_mask] = (
                    0.5 * current_image_np[bool_mask] + 0.5 * color_np
                ).astype(np.uint8)
                merged_overlay_pil = Image.fromarray(current_image_np)

            merged_overlay_file = os.path.join(
                dirs["overlay_merged"],
                f"{frame_num:06d}_merged_overlay.png"
            )
            merged_overlay_pil.save(merged_overlay_file)
            print(f"Saved merged overlay for frame {frame_num}")

    def _prepare_video_context(
        self, video_path: str, prompt: Union[str, List[str]]
    ) -> VideoProcessContext | None:
        """
        Extract frames, initialize SAM state, and prepare detection context for video processing.
        Returns VideoProcessContext or None if failed.
        """
        tmp_frames_dir = tempfile.mkdtemp(prefix="sowlv2_frames_")
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-r", str(self.config.fps),
                 os.path.join(tmp_frames_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
                check=True
            )
            initial_sam_state = self.sam.init_state(tmp_frames_dir)
            first_img_path = os.path.join(tmp_frames_dir, _FIRST_FRAME)
            if not os.path.exists(first_img_path):
                print(f"First frame {_FIRST_FRAME} not found in {tmp_frames_dir}.")
                shutil.rmtree(tmp_frames_dir, ignore_errors=True)
                return None

            first_pil_img = Image.open(first_img_path).convert("RGB")
            detection_details_for_video, updated_sam_state = self._initialize_video_tracking(
                first_pil_img, prompt, initial_sam_state
            )
            if not detection_details_for_video:
                shutil.rmtree(tmp_frames_dir, ignore_errors=True)
                return None

            return VideoProcessContext(
                tmp_frames_dir=tmp_frames_dir,
                initial_sam_state=initial_sam_state,
                first_img_path=first_img_path,
                first_pil_img=first_pil_img,
                detection_details_for_video=detection_details_for_video,
                updated_sam_state=updated_sam_state
            )
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"Error during video preparation: {e}")
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)
            return None

    def _initialize_video_tracking(
        self,
        first_pil_img: Image.Image,
        prompt: Union[str, List[str]],
        sam_state: Any
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Detect objects in first frame, assign colors, and initialize SAM tracking.
        """
        detection_details_for_video: List[Dict[str, Any]] = [] # Or List[VideoDetectionDetail]
        detections_owl = self.owl.detect(
            image=first_pil_img, prompt=prompt, threshold=self.config.threshold
        )

        if not detections_owl:
            print(f"No objects for prompt(s) '{prompt}' in first frame. Aborting video processing.")
            return [], sam_state # Return empty list and original state

        sam_obj_id_counter = 1
        for det in detections_owl:
            core_prompt = det["core_prompt"]
            object_color = self._get_color_for_prompt(core_prompt)
            box = det["box"]

            detection_details_for_video.append({ # Or VideoDetectionDetail(...)
                'sam_id': sam_obj_id_counter,
                'core_prompt': core_prompt,
                'color': object_color
            })

            # Update SAM state with the new box
            self.sam.add_new_box(
                state=sam_state, frame_idx=_FIRST_FRAME_IDX,
                box=box, obj_idx=sam_obj_id_counter
            )
            sam_obj_id_counter += 1

        return detection_details_for_video, sam_state

    def _create_overlay(self,
                        pil_image: Image.Image,
                        bool_mask_np: np.ndarray,
                        color: Tuple[int, int, int]):
        """
        Blend a colored mask with a PIL image and return the overlay as a PIL.Image.

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
