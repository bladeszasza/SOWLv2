"""
Core pipeline for SOWLv2: text-prompted detection (OWLv2) and segmentation (SAM2) for images/videos.
"""
import os
import subprocess
import shutil
import tempfile
from typing import Any, Union, List, Dict, Tuple
import cv2  # pylint: disable=import-error
import numpy as np
from PIL import Image
import torch
from sowlv2 import video_utils
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper
from sowlv2.data.config import (
    PipelineBaseData, MergedOverlayItem, PropagatedFrameOutput,
    SingleDetectionInput, VideoProcessContext
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

        mask_np = self.sam.segment(single_detection.pil_image, box) # Expects HxW uint8 numpy array
        if mask_np is None:
            print(f"SAM2 failed to segment object {single_detection.obj_idx} ({core_prompt}).")
            return None

        mask_img_pil = Image.fromarray(mask_np * 255).convert("L")
        mask_file = os.path.join(
            single_detection.output_dir,
            f"{single_detection.base_name}_{core_prompt.replace(' ','_')}_"
            f"{single_detection.obj_idx}_mask.png"
        )
        mask_img_pil.save(mask_file)

        individual_overlay_pil = self._create_overlay(
            single_detection.pil_image, mask_np > 0, color=object_color)
        overlay_file = os.path.join(
            single_detection.output_dir,
            f"{single_detection.base_name}_{core_prompt.replace(' ','_')}_"
            f"{single_detection.obj_idx}_overlay.png"
        )
        individual_overlay_pil.save(overlay_file)

        return MergedOverlayItem(mask=mask_np > 0, color=object_color, label=core_prompt)

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
        if not items_for_merge:
            return

        merged_overlay_pil = original_pil_image.copy()

        for item in items_for_merge:
            # The _create_overlay function does blending. For merging, we apply one by one.
            # We need to ensure the image we are drawing on is in the correct state.
            # This can be tricky if colors are semi-transparent.
            # A simpler approach for direct application:
            current_image_np = np.array(merged_overlay_pil)
            bool_mask_np = item.mask
            color_np = np.array(item.color, dtype=np.uint8)

            # Ensure current_image_np is 3-channel RGB for color application
            if current_image_np.ndim == 2: # Grayscale
                current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_GRAY2RGB)
            elif current_image_np.ndim == 3 and current_image_np.shape[2] == 4: # RGBA
                current_image_np = cv2.cvtColor(current_image_np, cv2.COLOR_RGBA2RGB)

            # Apply colored mask using blending (as in _create_overlay)
            # This ensures consistent blending if that's the desired visual effect
            current_image_np[bool_mask_np] = (
                0.5 * current_image_np[bool_mask_np] + 0.5 * color_np
            ).astype(np.uint8)
            merged_overlay_pil = Image.fromarray(current_image_np)

        merged_overlay_file = os.path.join(output_dir, f"{base_name}_merged_overlay.png")
        merged_overlay_pil.save(merged_overlay_file)
        print(f"Saved merged overlay: {merged_overlay_file}")

    def process_image(self, image_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Detect, segment, and save masks/overlays for a single image.
        """
        pil_image = Image.open(image_path).convert("RGB")
        # Use self.threshold from init, not self.config.threshold directly here for consistency
        detections = self.owl.detect(
            image=pil_image,
            prompt=prompt,
            threshold=self.threshold) # Use the pipeline's configured threshold
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
            image=first_pil_img, prompt=prompt, threshold=self.threshold
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

    def _process_propagated_frame_output(
        self,
        frame_output: PropagatedFrameOutput
    ):
        """
        Save masks/overlays for a single frame from SAM's propagation output.
        """
        if isinstance(frame_output.sam_obj_ids_tensor, torch.Tensor):
            sam_obj_ids_list = frame_output.sam_obj_ids_tensor.cpu().numpy().tolist()
        else:
            sam_obj_ids_list = list(frame_output.sam_obj_ids_tensor)

        for i, sam_id in enumerate(sam_obj_ids_list):
            detail = next(
                (d for d in frame_output.detection_details_map if d['sam_id'] == sam_id), None
            )

            if not detail:
                print(f"No details found for {sam_id} in frame {frame_output.frame_num}.")
                object_color = self.palette[0]
                core_prompt_str = f"unknown{sam_id}"
            else:
                object_color = detail['color']
                core_prompt_str = detail['core_prompt']

            mask_for_obj = frame_output.mask_logits_tensor[i]
            mask_binary_np = (mask_for_obj > 0.5).cpu().numpy().astype(np.uint8)
            mask_bool_np = np.squeeze(mask_binary_np > 0)

            mask_pil_img = Image.fromarray(np.squeeze(mask_binary_np) * 255).convert("L")
            mask_filename = (
                f"{frame_output.frame_num:06d}_obj{sam_id}_"
                f"{core_prompt_str.replace(' ','_')}_mask.png"
            )
            mask_pil_img.save(os.path.join(frame_output.output_dir, mask_filename))

            overlay_pil_img = self._create_overlay(
                frame_output.current_pil_img, mask_bool_np, color=object_color
            )
            overlay_filename = (
                f"{frame_output.frame_num:06d}_obj{sam_id}_"
                f"{core_prompt_str.replace(' ','_')}_overlay.png"
            )
            overlay_pil_img.save(os.path.join(frame_output.output_dir, overlay_filename))


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

    def process_video(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Detect and segment objects in a video, propagate masks,
        and save per-frame/per-object results.
        """
        video_ctx = self._prepare_video_context(video_path, prompt)
        if video_ctx is None:
            print("Video preparation failed. Aborting.")
            return

        try:
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
                    output_dir=output_dir
                )
                self._process_propagated_frame_output(frame_output)

            print(f"✅ Video frame segmentation finished; results in {output_dir}")

            # Step 5: Generate per-object summary videos
            video_utils.generate_per_object_videos(output_dir, fps=self.fps) # Use self.fps

        finally:
            shutil.rmtree(video_ctx.tmp_frames_dir, ignore_errors=True)
        print(f"✅ Video generation finished; results in {output_dir}")


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
