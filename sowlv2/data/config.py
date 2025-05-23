"""
Dataclasses for configuring the SOWLv2 object detection and segmentation pipeline.
"""
from dataclasses import dataclass
from typing import Any, Tuple, List, Dict
import numpy as np
from PIL import Image

@dataclass
class PipelineConfig:
    """
    Configuration class for pipeline settings.
    Attributes:
        merged (bool): Indicates whether the pipeline should operate in merged mode.
        binary (bool): Specifies if binary processing is enabled.
        overlay (bool): Determines if overlay functionality is active.
    """
    merged: bool
    binary: bool
    overlay: bool

@dataclass
class PipelineBaseData:
    """
    Stores all core configuration parameters for initializing and running the SOWLv2 pipeline.
    """
    owl_model: str
    sam_model: str
    threshold: float
    fps: int
    device: str
    pipeline_config: PipelineConfig


@dataclass
class SaveMaskOverlayConfig:
    """
    Configuration for saving masks and overlays for a frame.

    Attributes:
        pil_img (Any): The PIL image for the frame.
        frame_idx (int): The frame index.
        obj_ids (Any): Object IDs for the masks.
        masks (Any): Masks for the objects.
        out_dir (str): Output directory for saving results.
    """
    pil_img: Any
    frame_idx: int
    obj_ids: Any
    masks: Any
    out_dir: str

@dataclass
class MaskObject:
    """
    Stores the mask and its properties for a detected object.

    Attributes:
        mask_np (np.ndarray): The mask as a NumPy array.
        mask_img_pil (Image.Image): The mask as a PIL image.
        mask_file (str): Path to the saved mask file.
    """
    mask_np: np.ndarray
    mask_img_pil: Image.Image
    mask_file: str

@dataclass
class DetectionResult:
    """
    Stores the result of a single detection and segmentation.

    Attributes:
        box (Any): The bounding box for the detected object.
        core_prompt (str): The core prompt/label for the object.
        object_color (Tuple[int, int, int]): The assigned color for the object.
        mask_np (np.ndarray): The segmentation mask as a NumPy array.
        mask_img_pil (Image.Image): The mask as a PIL image.
        mask_file (str): Path to the saved mask file.
        individual_overlay_pil (Image.Image): The overlay as a PIL image.
        overlay_file (str): Path to the saved overlay file.
    """
    box: Any
    core_prompt: str
    object_color: Tuple[int, int, int]
    mask : MaskObject
    individual_overlay_pil: Image.Image
    overlay_file: str

@dataclass
class MergedOverlayItem:
    """
    Stores information for a single mask/color/label used in a merged overlay.

    Attributes:
        mask (np.ndarray): Boolean mask for the object.
        color (Tuple[int, int, int]): Color assigned to the object.
        label (str): The core prompt/label for the object.
    """
    mask: np.ndarray
    color: Tuple[int, int, int]
    label: str

@dataclass
class VideoDetectionDetail:
    """
    Stores details for each detected object in a video, including its SAM object ID,
    the core prompt/label, and the assigned color for consistent visualization.

    Attributes:
        sam_id (int): The unique SAM object ID assigned for tracking in the video.
        core_prompt (str): The core prompt/label for the detected object.
        color (Tuple[int, int, int]): The RGB color assigned to this object for overlays.
    """
    sam_id: int
    core_prompt: str
    color: Tuple[int, int, int]

@dataclass
class PropagatedFrameOutput:
    """
    Stores all relevant data for a single propagated frame in video segmentation.

    Attributes:
        current_pil_img (Image.Image): The current frame as a PIL image.
        frame_num (int): The frame number (1-based).
        sam_obj_ids_tensor (Any): Tensor or list of SAM object IDs for this frame.
        mask_logits_tensor (Any): Tensor of mask logits for this frame.
        detection_details_map (List[Dict[str, Any]]): List of detection details for mapping IDs.
        output_dir (str): Output directory for saving results.
    """
    current_pil_img: Image.Image
    frame_num: int
    sam_obj_ids_tensor: Any
    mask_logits_tensor: Any
    detection_details_map: List[Dict[str, Any]]
    output_dir: str

@dataclass
class SingleDetectionInput:
    """
    Stores all relevant data for processing a single detection in an image.

    Attributes:
        pil_image (Image.Image): The PIL image being processed.
        detection_detail (dict): The detection dictionary for the object.
        obj_idx (int): The index of the object in the detections list.
        base_name (str): The base name of the input image file.
        output_dir (str): Output directory for saving results.
    """
    pil_image: Image.Image
    detection_detail: dict
    obj_idx: int
    base_name: str
    output_dir: str

@dataclass
class VideoProcessContext:
    """
    Holds all context/state for processing a video in SOWLv2Pipeline.
    """
    tmp_frames_dir: str
    initial_sam_state: Any
    first_img_path: str
    first_pil_img: Image.Image
    detection_details_for_video: List[Dict[str, Any]]
    updated_sam_state: Any

@dataclass
class TempBinaryPaths:
    """Paths for temporary binary files."""
    path: str
    merged_path: str

@dataclass
class TempOverlayPaths:
    """Paths for temporary overlay files."""
    path: str
    merged_path: str

@dataclass
class TempVideoOutputPaths:
    """Paths for temporary video output files."""
    path: str
    binary_path: str
    overlay_path: str

@dataclass
class VideoDirectories:
    """
    Data class to store video processing directory paths.
    """
    temp_dir: str
    binary: TempBinaryPaths
    overlay: TempOverlayPaths
    video: TempVideoOutputPaths

@dataclass
class VideoProcessOptions:
    """
    Data class to store video processing options and flags.
    """
    binary: bool
    overlay: bool
    merged: bool
    fps: int

@dataclass
class MergedFrameItems:
    """
    Data class to store items for merged frame processing.
    """
    binary_items: List[np.ndarray]  # Store binary masks
    overlay_items: List[Tuple[np.ndarray, Tuple[int, int, int]]]  # Store (mask, color) pairs

@dataclass
class ObjectContext:
    """Context for a single object in a frame."""
    sam_id: int
    core_prompt_str: str
    object_color: Tuple[int, int, int]

@dataclass
class ObjectMaskProcessData:
    """
    Data class to store parameters for processing object masks.
    """
    mask_for_obj: Any  # The mask tensor/logits for the current object
    obj_ctx: ObjectContext  # Context of the object being processed
    frame_num: int
    pil_image: Image.Image  # The full frame PIL image
    dirs: Dict[str, str]  # Output directories for this frame's individual objects
    merged_items: MergedFrameItems # Collection for items to be merged later

@dataclass
class SaveOutputsConfig:
    """Configuration for saving detection outputs."""
    output_dir: str
    base_name: str
    core_prompt_slug: str
    obj_idx: int
    mask_np: np.ndarray
    pil_image: Image.Image
    object_color: Tuple[int, int, int]

@dataclass
class ProcessSingleObjectConfig:
    """Configuration for processing a single object in a frame."""
    frame_output: PropagatedFrameOutput
    sam_id: int
    obj_idx: int
    dirs_frame_output: Dict[str, str]
    merged_items: MergedFrameItems
