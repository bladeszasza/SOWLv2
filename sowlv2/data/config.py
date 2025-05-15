"""
Dataclasses for configuring the SOWLv2 object detection and segmentation pipeline.
"""

from dataclasses import dataclass
from typing import Any, Tuple, List, Dict
import numpy as np
from PIL import Image

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
    mask_np: np.ndarray
    mask_img_pil: Image.Image
    mask_file: str
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
