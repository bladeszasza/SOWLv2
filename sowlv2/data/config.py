"""
Dataclasses for configuring the SOWLv2 object detection and segmentation pipeline.
"""

from dataclasses import dataclass
from typing import Any

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
    owl_skip_frames: int

@dataclass
class SaveMaskOverlayConfig:
    """
    Configuration for saving masks and overlays for a frame.
    """
    pil_img: Any
    frame_idx: int
    obj_ids: Any
    masks: Any
    out_dir: str
