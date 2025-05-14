"""
Dataclasses for configuring the SOWLv2 object detection and segmentation pipeline.
"""

from dataclasses import dataclass

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
