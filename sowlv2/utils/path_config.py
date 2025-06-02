"""
Centralized configuration for file paths and naming patterns in SOWLv2.
"""
import os
import re
from enum import Enum, auto
from typing import Dict, List, Union

class OutputType(Enum):
    """Types of outputs that can be generated."""
    BINARY = auto()
    OVERLAY = auto()
    MERGED = auto()

class FilePattern:
    """File naming patterns for different types of outputs."""
    # Frame number format
    FRAME_NUM_FORMAT = "{:06d}"

    # Individual object file patterns
    INDIVIDUAL_MASK = "{frame_num}_obj{obj_id}_{prompt}_mask.png"
    INDIVIDUAL_OVERLAY = "{frame_num}_obj{obj_id}_{prompt}_overlay.png"

    # Merged file patterns
    MERGED_MASK = "{frame_num}_merged_mask.png"
    MERGED_OVERLAY = "{frame_num}_merged_overlay.png"

    # Video file patterns
    VIDEO_MASK = "{obj_id}_mask.mp4"
    VIDEO_OVERLAY = "{obj_id}_overlay.mp4"
    VIDEO_MERGED_MASK = "merged_mask.mp4"
    VIDEO_MERGED_OVERLAY = "merged_overlay.mp4"

    @classmethod
    def get_individual_patterns(cls) -> Dict[str, str]:
        """Get patterns for individual object outputs."""
        return {
            "mask": cls.INDIVIDUAL_MASK,
            "overlay": cls.INDIVIDUAL_OVERLAY
        }

    @classmethod
    def get_merged_patterns(cls) -> Dict[str, str]:
        """Get patterns for merged outputs."""
        return {
            "mask": cls.MERGED_MASK,
            "overlay": cls.MERGED_OVERLAY
        }

class DirectoryStructure:
    """Directory structure configuration."""
    # Base directories
    BINARY = "binary"
    OVERLAY = "overlay"
    VIDEO = "video"

    # Subdirectories
    FRAMES = "frames"
    MERGED = "merged"

    @classmethod
    def get_directory_map(cls, base_dir: str, include_video: bool = False) -> Dict[str, str]:
        """Get a mapping of directory types to their paths."""
        dirs = {
            cls.BINARY: os.path.join(base_dir, cls.BINARY),
            f"{cls.BINARY}_frames": os.path.join(base_dir, cls.BINARY, cls.FRAMES),
            f"{cls.BINARY}_merged": os.path.join(base_dir, cls.BINARY, cls.MERGED),
            cls.OVERLAY: os.path.join(base_dir, cls.OVERLAY),
            f"{cls.OVERLAY}_frames": os.path.join(base_dir, cls.OVERLAY, cls.FRAMES),
            f"{cls.OVERLAY}_merged": os.path.join(base_dir, cls.OVERLAY, cls.MERGED),
        }

        if include_video:
            dirs.update({
                cls.VIDEO: os.path.join(base_dir, cls.VIDEO),
                f"{cls.VIDEO}_binary": os.path.join(base_dir, cls.VIDEO, cls.BINARY),
                f"{cls.VIDEO}_overlay": os.path.join(base_dir, cls.VIDEO, cls.OVERLAY),
            })

        return dirs

    @classmethod
    def get_base_directories(cls) -> List[str]:
        """Get list of base directory names."""
        return [cls.BINARY, cls.OVERLAY, cls.VIDEO]

class FilePatternMatcher:
    """Pattern matching utilities for file operations."""

    @staticmethod
    def get_individual_mask_pattern() -> str:
        """Get the regex pattern for individual mask files."""
        return r"(\d+)_obj(\d+)_(.*?)_mask\.png"

    @staticmethod
    def get_simple_mask_pattern() -> str:
        """Get the regex pattern for simple mask files without prompt."""
        return r"(\d+)_(obj\d+)_mask\.png"

    @staticmethod
    def get_merged_mask_pattern() -> str:
        """Get the regex pattern for merged mask files."""
        return r"(\d+)_merged_mask\.png"

    @staticmethod
    def get_merged_overlay_pattern() -> str:
        """Get the regex pattern for merged overlay files."""
        return r"(\d+)_merged_overlay\.png"

    @staticmethod
    def natural_sort_key(s: str) -> List[Union[int, str]]:
        """Get a key for natural sorting of filenames."""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
