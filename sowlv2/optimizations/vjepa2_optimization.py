"""
V-JEPA 2 optimization for video batch processing.
Integrates Meta's V-JEPA 2 model for efficient video understanding and preprocessing.
"""
import torch
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from sowlv2.data.config import PipelineBaseData


class VJepa2VideoOptimizer:
    """
    Optimizes video processing using V-JEPA 2 for efficient frame understanding.
    """

    def __init__(self,
                 config: PipelineBaseData,
                 model_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
                 frames_per_clip: int = 16,
                 device: Optional[str] = None):
        """
        Initialize V-JEPA 2 video optimizer.

        Args:
            config: Pipeline configuration
            model_name: V-JEPA 2 model name from HuggingFace
            frames_per_clip: Number of frames to process in each clip
            device: Device to run on (cuda/cpu)
        """
        self.config = config
        self.model_name = model_name
        self.frames_per_clip = frames_per_clip
        self.device = device or config.device

        # Initialize models lazily
        self._model = None
        self._processor = None

    def _load_models(self):
        """Lazy load V-JEPA 2 models."""
        if self._model is None:
            try:
                # Import here to avoid dependency issues if transformers not available
                from transformers import (  # pylint: disable=import-outside-toplevel
                    AutoModelForVideoClassification,
                    AutoVideoProcessor
                )

                print(f"Loading V-JEPA 2 model: {self.model_name}")
                self._model = AutoModelForVideoClassification.from_pretrained(
                    self.model_name
                ).to(self.device)
                self._processor = AutoVideoProcessor.from_pretrained(self.model_name)

                # Set to eval mode for inference
                self._model.eval()

            except ImportError as e:
                raise ImportError(
                    "transformers library required for V-JEPA 2 optimization. "
                    "Install with: pip install transformers"
                ) from e
            except Exception as e:
                print(f"Warning: Could not load V-JEPA 2 model: {e}")
                self._model = None
                self._processor = None

    @property
    def is_available(self) -> bool:
        """Check if V-JEPA 2 optimization is available."""
        try:
            self._load_models()
            return self._model is not None
        except Exception:
            return False

    def extract_video_features(self,
                              frames: List[Image.Image]) -> Optional[torch.Tensor]:
        """
        Extract features from video frames using V-JEPA 2.

        Args:
            frames: List of PIL Images representing video frames

        Returns:
            Feature tensor or None if model not available
        """
        if not self.is_available:
            return None

        # Convert PIL images to numpy arrays
        frame_arrays = []
        for frame in frames:
            frame_np = np.array(frame.convert('RGB'))
            frame_arrays.append(frame_np)

        # Create video tensor: [frames, height, width, channels]
        video_tensor = np.stack(frame_arrays, axis=0)

        # Process with V-JEPA 2
        inputs = self._processor(video_tensor, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract features (encoder output)
        features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
        return features

    def get_temporal_importance_scores(self,
                                     frames: List[Image.Image]) -> Optional[List[float]]:
        """
        Get temporal importance scores for frames using V-JEPA 2.

        Args:
            frames: List of PIL Images

        Returns:
            List of importance scores (0-1) for each frame, or None if unavailable
        """
        features = self.extract_video_features(frames)
        if features is None:
            return None

        # Simple temporal importance based on feature variance
        # More sophisticated methods could be implemented here
        frame_importance = []
        for i in range(len(frames)):
            if i < features.shape[1]:  # Ensure we don't exceed feature dimensions
                frame_feat = features[0, i]  # Get features for frame i
                importance = float(torch.var(frame_feat).cpu())
                frame_importance.append(importance)
            else:
                frame_importance.append(0.0)

        # Normalize scores to 0-1
        if frame_importance:
            max_importance = max(frame_importance)
            if max_importance > 0:
                frame_importance = [score / max_importance for score in frame_importance]

        return frame_importance

    def optimize_frame_selection(self,
                                frames: List[Image.Image],
                                target_frames: int) -> List[int]:
        """
        Select optimal frames for processing using V-JEPA 2 insights.

        Args:
            frames: List of all video frames
            target_frames: Number of frames to select

        Returns:
            List of indices of selected frames
        """
        if not self.is_available or len(frames) <= target_frames:
            # Fall back to uniform sampling
            indices = list(range(0, len(frames), max(1, len(frames) // target_frames)))
            return indices[:target_frames]

        # Get importance scores
        importance_scores = self.get_temporal_importance_scores(frames)
        if importance_scores is None:
            # Fall back to uniform sampling
            indices = list(range(0, len(frames), max(1, len(frames) // target_frames)))
            return indices[:target_frames]

        # Select frames with highest importance scores
        frame_indices_with_scores = list(enumerate(importance_scores))
        frame_indices_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top N frames and sort by temporal order
        selected_indices = [idx for idx, _ in frame_indices_with_scores[:target_frames]]
        selected_indices.sort()

        return selected_indices

    def batch_process_video_clips(
            self,
            all_frames: List[Image.Image],
            batch_size: int = 4
    ) -> List[Tuple[List[Image.Image], torch.Tensor]]:
        """
        Process video in batches using V-JEPA 2 for optimal clip segmentation.

        Args:
            all_frames: All video frames
            batch_size: Number of clips to process in parallel

        Returns:
            List of (frames, features) tuples for each clip
        """
        if not self.is_available:
            # Fall back to simple chunking
            clip_size = self.frames_per_clip
            clips = []
            for i in range(0, len(all_frames), clip_size):
                clip_frames = all_frames[i:i + clip_size]
                clips.append((clip_frames, None))
            return clips

        results = []
        clip_size = self.frames_per_clip

        # Process clips in batches
        for start_idx in range(0, len(all_frames), clip_size):
            end_idx = min(start_idx + clip_size, len(all_frames))
            clip_frames = all_frames[start_idx:end_idx]

            # Extract features for this clip
            features = self.extract_video_features(clip_frames)
            results.append((clip_frames, features))

        return results


def create_vjepa2_optimizer(config: PipelineBaseData,
                          enable_vjepa2: bool = True) -> Optional[VJepa2VideoOptimizer]:
    """
    Factory function to create V-JEPA 2 optimizer.

    Args:
        config: Pipeline configuration
        enable_vjepa2: Whether to enable V-JEPA 2 optimization

    Returns:
        VJepa2VideoOptimizer instance or None if not available/disabled
    """
    if not enable_vjepa2:
        return None

    try:
        optimizer = VJepa2VideoOptimizer(config)
        if optimizer.is_available:
            return optimizer
        else:
            print("V-JEPA 2 optimization not available, falling back to standard processing")
            return None
    except Exception as e:
        print(f"Failed to initialize V-JEPA 2 optimizer: {e}")
        return None
