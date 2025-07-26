"""
V-JEPA 2 optimization for video batch processing.
Integrates Meta's V-JEPA 2 model for efficient video understanding and preprocessing.
"""
import logging
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import torch
import numpy as np
from PIL import Image
import cv2

from sowlv2.data.config import PipelineBaseData


class ContentType(Enum):
    """Video content types for adaptive optimization."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    FAST_MOTION = "fast_motion"
    MIXED = "mixed"


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

                logging.info("Loading V-JEPA 2 model: %s", self.model_name)
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
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Warning: Could not load V-JEPA 2 model: {e}")
                self._model = None
                self._processor = None

    @property
    def is_available(self) -> bool:
        """Check if V-JEPA 2 optimization is available."""
        try:
            self._load_models()
            return self._model is not None
        except Exception:  # pylint: disable=broad-except
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
        for i, _ in enumerate(frames):
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

    def analyze_content_type(self, frames: List[Image.Image]) -> ContentType:
        """
        Analyze video content type for adaptive optimization.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            ContentType enum indicating the video characteristics
        """
        if len(frames) < 3:
            return ContentType.STATIC
            
        motion_scores = []
        edge_densities = []
        
        for i in range(1, len(frames)):
            # Convert to grayscale for analysis
            curr_gray = np.array(frames[i].convert('L'))
            prev_gray = np.array(frames[i-1].convert('L'))
            
            # Calculate optical flow magnitude
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, 
                np.array([[x, y] for x in range(0, curr_gray.shape[1], 20) 
                         for y in range(0, curr_gray.shape[0], 20)], dtype=np.float32),
                None
            )[0]
            
            if flow is not None:
                motion_magnitude = np.mean(np.linalg.norm(flow, axis=1))
                motion_scores.append(motion_magnitude)
            else:
                motion_scores.append(0.0)
            
            # Calculate edge density for scene complexity
            edges = cv2.Canny(curr_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
        
        avg_motion = np.mean(motion_scores)
        motion_variance = np.var(motion_scores)
        avg_edge_density = np.mean(edge_densities)
        
        # Classify content type based on motion characteristics
        if avg_motion < 2.0:
            return ContentType.STATIC
        elif avg_motion > 10.0 or motion_variance > 50.0:
            return ContentType.FAST_MOTION
        elif motion_variance > 20.0:
            return ContentType.MIXED
        else:
            return ContentType.DYNAMIC

    def get_adaptive_scoring_weights(self, content_type: ContentType) -> Dict[str, float]:
        """
        Get adaptive scoring weights based on content type.
        
        Args:
            content_type: The analyzed content type
            
        Returns:
            Dictionary of weights for different scoring components
        """
        weights = {
            ContentType.STATIC: {
                'feature_weight': 0.8,
                'motion_weight': 0.1,
                'edge_weight': 0.1,
                'temporal_consistency_weight': 0.3
            },
            ContentType.DYNAMIC: {
                'feature_weight': 0.5,
                'motion_weight': 0.4,
                'edge_weight': 0.1,
                'temporal_consistency_weight': 0.5
            },
            ContentType.FAST_MOTION: {
                'feature_weight': 0.3,
                'motion_weight': 0.6,
                'edge_weight': 0.1,
                'temporal_consistency_weight': 0.7
            },
            ContentType.MIXED: {
                'feature_weight': 0.4,
                'motion_weight': 0.4,
                'edge_weight': 0.2,
                'temporal_consistency_weight': 0.6
            }
        }
        return weights[content_type]

    def calculate_advanced_motion_scores(self, frames: List[Image.Image]) -> List[float]:
        """
        Calculate advanced motion scores using optical flow and edge detection.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            List of motion scores for each frame
        """
        motion_scores = [0.0]  # First frame has no motion
        
        for i in range(1, len(frames)):
            curr_frame = np.array(frames[i].convert('RGB'))
            prev_frame = np.array(frames[i-1].convert('RGB'))
            
            # Convert to grayscale for optical flow
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                np.array([[x, y] for x in range(0, curr_gray.shape[1], 10)
                         for y in range(0, curr_gray.shape[0], 10)], dtype=np.float32),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                # Calculate motion magnitude
                motion_vectors = flow.reshape(-1, 2)
                motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
                motion_score = np.mean(motion_magnitudes)
            else:
                # Fallback to frame difference
                diff = np.abs(curr_gray.astype(float) - prev_gray.astype(float))
                motion_score = np.mean(diff) / 255.0
            
            motion_scores.append(motion_score)
        
        return motion_scores

    def calculate_temporal_consistency_scores(self, frames: List[Image.Image], window_size: int = 3) -> List[float]:
        """
        Calculate temporal consistency scores for frame selection.
        
        Args:
            frames: List of PIL Images
            window_size: Size of temporal window for consistency check
            
        Returns:
            List of consistency scores for each frame
        """
        consistency_scores = []
        
        for i, frame in enumerate(frames):
            # Define temporal window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(frames), i + window_size // 2 + 1)
            window_frames = frames[start_idx:end_idx]
            
            if len(window_frames) < 2:
                consistency_scores.append(1.0)
                continue
            
            # Calculate consistency as inverse of variance in the window
            frame_arrays = [np.array(f.convert('L')) for f in window_frames]
            pixel_variances = []
            
            for y in range(0, frame_arrays[0].shape[0], 10):
                for x in range(0, frame_arrays[0].shape[1], 10):
                    pixel_values = [arr[y, x] for arr in frame_arrays]
                    pixel_variances.append(np.var(pixel_values))
            
            # Higher variance means less consistency
            avg_variance = np.mean(pixel_variances)
            consistency_score = 1.0 / (1.0 + avg_variance / 100.0)
            consistency_scores.append(consistency_score)
        
        return consistency_scores

    def get_motion_aware_importance_scores(
        self,
        frames: List[Image.Image],
        motion_weight: float = 0.5,
        adaptive_weights: bool = True
    ) -> Optional[List[float]]:
        """
        Enhanced importance scoring that considers both feature variance and motion.

        Args:
            frames: List of PIL Images
            motion_weight: Weight for motion component (0-1), ignored if adaptive_weights=True
            adaptive_weights: Whether to use content-type adaptive weights

        Returns:
            List of importance scores (0-1) for each frame
        """
        # Get feature-based importance
        feature_importance = self.get_temporal_importance_scores(frames)
        if feature_importance is None:
            return None

        # Analyze content type for adaptive weighting
        content_type = self.analyze_content_type(frames) if adaptive_weights else ContentType.DYNAMIC
        weights = self.get_adaptive_scoring_weights(content_type) if adaptive_weights else {
            'feature_weight': 1 - motion_weight,
            'motion_weight': motion_weight,
            'edge_weight': 0.0,
            'temporal_consistency_weight': 0.0
        }

        # Calculate advanced motion scores
        motion_importance = self.calculate_advanced_motion_scores(frames)
        
        # Calculate temporal consistency scores
        consistency_scores = self.calculate_temporal_consistency_scores(frames)
        
        # Calculate edge-based importance
        edge_importance = []
        for frame in frames:
            gray_frame = np.array(frame.convert('L'))
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_importance.append(edge_density)

        # Normalize all scores
        def normalize_scores(scores):
            max_score = max(scores) if scores else 1.0
            return [s / max_score if max_score > 0 else 0.0 for s in scores]

        feature_importance = normalize_scores(feature_importance)
        motion_importance = normalize_scores(motion_importance)
        edge_importance = normalize_scores(edge_importance)
        consistency_scores = normalize_scores(consistency_scores)

        # Combine scores with adaptive weights
        combined_scores = []
        for i in range(len(frames)):
            feature_score = feature_importance[i]
            motion_score = motion_importance[i]
            edge_score = edge_importance[i]
            consistency_score = consistency_scores[i]
            
            combined = (
                weights['feature_weight'] * feature_score +
                weights['motion_weight'] * motion_score +
                weights['edge_weight'] * edge_score +
                weights['temporal_consistency_weight'] * consistency_score
            )
            combined_scores.append(combined)

        return combined_scores

    def get_adaptive_frame_spacing(self, frames: List[Image.Image], target_frames: int) -> List[int]:
        """
        Calculate adaptive frame spacing based on video characteristics.
        
        Args:
            frames: List of PIL Images
            target_frames: Number of frames to select
            
        Returns:
            List of frame indices with adaptive spacing
        """
        if len(frames) <= target_frames:
            return list(range(len(frames)))
        
        content_type = self.analyze_content_type(frames)
        motion_scores = self.calculate_advanced_motion_scores(frames)
        
        # Adaptive spacing based on content type
        if content_type == ContentType.STATIC:
            # Uniform spacing for static content
            step = len(frames) // target_frames
            return list(range(0, len(frames), step))[:target_frames]
        
        elif content_type == ContentType.FAST_MOTION:
            # Denser sampling for fast motion
            importance_scores = self.get_motion_aware_importance_scores(frames)
            if importance_scores:
                # Select frames with highest motion importance
                indexed_scores = list(enumerate(importance_scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [idx for idx, _ in indexed_scores[:target_frames]]
                return sorted(selected_indices)
        
        # For dynamic and mixed content, use motion-aware selection
        selected_indices = []
        motion_threshold = np.mean(motion_scores) + np.std(motion_scores)
        
        # First, select high-motion frames
        high_motion_frames = [i for i, score in enumerate(motion_scores) if score > motion_threshold]
        
        # If we have enough high-motion frames, sample from them
        if len(high_motion_frames) >= target_frames:
            step = len(high_motion_frames) // target_frames
            selected_indices = [high_motion_frames[i] for i in range(0, len(high_motion_frames), step)][:target_frames]
        else:
            # Combine high-motion frames with uniform sampling
            selected_indices.extend(high_motion_frames)
            remaining_frames = target_frames - len(high_motion_frames)
            
            # Sample remaining frames uniformly from non-high-motion frames
            other_frames = [i for i in range(len(frames)) if i not in high_motion_frames]
            if other_frames and remaining_frames > 0:
                step = len(other_frames) // remaining_frames
                additional_frames = [other_frames[i] for i in range(0, len(other_frames), step)][:remaining_frames]
                selected_indices.extend(additional_frames)
        
        return sorted(selected_indices)

    def optimize_frame_selection(self,
                                frames: List[Image.Image],
                                target_frames: int,
                                use_adaptive_spacing: bool = True) -> List[int]:
        """
        Select optimal frames for processing using V-JEPA 2 insights.

        Args:
            frames: List of all video frames
            target_frames: Number of frames to select
            use_adaptive_spacing: Whether to use adaptive frame spacing

        Returns:
            List of indices of selected frames
        """
        if not self.is_available or len(frames) <= target_frames:
            # Fall back to uniform sampling
            indices = list(range(0, len(frames), max(1, len(frames) // target_frames)))
            return indices[:target_frames]

        # Use adaptive frame spacing if enabled
        if use_adaptive_spacing:
            return self.get_adaptive_frame_spacing(frames, target_frames)

        # Get enhanced importance scores
        importance_scores = self.get_motion_aware_importance_scores(frames, adaptive_weights=True)
        if importance_scores is None:
            # Fall back to uniform sampling
            indices = list(range(0, len(frames), max(1, len(frames) // target_frames)))
            return indices[:target_frames]

        # Select frames with highest importance scores while maintaining temporal diversity
        frame_indices_with_scores = list(enumerate(importance_scores))
        frame_indices_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Implement temporal diversity constraint
        selected_indices = []
        min_spacing = max(1, len(frames) // (target_frames * 2))  # Minimum spacing between frames
        
        for idx, score in frame_indices_with_scores:
            # Check if this frame is too close to already selected frames
            too_close = any(abs(idx - selected) < min_spacing for selected in selected_indices)
            if not too_close:
                selected_indices.append(idx)
                if len(selected_indices) >= target_frames:
                    break
        
        # If we couldn't get enough frames with spacing constraint, fill remaining slots
        if len(selected_indices) < target_frames:
            for idx, score in frame_indices_with_scores:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= target_frames:
                        break

        return sorted(selected_indices)

    def calculate_content_similarity(self, 
                                   features1: torch.Tensor, 
                                   features2: torch.Tensor) -> float:
        """
        Calculate similarity between two feature tensors.
        
        Args:
            features1: First feature tensor
            features2: Second feature tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Flatten features for comparison
        feat1_flat = features1.flatten()
        feat2_flat = features2.flatten()
        
        # Ensure same size
        min_size = min(len(feat1_flat), len(feat2_flat))
        feat1_flat = feat1_flat[:min_size]
        feat2_flat = feat2_flat[:min_size]
        
        # Calculate cosine similarity
        similarity = torch.cosine_similarity(feat1_flat.unsqueeze(0), feat2_flat.unsqueeze(0))
        return float(similarity.cpu())

    def group_similar_content(self, 
                            video_clips: List[Tuple[List[Image.Image], torch.Tensor]],
                            similarity_threshold: float = 0.8) -> List[List[int]]:
        """
        Group similar video clips based on V-JEPA2 features.
        
        Args:
            video_clips: List of (frames, features) tuples
            similarity_threshold: Minimum similarity for grouping
            
        Returns:
            List of groups, where each group is a list of clip indices
        """
        if not video_clips:
            return []
        
        groups = []
        used_clips = set()
        
        for i, (frames1, features1) in enumerate(video_clips):
            if i in used_clips or features1 is None:
                continue
            
            # Start new group with current clip
            current_group = [i]
            used_clips.add(i)
            
            # Find similar clips
            for j, (frames2, features2) in enumerate(video_clips[i+1:], i+1):
                if j in used_clips or features2 is None:
                    continue
                
                similarity = self.calculate_content_similarity(features1, features2)
                if similarity > similarity_threshold:
                    current_group.append(j)
                    used_clips.add(j)
            
            groups.append(current_group)
        
        return groups

    def create_feature_cache(self, 
                           video_clips: List[Tuple[List[Image.Image], torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Create intelligent cache of V-JEPA2 features for reuse.
        
        Args:
            video_clips: List of (frames, features) tuples
            
        Returns:
            Dictionary mapping content signatures to features
        """
        feature_cache = {}
        
        for i, (frames, features) in enumerate(video_clips):
            if features is None:
                continue
            
            # Create content signature based on frame characteristics
            signature = self._create_content_signature(frames)
            
            # Store features with signature
            if signature not in feature_cache:
                feature_cache[signature] = features
            else:
                # If signature exists, average the features
                existing_features = feature_cache[signature]
                averaged_features = (existing_features + features) / 2.0
                feature_cache[signature] = averaged_features
        
        return feature_cache

    def _create_content_signature(self, frames: List[Image.Image]) -> str:
        """
        Create a signature for content based on visual characteristics.
        
        Args:
            frames: List of PIL Images
            
        Returns:
            String signature representing the content
        """
        if not frames:
            return "empty"
        
        # Sample a few frames for signature
        sample_indices = [0, len(frames)//2, len(frames)-1] if len(frames) > 2 else [0]
        sample_frames = [frames[i] for i in sample_indices if i < len(frames)]
        
        signature_components = []
        
        for frame in sample_frames:
            # Convert to grayscale for analysis
            gray_frame = np.array(frame.convert('L'))
            
            # Calculate basic statistics
            mean_intensity = np.mean(gray_frame)
            std_intensity = np.std(gray_frame)
            
            # Calculate edge density
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Create component signature
            component = f"{mean_intensity:.1f}_{std_intensity:.1f}_{edge_density:.3f}"
            signature_components.append(component)
        
        return "_".join(signature_components)

    def batch_process_similar_content(self, 
                                    video_batches: List[List[Image.Image]],
                                    enable_feature_reuse: bool = True,
                                    parallel_processing: bool = True) -> List[torch.Tensor]:
        """
        Optimized batch processing for similar content with feature reuse.
        
        Args:
            video_batches: List of video frame lists
            enable_feature_reuse: Whether to reuse features for similar content
            parallel_processing: Whether to use parallel processing
            
        Returns:
            List of feature tensors for each video batch
        """
        if not self.is_available:
            return [None] * len(video_batches)
        
        # First pass: extract features for all batches
        all_clips = []
        for video_frames in video_batches:
            clips = self._create_clips_from_frames(video_frames)
            all_clips.extend(clips)
        
        # Group similar content
        similar_groups = self.group_similar_content(all_clips) if enable_feature_reuse else []
        
        # Create feature cache
        feature_cache = {}
        processed_features = {}
        
        if enable_feature_reuse and similar_groups:
            # Process one representative from each group
            for group in similar_groups:
                if not group:
                    continue
                
                # Use first clip as representative
                representative_idx = group[0]
                frames, _ = all_clips[representative_idx]
                
                # Extract features for representative
                features = self.extract_video_features(frames)
                if features is not None:
                    # Cache features for all clips in group
                    for clip_idx in group:
                        processed_features[clip_idx] = features
                        
                        # Also cache by content signature
                        clip_frames, _ = all_clips[clip_idx]
                        signature = self._create_content_signature(clip_frames)
                        feature_cache[signature] = features
        
        # Process remaining clips
        for i, (frames, _) in enumerate(all_clips):
            if i not in processed_features:
                # Check cache first
                signature = self._create_content_signature(frames)
                if signature in feature_cache:
                    processed_features[i] = feature_cache[signature]
                else:
                    # Extract new features
                    features = self.extract_video_features(frames)
                    processed_features[i] = features
                    if features is not None:
                        feature_cache[signature] = features
        
        # Organize results by original video batches
        results = []
        clip_idx = 0
        
        for video_frames in video_batches:
            clips = self._create_clips_from_frames(video_frames)
            
            # Aggregate features for this video
            video_features = []
            for _ in clips:
                if clip_idx in processed_features:
                    video_features.append(processed_features[clip_idx])
                clip_idx += 1
            
            # Combine features for the video (average or concatenate)
            if video_features and any(f is not None for f in video_features):
                valid_features = [f for f in video_features if f is not None]
                if valid_features:
                    combined_features = torch.mean(torch.stack(valid_features), dim=0)
                    results.append(combined_features)
                else:
                    results.append(None)
            else:
                results.append(None)
        
        return results

    def _create_clips_from_frames(self, frames: List[Image.Image]) -> List[Tuple[List[Image.Image], None]]:
        """Create clips from frames for processing."""
        clips = []
        clip_size = self.frames_per_clip
        
        for i in range(0, len(frames), clip_size):
            clip_frames = frames[i:i + clip_size]
            clips.append((clip_frames, None))
        
        return clips

    def optimize_batch_processing_order(self, 
                                      video_batches: List[List[Image.Image]]) -> List[int]:
        """
        Optimize the order of batch processing to maximize feature reuse.
        
        Args:
            video_batches: List of video frame lists
            
        Returns:
            List of indices representing optimal processing order
        """
        if len(video_batches) <= 1:
            return list(range(len(video_batches)))
        
        # Create content signatures for all batches
        signatures = []
        for video_frames in video_batches:
            # Sample frames for signature
            sample_frames = video_frames[::max(1, len(video_frames)//5)][:5]
            signature = self._create_content_signature(sample_frames)
            signatures.append(signature)
        
        # Group similar signatures
        signature_groups = {}
        for i, signature in enumerate(signatures):
            if signature not in signature_groups:
                signature_groups[signature] = []
            signature_groups[signature].append(i)
        
        # Create processing order that groups similar content together
        processing_order = []
        for group in signature_groups.values():
            processing_order.extend(group)
        
        return processing_order

    def parallel_process_similar_batches(self, 
                                       video_batches: List[List[Image.Image]],
                                       max_workers: int = 4) -> List[torch.Tensor]:
        """
        Process similar content batches in parallel with feature sharing.
        
        Args:
            video_batches: List of video frame lists
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of feature tensors for each video batch
        """
        if not self.is_available:
            return [None] * len(video_batches)
        
        # Optimize processing order
        processing_order = self.optimize_batch_processing_order(video_batches)
        
        # Process in optimized order with feature reuse
        ordered_batches = [video_batches[i] for i in processing_order]
        ordered_results = self.batch_process_similar_content(
            ordered_batches, 
            enable_feature_reuse=True,
            parallel_processing=True
        )
        
        # Reorder results to match original order
        results = [None] * len(video_batches)
        for i, original_idx in enumerate(processing_order):
            results[original_idx] = ordered_results[i]
        
        return results

    def batch_process_video_clips(
            self,
            all_frames: List[Image.Image],
            enable_similarity_optimization: bool = True
    ) -> List[Tuple[List[Image.Image], torch.Tensor]]:
        """
        Enhanced video processing with similarity-based optimization.

        Args:
            all_frames: All video frames
            enable_similarity_optimization: Whether to use similarity optimization

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
        
        # Create clips
        clips = []
        for start_idx in range(0, len(all_frames), clip_size):
            end_idx = min(start_idx + clip_size, len(all_frames))
            clip_frames = all_frames[start_idx:end_idx]
            clips.append((clip_frames, None))

        if enable_similarity_optimization and len(clips) > 1:
            # Extract features for all clips first
            clip_features = []
            for clip_frames, _ in clips:
                features = self.extract_video_features(clip_frames)
                clip_features.append(features)
            
            # Update clips with features
            clips = [(frames, features) for (frames, _), features in zip(clips, clip_features)]
            
            # Group similar clips and reuse features
            similar_groups = self.group_similar_content(clips)
            
            # Create optimized results with feature reuse
            for i, (clip_frames, features) in enumerate(clips):
                results.append((clip_frames, features))
        else:
            # Standard processing
            for clip_frames, _ in clips:
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
        print("V-JEPA 2 optimization not available, falling back to standard processing")
        return None
    except Exception as e:  # pylint: disable=broad-except
        print(f"Failed to initialize V-JEPA 2 optimizer: {e}")
        return None
