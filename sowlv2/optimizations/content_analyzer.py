"""
Content-aware optimization module for adaptive video processing.
Analyzes video content characteristics to optimize processing parameters.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from PIL import Image
import logging

from sowlv2.optimizations.vjepa2_optimization import ContentType


@dataclass
class ContentAnalysis:
    """Results of video content analysis."""
    content_type: ContentType
    motion_characteristics: Dict[str, float]
    scene_complexity: Dict[str, float]
    temporal_characteristics: Dict[str, float]
    optimization_recommendations: Dict[str, Any]


@dataclass
class OptimizationProfile:
    """Optimization profile for specific content types."""
    name: str
    content_type: ContentType
    frame_sampling_rate: float  # Fraction of frames to process
    batch_size_multiplier: float  # Multiplier for default batch size
    motion_threshold: float  # Threshold for motion detection
    consistency_weight: float  # Weight for temporal consistency
    feature_cache_size: int  # Size of feature cache
    parallel_processing: bool  # Whether to use parallel processing
    streaming_chunk_size: int  # Chunk size for streaming processing


class ContentAnalyzer:
    """
    Analyzes video content characteristics for adaptive optimization.
    """

    def __init__(self):
        """Initialize the content analyzer."""
        self.optimization_profiles = self._create_optimization_profiles()

    def _create_optimization_profiles(self) -> Dict[ContentType, OptimizationProfile]:
        """Create predefined optimization profiles for different content types."""
        profiles = {
            ContentType.STATIC: OptimizationProfile(
                name="Static Content",
                content_type=ContentType.STATIC,
                frame_sampling_rate=0.1,  # Process fewer frames
                batch_size_multiplier=2.0,  # Larger batches
                motion_threshold=2.0,
                consistency_weight=0.3,
                feature_cache_size=50,
                parallel_processing=True,
                streaming_chunk_size=200
            ),
            ContentType.DYNAMIC: OptimizationProfile(
                name="Dynamic Content",
                content_type=ContentType.DYNAMIC,
                frame_sampling_rate=0.3,  # Moderate frame sampling
                batch_size_multiplier=1.0,  # Standard batch size
                motion_threshold=5.0,
                consistency_weight=0.5,
                feature_cache_size=100,
                parallel_processing=True,
                streaming_chunk_size=100
            ),
            ContentType.FAST_MOTION: OptimizationProfile(
                name="Fast Motion",
                content_type=ContentType.FAST_MOTION,
                frame_sampling_rate=0.5,  # Process more frames
                batch_size_multiplier=0.7,  # Smaller batches
                motion_threshold=10.0,
                consistency_weight=0.7,
                feature_cache_size=150,
                parallel_processing=True,
                streaming_chunk_size=50
            ),
            ContentType.MIXED: OptimizationProfile(
                name="Mixed Content",
                content_type=ContentType.MIXED,
                frame_sampling_rate=0.4,  # Adaptive sampling
                batch_size_multiplier=0.8,
                motion_threshold=7.0,
                consistency_weight=0.6,
                feature_cache_size=120,
                parallel_processing=True,
                streaming_chunk_size=75
            )
        }
        return profiles

    def analyze_video_content(self, frames: List[Image.Image]) -> ContentAnalysis:
        """
        Comprehensive analysis of video content characteristics.

        Args:
            frames: List of PIL Images representing video frames

        Returns:
            ContentAnalysis object with detailed analysis results
        """
        if len(frames) < 2:
            return self._create_default_analysis()

        # Analyze motion characteristics
        motion_characteristics = self._analyze_motion_characteristics(frames)

        # Analyze scene complexity
        scene_complexity = self._analyze_scene_complexity(frames)

        # Analyze temporal characteristics
        temporal_characteristics = self._analyze_temporal_characteristics(frames)

        # Determine content type
        content_type = self._classify_content_type(
            motion_characteristics, scene_complexity, temporal_characteristics
        )

        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(
            content_type, motion_characteristics, scene_complexity, temporal_characteristics
        )

        return ContentAnalysis(
            content_type=content_type,
            motion_characteristics=motion_characteristics,
            scene_complexity=scene_complexity,
            temporal_characteristics=temporal_characteristics,
            optimization_recommendations=optimization_recommendations
        )

    def _analyze_motion_characteristics(self, frames: List[Image.Image]) -> Dict[str, float]:
        """Analyze motion characteristics of the video."""
        motion_scores = []
        motion_directions = []
        motion_accelerations = []

        prev_gray = None
        prev_flow = None

        for i, frame in enumerate(frames):
            curr_gray = np.array(frame.convert('L'))

            if prev_gray is not None:
                # Calculate optical flow
                try:
                    # Use sparse optical flow for efficiency
                    corners = cv2.goodFeaturesToTrack(
                        prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10
                    )

                    if corners is not None and len(corners) > 0:
                        flow, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, curr_gray, corners, None
                        )

                        # Filter good points
                        good_flow = flow[status == 1]
                        good_corners = corners[status == 1]

                        if len(good_flow) > 0:
                            # Calculate motion vectors
                            motion_vectors = good_flow - good_corners.reshape(-1, 2)
                            motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)

                            # Motion score
                            motion_score = np.mean(motion_magnitudes)
                            motion_scores.append(motion_score)

                            # Motion direction consistency
                            if len(motion_vectors) > 1:
                                angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
                                direction_consistency = 1.0 - np.std(angles) / np.pi
                                motion_directions.append(direction_consistency)

                            # Motion acceleration (if we have previous flow)
                            if prev_flow is not None and len(prev_flow) > 0:
                                # Simple acceleration estimation
                                acceleration = np.mean(np.abs(motion_magnitudes - prev_flow))
                                motion_accelerations.append(acceleration)

                            prev_flow = motion_magnitudes
                        else:
                            motion_scores.append(0.0)
                    else:
                        motion_scores.append(0.0)

                except Exception as e:
                    logging.warning(f"Motion analysis failed for frame {i}: {e}")
                    motion_scores.append(0.0)

            prev_gray = curr_gray

        return {
            'average_motion': np.mean(motion_scores) if motion_scores else 0.0,
            'motion_variance': np.var(motion_scores) if motion_scores else 0.0,
            'max_motion': np.max(motion_scores) if motion_scores else 0.0,
            'motion_consistency': np.mean(motion_directions) if motion_directions else 0.0,
            'motion_acceleration': np.mean(motion_accelerations) if motion_accelerations else 0.0
        }

    def _analyze_scene_complexity(self, frames: List[Image.Image]) -> Dict[str, float]:
        """Analyze scene complexity characteristics."""
        edge_densities = []
        texture_complexities = []
        color_diversities = []
        contrast_levels = []

        for frame in frames:
            # Convert to different formats for analysis
            gray_frame = np.array(frame.convert('L'))
            rgb_frame = np.array(frame.convert('RGB'))

            # Edge density
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)

            # Texture complexity using local binary patterns
            try:
                # Simple texture measure using gradient magnitude
                grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                texture_complexity = np.mean(gradient_magnitude)
                texture_complexities.append(texture_complexity)
            except Exception:
                texture_complexities.append(0.0)

            # Color diversity
            try:
                # Calculate color histogram entropy
                hist_r = cv2.calcHist([rgb_frame], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([rgb_frame], [1], None, [256], [0, 256])
                hist_b = cv2.calcHist([rgb_frame], [2], None, [256], [0, 256])

                # Normalize histograms
                hist_r = hist_r / np.sum(hist_r)
                hist_g = hist_g / np.sum(hist_g)
                hist_b = hist_b / np.sum(hist_b)

                # Calculate entropy
                entropy_r = -np.sum(hist_r * np.log2(hist_r + 1e-10))
                entropy_g = -np.sum(hist_g * np.log2(hist_g + 1e-10))
                entropy_b = -np.sum(hist_b * np.log2(hist_b + 1e-10))

                color_diversity = (entropy_r + entropy_g + entropy_b) / 3.0
                color_diversities.append(color_diversity)
            except Exception:
                color_diversities.append(0.0)

            # Contrast level
            contrast = np.std(gray_frame)
            contrast_levels.append(contrast)

        return {
            'average_edge_density': np.mean(edge_densities),
            'edge_density_variance': np.var(edge_densities),
            'average_texture_complexity': np.mean(texture_complexities),
            'average_color_diversity': np.mean(color_diversities),
            'average_contrast': np.mean(contrast_levels),
            'contrast_variance': np.var(contrast_levels)
        }

    def _analyze_temporal_characteristics(self, frames: List[Image.Image]) -> Dict[str, float]:
        """Analyze temporal characteristics of the video."""
        frame_differences = []
        scene_changes = []
        temporal_consistency = []

        prev_frame = None

        for i, frame in enumerate(frames):
            curr_frame = np.array(frame.convert('RGB'))

            if prev_frame is not None:
                # Frame difference
                diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
                frame_differences.append(diff)

                # Scene change detection (large frame difference)
                scene_change = 1.0 if diff > 50.0 else 0.0
                scene_changes.append(scene_change)

                # Temporal consistency (inverse of frame difference variance in local window)
                window_start = max(0, i - 5)
                window_diffs = frame_differences[window_start:]
                if len(window_diffs) > 1:
                    consistency = 1.0 / (1.0 + np.var(window_diffs))
                    temporal_consistency.append(consistency)

            prev_frame = curr_frame

        return {
            'average_frame_difference': np.mean(frame_differences) if frame_differences else 0.0,
            'frame_difference_variance': np.var(frame_differences) if frame_differences else 0.0,
            'scene_change_rate': np.mean(scene_changes) if scene_changes else 0.0,
            'temporal_consistency': np.mean(temporal_consistency) if temporal_consistency else 1.0,
            'temporal_stability': 1.0 - np.var(frame_differences) / (np.mean(frame_differences) + 1e-10) if frame_differences else 1.0
        }

    def _classify_content_type(self,
                              motion_characteristics: Dict[str, float],
                              scene_complexity: Dict[str, float],
                              temporal_characteristics: Dict[str, float]) -> ContentType:
        """Classify content type based on analysis results."""

        avg_motion = motion_characteristics['average_motion']
        motion_variance = motion_characteristics['motion_variance']
        scene_change_rate = temporal_characteristics['scene_change_rate']
        edge_density = scene_complexity['average_edge_density']

        # Classification logic
        if avg_motion < 3.0 and motion_variance < 10.0 and scene_change_rate < 0.1:
            return ContentType.STATIC
        elif avg_motion > 15.0 or motion_variance > 100.0 or scene_change_rate > 0.3:
            return ContentType.FAST_MOTION
        elif motion_variance > 30.0 or scene_change_rate > 0.15:
            return ContentType.MIXED
        else:
            return ContentType.DYNAMIC

    def _generate_optimization_recommendations(self,
                                             content_type: ContentType,
                                             motion_characteristics: Dict[str, float],
                                             scene_complexity: Dict[str, float],
                                             temporal_characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Generate optimization recommendations based on content analysis."""

        profile = self.optimization_profiles[content_type]

        # Base recommendations from profile
        recommendations = {
            'frame_sampling_rate': profile.frame_sampling_rate,
            'batch_size_multiplier': profile.batch_size_multiplier,
            'motion_threshold': profile.motion_threshold,
            'consistency_weight': profile.consistency_weight,
            'feature_cache_size': profile.feature_cache_size,
            'parallel_processing': profile.parallel_processing,
            'streaming_chunk_size': profile.streaming_chunk_size
        }

        # Fine-tune based on specific characteristics
        avg_motion = motion_characteristics['average_motion']
        edge_density = scene_complexity['average_edge_density']
        temporal_consistency = temporal_characteristics['temporal_consistency']

        # Adjust frame sampling based on motion
        if avg_motion > 20.0:
            recommendations['frame_sampling_rate'] = min(0.8, recommendations['frame_sampling_rate'] * 1.5)
        elif avg_motion < 1.0:
            recommendations['frame_sampling_rate'] = max(0.05, recommendations['frame_sampling_rate'] * 0.5)

        # Adjust batch size based on complexity
        if edge_density > 0.3:  # High complexity
            recommendations['batch_size_multiplier'] *= 0.8
        elif edge_density < 0.1:  # Low complexity
            recommendations['batch_size_multiplier'] *= 1.2

        # Adjust consistency weight based on temporal stability
        if temporal_consistency > 0.8:
            recommendations['consistency_weight'] *= 0.8  # Less emphasis on consistency
        elif temporal_consistency < 0.3:
            recommendations['consistency_weight'] *= 1.5  # More emphasis on consistency

        # Additional recommendations
        recommendations.update({
            'use_motion_prediction': avg_motion > 5.0,
            'enable_scene_change_detection': temporal_characteristics['scene_change_rate'] > 0.1,
            'use_adaptive_thresholding': scene_complexity['contrast_variance'] > 1000.0,
            'enable_feature_reuse': temporal_consistency > 0.6,
            'recommended_detection_interval': max(1, int(10 / (avg_motion + 1))),
            'use_temporal_smoothing': motion_characteristics['motion_variance'] > 50.0
        })

        return recommendations

    def _create_default_analysis(self) -> ContentAnalysis:
        """Create default analysis for insufficient data."""
        return ContentAnalysis(
            content_type=ContentType.DYNAMIC,
            motion_characteristics={
                'average_motion': 5.0,
                'motion_variance': 25.0,
                'max_motion': 10.0,
                'motion_consistency': 0.5,
                'motion_acceleration': 2.0
            },
            scene_complexity={
                'average_edge_density': 0.2,
                'edge_density_variance': 0.01,
                'average_texture_complexity': 50.0,
                'average_color_diversity': 6.0,
                'average_contrast': 40.0,
                'contrast_variance': 200.0
            },
            temporal_characteristics={
                'average_frame_difference': 20.0,
                'frame_difference_variance': 100.0,
                'scene_change_rate': 0.05,
                'temporal_consistency': 0.7,
                'temporal_stability': 0.6
            },
            optimization_recommendations=self.optimization_profiles[ContentType.DYNAMIC].__dict__
        )

    def get_optimization_profile(self, content_type: ContentType) -> OptimizationProfile:
        """Get optimization profile for a specific content type."""
        return self.optimization_profiles[content_type]

    def tune_parameters_for_content(self,
                                   base_params: Dict[str, Any],
                                   content_analysis: ContentAnalysis) -> Dict[str, Any]:
        """
        Automatically tune processing parameters based on content analysis.

        Args:
            base_params: Base processing parameters
            content_analysis: Results of content analysis

        Returns:
            Tuned parameters optimized for the content
        """
        tuned_params = base_params.copy()
        recommendations = content_analysis.optimization_recommendations

        # Apply recommendations to parameters
        if 'batch_size' in tuned_params:
            tuned_params['batch_size'] = int(
                tuned_params['batch_size'] * recommendations['batch_size_multiplier']
            )

        if 'frame_sampling_rate' in tuned_params:
            tuned_params['frame_sampling_rate'] = recommendations['frame_sampling_rate']

        if 'motion_threshold' in tuned_params:
            tuned_params['motion_threshold'] = recommendations['motion_threshold']

        if 'consistency_weight' in tuned_params:
            tuned_params['consistency_weight'] = recommendations['consistency_weight']

        # Add new parameters based on recommendations
        tuned_params.update({
            'use_motion_prediction': recommendations.get('use_motion_prediction', False),
            'enable_scene_change_detection': recommendations.get('enable_scene_change_detection', False),
            'use_adaptive_thresholding': recommendations.get('use_adaptive_thresholding', False),
            'enable_feature_reuse': recommendations.get('enable_feature_reuse', False),
            'detection_interval': recommendations.get('recommended_detection_interval', 5),
            'use_temporal_smoothing': recommendations.get('use_temporal_smoothing', False)
        })

        return tuned_params

    def create_content_report(self, content_analysis: ContentAnalysis) -> Dict[str, Any]:
        """Create a comprehensive content analysis report."""

        report = {
            'content_type': content_analysis.content_type.value,
            'analysis_summary': {
                'motion_level': self._categorize_motion_level(
                    content_analysis.motion_characteristics['average_motion']
                ),
                'scene_complexity': self._categorize_scene_complexity(
                    content_analysis.scene_complexity['average_edge_density']
                ),
                'temporal_stability': self._categorize_temporal_stability(
                    content_analysis.temporal_characteristics['temporal_consistency']
                )
            },
            'detailed_metrics': {
                'motion': content_analysis.motion_characteristics,
                'scene': content_analysis.scene_complexity,
                'temporal': content_analysis.temporal_characteristics
            },
            'optimization_recommendations': content_analysis.optimization_recommendations,
            'processing_suggestions': self._generate_processing_suggestions(content_analysis)
        }

        return report

    def _categorize_motion_level(self, avg_motion: float) -> str:
        """Categorize motion level for reporting."""
        if avg_motion < 2.0:
            return "Very Low"
        elif avg_motion < 5.0:
            return "Low"
        elif avg_motion < 10.0:
            return "Moderate"
        elif avg_motion < 20.0:
            return "High"
        else:
            return "Very High"

    def _categorize_scene_complexity(self, edge_density: float) -> str:
        """Categorize scene complexity for reporting."""
        if edge_density < 0.1:
            return "Simple"
        elif edge_density < 0.2:
            return "Moderate"
        elif edge_density < 0.3:
            return "Complex"
        else:
            return "Very Complex"

    def _categorize_temporal_stability(self, consistency: float) -> str:
        """Categorize temporal stability for reporting."""
        if consistency > 0.8:
            return "Very Stable"
        elif consistency > 0.6:
            return "Stable"
        elif consistency > 0.4:
            return "Moderate"
        elif consistency > 0.2:
            return "Unstable"
        else:
            return "Very Unstable"

    def _generate_processing_suggestions(self, content_analysis: ContentAnalysis) -> List[str]:
        """Generate human-readable processing suggestions."""
        suggestions = []

        content_type = content_analysis.content_type
        motion_chars = content_analysis.motion_characteristics
        scene_chars = content_analysis.scene_complexity
        temporal_chars = content_analysis.temporal_characteristics

        # Motion-based suggestions
        if motion_chars['average_motion'] > 15.0:
            suggestions.append("Use higher frame sampling rate for fast motion content")
            suggestions.append("Enable motion prediction for better tracking")
        elif motion_chars['average_motion'] < 2.0:
            suggestions.append("Use lower frame sampling rate for static content")
            suggestions.append("Increase batch size for better efficiency")

        # Scene complexity suggestions
        if scene_chars['average_edge_density'] > 0.3:
            suggestions.append("Reduce batch size for complex scenes")
            suggestions.append("Enable adaptive thresholding for better detection")

        # Temporal stability suggestions
        if temporal_chars['temporal_consistency'] < 0.4:
            suggestions.append("Increase temporal consistency weight")
            suggestions.append("Enable temporal smoothing for unstable content")

        # Content type specific suggestions
        if content_type == ContentType.STATIC:
            suggestions.append("Consider using larger processing chunks")
            suggestions.append("Enable aggressive feature caching")
        elif content_type == ContentType.FAST_MOTION:
            suggestions.append("Use smaller processing chunks")
            suggestions.append("Enable parallel processing for better performance")

        return suggestions
