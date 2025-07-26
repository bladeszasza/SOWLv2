"""
Unit tests for temporal detection functionality.
Tests object tracking, detection merging, and validation across frames.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from sowlv2.optimizations.temporal_detection import (
    TemporalDetection, TrackedObject, compute_iou, compute_box_center,
    compute_box_distance, estimate_velocity, predict_next_position,
    calculate_temporal_consistency_score, merge_temporal_detections,
    validate_multi_frame_detections, select_key_frames_for_detection,
    create_detection_validation_report
)


class TestTemporalDetection:
    """Test suite for TemporalDetection dataclass."""
    
    def test_temporal_detection_creation(self):
        """Test TemporalDetection creation with all parameters."""
        detection = TemporalDetection(
            frame_idx=5,
            box=[100, 100, 200, 200],
            score=0.85,
            core_prompt="cat",
            sam_id=1,
            features=np.array([1, 2, 3]),
            velocity=(2.5, -1.0)
        )
        
        assert detection.frame_idx == 5
        assert detection.box == [100, 100, 200, 200]
        assert detection.score == 0.85
        assert detection.core_prompt == "cat"
        assert detection.sam_id == 1
        assert np.array_equal(detection.features, np.array([1, 2, 3]))
        assert detection.velocity == (2.5, -1.0)
    
    def test_temporal_detection_defaults(self):
        """Test TemporalDetection creation with default values."""
        detection = TemporalDetection(
            frame_idx=0,
            box=[0, 0, 50, 50],
            score=0.5,
            core_prompt="dog"
        )
        
        assert detection.sam_id is None
        assert detection.features is None
        assert detection.velocity is None


class TestTrackedObject:
    """Test suite for TrackedObject dataclass."""
    
    def test_tracked_object_creation(self):
        """Test TrackedObject creation with all parameters."""
        detection = TemporalDetection(0, [10, 10, 50, 50], 0.9, "cat")
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=[detection],
            color=(255, 0, 0),
            best_detection_idx=0,
            trajectory=[(30, 30)],
            confidence_history=[0.9],
            temporal_consistency_score=0.8,
            predicted_next_box=[15, 15, 55, 55]
        )
        
        assert tracked_obj.object_id == 1
        assert tracked_obj.core_prompt == "cat"
        assert len(tracked_obj.detections) == 1
        assert tracked_obj.color == (255, 0, 0)
        assert tracked_obj.best_detection_idx == 0
        assert tracked_obj.trajectory == [(30, 30)]
        assert tracked_obj.confidence_history == [0.9]
        assert tracked_obj.temporal_consistency_score == 0.8
        assert tracked_obj.predicted_next_box == [15, 15, 55, 55]
    
    def test_tracked_object_defaults(self):
        """Test TrackedObject creation with default values."""
        detection = TemporalDetection(0, [10, 10, 50, 50], 0.9, "cat")
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=[detection],
            color=(255, 0, 0),
            best_detection_idx=0
        )
        
        assert tracked_obj.trajectory == []
        assert tracked_obj.confidence_history == []
        assert tracked_obj.temporal_consistency_score == 0.0
        assert tracked_obj.predicted_next_box is None


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_compute_iou_perfect_overlap(self):
        """Test IoU computation with perfect overlap."""
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        
        iou = compute_iou(box1, box2)
        
        assert iou == 1.0
    
    def test_compute_iou_no_overlap(self):
        """Test IoU computation with no overlap."""
        box1 = [10, 10, 50, 50]
        box2 = [60, 60, 100, 100]
        
        iou = compute_iou(box1, box2)
        
        assert iou == 0.0
    
    def test_compute_iou_partial_overlap(self):
        """Test IoU computation with partial overlap."""
        box1 = [10, 10, 50, 50]
        box2 = [30, 30, 70, 70]
        
        iou = compute_iou(box1, box2)
        
        # Intersection: 20x20 = 400
        # Union: 40x40 + 40x40 - 400 = 3200 - 400 = 2800
        # IoU: 400/2800 = 1/7 ≈ 0.143
        assert abs(iou - (1/7)) < 0.001
    
    def test_compute_box_center(self):
        """Test box center computation."""
        box = [10, 20, 50, 80]
        
        center = compute_box_center(box)
        
        assert center == (30.0, 50.0)  # (10+50)/2, (20+80)/2
    
    def test_compute_box_distance(self):
        """Test distance computation between box centers."""
        box1 = [0, 0, 20, 20]  # Center: (10, 10)
        box2 = [30, 40, 50, 60]  # Center: (40, 50)
        
        distance = compute_box_distance(box1, box2)
        
        # Distance: sqrt((40-10)^2 + (50-10)^2) = sqrt(900 + 1600) = sqrt(2500) = 50
        assert distance == 50.0
    
    def test_estimate_velocity(self):
        """Test velocity estimation between detections."""
        det1 = TemporalDetection(0, [10, 10, 30, 30], 0.9, "cat")  # Center: (20, 20)
        det2 = TemporalDetection(2, [30, 50, 50, 70], 0.8, "cat")  # Center: (40, 60)
        
        velocity = estimate_velocity(det1, det2)
        
        # Velocity: ((40-20)/(2-0), (60-20)/(2-0)) = (10, 20)
        assert velocity == (10.0, 20.0)
    
    def test_estimate_velocity_same_frame(self):
        """Test velocity estimation with same frame indices."""
        det1 = TemporalDetection(5, [10, 10, 30, 30], 0.9, "cat")
        det2 = TemporalDetection(5, [30, 50, 50, 70], 0.8, "cat")
        
        velocity = estimate_velocity(det1, det2)
        
        assert velocity == (0.0, 0.0)
    
    def test_estimate_velocity_reverse_order(self):
        """Test velocity estimation with reverse frame order."""
        det1 = TemporalDetection(5, [10, 10, 30, 30], 0.9, "cat")
        det2 = TemporalDetection(3, [30, 50, 50, 70], 0.8, "cat")
        
        velocity = estimate_velocity(det1, det2)
        
        assert velocity == (0.0, 0.0)


class TestPredictNextPosition:
    """Test suite for position prediction functionality."""
    
    def test_predict_next_position_success(self):
        """Test successful position prediction."""
        det1 = TemporalDetection(0, [10, 10, 30, 30], 0.9, "cat")  # Center: (20, 20)
        det2 = TemporalDetection(1, [25, 30, 45, 50], 0.8, "cat")  # Center: (35, 40)
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=[det1, det2],
            color=(255, 0, 0),
            best_detection_idx=1
        )
        
        predicted_box = predict_next_position(tracked_obj, target_frame=2)
        
        # Velocity: (15, 20) per frame
        # Predicted center at frame 2: (35, 40) + (15, 20) = (50, 60)
        # Box size: 20x20, so predicted box: [40, 50, 60, 70]
        assert predicted_box is not None
        assert len(predicted_box) == 4
        assert predicted_box[0] == 40.0  # 50 - 10
        assert predicted_box[1] == 50.0  # 60 - 10
        assert predicted_box[2] == 60.0  # 50 + 10
        assert predicted_box[3] == 70.0  # 60 + 10
    
    def test_predict_next_position_insufficient_data(self):
        """Test position prediction with insufficient detections."""
        det1 = TemporalDetection(0, [10, 10, 30, 30], 0.9, "cat")
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=[det1],
            color=(255, 0, 0),
            best_detection_idx=0
        )
        
        predicted_box = predict_next_position(tracked_obj, target_frame=1)
        
        assert predicted_box is None


class TestTemporalConsistency:
    """Test suite for temporal consistency scoring."""
    
    def test_calculate_temporal_consistency_score_smooth_trajectory(self):
        """Test consistency score for smooth trajectory."""
        # Create detections with smooth movement
        detections = [
            TemporalDetection(0, [10, 10, 30, 30], 0.9, "cat"),
            TemporalDetection(1, [15, 15, 35, 35], 0.85, "cat"),
            TemporalDetection(2, [20, 20, 40, 40], 0.8, "cat"),
            TemporalDetection(3, [25, 25, 45, 45], 0.85, "cat")
        ]
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=detections,
            color=(255, 0, 0),
            best_detection_idx=0,
            trajectory=[(20, 20), (25, 25), (30, 30), (35, 35)],
            confidence_history=[0.9, 0.85, 0.8, 0.85]
        )
        
        score = calculate_temporal_consistency_score(tracked_obj)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be relatively high for smooth trajectory
    
    def test_calculate_temporal_consistency_score_insufficient_data(self):
        """Test consistency score with insufficient data."""
        detections = [
            TemporalDetection(0, [10, 10, 30, 30], 0.9, "cat"),
            TemporalDetection(1, [15, 15, 35, 35], 0.85, "cat")
        ]
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=detections,
            color=(255, 0, 0),
            best_detection_idx=0
        )
        
        score = calculate_temporal_consistency_score(tracked_obj)
        
        assert score == 1.0  # Default for insufficient data


class TestMergeTemporalDetections:
    """Test suite for temporal detection merging."""
    
    def test_merge_temporal_detections_basic(self):
        """Test basic temporal detection merging."""
        detections_by_frame = {
            0: [{'box': [10, 10, 30, 30], 'score': 0.9, 'core_prompt': 'cat'}],
            1: [{'box': [15, 15, 35, 35], 'score': 0.85, 'core_prompt': 'cat'}],
            2: [{'box': [20, 20, 40, 40], 'score': 0.8, 'core_prompt': 'cat'}]
        }
        
        tracked_objects = merge_temporal_detections(detections_by_frame)
        
        assert len(tracked_objects) == 1
        tracked_obj = tracked_objects[0]
        assert tracked_obj.core_prompt == 'cat'
        assert len(tracked_obj.detections) == 3
        assert len(tracked_obj.trajectory) == 3
        assert len(tracked_obj.confidence_history) == 3
    
    def test_merge_temporal_detections_multiple_objects(self):
        """Test merging with multiple different objects."""
        detections_by_frame = {
            0: [
                {'box': [10, 10, 30, 30], 'score': 0.9, 'core_prompt': 'cat'},
                {'box': [100, 100, 120, 120], 'score': 0.8, 'core_prompt': 'dog'}
            ],
            1: [
                {'box': [15, 15, 35, 35], 'score': 0.85, 'core_prompt': 'cat'},
                {'box': [105, 105, 125, 125], 'score': 0.75, 'core_prompt': 'dog'}
            ]
        }
        
        tracked_objects = merge_temporal_detections(detections_by_frame)
        
        assert len(tracked_objects) == 2
        
        # Find cat and dog objects
        cat_obj = next(obj for obj in tracked_objects if obj.core_prompt == 'cat')
        dog_obj = next(obj for obj in tracked_objects if obj.core_prompt == 'dog')
        
        assert len(cat_obj.detections) == 2
        assert len(dog_obj.detections) == 2
    
    def test_merge_temporal_detections_different_prompts(self):
        """Test that objects with different prompts are not merged."""
        detections_by_frame = {
            0: [{'box': [10, 10, 30, 30], 'score': 0.9, 'core_prompt': 'cat'}],
            1: [{'box': [15, 15, 35, 35], 'score': 0.85, 'core_prompt': 'dog'}]  # Different prompt
        }
        
        tracked_objects = merge_temporal_detections(detections_by_frame)
        
        assert len(tracked_objects) == 2  # Should create separate objects
    
    def test_merge_temporal_detections_large_frame_gap(self):
        """Test merging with large frame gaps."""
        detections_by_frame = {
            0: [{'box': [10, 10, 30, 30], 'score': 0.9, 'core_prompt': 'cat'}],
            10: [{'box': [15, 15, 35, 35], 'score': 0.85, 'core_prompt': 'cat'}]  # Large gap
        }
        
        tracked_objects = merge_temporal_detections(detections_by_frame, max_frame_gap=5)
        
        assert len(tracked_objects) == 2  # Should create separate objects due to large gap
    
    def test_merge_temporal_detections_low_threshold(self):
        """Test merging with low similarity threshold."""
        detections_by_frame = {
            0: [{'box': [10, 10, 30, 30], 'score': 0.9, 'core_prompt': 'cat'}],
            1: [{'box': [100, 100, 120, 120], 'score': 0.85, 'core_prompt': 'cat'}]  # Far apart
        }
        
        tracked_objects = merge_temporal_detections(detections_by_frame, merge_threshold=0.1)
        
        # With low threshold, even distant objects might be merged
        assert len(tracked_objects) >= 1
    
    def test_merge_temporal_detections_empty_input(self):
        """Test merging with empty input."""
        tracked_objects = merge_temporal_detections({})
        
        assert tracked_objects == []


class TestValidateMultiFrameDetections:
    """Test suite for multi-frame detection validation."""
    
    def test_validate_multi_frame_detections_valid_objects(self):
        """Test validation with valid tracked objects."""
        # Create valid tracked objects
        detections = [
            TemporalDetection(i, [10+i*5, 10+i*5, 30+i*5, 30+i*5], 0.8+i*0.01, "cat")
            for i in range(5)
        ]
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=detections,
            color=(255, 0, 0),
            best_detection_idx=0,
            trajectory=[(20+i*5, 20+i*5) for i in range(5)],
            confidence_history=[0.8+i*0.01 for i in range(5)],
            temporal_consistency_score=0.8
        )
        
        validated = validate_multi_frame_detections([tracked_obj])
        
        assert len(validated) == 1
        assert validated[0] == tracked_obj
    
    def test_validate_multi_frame_detections_insufficient_frames(self):
        """Test validation with insufficient frames."""
        detections = [TemporalDetection(0, [10, 10, 30, 30], 0.9, "cat")]
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=detections,
            color=(255, 0, 0),
            best_detection_idx=0,
            temporal_consistency_score=0.8
        )
        
        validated = validate_multi_frame_detections([tracked_obj], min_frames=3)
        
        assert len(validated) == 0  # Should be rejected
    
    def test_validate_multi_frame_detections_low_consistency(self):
        """Test validation with low consistency score."""
        detections = [
            TemporalDetection(i, [10, 10, 30, 30], 0.8, "cat")
            for i in range(5)
        ]
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=detections,
            color=(255, 0, 0),
            best_detection_idx=0,
            temporal_consistency_score=0.2  # Low consistency
        )
        
        validated = validate_multi_frame_detections([tracked_obj], consistency_threshold=0.5)
        
        assert len(validated) == 0  # Should be rejected


class TestSelectKeyFrames:
    """Test suite for key frame selection."""
    
    def test_select_key_frames_for_detection_basic(self):
        """Test basic key frame selection."""
        importance_scores = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5]
        
        selected_indices = select_key_frames_for_detection(
            importance_scores, num_frames=3, min_spacing=2
        )
        
        assert len(selected_indices) == 3
        assert selected_indices == sorted(selected_indices)
        
        # Check minimum spacing
        for i in range(1, len(selected_indices)):
            assert selected_indices[i] - selected_indices[i-1] >= 2
    
    def test_select_key_frames_for_detection_insufficient_frames(self):
        """Test key frame selection with fewer frames than requested."""
        importance_scores = [0.8, 0.6]
        
        selected_indices = select_key_frames_for_detection(
            importance_scores, num_frames=5
        )
        
        assert len(selected_indices) == 2
        assert selected_indices == [0, 1]
    
    def test_select_key_frames_for_detection_adaptive_spacing(self):
        """Test key frame selection with adaptive spacing."""
        # High variance scores should use stricter spacing
        importance_scores = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        
        selected_indices = select_key_frames_for_detection(
            importance_scores, num_frames=3, min_spacing=1, use_adaptive_spacing=True
        )
        
        assert len(selected_indices) == 3
        assert selected_indices == sorted(selected_indices)
    
    def test_select_key_frames_for_detection_relaxed_spacing(self):
        """Test key frame selection with spacing relaxation."""
        importance_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        # Request more frames than can fit with strict spacing
        selected_indices = select_key_frames_for_detection(
            importance_scores, num_frames=4, min_spacing=3
        )
        
        assert len(selected_indices) == 4
        # Should relax spacing to fit all requested frames


class TestDetectionValidationReport:
    """Test suite for detection validation reporting."""
    
    def test_create_detection_validation_report_basic(self):
        """Test basic validation report creation."""
        detections = [
            TemporalDetection(i, [10, 10, 30, 30], 0.8, "cat")
            for i in range(3)
        ]
        
        tracked_obj = TrackedObject(
            object_id=1,
            core_prompt="cat",
            detections=detections,
            color=(255, 0, 0),
            best_detection_idx=0,
            confidence_history=[0.8, 0.85, 0.9],
            temporal_consistency_score=0.7
        )
        
        report = create_detection_validation_report([tracked_obj])
        
        assert report['total_objects'] == 1
        assert 'cat' in report['objects_by_prompt']
        assert report['objects_by_prompt']['cat'] == [1]
        assert report['average_track_length'] == 3.0
        assert report['average_consistency_score'] == 0.7
        assert 'temporal_coverage' in report
        assert 'quality_metrics' in report
    
    def test_create_detection_validation_report_empty(self):
        """Test validation report with empty input."""
        report = create_detection_validation_report([])
        
        assert report['total_objects'] == 0
        assert report['objects_by_prompt'] == {}
        assert report['average_track_length'] == 0.0
        assert report['average_consistency_score'] == 0.0
    
    def test_create_detection_validation_report_multiple_prompts(self):
        """Test validation report with multiple prompts."""
        cat_detections = [TemporalDetection(i, [10, 10, 30, 30], 0.8, "cat") for i in range(3)]
        dog_detections = [TemporalDetection(i, [50, 50, 70, 70], 0.9, "dog") for i in range(2)]
        
        cat_obj = TrackedObject(
            object_id=1, core_prompt="cat", detections=cat_detections,
            color=(255, 0, 0), best_detection_idx=0,
            confidence_history=[0.8, 0.85, 0.9], temporal_consistency_score=0.7
        )
        
        dog_obj = TrackedObject(
            object_id=2, core_prompt="dog", detections=dog_detections,
            color=(0, 255, 0), best_detection_idx=0,
            confidence_history=[0.9, 0.95], temporal_consistency_score=0.9
        )
        
        report = create_detection_validation_report([cat_obj, dog_obj])
        
        assert report['total_objects'] == 2
        assert len(report['objects_by_prompt']) == 2
        assert 'cat' in report['objects_by_prompt']
        assert 'dog' in report['objects_by_prompt']
        assert report['average_track_length'] == 2.5  # (3 + 2) / 2
        assert report['average_consistency_score'] == 0.8  # (0.7 + 0.9) / 2
    
    def test_create_detection_validation_report_quality_metrics(self):
        """Test validation report quality metrics calculation."""
        # Create objects with different quality levels
        high_quality_obj = TrackedObject(
            object_id=1, core_prompt="cat",
            detections=[TemporalDetection(i, [10, 10, 30, 30], 0.9, "cat") for i in range(6)],
            color=(255, 0, 0), best_detection_idx=0,
            confidence_history=[0.9] * 6, temporal_consistency_score=0.8
        )
        
        low_quality_obj = TrackedObject(
            object_id=2, core_prompt="dog",
            detections=[TemporalDetection(i, [50, 50, 70, 70], 0.6, "dog") for i in range(2)],
            color=(0, 255, 0), best_detection_idx=0,
            confidence_history=[0.6, 0.65], temporal_consistency_score=0.3
        )
        
        report = create_detection_validation_report([high_quality_obj, low_quality_obj])
        
        quality_metrics = report['quality_metrics']
        assert quality_metrics['high_quality_tracks'] == 1  # Only high_quality_obj > 0.7
        assert quality_metrics['long_tracks'] == 1  # Only high_quality_obj >= 5 frames
        assert quality_metrics['quality_ratio'] == 0.5  # 1/2
        assert quality_metrics['average_confidence'] == 0.775  # (0.9 + 0.625) / 2