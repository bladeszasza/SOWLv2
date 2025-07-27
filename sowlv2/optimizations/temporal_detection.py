"""
Temporal detection module for multi-frame object detection and tracking.
"""
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class TemporalDetection:
    """Container for detection across time."""
    frame_idx: int
    box: List[float]
    score: float
    core_prompt: str
    sam_id: Optional[int] = None
    features: Optional[np.ndarray] = None  # Visual features for better matching
    velocity: Optional[Tuple[float, float]] = None  # Estimated velocity (dx, dy)


@dataclass
class TrackedObject:
    """Represents an object tracked across frames."""
    object_id: int
    core_prompt: str
    detections: List[TemporalDetection]
    color: Tuple[int, int, int]
    best_detection_idx: int  # Frame with highest confidence
    trajectory: List[Tuple[float, float]] = field(default_factory=list)  # Center points over time
    confidence_history: List[float] = field(default_factory=list)  # Confidence scores over time
    temporal_consistency_score: float = 0.0  # Overall consistency metric
    predicted_next_box: Optional[List[float]] = None  # Predicted next position


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def compute_box_center(box: List[float]) -> Tuple[float, float]:
    """Compute center point of a bounding box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def compute_box_distance(box1: List[float], box2: List[float]) -> float:
    """Compute Euclidean distance between box centers."""
    center1 = compute_box_center(box1)
    center2 = compute_box_center(box2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def estimate_velocity(detection1: TemporalDetection, detection2: TemporalDetection) -> Tuple[float, float]:
    """Estimate velocity between two detections."""
    if detection2.frame_idx <= detection1.frame_idx:
        return (0.0, 0.0)

    center1 = compute_box_center(detection1.box)
    center2 = compute_box_center(detection2.box)
    frame_diff = detection2.frame_idx - detection1.frame_idx

    dx = (center2[0] - center1[0]) / frame_diff
    dy = (center2[1] - center1[1]) / frame_diff

    return (dx, dy)


def predict_next_position(tracked_obj: TrackedObject, target_frame: int) -> Optional[List[float]]:
    """Predict object position at target frame using trajectory analysis."""
    if len(tracked_obj.detections) < 2:
        return None

    # Use last two detections for prediction
    last_detection = tracked_obj.detections[-1]
    prev_detection = tracked_obj.detections[-2]

    # Estimate velocity
    velocity = estimate_velocity(prev_detection, last_detection)

    # Predict center position
    last_center = compute_box_center(last_detection.box)
    frame_diff = target_frame - last_detection.frame_idx

    predicted_center = (
        last_center[0] + velocity[0] * frame_diff,
        last_center[1] + velocity[1] * frame_diff
    )

    # Use last detection's box size
    box_width = last_detection.box[2] - last_detection.box[0]
    box_height = last_detection.box[3] - last_detection.box[1]

    predicted_box = [
        predicted_center[0] - box_width / 2,
        predicted_center[1] - box_height / 2,
        predicted_center[0] + box_width / 2,
        predicted_center[1] + box_height / 2
    ]

    return predicted_box


def calculate_temporal_consistency_score(tracked_obj: TrackedObject) -> float:
    """Calculate temporal consistency score for a tracked object."""
    if len(tracked_obj.detections) < 3:
        return 1.0  # Not enough data for consistency check

    # Calculate consistency based on trajectory smoothness
    trajectory_consistency = 0.0
    if len(tracked_obj.trajectory) >= 3:
        # Calculate trajectory smoothness using second derivatives
        smoothness_scores = []
        for i in range(2, len(tracked_obj.trajectory)):
            p1, p2, p3 = tracked_obj.trajectory[i-2:i+1]

            # Calculate acceleration (second derivative)
            acc_x = p3[0] - 2*p2[0] + p1[0]
            acc_y = p3[1] - 2*p2[1] + p1[1]
            acceleration = np.sqrt(acc_x**2 + acc_y**2)

            # Lower acceleration means smoother trajectory
            smoothness_scores.append(1.0 / (1.0 + acceleration))

        trajectory_consistency = np.mean(smoothness_scores)

    # Calculate confidence consistency
    confidence_consistency = 0.0
    if tracked_obj.confidence_history:
        confidence_std = np.std(tracked_obj.confidence_history)
        confidence_consistency = 1.0 / (1.0 + confidence_std)

    # Calculate size consistency
    size_consistency = 0.0
    if len(tracked_obj.detections) >= 2:
        sizes = []
        for detection in tracked_obj.detections:
            width = detection.box[2] - detection.box[0]
            height = detection.box[3] - detection.box[1]
            sizes.append(width * height)

        size_std = np.std(sizes)
        size_consistency = 1.0 / (1.0 + size_std / np.mean(sizes))

    # Combine consistency metrics
    overall_consistency = (trajectory_consistency + confidence_consistency + size_consistency) / 3.0
    return overall_consistency


def compute_confidence_weighted_score(detections: List[TemporalDetection],
                                    iou_scores: List[float]) -> float:
    """Compute confidence-weighted matching score."""
    if not detections or not iou_scores:
        return 0.0

    weighted_scores = []
    for detection, iou in zip(detections, iou_scores):
        # Weight IoU by detection confidence
        weighted_score = iou * detection.score
        weighted_scores.append(weighted_score)

    return np.mean(weighted_scores)


def merge_temporal_detections(
    detections_by_frame: Dict[int, List[Dict[str, Any]]],
    merge_threshold: float = 0.5,
    confidence_weight: float = 0.3,
    trajectory_weight: float = 0.4,
    max_frame_gap: int = 5
) -> List[TrackedObject]:
    """
    Enhanced merge detections across frames with improved object tracking.
    Uses IoU, confidence weighting, and trajectory prediction for association.

    Args:
        detections_by_frame: Dictionary mapping frame indices to detection lists
        merge_threshold: Minimum similarity score for merging detections
        confidence_weight: Weight for confidence in matching score
        trajectory_weight: Weight for trajectory prediction in matching
        max_frame_gap: Maximum frame gap to consider for tracking
    """
    tracked_objects: List[TrackedObject] = []
    object_id_counter = 1

    # Process frames in order
    for frame_idx in sorted(detections_by_frame.keys()):
        frame_detections = detections_by_frame[frame_idx]

        for detection in frame_detections:
            temporal_det = TemporalDetection(
                frame_idx=frame_idx,
                box=detection['box'],
                score=detection['score'],
                core_prompt=detection['core_prompt'],
                features=detection.get('features')
            )

            # Find best matching tracked object
            best_match = None
            best_score = 0.0

            for tracked_obj in tracked_objects:
                # Only match if same prompt
                if tracked_obj.core_prompt != temporal_det.core_prompt:
                    continue

                # Skip if frame gap is too large
                last_frame = tracked_obj.detections[-1].frame_idx
                if frame_idx - last_frame > max_frame_gap:
                    continue

                # Calculate matching score using multiple criteria
                matching_score = calculate_matching_score(
                    tracked_obj, temporal_det, confidence_weight, trajectory_weight
                )

                if matching_score > best_score:
                    best_score = matching_score
                    best_match = tracked_obj

            # Add to existing object or create new
            if best_match and best_score > merge_threshold:
                # Update tracked object
                best_match.detections.append(temporal_det)

                # Update trajectory
                center = compute_box_center(temporal_det.box)
                best_match.trajectory.append(center)
                best_match.confidence_history.append(temporal_det.score)

                # Update velocity for the detection
                if len(best_match.detections) >= 2:
                    prev_detection = best_match.detections[-2]
                    temporal_det.velocity = estimate_velocity(prev_detection, temporal_det)

                # Update best detection if this has higher score
                best_det = best_match.detections[best_match.best_detection_idx]
                if temporal_det.score > best_det.score:
                    best_match.best_detection_idx = len(best_match.detections) - 1

                # Update temporal consistency score
                best_match.temporal_consistency_score = calculate_temporal_consistency_score(best_match)

                # Update predicted next position
                best_match.predicted_next_box = predict_next_position(best_match, frame_idx + 1)

            else:
                # Create new tracked object
                center = compute_box_center(temporal_det.box)
                new_object = TrackedObject(
                    object_id=object_id_counter,
                    core_prompt=temporal_det.core_prompt,
                    detections=[temporal_det],
                    color=(0, 0, 0),  # Will be assigned later
                    best_detection_idx=0,
                    trajectory=[center],
                    confidence_history=[temporal_det.score],
                    temporal_consistency_score=1.0
                )
                tracked_objects.append(new_object)
                object_id_counter += 1

    # Post-process: validate and merge similar tracks
    tracked_objects = validate_and_merge_tracks(tracked_objects, merge_threshold)

    return tracked_objects


def calculate_matching_score(tracked_obj: TrackedObject,
                           detection: TemporalDetection,
                           confidence_weight: float,
                           trajectory_weight: float) -> float:
    """Calculate comprehensive matching score between tracked object and detection."""

    # IoU with most recent detection
    recent_detection = tracked_obj.detections[-1]
    iou_score = compute_iou(recent_detection.box, detection.box)

    # Confidence-weighted IoU
    confidence_factor = (recent_detection.score + detection.score) / 2.0
    confidence_weighted_iou = iou_score * (1.0 + confidence_weight * confidence_factor)

    # Trajectory prediction score
    trajectory_score = 0.0
    if tracked_obj.predicted_next_box:
        predicted_iou = compute_iou(tracked_obj.predicted_next_box, detection.box)
        trajectory_score = predicted_iou * trajectory_weight

    # Distance penalty (closer is better)
    distance = compute_box_distance(recent_detection.box, detection.box)
    distance_penalty = 1.0 / (1.0 + distance / 100.0)  # Normalize by image size assumption

    # Combine scores
    total_score = (
        confidence_weighted_iou * 0.4 +
        trajectory_score * 0.3 +
        distance_penalty * 0.3
    )

    return total_score


def validate_and_merge_tracks(tracked_objects: List[TrackedObject],
                            merge_threshold: float) -> List[TrackedObject]:
    """Validate tracks and merge similar ones that might represent the same object."""

    # Remove short tracks (likely false positives)
    min_track_length = 2
    valid_tracks = [obj for obj in tracked_objects if len(obj.detections) >= min_track_length]

    # Merge tracks that might represent the same object
    merged_tracks = []
    used_indices = set()

    for i, track1 in enumerate(valid_tracks):
        if i in used_indices:
            continue

        # Look for similar tracks to merge
        tracks_to_merge = [track1]
        used_indices.add(i)

        for j, track2 in enumerate(valid_tracks[i+1:], i+1):
            if j in used_indices:
                continue

            # Check if tracks should be merged
            if should_merge_tracks(track1, track2, merge_threshold):
                tracks_to_merge.append(track2)
                used_indices.add(j)

        # Merge tracks if multiple found
        if len(tracks_to_merge) > 1:
            merged_track = merge_tracks(tracks_to_merge)
            merged_tracks.append(merged_track)
        else:
            merged_tracks.append(track1)

    return merged_tracks


def should_merge_tracks(track1: TrackedObject, track2: TrackedObject, threshold: float) -> bool:
    """Determine if two tracks should be merged."""

    # Must have same prompt
    if track1.core_prompt != track2.core_prompt:
        return False

    # Check temporal overlap or proximity
    frames1 = {det.frame_idx for det in track1.detections}
    frames2 = {det.frame_idx for det in track2.detections}

    # If tracks overlap in time, check spatial similarity
    if frames1 & frames2:
        # Find overlapping frames and check IoU
        overlapping_frames = frames1 & frames2
        ious = []

        for frame_idx in overlapping_frames:
            det1 = next(det for det in track1.detections if det.frame_idx == frame_idx)
            det2 = next(det for det in track2.detections if det.frame_idx == frame_idx)
            ious.append(compute_iou(det1.box, det2.box))

        return np.mean(ious) > threshold

    # If tracks are temporally adjacent, check spatial continuity
    max_frame1 = max(frames1)
    min_frame2 = min(frames2)

    if abs(max_frame1 - min_frame2) <= 3:  # Small temporal gap
        # Check if last detection of track1 is close to first detection of track2
        last_det1 = next(det for det in track1.detections if det.frame_idx == max_frame1)
        first_det2 = next(det for det in track2.detections if det.frame_idx == min_frame2)

        distance = compute_box_distance(last_det1.box, first_det2.box)
        return distance < 50  # Threshold for spatial continuity

    return False


def merge_tracks(tracks: List[TrackedObject]) -> TrackedObject:
    """Merge multiple tracks into a single track."""

    # Use the track with highest average confidence as base
    base_track = max(tracks, key=lambda t: np.mean(t.confidence_history))

    # Collect all detections and sort by frame
    all_detections = []
    for track in tracks:
        all_detections.extend(track.detections)

    all_detections.sort(key=lambda d: d.frame_idx)

    # Remove duplicate detections in same frame (keep highest confidence)
    merged_detections = []
    current_frame = None
    frame_detections = []

    for detection in all_detections:
        if current_frame is None or detection.frame_idx == current_frame:
            frame_detections.append(detection)
            current_frame = detection.frame_idx
        else:
            # Process previous frame's detections
            if frame_detections:
                best_detection = max(frame_detections, key=lambda d: d.score)
                merged_detections.append(best_detection)

            # Start new frame
            frame_detections = [detection]
            current_frame = detection.frame_idx

    # Don't forget the last frame
    if frame_detections:
        best_detection = max(frame_detections, key=lambda d: d.score)
        merged_detections.append(best_detection)

    # Create merged track
    merged_track = TrackedObject(
        object_id=base_track.object_id,
        core_prompt=base_track.core_prompt,
        detections=merged_detections,
        color=base_track.color,
        best_detection_idx=0,
        trajectory=[],
        confidence_history=[],
        temporal_consistency_score=0.0
    )

    # Rebuild trajectory and confidence history
    for detection in merged_detections:
        center = compute_box_center(detection.box)
        merged_track.trajectory.append(center)
        merged_track.confidence_history.append(detection.score)

    # Find best detection index
    best_score = 0.0
    for i, detection in enumerate(merged_detections):
        if detection.score > best_score:
            best_score = detection.score
            merged_track.best_detection_idx = i

    # Calculate temporal consistency
    merged_track.temporal_consistency_score = calculate_temporal_consistency_score(merged_track)

    return merged_track


def validate_multi_frame_detections(
    tracked_objects: List[TrackedObject],
    min_frames: int = 3,
    consistency_threshold: float = 0.5
) -> List[TrackedObject]:
    """
    Validate tracked objects across multiple frames.

    Args:
        tracked_objects: List of tracked objects to validate
        min_frames: Minimum number of frames for a valid track
        consistency_threshold: Minimum consistency score for validation

    Returns:
        List of validated tracked objects
    """
    validated_objects = []

    for tracked_obj in tracked_objects:
        # Check minimum frame requirement
        if len(tracked_obj.detections) < min_frames:
            logging.debug(f"Object {tracked_obj.object_id} rejected: insufficient frames "
                         f"({len(tracked_obj.detections)} < {min_frames})")
            continue

        # Check temporal consistency
        if tracked_obj.temporal_consistency_score < consistency_threshold:
            logging.debug(f"Object {tracked_obj.object_id} rejected: low consistency "
                         f"({tracked_obj.temporal_consistency_score:.3f} < {consistency_threshold})")
            continue

        # Check for reasonable trajectory (not too erratic)
        if len(tracked_obj.trajectory) >= 3:
            trajectory_variance = calculate_trajectory_variance(tracked_obj.trajectory)
            if trajectory_variance > 1000:  # Threshold for erratic movement
                logging.debug(f"Object {tracked_obj.object_id} rejected: erratic trajectory "
                             f"(variance: {trajectory_variance:.2f})")
                continue

        # Check confidence stability
        if tracked_obj.confidence_history:
            confidence_std = np.std(tracked_obj.confidence_history)
            confidence_mean = np.mean(tracked_obj.confidence_history)
            if confidence_std / confidence_mean > 0.5:  # High relative variance
                logging.debug(f"Object {tracked_obj.object_id} rejected: unstable confidence")
                continue

        validated_objects.append(tracked_obj)

    return validated_objects


def calculate_trajectory_variance(trajectory: List[Tuple[float, float]]) -> float:
    """Calculate variance in trajectory movement."""
    if len(trajectory) < 3:
        return 0.0

    # Calculate movement vectors
    movements = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        movement_magnitude = np.sqrt(dx**2 + dy**2)
        movements.append(movement_magnitude)

    return np.var(movements)


def select_key_frames_for_detection(
    importance_scores: List[float],
    num_frames: int,
    min_spacing: int = 10,
    use_adaptive_spacing: bool = True
) -> List[int]:
    """
    Enhanced key frame selection with adaptive spacing.
    Ensures temporal diversity by enforcing minimum spacing.

    Args:
        importance_scores: Importance score for each frame
        num_frames: Number of frames to select
        min_spacing: Minimum spacing between selected frames
        use_adaptive_spacing: Whether to use adaptive spacing based on content
    """
    if len(importance_scores) <= num_frames:
        return list(range(len(importance_scores)))

    # Create (index, score) pairs and sort by score
    indexed_scores = list(enumerate(importance_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    selected_indices = []

    if use_adaptive_spacing:
        # Adaptive spacing based on importance score distribution
        score_variance = np.var(importance_scores)
        if score_variance > 0.1:  # High variance - use stricter spacing
            min_spacing = max(min_spacing, len(importance_scores) // (num_frames * 2))
        else:  # Low variance - can use closer spacing
            min_spacing = max(5, min_spacing // 2)

    for idx, score in indexed_scores:
        # Check minimum spacing constraint
        too_close = any(abs(idx - selected) < min_spacing for selected in selected_indices)
        if not too_close:
            selected_indices.append(idx)
            if len(selected_indices) >= num_frames:
                break

    # If we couldn't get enough frames with spacing, relax constraint progressively
    if len(selected_indices) < num_frames:
        relaxed_spacing = min_spacing
        while len(selected_indices) < num_frames and relaxed_spacing > 1:
            relaxed_spacing = max(1, relaxed_spacing // 2)

            for idx, score in indexed_scores:
                if idx in selected_indices:
                    continue

                too_close = any(abs(idx - selected) < relaxed_spacing for selected in selected_indices)
                if not too_close:
                    selected_indices.append(idx)
                    if len(selected_indices) >= num_frames:
                        break

    # Final fallback: fill remaining slots with highest scoring frames
    if len(selected_indices) < num_frames:
        for idx, score in indexed_scores:
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) >= num_frames:
                    break

    return sorted(selected_indices)


def create_detection_validation_report(tracked_objects: List[TrackedObject]) -> Dict[str, Any]:
    """Create a comprehensive validation report for tracked objects."""

    report = {
        'total_objects': len(tracked_objects),
        'objects_by_prompt': {},
        'average_track_length': 0.0,
        'average_consistency_score': 0.0,
        'temporal_coverage': {},
        'quality_metrics': {}
    }

    if not tracked_objects:
        return report

    # Group by prompt
    for obj in tracked_objects:
        prompt = obj.core_prompt
        if prompt not in report['objects_by_prompt']:
            report['objects_by_prompt'][prompt] = []
        report['objects_by_prompt'][prompt].append(obj.object_id)

    # Calculate averages
    track_lengths = [len(obj.detections) for obj in tracked_objects]
    consistency_scores = [obj.temporal_consistency_score for obj in tracked_objects]

    report['average_track_length'] = np.mean(track_lengths)
    report['average_consistency_score'] = np.mean(consistency_scores)

    # Temporal coverage analysis
    all_frames = set()
    for obj in tracked_objects:
        for detection in obj.detections:
            all_frames.add(detection.frame_idx)

    if all_frames:
        report['temporal_coverage'] = {
            'total_frames_with_detections': len(all_frames),
            'frame_range': (min(all_frames), max(all_frames)),
            'coverage_density': len(all_frames) / (max(all_frames) - min(all_frames) + 1)
        }

    # Quality metrics
    high_quality_tracks = [obj for obj in tracked_objects if obj.temporal_consistency_score > 0.7]
    long_tracks = [obj for obj in tracked_objects if len(obj.detections) >= 5]

    report['quality_metrics'] = {
        'high_quality_tracks': len(high_quality_tracks),
        'long_tracks': len(long_tracks),
        'quality_ratio': len(high_quality_tracks) / len(tracked_objects),
        'average_confidence': np.mean([np.mean(obj.confidence_history) for obj in tracked_objects])
    }

    return report
