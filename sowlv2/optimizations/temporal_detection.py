"""
Temporal detection module for multi-frame object detection and tracking.
"""
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class TemporalDetection:
    """Container for detection across time."""
    frame_idx: int
    box: List[float]
    score: float
    core_prompt: str
    sam_id: Optional[int] = None


@dataclass
class TrackedObject:
    """Represents an object tracked across frames."""
    object_id: int
    core_prompt: str
    detections: List[TemporalDetection]
    color: Tuple[int, int, int]
    best_detection_idx: int  # Frame with highest confidence


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


def merge_temporal_detections(
    detections_by_frame: Dict[int, List[Dict[str, Any]]],
    merge_threshold: float = 0.7
) -> List[TrackedObject]:
    """
    Merge detections across frames to identify unique objects.
    Uses IoU and prompt matching to associate detections.
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
                core_prompt=detection['core_prompt']
            )

            # Find matching tracked object
            matched_object = None
            best_iou = 0

            for tracked_obj in tracked_objects:
                # Only match if same prompt
                if tracked_obj.core_prompt != temporal_det.core_prompt:
                    continue

                # Compare with recent detections
                for recent_det in tracked_obj.detections[-3:]:  # Look at last 3 frames
                    iou = compute_iou(temporal_det.box, recent_det.box)
                    if iou > best_iou:
                        best_iou = iou
                        matched_object = tracked_obj

            # Add to existing object or create new
            if matched_object and best_iou > merge_threshold:
                matched_object.detections.append(temporal_det)
                # Update best detection if this has higher score
                best_det = matched_object.detections[matched_object.best_detection_idx]
                if temporal_det.score > best_det.score:
                    matched_object.best_detection_idx = len(matched_object.detections) - 1
            else:
                # Create new tracked object
                new_object = TrackedObject(
                    object_id=object_id_counter,
                    core_prompt=temporal_det.core_prompt,
                    detections=[temporal_det],
                    color=(0, 0, 0),  # Will be assigned later
                    best_detection_idx=0
                )
                tracked_objects.append(new_object)
                object_id_counter += 1

    return tracked_objects


def select_key_frames_for_detection(
    importance_scores: List[float],
    num_frames: int,
    min_spacing: int = 10
) -> List[int]:
    """
    Select key frames for detection based on importance scores.
    Ensures temporal diversity by enforcing minimum spacing.
    """
    if len(importance_scores) <= num_frames:
        return list(range(len(importance_scores)))

    # Create (index, score) pairs and sort by score
    indexed_scores = list(enumerate(importance_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    selected_indices = []
    for idx, score in indexed_scores:
        # Check minimum spacing constraint
        too_close = any(abs(idx - selected) < min_spacing for selected in selected_indices)
        if not too_close:
            selected_indices.append(idx)
            if len(selected_indices) >= num_frames:
                break

    # If we couldn't get enough frames with spacing, relax constraint
    if len(selected_indices) < num_frames:
        for idx, score in indexed_scores:
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) >= num_frames:
                    break

    return sorted(selected_indices) 