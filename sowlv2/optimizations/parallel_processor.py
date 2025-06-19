"""
Parallel processing for optimized SOWLv2 pipeline.
"""
import os
import threading
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from PIL import Image

from sowlv2.models import OWLV2Wrapper, SAM2Wrapper


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None
    detection_batch_size: int = 4
    segmentation_batch_size: int = 2
    io_batch_size: int = 8
    enable_threading: bool = True
    thread_safety: bool = True


@dataclass
class BatchDetectionResult:
    """Result from batch detection processing."""
    detections: List[Dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)


class ParallelDetectionProcessor:
    """Handles parallel object detection across multiple prompts and images."""

    def __init__(self, owl_model: OWLV2Wrapper, sam_model: SAM2Wrapper,
                 parallel_config: ParallelConfig):
        self.owl = owl_model
        self.sam = sam_model
        self.config = parallel_config
        self._thread_lock = threading.Lock() if parallel_config.thread_safety else None

    def detect_multiple_prompts_parallel(self,
                                       image: Image.Image,
                                       prompts: List[str],
                                       threshold: float) -> List[BatchDetectionResult]:
        """
        Run detection for multiple prompts in parallel.
        """
        if not prompts:
            return []

        results = []

        if self.config.enable_threading and len(prompts) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                for prompt in prompts:
                    future = executor.submit(self._detect_single_prompt_safe,
                                           image, prompt, threshold)
                    futures[future] = prompt

                for future in as_completed(futures):
                    prompt = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:  # pylint: disable=broad-except
                        error_result = BatchDetectionResult()
                        error_result.error_count = 1
                        error_result.errors = [f"Detection failed for '{prompt}': {str(e)}"]
                        results.append(error_result)
        else:
            # Sequential processing
            for prompt in prompts:
                result = self._detect_single_prompt_safe(image, prompt, threshold)
                results.append(result)

        return results

    def _detect_single_prompt_safe(self,
                                 image: Image.Image,
                                 prompt: str,
                                 threshold: float) -> BatchDetectionResult:
        """
        Thread-safe detection for a single prompt.
        """
        result = BatchDetectionResult()

        try:
            if self._thread_lock:
                with self._thread_lock:
                    detections = self._detect_single_prompt(image, prompt, threshold)
            else:
                detections = self._detect_single_prompt(image, prompt, threshold)

            result.detections = detections
            result.success_count = len(detections)

        except Exception as e:  # pylint: disable=broad-except
            result.error_count = 1
            result.errors = [f"Detection failed for '{prompt}': {str(e)}"]

        return result

    def _detect_single_prompt(self,
                            image: Image.Image,
                            prompt: str,
                            threshold: float) -> List[Dict[str, Any]]:
        """
        Core detection logic for a single prompt.
        """
        # Use OWL model for detection
        detections = self.owl.detect(image=image, prompt=[prompt], threshold=threshold)

        # Filter by threshold and format
        valid_detections = []
        for detection in detections:
            if detection['score'] >= threshold:
                detection['core_prompt'] = prompt
                valid_detections.append(detection)

        return valid_detections


class ParallelSegmentationProcessor:
    """Handles parallel segmentation of detected objects."""

    def __init__(self, sam_model: SAM2Wrapper, parallel_config: ParallelConfig):
        self.sam = sam_model
        self.config = parallel_config
        self._thread_lock = threading.Lock() if parallel_config.thread_safety else None

    def segment_detections_parallel(self,
                                  image: Image.Image,
                                  detections: List[Dict[str, Any]]) -> List[
                                      Tuple[Dict[str, Any], Any]]:
        """
        Segment multiple detections in parallel.
        """
        if not detections:
            return []

        results = []

        if self.config.enable_threading and len(detections) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                for i, detection in enumerate(detections):
                    future = executor.submit(self._segment_single_detection_safe,
                                           image, detection, i)
                    futures[future] = (i, detection)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:  # pylint: disable=broad-except
                        detection_idx, detection = futures[future]
                        print(f"Segmentation failed for detection {detection_idx}: {e}")
                        results.append((detection, None))
        else:
            # Sequential processing
            for i, detection in enumerate(detections):
                result = self._segment_single_detection_safe(image, detection, i)
                results.append(result)

        return results

    def _segment_single_detection_safe(self,
                                     image: Image.Image,
                                     detection: Dict[str, Any],
                                     detection_idx: int) -> Tuple[Dict[str, Any], Any]:
        """
        Thread-safe segmentation for a single detection.
        """
        try:
            if self._thread_lock:
                with self._thread_lock:
                    mask = self._segment_single_detection(image, detection)
            else:
                mask = self._segment_single_detection(image, detection)

            return (detection, mask)

        except Exception as e:  # pylint: disable=broad-except
            print(f"Segmentation failed for detection {detection_idx}: {e}")
            return (detection, None)

    def _segment_single_detection(self,
                                image: Image.Image,
                                detection: Dict[str, Any]):
        """
        Core segmentation logic for a single detection.
        """
        # Use SAM model for segmentation
        return self.sam.segment(image, detection['box'])


class ParallelIOProcessor:
    """Handles parallel I/O operations for saving outputs."""

    def __init__(self, parallel_config: ParallelConfig):
        self.config = parallel_config

    def save_outputs_parallel(self, save_tasks: List[Tuple[str, Image.Image]]):
        """
        Save multiple outputs in parallel.

        Args:
            save_tasks: List of (file_path, image) tuples to save
        """
        if not save_tasks:
            return

        if self.config.enable_threading and len(save_tasks) > 1:
            # Parallel saving
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for file_path, image in save_tasks:
                    future = executor.submit(self._save_single_output, file_path, image)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Save operation failed: {e}")
        else:
            # Sequential saving
            for file_path, image in save_tasks:
                self._save_single_output(file_path, image)

    def _save_single_output(self, file_path: str, image: Image.Image):
        """
        Save a single output file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save image
        image.save(file_path)


class ParallelFrameProcessor:
    """Handles parallel processing of video frames."""

    def __init__(self, detection_processor: ParallelDetectionProcessor,
                 segmentation_processor: ParallelSegmentationProcessor,
                 parallel_config: ParallelConfig):
        self.detection_processor = detection_processor
        self.segmentation_processor = segmentation_processor
        self.config = parallel_config

    def process_frames_parallel(self,
                              frames: List[Image.Image],
                              prompts: List[str],
                              threshold: float) -> List[Dict[str, Any]]:
        """
        Process multiple frames in parallel.
        """
        results = []

        if self.config.enable_threading and len(frames) > 1:
            # Parallel frame processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                for i, frame in enumerate(frames):
                    future = executor.submit(self._process_single_frame,
                                           frame, prompts, threshold, i)
                    futures[future] = i

                for future in as_completed(futures):
                    frame_idx = futures[future]
                    try:
                        result = future.result()
                        result['frame_idx'] = frame_idx
                        results.append(result)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Frame {frame_idx} processing failed: {e}")
                        results.append({'frame_idx': frame_idx, 'detections': [], 'error': str(e)})
        else:
            # Sequential processing
            for i, frame in enumerate(frames):
                result = self._process_single_frame(frame, prompts, threshold, i)
                result['frame_idx'] = i
                results.append(result)

        return results

    def _process_single_frame(self,
                            frame: Image.Image,
                            prompts: List[str],
                            threshold: float,
                            frame_idx: int) -> Dict[str, Any]:
        """
        Process a single frame with detection and segmentation.
        """
        # Run detection
        detection_results = self.detection_processor.detect_multiple_prompts_parallel(
            frame, prompts, threshold
        )

        # Collect all detections
        all_detections = []
        for batch_result in detection_results:
            all_detections.extend(batch_result.detections)

        # Run segmentation if detections found
        if all_detections:
            segmentation_results = self.segmentation_processor.segment_detections_parallel(
                frame, all_detections
            )
        else:
            segmentation_results = []

        return {
            'frame_idx': frame_idx,
            'detections': all_detections,
            'segmentations': segmentation_results
        }
