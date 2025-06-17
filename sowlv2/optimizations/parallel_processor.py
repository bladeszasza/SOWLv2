"""
Parallel processing optimizations for SOWLv2 pipeline.
Implements multiprocessing for multiple prompts and batch processing.
"""
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass
import torch
from PIL import Image
import numpy as np

from sowlv2.data.config import SingleDetectionInput, MergedOverlayItem
from sowlv2.utils.pipeline_utils import validate_mask, create_overlay


@dataclass
class BatchDetectionResult:
    """Container for batch detection results."""
    prompt: str
    detections: List[Dict[str, Any]]
    prompt_idx: int


@dataclass 
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None  # None = use CPU count
    batch_size: int = 4
    use_gpu_batching: bool = True
    thread_pool_size: int = 8  # For I/O operations


class ParallelDetectionProcessor:
    """Handles parallel detection processing for multiple prompts."""
    
    def __init__(self, owl_model, sam_model, config: ParallelConfig = None):
        """Initialize parallel processor with models."""
        self.owl_model = owl_model
        self.sam_model = sam_model
        self.config = config or ParallelConfig()
        self.device = owl_model.device
        
    def detect_multiple_prompts_parallel(
        self, 
        image: Image.Image, 
        prompts: List[str], 
        threshold: float
    ) -> List[BatchDetectionResult]:
        """
        Process multiple prompts in parallel using batch processing.
        
        Args:
            image: Input PIL image
            prompts: List of text prompts
            threshold: Detection threshold
            
        Returns:
            List of BatchDetectionResult objects
        """
        if len(prompts) == 1:
            # Single prompt, no parallelization needed
            detections = self.owl_model.detect(
                image=image, prompt=prompts[0], threshold=threshold
            )
            return [BatchDetectionResult(prompts[0], detections, 0)]
        
        # Batch process prompts for GPU efficiency
        if self.config.use_gpu_batching and self.device != "cpu":
            return self._batch_detect_gpu(image, prompts, threshold)
        else:
            # CPU parallel processing
            return self._parallel_detect_cpu(image, prompts, threshold)
    
    def _batch_detect_gpu(
        self, 
        image: Image.Image, 
        prompts: List[str], 
        threshold: float
    ) -> List[BatchDetectionResult]:
        """Batch process prompts on GPU for efficiency."""
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i:i + self.config.batch_size]
            
            # OWLv2 can handle multiple prompts at once
            batch_detections = self.owl_model.detect(
                image=image, 
                prompt=batch_prompts, 
                threshold=threshold
            )
            
            # Group detections by prompt
            prompt_detections = {p: [] for p in batch_prompts}
            for det in batch_detections:
                prompt_detections[det['core_prompt']].append(det)
            
            # Create results
            for j, prompt in enumerate(batch_prompts):
                results.append(BatchDetectionResult(
                    prompt, 
                    prompt_detections[prompt], 
                    i + j
                ))
        
        return sorted(results, key=lambda x: x.prompt_idx)
    
    def _parallel_detect_cpu(
        self, 
        image: Image.Image, 
        prompts: List[str], 
        threshold: float
    ) -> List[BatchDetectionResult]:
        """Process prompts in parallel on CPU."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit detection tasks
            future_to_prompt = {
                executor.submit(
                    self._detect_single_prompt, 
                    image, prompt, threshold, idx
                ): (prompt, idx)
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results
            for future in as_completed(future_to_prompt):
                prompt, idx = future_to_prompt[future]
                try:
                    detections = future.result()
                    results.append(BatchDetectionResult(prompt, detections, idx))
                except Exception as e:
                    print(f"Error detecting prompt '{prompt}': {e}")
                    results.append(BatchDetectionResult(prompt, [], idx))
        
        return sorted(results, key=lambda x: x.prompt_idx)
    
    def _detect_single_prompt(
        self, 
        image: Image.Image, 
        prompt: str, 
        threshold: float,
        idx: int
    ) -> List[Dict[str, Any]]:
        """Helper for parallel detection of single prompt."""
        return self.owl_model.detect(
            image=image, prompt=prompt, threshold=threshold
        )


class ParallelSegmentationProcessor:
    """Handles parallel segmentation processing."""
    
    def __init__(self, sam_model, config: ParallelConfig = None):
        """Initialize parallel segmentation processor."""
        self.sam_model = sam_model
        self.config = config or ParallelConfig()
        
    def segment_detections_parallel(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Optional[np.ndarray]]]:
        """
        Process multiple detections in parallel for segmentation.
        
        Args:
            image: Input PIL image
            detections: List of detection dictionaries
            
        Returns:
            List of tuples (detection, mask)
        """
        if len(detections) <= 1:
            # Single detection, no parallelization needed
            if detections:
                mask = self.sam_model.segment(image, detections[0]['box'])
                return [(detections[0], mask)]
            return []
        
        # Use ThreadPoolExecutor for I/O-bound SAM operations
        indexed_detections = [(idx, det) for idx, det in enumerate(detections)]
        results = []
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            future_to_det = {
                executor.submit(
                    self._segment_single_detection,
                    image, det
                ): (idx, det)
                for idx, det in indexed_detections
            }
            
            for future in as_completed(future_to_det):
                idx, det = future_to_det[future]
                try:
                    mask = future.result()
                    results.append((idx, det, mask))
                except Exception as e:
                    print(f"Error segmenting detection: {e}")
                    results.append((idx, det, None))
        
        # Sort results by original index to ensure deterministic output
        results.sort(key=lambda x: x[0])
        return [(det, mask) for _, det, mask in results]
    
    def _segment_single_detection(
        self,
        image: Image.Image,
        detection: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Helper for parallel segmentation of single detection."""
        return self.sam_model.segment(image, detection['box'])


class ParallelFrameProcessor:
    """Handles parallel frame processing for videos."""
    
    def __init__(self, config: ParallelConfig = None):
        """Initialize parallel frame processor."""
        self.config = config or ParallelConfig()
        
    def process_frames_parallel(
        self,
        frame_paths: List[str],
        process_func,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process multiple frames in parallel.
        
        Args:
            frame_paths: List of frame file paths
            process_func: Function to process each frame
            *args, **kwargs: Additional arguments for process_func
            
        Returns:
            List of processing results
        """
        results = [None] * len(frame_paths)
        
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            future_to_idx = {
                executor.submit(
                    process_func,
                    frame_path,
                    *args,
                    **kwargs
                ): idx
                for idx, frame_path in enumerate(frame_paths)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    results[idx] = None
        
        return results


class ParallelIOProcessor:
    """Handles parallel I/O operations for saving outputs."""
    
    def __init__(self, config: ParallelConfig = None):
        """Initialize parallel I/O processor."""
        self.config = config or ParallelConfig()
        
    def save_outputs_parallel(
        self,
        save_tasks: List[Tuple[str, Image.Image]]
    ):
        """
        Save multiple images in parallel.
        
        Args:
            save_tasks: List of (filepath, image) tuples
        """
        with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
            futures = [
                executor.submit(self._save_single_image, filepath, img)
                for filepath, img in save_tasks
            ]
            
            # Wait for all saves to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error saving image: {e}")
    
    def _save_single_image(self, filepath: str, image: Image.Image):
        """Helper to save single image."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath) 