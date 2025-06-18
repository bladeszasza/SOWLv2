"""
Optimized SOWLv2 pipeline with parallel processing and performance improvements.
"""
import os
import time
from typing import Union, List
from PIL import Image
import torch

from sowlv2.pipeline import SOWLv2Pipeline
from sowlv2.data.config import PipelineBaseData, MergedOverlayItem
from sowlv2.models import OWLV2Wrapper, SAM2Wrapper
from sowlv2.utils.pipeline_utils import validate_mask
# Image pipeline imports added when needed
from sowlv2.utils.filesystem_utils import remove_empty_folders

from .parallel_processor import (
    ParallelConfig, ParallelDetectionProcessor,
    ParallelSegmentationProcessor, ParallelIOProcessor,
    BatchDetectionResult
)


class OptimizedSOWLv2Pipeline(SOWLv2Pipeline):
    """
    Optimized version of SOWLv2 pipeline with parallel processing and performance improvements.
    """

    def __init__(self, config: PipelineBaseData = None, parallel_config: ParallelConfig = None):
        """
        Initialize optimized pipeline with parallel processing support.

        Args:
            config: Pipeline configuration
            parallel_config: Parallel processing configuration
        """
        super().__init__(config)

        # Initialize parallel processors
        self.parallel_config = parallel_config or ParallelConfig()
        self.detection_processor = ParallelDetectionProcessor(
            self.owl, self.sam, self.parallel_config
        )
        self.segmentation_processor = ParallelSegmentationProcessor(
            self.sam, self.parallel_config
        )
        self.io_processor = ParallelIOProcessor(self.parallel_config)

        # Enable model optimizations
        self._optimize_models()

    def _optimize_models(self):
        """Apply model-specific optimizations."""
        if self.config.device != "cpu" and torch.cuda.is_available():
            # Enable mixed precision for faster inference
            self.use_amp = True

            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

            # Compile models if using PyTorch 2.0+
            if hasattr(torch, 'compile'):
                try:
                    print("Compiling models with torch.compile()...")
                    self.owl.model = torch.compile(self.owl.model, mode="reduce-overhead")
                    self.sam.model = torch.compile(self.sam.model, mode="reduce-overhead")
                except Exception as e:
                    print(f"Model compilation failed: {e}")
        else:
            self.use_amp = False

    def process_image(self, image_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Optimized image processing with parallel detection and segmentation.
        """
        start_time = time.time()

        # Load image once
        pil_image = Image.open(image_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Convert prompt to list if needed
        prompts = [prompt] if isinstance(prompt, str) else prompt

        # Parallel detection for multiple prompts
        print(f"Processing {len(prompts)} prompt(s) in parallel...")
        batch_results = self.detection_processor.detect_multiple_prompts_parallel(
            pil_image, prompts, self.config.threshold
        )

        # Collect all detections
        all_detections = []
        for batch_result in batch_results:
            all_detections.extend(batch_result.detections)

        if not all_detections:
            print(f"No objects detected for prompt(s) '{prompt}' in image '{image_path}'.")
            return

        print(f"Found {len(all_detections)} total detections")

        # Parallel segmentation
        segmentation_results = self.segmentation_processor.segment_detections_parallel(
            pil_image, all_detections
        )

        # Process results and prepare for saving
        items_for_merged_overlay: List[MergedOverlayItem] = []
        save_tasks = []

        for idx, (det_detail, mask) in enumerate(segmentation_results):
            if mask is None:
                print(f"SAM2 failed to segment object {idx} ({det_detail['core_prompt']}).")
                continue

            # Update detection detail
            det_detail['mask'] = mask
            det_detail['color'] = self._get_color_for_prompt(det_detail['core_prompt'])

            # Prepare for merged overlay
            merged_item = MergedOverlayItem(
                mask=mask,
                color=det_detail['color'],
                label=det_detail['core_prompt']
            )
            items_for_merged_overlay.append(merged_item)

            # Prepare save tasks for parallel I/O
            prompt_slug = det_detail['core_prompt'].replace(' ', '_')
            base_name_slug = base_name.replace(' ', '_')

            # Binary mask path
            binary_path = os.path.join(
                output_dir, "binary", "frames",
                f"{base_name_slug}_obj{idx}_{prompt_slug}_mask.png"
            )
            save_tasks.append((binary_path, Image.fromarray(mask)))

            # Overlay path
            from sowlv2.utils.pipeline_utils import create_overlay  # pylint: disable=import-outside-toplevel
            overlay_img = create_overlay(pil_image, mask, det_detail['color'])
            overlay_path = os.path.join(
                output_dir, "overlay", "frames",
                f"{base_name_slug}_obj{idx}_{prompt_slug}_overlay.png"
            )
            save_tasks.append((overlay_path, overlay_img))

        # Save all outputs in parallel
        print(f"Saving {len(save_tasks)} outputs in parallel...")
        self.io_processor.save_outputs_parallel(save_tasks)

        # Create merged overlay
        from sowlv2.image_pipeline import create_and_save_merged_overlay  # pylint: disable=import-outside-toplevel
        create_and_save_merged_overlay(
            items_for_merged_overlay,
            pil_image,
            output_dir,
            int(base_name) if base_name.isdigit() else 0
        )

        # Apply output filtering
        self._filter_outputs_by_flags(output_dir)
        remove_empty_folders(output_dir)

        elapsed_time = time.time() - start_time
        print(f"✅ Image processing completed in {elapsed_time:.2f} seconds")

    def process_video_optimized(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Optimized video processing with frame batching and parallel processing.
        """
        # TODO: Implement optimized video processing with:
        # - Batch frame processing
        # - Parallel mask propagation
        # - Optimized video encoding

        # For now, fall back to parent implementation
        print("Using standard video processing (optimization coming soon)...")
        super().process_video(video_path, prompt, output_dir)


class ModelOptimizations:
    """Additional model-specific optimizations."""

    @staticmethod
    def optimize_sam_for_video(sam_model: SAM2Wrapper):
        """
        Apply SAM-specific optimizations for video processing.
        """
        if hasattr(sam_model.model, 'image_encoder'):
            # Cache image embeddings for video frames
            sam_model.model.image_encoder.eval()

            # Enable gradient checkpointing if available
            if hasattr(sam_model.model, 'enable_gradient_checkpointing'):
                sam_model.model.enable_gradient_checkpointing()

    @staticmethod
    def optimize_owl_batch_processing(owl_model: OWLV2Wrapper):
        """
        Optimize OWL model for batch processing.
        """
        # Set model to eval mode
        owl_model.model.eval()

        # Disable gradient computation
        for param in owl_model.model.parameters():
            param.requires_grad = False


class CachedModelWrapper:
    """
    Wrapper to add caching capabilities to models.
    """

    def __init__(self, model, cache_size: int = 100):
        """Initialize cached model wrapper."""
        self.model = model
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []

    def _get_cache_key(self, *args, **kwargs):
        """Generate cache key from arguments."""
        # Simple hash-based key (can be improved)
        return hash(str(args) + str(kwargs))

    def cached_forward(self, *args, **kwargs):
        """Forward with caching."""
        key = self._get_cache_key(*args, **kwargs)

        if key in self._cache:
            # Move to end (LRU)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]

        # Compute result
        result = self.model(*args, **kwargs)

        # Add to cache
        self._cache[key] = result
        self._cache_order.append(key)

        # Evict oldest if cache is full
        if len(self._cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        return result
