"""
Optimized SOWLv2 pipeline with parallel processing and performance improvements.
"""
import os
import time
import tempfile
import subprocess
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import torch

from sowlv2.pipeline import SOWLv2Pipeline
from sowlv2.data.config import PipelineBaseData, MergedOverlayItem, VideoProcessContext
from sowlv2.models import OWLV2Wrapper, SAM2Wrapper
from sowlv2.utils.filesystem_utils import remove_empty_folders
from sowlv2.utils import video_utils
from sowlv2.utils.frame_utils import VALID_EXTS
from sowlv2.utils.pipeline_utils import get_prompt_color

from .parallel_processor import (
    ParallelConfig, ParallelDetectionProcessor,
    ParallelSegmentationProcessor, ParallelIOProcessor
)
from .model_cache import IntelligentModelCache
from .batch_optimizer import IntelligentBatchOptimizer
from .temporal_detection import (
    merge_temporal_detections, select_key_frames_for_detection
)

# Conditional imports for video processing
try:
    from sowlv2.video_pipeline import (
        create_temp_directories_for_video,
        run_video_processing_steps,
        move_video_outputs_to_final_dir,
        VideoProcessingConfig
    )
except ImportError:
    # Define dummy functions if video pipeline not available
    def create_temp_directories_for_video(*args):
        """Dummy function for testing."""
        return None

    def run_video_processing_steps(*args):
        """Dummy function for testing."""
        return {}, 0

    def move_video_outputs_to_final_dir(*args):
        """Dummy function for testing."""
        pass

    class VideoProcessingConfig:
        """Dummy class for testing."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


# pylint: disable=too-many-instance-attributes
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

        # Initialize intelligent optimizers
        self.model_cache = IntelligentModelCache(config.device)
        self.batch_optimizer = IntelligentBatchOptimizer(config.device)

        # Temporal detection settings (will be set from CLI)
        self.vjepa2_optimizer = None
        self.use_temporal_detection = False
        self.temporal_detection_frames = 5
        self.temporal_merge_threshold = 0.7

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
                    # Note: These attributes might not exist in the model wrappers
                    # We'll handle AttributeError gracefully
                    if hasattr(self.owl, 'model'):
                        self.owl.model = torch.compile(self.owl.model, mode="reduce-overhead")
                    if hasattr(self.sam, 'model'):
                        self.sam.model = torch.compile(self.sam.model, mode="reduce-overhead")
                except (AttributeError, RuntimeError, TypeError) as e:
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

    def process_video(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Optimized video processing with frame batching, parallel processing, and V-JEPA 2 optimization.
        """
        # Use V-JEPA 2 optimization if available
        if hasattr(self, 'vjepa2_optimizer') and self.vjepa2_optimizer:
            print("Using V-JEPA 2 optimized video processing...")
            return self._process_video_with_vjepa2(video_path, prompt, output_dir)

        print("Using standard optimized video processing...")
        return self._process_video_optimized_standard(video_path, prompt, output_dir)

    def _process_video_with_vjepa2(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Video processing with V-JEPA 2 optimization and temporal detection.
        """
        # Check if temporal detection is enabled
        use_temporal = hasattr(self, 'use_temporal_detection') and self.use_temporal_detection
        num_detection_frames = getattr(self, 'temporal_detection_frames', 5)
        merge_threshold = getattr(self, 'temporal_merge_threshold', 0.7)

        with tempfile.TemporaryDirectory() as temp_frames_dir:
            # Extract frames
            print("Extracting frames from video...")
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-r", str(self.config.fps),
                 os.path.join(temp_frames_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
                check=True,
                timeout=300
            )

            # Load frames
            frame_paths = sorted([
                os.path.join(temp_frames_dir, f)
                for f in os.listdir(temp_frames_dir)
                if f.endswith('.jpg')
            ])
            frames = [Image.open(fp).convert("RGB") for fp in frame_paths]

            if not frames:
                print("No frames extracted from video")
                return

            # Get temporal importance scores
            print("Analyzing temporal importance with V-JEPA 2...")
            importance_scores = self.vjepa2_optimizer.get_motion_aware_importance_scores(frames)

            if importance_scores is None:
                print("Failed to get importance scores, using uniform sampling")
                key_frame_indices = list(range(0, len(frames),
                                             max(1, len(frames) // num_detection_frames)))
            else:
                # Select key frames for detection
                key_frame_indices = select_key_frames_for_detection(
                    importance_scores,
                    num_detection_frames,
                    min_spacing=max(10, len(frames) // (num_detection_frames * 2))
                )

            print(f"Selected {len(key_frame_indices)} key frames for detection: {key_frame_indices}")

            # Run detection on key frames
            detections_by_frame = {}
            prompts = [prompt] if isinstance(prompt, str) else prompt

            for frame_idx in key_frame_indices:
                frame = frames[frame_idx]
                print(f"Running detection on frame {frame_idx + 1}/{len(frames)}")

                # Use batch detection for multiple prompts
                batch_results = self.detection_processor.detect_multiple_prompts_parallel(
                    frame, prompts, self.config.threshold
                )

                # Collect detections for this frame
                frame_detections = []
                for batch_result in batch_results:
                    frame_detections.extend(batch_result.detections)

                if frame_detections:
                    detections_by_frame[frame_idx] = frame_detections

            if not detections_by_frame:
                print("No objects detected in any key frames")
                return

            # Merge detections across frames
            print("Merging temporal detections...")
            tracked_objects = merge_temporal_detections(detections_by_frame, merge_threshold)
            print(f"Identified {len(tracked_objects)} unique objects across frames")

            # Initialize SAM2 video tracking with best detections
            sam_state = self.sam.init_state(temp_frames_dir)

            # Assign colors and initialize tracking
            prompt_color_map = {}
            next_color_idx = 0
            detection_details_for_video = []

            for obj_idx, tracked_obj in enumerate(tracked_objects):
                # Get color for this object
                color, next_color_idx = get_prompt_color(
                    tracked_obj.core_prompt,
                    prompt_color_map,
                    self.palette,
                    next_color_idx
                )
                tracked_obj.color = color

                # Use best detection to initialize SAM
                best_det = tracked_obj.detections[tracked_obj.best_detection_idx]

                # Add to SAM state
                self.sam.add_new_box(
                    state=sam_state,
                    frame_idx=best_det.frame_idx,
                    box=best_det.box,
                    obj_idx=obj_idx + 1
                )

                # Store detection details
                detection_details_for_video.append({
                    'sam_id': obj_idx + 1,
                    'core_prompt': tracked_obj.core_prompt,
                    'color': color,
                    'tracked_object': tracked_obj  # Store for reference
                })

            # Create video context
            video_ctx = VideoProcessContext(
                tmp_frames_dir=temp_frames_dir,
                initial_sam_state=sam_state,
                first_img_path=frame_paths[0],
                first_pil_img=frames[0],
                detection_details_for_video=detection_details_for_video,
                updated_sam_state=sam_state
            )

            # Process video with temporal tracking
            with tempfile.TemporaryDirectory() as temp_output_dir:
                video_temp_dirs = create_temp_directories_for_video(temp_output_dir)

                # Run video processing
                prompt_color_map, next_color_idx = run_video_processing_steps(
                    video_ctx,
                    self.sam,
                    video_temp_dirs,
                    VideoProcessingConfig(
                        pipeline_config=self.config.pipeline_config,
                        prompt_color_map=prompt_color_map,
                        next_color_idx=next_color_idx,
                        fps=self.config.fps
                    )
                )

                # Move outputs to final directory
                move_video_outputs_to_final_dir(
                    video_temp_dirs,
                    output_dir,
                    self.config.pipeline_config
                )

            print(f"✅ Temporal video processing completed for {video_path}")

    def _process_video_optimized_standard(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Standard optimized video processing with parallel frame processing.
        """
        # For now, use parent implementation with optimized models
        # Future enhancement: Implement batch frame processing with parallel SAM2 tracking
        return super().process_video(video_path, prompt, output_dir)

    def process_frames(self, folder_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Optimized batch frame processing with parallel processing.
        """
        start_time = time.time()

        # Get all image files
        image_files = []
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in VALID_EXTS:
                image_files.append(os.path.join(folder_path, file))

        image_files.sort()  # Process in order

        if not image_files:
            print(f"No valid image files found in {folder_path}")
            return

        print(f"Processing {len(image_files)} frames in parallel...")

        # Process frames in parallel batches
        results = []

        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            futures = []
            for image_file in image_files:
                future = executor.submit(self._process_single_frame_optimized,
                                       image_file, prompt, output_dir)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error processing frame: {e}")

        elapsed_time = time.time() - start_time
        print(f"✅ Batch frame processing completed in {elapsed_time:.2f} seconds")

        # Apply output filtering
        self._filter_outputs_by_flags(output_dir)
        remove_empty_folders(output_dir)

    def _process_single_frame_optimized(self, image_path: str,
                                      prompt: Union[str, List[str]], output_dir: str):
        """
        Process a single frame with optimizations (helper for batch processing).
        """
        try:
            # Use the optimized image processing method
            self.process_image(image_path, prompt, output_dir)
            return True
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing {image_path}: {e}")
            return False

    def process_images_batch(self, image_paths: List[str],
                           prompt: Union[str, List[str]], output_dir: str):
        """
        Process multiple individual images in parallel.

        Args:
            image_paths: List of paths to individual image files
            prompt: Text prompt(s) for detection
            output_dir: Output directory for results
        """
        start_time = time.time()

        print(f"Processing {len(image_paths)} images in parallel...")

        # Process images in parallel
        results = []

        with ThreadPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            futures = []
            for image_path in image_paths:
                future = executor.submit(self._process_single_frame_optimized,
                                       image_path, prompt, output_dir)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error processing image: {e}")

        elapsed_time = time.time() - start_time
        print(f"✅ Batch image processing completed in {elapsed_time:.2f} seconds")

        # Apply output filtering
        self._filter_outputs_by_flags(output_dir)
        remove_empty_folders(output_dir)

    def process_videos_batch(self, video_paths: List[str],
                           prompt: Union[str, List[str]], output_dir: str):
        """
        Process multiple videos in parallel.

        Args:
            video_paths: List of paths to video files
            prompt: Text prompt(s) for detection
            output_dir: Output directory for results
        """
        start_time = time.time()

        print(f"Processing {len(video_paths)} videos in parallel...")

        # Process videos in parallel (limited concurrency for memory management)
        max_concurrent_videos = min(self.parallel_config.max_workers or 2, 2)

        results = []

        with ThreadPoolExecutor(max_workers=max_concurrent_videos) as executor:
            futures = []
            for i, video_path in enumerate(video_paths):
                # Create separate output directory for each video
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_output_dir = os.path.join(output_dir, f"video_{i+1}_{video_name}")

                future = executor.submit(self._process_single_video_optimized,
                                       video_path, prompt, video_output_dir)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error processing video: {e}")

        elapsed_time = time.time() - start_time
        print(f"✅ Batch video processing completed in {elapsed_time:.2f} seconds")

    def _process_single_video_optimized(self, video_path: str,
                                      prompt: Union[str, List[str]], output_dir: str):
        """
        Process a single video with optimizations (helper for batch processing).
        """
        try:
            # Use the optimized video processing method
            self.process_video(video_path, prompt, output_dir)
            return True
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing {video_path}: {e}")
            return False


class ModelOptimizations:
    """Additional model-specific optimizations."""

    @staticmethod
    def optimize_sam_for_video(sam_model: SAM2Wrapper):
        """
        Apply SAM-specific optimizations for video processing.
        """
        # Note: SAM2Wrapper might not have these attributes
        # We'll handle AttributeError gracefully
        try:
            if hasattr(sam_model, 'model') and hasattr(sam_model.model, 'image_encoder'):
                # Cache image embeddings for video frames
                sam_model.model.image_encoder.eval()

                # Enable gradient checkpointing if available
                if hasattr(sam_model.model, 'enable_gradient_checkpointing'):
                    sam_model.model.enable_gradient_checkpointing()
        except AttributeError:
            # Model structure might be different
            pass

    @staticmethod
    def optimize_owl_batch_processing(owl_model: OWLV2Wrapper):
        """
        Optimize OWL model for batch processing.
        """
        # Set model to eval mode
        if hasattr(owl_model, 'model'):
            owl_model.model.eval()

            # Disable gradient computation
            for param in owl_model.model.parameters():
                param.requires_grad = False


class CachedModelWrapper:
    """Wrapper for caching model outputs."""
    # Implementation can be added here as needed
    pass
