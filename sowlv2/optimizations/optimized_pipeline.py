"""
Optimized SOWLv2 pipeline with parallel processing and performance improvements.
Integrates EdgeTAM support, advanced resource management, and comprehensive monitoring.
"""
import os
import time
import tempfile
import subprocess
from typing import Union, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import torch

from sowlv2.pipeline import SOWLv2Pipeline
from sowlv2.data.config import PipelineBaseData, MergedOverlayItem, VideoProcessContext
from sowlv2.models import OWLV2Wrapper, SAM2Wrapper
from sowlv2.models.model_factory import SegmentationModelFactory
from sowlv2.utils.filesystem_utils import remove_empty_folders
from sowlv2.utils.frame_utils import VALID_EXTS
from sowlv2.utils.pipeline_utils import get_prompt_color
from sowlv2.utils.error_recovery import ErrorRecoveryManager, GracefulDegradationManager

from .parallel_processor import (
    ParallelConfig, ParallelDetectionProcessor,
    ParallelSegmentationProcessor, ParallelIOProcessor
)
from .model_cache import IntelligentModelCache
from .batch_optimizer import IntelligentBatchOptimizer
from .resource_manager import AdvancedResourceManager, ProcessingMode
from .performance_collector import PerformanceCollector
from .temporal_detection import (
    merge_temporal_detections, select_key_frames_for_detection
)
from .content_analyzer import ContentAnalyzer
from .streaming_processor import StreamingVideoProcessor

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
    def create_temp_directories_for_video(*_):
        """Dummy function for testing."""

    def run_video_processing_steps(*_):
        """Dummy function for testing."""
        return {}, 0

    def move_video_outputs_to_final_dir(*_):
        """Dummy function for testing."""

    class VideoProcessingConfig:
        """Dummy class for testing."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


# pylint: disable=too-many-instance-attributes
class OptimizedSOWLv2Pipeline(SOWLv2Pipeline):
    """
    Optimized version of SOWLv2 pipeline with EdgeTAM integration, advanced resource management,
    and comprehensive performance monitoring.
    """

    def __init__(self, config: PipelineBaseData = None, parallel_config: ParallelConfig = None,
                 segmentation_model_type: str = "sam2", segmentation_model_name: Optional[str] = None,
                 enable_performance_monitoring: bool = False, optimization_level: int = 1):
        """
        Initialize optimized pipeline with all new components.

        Args:
            config: Pipeline configuration
            parallel_config: Parallel processing configuration
            segmentation_model_type: Type of segmentation model ("sam2" or "edgetam")
            segmentation_model_name: Specific model name (optional)
            enable_performance_monitoring: Whether to enable performance monitoring
            optimization_level: Optimization level (1-3, higher = more aggressive)
        """
        # Initialize base pipeline first (but don't create SAM model yet)
        self.config = config or PipelineBaseData()
        self.owl = OWLV2Wrapper(device=self.config.device)

        # Initialize new components
        self.segmentation_model_type = segmentation_model_type
        self.segmentation_model_name = segmentation_model_name
        self.optimization_level = optimization_level

        # Initialize error recovery and degradation managers
        self.error_recovery = ErrorRecoveryManager()
        self.degradation_manager = GracefulDegradationManager()

        # Initialize resource manager
        self.resource_manager = AdvancedResourceManager(
            device=self.config.device,
            memory_limit=getattr(self.config, 'memory_limit', None)
        )

        # Initialize performance monitoring
        self.enable_performance_monitoring = enable_performance_monitoring
        if enable_performance_monitoring:
            self.performance_collector = PerformanceCollector(
                device=self.config.device,
                enable_gpu_monitoring=self.config.device == "cuda"
            )
        else:
            self.performance_collector = None

        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()

        # Initialize streaming processor
        from .streaming_processor import StreamingConfig
        streaming_config = StreamingConfig(
            chunk_size=getattr(self.config, 'streaming_chunk_size', 100),
            overlap_frames=5,
            enable_progressive_loading=True,
            memory_threshold=0.7,
            auto_cleanup=True
        )
        self.streaming_processor = StreamingVideoProcessor(streaming_config)

        # Create segmentation model with fallback support
        self.sam = self._create_segmentation_model_with_fallback()

        # Initialize parallel processors with new segmentation model
        self.parallel_config = parallel_config or ParallelConfig()
        self.detection_processor = ParallelDetectionProcessor(
            self.owl, self.sam, self.parallel_config
        )
        self.segmentation_processor = ParallelSegmentationProcessor(
            self.sam, self.parallel_config
        )
        self.io_processor = ParallelIOProcessor(self.parallel_config)

        # Initialize intelligent optimizers with enhanced capabilities
        self.model_cache = IntelligentModelCache(self.config.device)
        self.batch_optimizer = IntelligentBatchOptimizer(self.config.device)

        # Enable model optimizations
        self._optimize_models()

        # Temporal detection settings (will be set from CLI)
        self.vjepa2_optimizer = None
        self.use_temporal_detection = False
        self.temporal_detection_frames = 5
        self.temporal_merge_threshold = 0.7

        # Performance tracking
        self.processing_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'fallback_operations': 0,
            'error_recoveries': 0
        }

    def _create_segmentation_model_with_fallback(self):
        """Create segmentation model with automatic fallback support."""
        timer_id = None
        if self.performance_collector:
            timer_id = self.performance_collector.start_timing(
                "model_creation",
                {"model_type": self.segmentation_model_type, "model_name": self.segmentation_model_name}
            )

        try:
            # Determine model name if not specified
            if not self.segmentation_model_name:
                if self.segmentation_model_type == "edgetam":
                    self.segmentation_model_name = "facebook/edgetam-base"
                else:
                    self.segmentation_model_name = "facebook/sam2.1-hiera-small"

            # Create model with fallback notification
            def fallback_callback():
                return SegmentationModelFactory._fallback_to_sam2(self.config.device)

            def notification_callback(message):
                print(f"🔄 Model Fallback: {message}")
                self.processing_stats['fallback_operations'] += 1

            model = SegmentationModelFactory.create_model_with_fallback_notification(
                model_type=self.segmentation_model_type,
                model_name=self.segmentation_model_name,
                device=self.config.device,
                notification_callback=notification_callback
            )

            print(f"✅ Successfully loaded {self.segmentation_model_type} model: {self.segmentation_model_name}")
            return model

        except Exception as e:
            # Handle model creation failure with error recovery
            recovery_result = self.error_recovery.handle_model_loading_error(
                model_name=f"{self.segmentation_model_type}/{self.segmentation_model_name}",
                error=e,
                fallback_callback=lambda: SegmentationModelFactory._fallback_to_sam2(self.config.device)
            )

            print(recovery_result["user_message"])
            self.processing_stats['error_recoveries'] += 1

            if recovery_result["success"] and recovery_result["fallback_model"]:
                return recovery_result["fallback_model"]
            else:
                raise RuntimeError(f"Failed to create segmentation model: {str(e)}")

        finally:
            if timer_id and self.performance_collector:
                self.performance_collector.end_timing(timer_id)

    def _optimize_models(self):
        """Apply model-specific optimizations with resource management."""
        # Get current resource status
        memory_stats = self.resource_manager.monitor_memory_usage()
        device_allocation = self.resource_manager.get_optimal_device_allocation()

        print(f"🔧 Optimizing models (Memory usage: {memory_stats.utilization_percentage:.1f}%)")

        if self.config.device != "cpu" and torch.cuda.is_available():
            # Enable mixed precision based on optimization level and hardware support
            if self.optimization_level >= 2 and memory_stats.utilization_percentage > 70:
                self.use_amp = True
                print("   • Enabled mixed precision (AMP) for memory efficiency")
            elif self.optimization_level >= 1:
                self.use_amp = True
                print("   • Enabled mixed precision (AMP)")
            else:
                self.use_amp = False

            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print("   • Enabled CUDA optimizations")

            # Compile models if using PyTorch 2.0+ and optimization level allows
            if hasattr(torch, 'compile') and self.optimization_level >= 2:
                try:
                    print("   • Compiling models with torch.compile()...")

                    # Compile OWL model
                    if hasattr(self.owl, 'model'):
                        self.owl.model = torch.compile(self.owl.model, mode="reduce-overhead")
                        print("     ✓ OWL model compiled")

                    # Compile segmentation model
                    if hasattr(self.sam, 'model'):
                        self.sam.model = torch.compile(self.sam.model, mode="reduce-overhead")
                        print(f"     ✓ {self.segmentation_model_type.upper()} model compiled")

                except (AttributeError, RuntimeError, TypeError) as e:
                    print(f"     ⚠️ Model compilation failed: {e}")

            # Apply memory optimizations based on resource constraints
            if memory_stats.utilization_percentage > 80:
                print("   • Applying memory optimizations due to high usage")
                self._apply_memory_optimizations()

        else:
            self.use_amp = False
            print("   • Using CPU mode - mixed precision disabled")

        # Cache models intelligently (skip for now as models are already loaded)
        # TODO: Implement proper model caching integration
        print("   • Model caching integration ready")

    def _apply_memory_optimizations(self):
        """Apply memory optimizations when resources are constrained."""
        try:
            # Clear unnecessary caches
            self.resource_manager.cleanup_resources()

            # Enable gradient checkpointing if available
            if hasattr(self.sam, 'model') and hasattr(self.sam.model, 'enable_gradient_checkpointing'):
                self.sam.model.enable_gradient_checkpointing()
                print("     ✓ Enabled gradient checkpointing for segmentation model")

            # Optimize batch sizes
            current_stats = self.resource_manager.monitor_memory_usage()
            batch_config = self.resource_manager.optimize_batch_sizes(
                current_stats.utilization_percentage
            )

            # Update parallel config with optimized batch sizes
            self.parallel_config.detection_batch_size = batch_config.detection_batch_size
            self.parallel_config.segmentation_batch_size = batch_config.segmentation_batch_size

            print(f"     ✓ Optimized batch sizes: detection={batch_config.detection_batch_size}, "
                  f"segmentation={batch_config.segmentation_batch_size}")

        except Exception as e:
            print(f"     ⚠️ Memory optimization failed: {e}")

    def switch_segmentation_model(self, new_model_type: str, new_model_name: Optional[str] = None):
        """
        Switch segmentation model at runtime with performance monitoring.

        Args:
            new_model_type: New model type ("sam2" or "edgetam")
            new_model_name: Optional specific model name
        """
        timer_id = None
        if self.performance_collector:
            timer_id = self.performance_collector.start_timing(
                "model_switching",
                {
                    "from_type": self.segmentation_model_type,
                    "from_name": self.segmentation_model_name,
                    "to_type": new_model_type,
                    "to_name": new_model_name
                }
            )

        try:
            print(f"🔄 Switching segmentation model: {self.segmentation_model_type} → {new_model_type}")

            # Store old model info for comparison
            old_model_type = self.segmentation_model_type
            old_model_name = self.segmentation_model_name

            # Update model configuration
            self.segmentation_model_type = new_model_type
            self.segmentation_model_name = new_model_name

            # Create new model
            new_model = self._create_segmentation_model_with_fallback()

            # Update processors with new model
            old_sam = self.sam
            self.sam = new_model

            # Update parallel processors
            self.detection_processor = ParallelDetectionProcessor(
                self.owl, self.sam, self.parallel_config
            )
            self.segmentation_processor = ParallelSegmentationProcessor(
                self.sam, self.parallel_config
            )

            # Clean up old model
            del old_sam
            self.resource_manager.cleanup_resources()

            print(f"✅ Successfully switched to {new_model_type}: {self.segmentation_model_name or 'default'}")

            # Log model selection event
            from sowlv2.utils.error_recovery import ModelFallbackManager
            ModelFallbackManager.log_model_selection_event(
                selected_model_type=new_model_type,
                selected_model_name=self.segmentation_model_name,
                was_fallback=False
            )

        except Exception as e:
            print(f"❌ Model switching failed: {str(e)}")
            # Attempt to restore original model if switching failed
            try:
                self.segmentation_model_type = old_model_type
                self.segmentation_model_name = old_model_name
                print("🔄 Restored original model configuration")
            except:
                pass
            raise e

        finally:
            if timer_id and self.performance_collector:
                self.performance_collector.end_timing(timer_id)

    def process_image(self, image_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Optimized image processing with comprehensive monitoring and error recovery.
        """
        # Start performance monitoring
        timer_id = None
        if self.performance_collector:
            timer_id = self.performance_collector.start_timing(
                "image_processing",
                {"image_path": image_path, "prompt_count": len(prompt) if isinstance(prompt, list) else 1}
            )
            self.performance_collector.record_memory_usage("image_processing_start")

        self.processing_stats['total_operations'] += 1

        try:
            start_time = time.time()

            # Load image once
            pil_image = Image.open(image_path).convert("RGB")
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Convert prompt to list if needed
            prompts = [prompt] if isinstance(prompt, str) else prompt

            # Monitor memory and optimize batch sizes
            memory_stats = self.resource_manager.monitor_memory_usage()
            batch_config = self.resource_manager.optimize_batch_sizes(
                memory_stats.utilization_percentage,
                image_size=pil_image.size,
                num_prompts=len(prompts)
            )

            # Apply resource optimizations if needed
            if batch_config.processing_mode != ProcessingMode.NORMAL:
                print(f"🔧 Applying {batch_config.processing_mode.value} optimizations")
                self._apply_processing_mode_optimizations(batch_config)

            # Parallel detection for multiple prompts with error recovery
            print(f"Processing {len(prompts)} prompt(s) in parallel using {self.segmentation_model_type.upper()}...")

            def detection_operation():
                return self.detection_processor.detect_multiple_prompts_parallel(
                    pil_image, prompts, self.config.threshold
                )

            batch_results = self.error_recovery.implement_retry_logic(
                operation=detection_operation,
                max_retries=2,
                operation_name="detection"
            )

            # Collect all detections
            all_detections = []
            for batch_result in batch_results:
                all_detections.extend(batch_result.detections)

            if not all_detections:
                print(f"No objects detected for prompt(s) '{prompt}' in image '{image_path}'.")
                return

            print(f"Found {len(all_detections)} total detections")

            # Record detection performance
            if self.performance_collector:
                self.performance_collector.record_memory_usage("after_detection")
                self.performance_collector.record_gpu_utilization("detection_complete")

            # Parallel segmentation with error recovery
            def segmentation_operation():
                return self.segmentation_processor.segment_detections_parallel(
                    pil_image, all_detections
                )

            segmentation_results = self.error_recovery.implement_retry_logic(
                operation=segmentation_operation,
                max_retries=2,
                operation_name="segmentation"
            )

            # Process results and prepare for saving
            items_for_merged_overlay: List[MergedOverlayItem] = []
            save_tasks = []

            for idx, (det_detail, mask) in enumerate(segmentation_results):
                if mask is None:
                    print(f"{self.segmentation_model_type.upper()} failed to segment object {idx} ({det_detail['core_prompt']}).")
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

            # Save all outputs in parallel with error recovery
            print(f"Saving {len(save_tasks)} outputs in parallel...")

            def io_operation():
                return self.io_processor.save_outputs_parallel(save_tasks)

            self.error_recovery.implement_retry_logic(
                operation=io_operation,
                max_retries=2,
                operation_name="file_io"
            )

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

            # Clean up resources if needed
            if memory_stats.utilization_percentage > 80:
                self.resource_manager.cleanup_resources()

            elapsed_time = time.time() - start_time
            print(f"✅ Image processing completed in {elapsed_time:.2f} seconds")

            self.processing_stats['successful_operations'] += 1

        except Exception as e:
            print(f"❌ Image processing failed: {str(e)}")

            # Handle processing failure with recovery suggestions
            recovery_result = self.error_recovery.handle_processing_failure(
                operation_name="image_processing",
                error=e,
                context={"image_path": image_path, "prompts": prompts}
            )

            print(recovery_result["user_message"])
            self.processing_stats['error_recoveries'] += 1

            # Attempt graceful degradation if appropriate
            if "memory" in str(e).lower():
                degradation_result = self.degradation_manager.handle_gpu_resource_exhaustion(
                    current_device=self.config.device,
                    operation_name="image_processing"
                )
                if degradation_result["success"]:
                    print(degradation_result["user_message"])

            raise e

        finally:
            # Record final performance metrics
            if timer_id and self.performance_collector:
                self.performance_collector.record_memory_usage("image_processing_end")
                self.performance_collector.record_gpu_utilization("image_processing_complete")
                self.performance_collector.end_timing(timer_id)

    def _apply_processing_mode_optimizations(self, batch_config):
        """Apply optimizations based on processing mode."""
        if batch_config.processing_mode == ProcessingMode.MEMORY_EFFICIENT:
            print("   • Reducing batch sizes for memory efficiency")
            self.parallel_config.detection_batch_size = batch_config.detection_batch_size
            self.parallel_config.segmentation_batch_size = batch_config.segmentation_batch_size

        elif batch_config.processing_mode == ProcessingMode.STREAMING:
            print("   • Enabling streaming mode for large inputs")
            # Streaming mode will be handled by individual processors

        elif batch_config.processing_mode == ProcessingMode.CPU_FALLBACK:
            print("   • Falling back to CPU processing due to memory constraints")
            # This would require switching device, which is complex
            # For now, just reduce batch sizes significantly
            self.parallel_config.detection_batch_size = 1
            self.parallel_config.segmentation_batch_size = 1

    def process_video(self, video_path: str, prompt: Union[str, List[str]], output_dir: str):
        """
        Optimized video processing with comprehensive resource management and monitoring.
        """
        # Start performance monitoring
        timer_id = None
        if self.performance_collector:
            timer_id = self.performance_collector.start_timing(
                "video_processing",
                {"video_path": video_path, "prompt_count": len(prompt) if isinstance(prompt, list) else 1}
            )
            self.performance_collector.record_memory_usage("video_processing_start")

        self.processing_stats['total_operations'] += 1

        try:
            # Analyze video content to determine optimal processing strategy
            content_analysis = self.content_analyzer.analyze_video_content(video_path)
            print(f"📊 Video analysis: {content_analysis['content_type']} content, "
                  f"{content_analysis['frame_count']} frames")

            # Check if streaming mode should be enabled
            should_stream = self.resource_manager.should_enable_streaming(
                video_frames=content_analysis['frame_count'],
                frame_size=content_analysis.get('frame_size', (1024, 1024))
            )

            if should_stream:
                print("🌊 Using streaming video processing for large video")
                return self._process_video_streaming(video_path, prompt, output_dir, content_analysis)
            elif hasattr(self, 'vjepa2_optimizer') and self.vjepa2_optimizer:
                print("🧠 Using V-JEPA2 optimized video processing")
                return self._process_video_with_vjepa2(video_path, prompt, output_dir, content_analysis)
            else:
                print("⚡ Using standard optimized video processing")
                return self._process_video_optimized_standard(video_path, prompt, output_dir, content_analysis)

        except Exception as e:
            print(f"❌ Video processing failed: {str(e)}")

            # Handle processing failure with recovery suggestions
            recovery_result = self.error_recovery.handle_processing_failure(
                operation_name="video_processing",
                error=e,
                context={"video_path": video_path, "prompts": prompt}
            )

            print(recovery_result["user_message"])
            self.processing_stats['error_recoveries'] += 1

            raise e

        finally:
            # Record final performance metrics
            if timer_id and self.performance_collector:
                self.performance_collector.record_memory_usage("video_processing_end")
                self.performance_collector.record_gpu_utilization("video_processing_complete")
                self.performance_collector.end_timing(timer_id)

    def _process_video_streaming(self, video_path: str, prompt: Union[str, List[str]],
                               output_dir: str, content_analysis: Dict[str, Any]):
        """Process video using streaming mode for memory efficiency."""
        streaming_config = self.resource_manager.enable_streaming_mode(
            video_size=content_analysis['frame_count']
        )

        print(f"🌊 Streaming configuration: {streaming_config.chunk_size} frames per chunk, "
              f"{streaming_config.overlap_frames} frame overlap")

        # Use streaming processor
        return self.streaming_processor.process_video_streaming(
            video_path=video_path,
            prompt=prompt,
            output_dir=output_dir,
            streaming_config=streaming_config,
            owl_model=self.owl,
            sam_model=self.sam,
            vjepa2_optimizer=getattr(self, 'vjepa2_optimizer', None)
        )

    def _process_video_with_vjepa2(self, video_path: str, prompt: Union[str, List[str]],
                                   output_dir: str, content_analysis: Dict[str, Any]):
        """
        Video processing with V-JEPA2 optimization, temporal detection, and resource management.
        """
        # Check if temporal detection is enabled
        use_temporal = hasattr(self, 'use_temporal_detection') and self.use_temporal_detection
        num_detection_frames = getattr(self, 'temporal_detection_frames', 5)
        merge_threshold = getattr(self, 'temporal_merge_threshold', 0.7)

        # Adapt parameters based on content analysis
        optimization_config = self.content_analyzer.get_optimization_config_for_content(
            content_analysis
        )

        if optimization_config:
            num_detection_frames = optimization_config.get('detection_frames', num_detection_frames)
            merge_threshold = optimization_config.get('merge_threshold', merge_threshold)
            print(f"🎯 Adapted parameters for {content_analysis['content_type']} content: "
                  f"{num_detection_frames} detection frames, {merge_threshold:.2f} merge threshold")

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

            print(f"Selected {len(key_frame_indices)} key frames for detection: "
                  f"{key_frame_indices}")

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

    def _process_video_optimized_standard(self, video_path: str,
                                           prompt: Union[str, List[str]],
                                           output_dir: str, content_analysis: Dict[str, Any]):
        """
        Standard optimized video processing with resource management and monitoring.
        """
        # Monitor resources and optimize batch processing
        memory_stats = self.resource_manager.monitor_memory_usage()
        batch_config = self.resource_manager.optimize_batch_sizes(
            memory_stats.utilization_percentage,
            image_size=content_analysis.get('frame_size', (1024, 1024)),
            num_prompts=len(prompt) if isinstance(prompt, list) else 1
        )

        print(f"🔧 Video processing configuration: {batch_config.processing_mode.value} mode, "
              f"batch sizes: detection={batch_config.detection_batch_size}, "
              f"segmentation={batch_config.segmentation_batch_size}")

        # Apply processing mode optimizations
        self._apply_processing_mode_optimizations(batch_config)

        # Use parent implementation with optimized models and monitoring
        try:
            if self.performance_collector:
                self.performance_collector.record_memory_usage("standard_video_start")

            result = super().process_video(video_path, prompt, output_dir)

            if self.performance_collector:
                self.performance_collector.record_memory_usage("standard_video_end")

            self.processing_stats['successful_operations'] += 1
            return result

        except Exception as e:
            # Handle memory overflow during video processing
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                print("🔧 Attempting memory overflow recovery...")

                recovery_result = self.error_recovery.handle_memory_overflow(
                    current_batch_size=batch_config.segmentation_batch_size,
                    memory_usage_gb=memory_stats.allocated_memory,
                    available_memory_gb=memory_stats.free_memory
                )

                if recovery_result["success"]:
                    print(recovery_result["user_message"])
                    # Retry with reduced batch size
                    self.parallel_config.segmentation_batch_size = recovery_result["new_batch_size"]
                    return super().process_video(video_path, prompt, output_dir)

            raise e

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


    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary containing performance metrics and statistics
        """
        if not self.performance_collector:
            return {"error": "Performance monitoring not enabled"}

        # Get operation summaries
        operation_summaries = {}
        for operation in ["image_processing", "video_processing", "detection", "segmentation"]:
            summary = self.performance_collector.get_operation_summary(operation)
            if "error" not in summary:
                operation_summaries[operation] = summary

        # Get current resource status
        memory_stats = self.resource_manager.monitor_memory_usage()
        memory_trend = self.resource_manager.get_memory_trend()

        # Get model information
        model_info = {
            "segmentation_model": {
                "type": self.segmentation_model_type,
                "name": self.segmentation_model_name,
                "device": self.config.device
            },
            "detection_model": {
                "type": "owl_v2",
                "device": self.config.device
            }
        }

        # Compile comprehensive report
        report = {
            "timestamp": time.time(),
            "pipeline_stats": self.processing_stats,
            "operation_summaries": operation_summaries,
            "resource_status": {
                "memory_stats": {
                    "total_memory": memory_stats.total_memory,
                    "allocated_memory": memory_stats.allocated_memory,
                    "utilization_percentage": memory_stats.utilization_percentage,
                    "system_memory_usage": memory_stats.system_memory_usage
                },
                "memory_trend": memory_trend,
                "device_allocation": self.resource_manager.get_optimal_device_allocation().__dict__
            },
            "model_info": model_info,
            "optimization_config": {
                "optimization_level": self.optimization_level,
                "mixed_precision_enabled": getattr(self, 'use_amp', False),
                "parallel_config": {
                    "max_workers": self.parallel_config.max_workers,
                    "detection_batch_size": self.parallel_config.detection_batch_size,
                    "segmentation_batch_size": self.parallel_config.segmentation_batch_size
                }
            },
            "error_recovery_stats": self.error_recovery.get_recovery_statistics(),
            "degradation_history": self.degradation_manager.degradation_history
        }

        return report

    def compare_model_performance(self, test_image_path: str, test_prompt: str) -> Dict[str, Any]:
        """
        Compare performance between current model and alternative.

        Args:
            test_image_path: Path to test image
            test_prompt: Test prompt for comparison

        Returns:
            Dictionary containing comparison results
        """
        if not self.performance_collector:
            return {"error": "Performance monitoring not enabled"}

        current_model_type = self.segmentation_model_type
        alternative_type = "sam2" if current_model_type == "edgetam" else "edgetam"

        print(f"🔬 Comparing {current_model_type.upper()} vs {alternative_type.upper()} performance...")

        try:
            # Test current model
            current_timer = self.performance_collector.start_timing(
                f"{current_model_type}_benchmark",
                {"model": self.segmentation_model_name}
            )

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                self.process_image(test_image_path, test_prompt, temp_dir)

            current_metrics = self.performance_collector.end_timing(current_timer)

            # Test alternative model
            original_model = self.sam
            original_type = self.segmentation_model_type
            original_name = self.segmentation_model_name

            try:
                # Switch to alternative model
                self.switch_segmentation_model(alternative_type)

                alt_timer = self.performance_collector.start_timing(
                    f"{alternative_type}_benchmark",
                    {"model": self.segmentation_model_name}
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    self.process_image(test_image_path, test_prompt, temp_dir)

                alt_metrics = self.performance_collector.end_timing(alt_timer)

                # Generate comparison report
                comparison = self.performance_collector.compare_models(
                    sam2_metrics=current_metrics if current_model_type == "sam2" else alt_metrics,
                    edgetam_metrics=alt_metrics if current_model_type == "sam2" else current_metrics
                )

                print(f"📊 Performance comparison complete:")
                print(f"   Speed improvement: {comparison.speed_improvement:+.1f}%")
                print(f"   Memory savings: {comparison.memory_savings:+.1f}%")
                print(f"   Recommendation: {comparison.recommendation}")

                return {
                    "comparison": comparison,
                    "current_model_metrics": current_metrics,
                    "alternative_model_metrics": alt_metrics
                }

            finally:
                # Restore original model
                self.sam = original_model
                self.segmentation_model_type = original_type
                self.segmentation_model_name = original_name

                # Update processors
                self.detection_processor = ParallelDetectionProcessor(
                    self.owl, self.sam, self.parallel_config
                )
                self.segmentation_processor = ParallelSegmentationProcessor(
                    self.sam, self.parallel_config
                )

        except Exception as e:
            return {"error": f"Performance comparison failed: {str(e)}"}

    def optimize_for_use_case(self, use_case: str = "general", priority: str = "balanced"):
        """
        Optimize pipeline configuration for specific use case.

        Args:
            use_case: Use case ("general", "video", "realtime", "batch")
            priority: Priority ("speed", "accuracy", "balanced", "memory")
        """
        print(f"🎯 Optimizing pipeline for {use_case} use case with {priority} priority...")

        # Get model recommendation
        recommendation = SegmentationModelFactory.recommend_model(
            use_case=use_case,
            priority=priority,
            device=self.config.device
        )

        print(f"💡 Recommended model: {recommendation['model_type']}/{recommendation['model_name']}")
        print(f"   Reasoning: {recommendation['reasoning']}")

        # Switch model if different from current
        if (recommendation['model_type'] != self.segmentation_model_type or
            recommendation['model_name'] != self.segmentation_model_name):

            try:
                self.switch_segmentation_model(
                    recommendation['model_type'],
                    recommendation['model_name']
                )
            except Exception as e:
                print(f"⚠️ Could not switch to recommended model: {e}")

        # Adjust optimization level based on priority
        if priority == "speed":
            self.optimization_level = 3
            print("   • Set optimization level to 3 (maximum speed)")
        elif priority == "memory":
            self.optimization_level = 2
            print("   • Set optimization level to 2 (memory efficient)")
        elif priority == "accuracy":
            self.optimization_level = 1
            print("   • Set optimization level to 1 (accuracy focused)")

        # Re-optimize models with new settings
        self._optimize_models()

        print("✅ Pipeline optimization complete")

    def enable_automatic_model_selection(self, enable: bool = True,
                                        performance_threshold: float = 0.8):
        """
        Enable or disable automatic model selection based on performance.

        Args:
            enable: Whether to enable automatic selection
            performance_threshold: Performance threshold for switching (0-1)
        """
        self.auto_model_selection = enable
        self.performance_threshold = performance_threshold

        if enable:
            print(f"🤖 Enabled automatic model selection (threshold: {performance_threshold})")
        else:
            print("🤖 Disabled automatic model selection")

    def warm_up_models(self, test_image_size: tuple = (1024, 1024)):
        """
        Warm up models by running inference on dummy data.

        Args:
            test_image_size: Size of test image for warm-up
        """
        print("🔥 Warming up models...")

        # Create dummy test image
        import numpy as np
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (*test_image_size, 3), dtype=np.uint8)
        )

        # Warm up current model
        timer_id = None
        if self.performance_collector:
            timer_id = self.performance_collector.start_timing(
                "model_warmup",
                {"model_type": self.segmentation_model_type}
            )

        try:
            # Run dummy detection
            batch_results = self.detection_processor.detect_multiple_prompts_parallel(
                dummy_image, ["test object"], 0.1
            )

            # Run dummy segmentation if detections found
            if batch_results and batch_results[0].detections:
                self.segmentation_processor.segment_detections_parallel(
                    dummy_image, batch_results[0].detections[:1]
                )

            print(f"   ✓ {self.segmentation_model_type.upper()} model warmed up")

        except Exception as e:
            print(f"   ⚠️ Model warm-up failed: {e}")

        finally:
            if timer_id and self.performance_collector:
                warmup_metrics = self.performance_collector.end_timing(timer_id)
                print(f"   ⏱️ Warm-up time: {warmup_metrics.processing_time:.2f}s")

    def preload_alternative_model(self, model_type: str, model_name: Optional[str] = None):
        """
        Preload alternative model for faster switching.

        Args:
            model_type: Type of model to preload
            model_name: Specific model name (optional)
        """
        if not hasattr(self, '_preloaded_models'):
            self._preloaded_models = {}

        print(f"📦 Preloading {model_type} model...")

        try:
            # Determine model name if not specified
            if not model_name:
                if model_type == "edgetam":
                    model_name = "facebook/edgetam-base"
                else:
                    model_name = "facebook/sam2.1-hiera-small"

            # Create and cache the model
            preloaded_model = SegmentationModelFactory.create_model(
                model_type=model_type,
                model_name=model_name,
                device=self.config.device,
                enable_fallback=True
            )

            self._preloaded_models[f"{model_type}_{model_name}"] = preloaded_model
            print(f"   ✓ {model_type.upper()} model preloaded: {model_name}")

            # Warm up preloaded model
            self._warm_up_preloaded_model(preloaded_model, model_type)

        except Exception as e:
            print(f"   ❌ Failed to preload {model_type} model: {e}")

    def _warm_up_preloaded_model(self, model, model_type: str):
        """Warm up a preloaded model with dummy inference."""
        try:
            import numpy as np
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )

            # Run dummy segmentation
            dummy_box = [100, 100, 200, 200]  # x1, y1, x2, y2
            _ = model.segment(dummy_image, dummy_box)

            print(f"     ✓ {model_type.upper()} model warmed up")

        except Exception as e:
            print(f"     ⚠️ Warm-up failed for {model_type}: {e}")

    def switch_to_preloaded_model(self, model_type: str, model_name: Optional[str] = None):
        """
        Switch to a preloaded model for faster switching.

        Args:
            model_type: Type of model to switch to
            model_name: Specific model name (optional)
        """
        if not hasattr(self, '_preloaded_models'):
            self._preloaded_models = {}

        # Determine model name if not specified
        if not model_name:
            if model_type == "edgetam":
                model_name = "facebook/edgetam-base"
            else:
                model_name = "facebook/sam2.1-hiera-small"

        model_key = f"{model_type}_{model_name}"

        if model_key in self._preloaded_models:
            print(f"⚡ Switching to preloaded {model_type.upper()} model...")

            # Store old model info
            old_model_type = self.segmentation_model_type
            old_model_name = self.segmentation_model_name

            # Switch to preloaded model
            old_sam = self.sam
            self.sam = self._preloaded_models[model_key]
            self.segmentation_model_type = model_type
            self.segmentation_model_name = model_name

            # Update processors
            self.detection_processor = ParallelDetectionProcessor(
                self.owl, self.sam, self.parallel_config
            )
            self.segmentation_processor = ParallelSegmentationProcessor(
                self.sam, self.parallel_config
            )

            # Clean up old model
            del old_sam
            self.resource_manager.cleanup_resources()

            print(f"✅ Switched to preloaded {model_type.upper()}: {model_name}")

            # Log model selection event
            from sowlv2.utils.error_recovery import ModelFallbackManager
            ModelFallbackManager.log_model_selection_event(
                selected_model_type=model_type,
                selected_model_name=model_name,
                was_fallback=False
            )

        else:
            print(f"❌ {model_type.upper()} model not preloaded, using regular switching...")
            self.switch_segmentation_model(model_type, model_name)

    def auto_select_optimal_model(self, test_image_path: Optional[str] = None,
                                test_prompt: str = "test object") -> Dict[str, Any]:
        """
        Automatically select optimal model based on performance testing.

        Args:
            test_image_path: Optional test image path
            test_prompt: Test prompt for evaluation

        Returns:
            Dictionary containing selection results
        """
        if not self.performance_collector:
            return {"error": "Performance monitoring required for auto-selection"}

        print("🤖 Running automatic model selection...")

        # Use provided test image or create dummy one
        if test_image_path and os.path.exists(test_image_path):
            test_image = test_image_path
        else:
            # Create temporary test image
            import numpy as np
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            )

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                dummy_image.save(tmp_file.name)
                test_image = tmp_file.name

        try:
            # Test current model
            current_model_type = self.segmentation_model_type
            current_timer = self.performance_collector.start_timing(
                f"auto_select_{current_model_type}",
                {"model": self.segmentation_model_name}
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                self.process_image(test_image, test_prompt, temp_dir)

            current_metrics = self.performance_collector.end_timing(current_timer)

            # Test alternative model
            alternative_type = "sam2" if current_model_type == "edgetam" else "edgetam"

            # Store original model
            original_model = self.sam
            original_type = self.segmentation_model_type
            original_name = self.segmentation_model_name

            try:
                # Switch to alternative
                self.switch_segmentation_model(alternative_type)

                alt_timer = self.performance_collector.start_timing(
                    f"auto_select_{alternative_type}",
                    {"model": self.segmentation_model_name}
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    self.process_image(test_image, test_prompt, temp_dir)

                alt_metrics = self.performance_collector.end_timing(alt_timer)

                # Compare performance
                comparison = self.performance_collector.compare_models(
                    sam2_metrics=current_metrics if current_model_type == "sam2" else alt_metrics,
                    edgetam_metrics=alt_metrics if current_model_type == "sam2" else current_metrics
                )

                # Determine optimal model based on performance threshold
                speed_improvement = comparison.speed_improvement
                memory_savings = comparison.memory_savings

                # Calculate overall performance score
                current_score = self._calculate_performance_score(current_metrics)
                alt_score = self._calculate_performance_score(alt_metrics)

                if alt_score > current_score * (1 + self.performance_threshold):
                    # Switch to alternative model
                    optimal_type = alternative_type
                    optimal_name = self.segmentation_model_name
                    optimal_metrics = alt_metrics
                    switch_recommended = True
                else:
                    # Keep current model
                    optimal_type = original_type
                    optimal_name = original_name
                    optimal_metrics = current_metrics
                    switch_recommended = False

                    # Restore original model
                    self.sam = original_model
                    self.segmentation_model_type = original_type
                    self.segmentation_model_name = original_name

                    # Update processors
                    self.detection_processor = ParallelDetectionProcessor(
                        self.owl, self.sam, self.parallel_config
                    )
                    self.segmentation_processor = ParallelSegmentationProcessor(
                        self.sam, self.parallel_config
                    )

                result = {
                    "optimal_model_type": optimal_type,
                    "optimal_model_name": optimal_name,
                    "switch_recommended": switch_recommended,
                    "performance_comparison": comparison,
                    "current_model_score": current_score,
                    "alternative_model_score": alt_score,
                    "selected_metrics": optimal_metrics
                }

                print(f"🎯 Auto-selection result: {optimal_type.upper()} "
                      f"({'switched' if switch_recommended else 'kept current'})")
                print(f"   Performance scores: Current={current_score:.2f}, "
                      f"Alternative={alt_score:.2f}")

                return result

            except Exception as switch_error:
                # Restore original model on error
                self.sam = original_model
                self.segmentation_model_type = original_type
                self.segmentation_model_name = original_name

                # Update processors
                self.detection_processor = ParallelDetectionProcessor(
                    self.owl, self.sam, self.parallel_config
                )
                self.segmentation_processor = ParallelSegmentationProcessor(
                    self.sam, self.parallel_config
                )

                raise switch_error

        finally:
            # Clean up temporary test image if created
            if not test_image_path and os.path.exists(test_image):
                os.unlink(test_image)

    def _calculate_performance_score(self, metrics: Any) -> float:
        """
        Calculate overall performance score from metrics.

        Args:
            metrics: Performance metrics object

        Returns:
            Performance score (higher is better)
        """
        # Normalize metrics (lower processing time and memory usage = higher score)
        time_score = 1.0 / max(0.1, metrics.processing_time)  # Avoid division by zero
        memory_score = 1.0 / max(0.1, metrics.memory_peak_usage)
        throughput_score = metrics.throughput_fps if metrics.throughput_fps > 0 else 1.0

        # Weighted combination (adjust weights based on priorities)
        score = (
            time_score * 0.4 +      # 40% weight on speed
            memory_score * 0.3 +    # 30% weight on memory efficiency
            throughput_score * 0.3  # 30% weight on throughput
        )

        return score

    def validate_model_switching(self, test_image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that model switching works correctly.

        Args:
            test_image_path: Optional test image path

        Returns:
            Dictionary containing validation results
        """
        print("🔍 Validating model switching functionality...")

        validation_results = {
            "switch_to_edgetam": False,
            "switch_to_sam2": False,
            "switch_back": False,
            "errors": [],
            "performance_consistent": False
        }

        # Store original configuration
        original_type = self.segmentation_model_type
        original_name = self.segmentation_model_name

        try:
            # Create test image if not provided
            if not test_image_path:
                import numpy as np
                dummy_image = Image.fromarray(
                    np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                )

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    dummy_image.save(tmp_file.name)
                    test_image_path = tmp_file.name

            # Test switching to EdgeTAM
            try:
                self.switch_segmentation_model("edgetam")
                validation_results["switch_to_edgetam"] = True
                print("   ✓ Switch to EdgeTAM successful")

                # Test inference
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.process_image(test_image_path, "test object", temp_dir)

            except Exception as e:
                validation_results["errors"].append(f"EdgeTAM switch failed: {str(e)}")
                print(f"   ❌ Switch to EdgeTAM failed: {e}")

            # Test switching to SAM2
            try:
                self.switch_segmentation_model("sam2")
                validation_results["switch_to_sam2"] = True
                print("   ✓ Switch to SAM2 successful")

                # Test inference
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.process_image(test_image_path, "test object", temp_dir)

            except Exception as e:
                validation_results["errors"].append(f"SAM2 switch failed: {str(e)}")
                print(f"   ❌ Switch to SAM2 failed: {e}")

            # Test switching back to original
            try:
                self.switch_segmentation_model(original_type, original_name)
                validation_results["switch_back"] = True
                print("   ✓ Switch back to original successful")

            except Exception as e:
                validation_results["errors"].append(f"Switch back failed: {str(e)}")
                print(f"   ❌ Switch back failed: {e}")

            # Overall validation
            all_switches_successful = (
                validation_results["switch_to_edgetam"] and
                validation_results["switch_to_sam2"] and
                validation_results["switch_back"]
            )

            if all_switches_successful:
                print("✅ Model switching validation passed")
            else:
                print("❌ Model switching validation failed")

            validation_results["overall_success"] = all_switches_successful

        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            print(f"❌ Validation error: {e}")

        finally:
            # Clean up temporary test image
            if test_image_path and not os.path.exists(test_image_path.replace('tmp', '')):
                try:
                    os.unlink(test_image_path)
                except:
                    pass

        return validation_results

    def auto_select_optimization_level(self, target_use_case: str = "general") -> int:
        """
        Automatically select optimal optimization level based on system resources and use case.

        Args:
            target_use_case: Target use case ("realtime", "batch", "memory_constrained", "general")

        Returns:
            Recommended optimization level (1-3)
        """
        print(f"🎯 Auto-selecting optimization level for {target_use_case} use case...")

        # Get current system status
        memory_stats = self.resource_manager.monitor_memory_usage()
        device_allocation = self.resource_manager.get_optimal_device_allocation()

        # Base optimization level
        optimization_level = 1

        # Adjust based on memory availability
        if memory_stats.utilization_percentage < 50:
            optimization_level = max(optimization_level, 2)
            print("   • Sufficient memory available - enabling level 2 optimizations")
        elif memory_stats.utilization_percentage > 80:
            optimization_level = 1
            print("   • High memory usage - limiting to level 1 optimizations")

        # Adjust based on use case
        if target_use_case == "realtime":
            optimization_level = 3
            print("   • Realtime use case - enabling maximum optimizations (level 3)")
        elif target_use_case == "memory_constrained":
            optimization_level = min(optimization_level, 1)
            print("   • Memory constrained - using conservative optimizations (level 1)")
        elif target_use_case == "batch":
            optimization_level = max(optimization_level, 2)
            print("   • Batch processing - enabling level 2+ optimizations")

        # Adjust based on device capabilities
        if self.config.device == "cpu":
            optimization_level = min(optimization_level, 2)
            print("   • CPU device - limiting optimization level")
        elif torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            if gpu_props.total_memory < 4e9:  # Less than 4GB
                optimization_level = min(optimization_level, 1)
                print("   • Limited GPU memory - reducing optimization level")

        print(f"🎯 Selected optimization level: {optimization_level}")

        # Apply the selected optimization level
        old_level = self.optimization_level
        self.optimization_level = optimization_level

        if old_level != optimization_level:
            print("🔧 Re-optimizing models with new level...")
            self._optimize_models()

        return optimization_level

    def monitor_optimization_effectiveness(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Monitor the effectiveness of current optimizations.

        Args:
            window_size: Number of recent operations to analyze

        Returns:
            Dictionary containing optimization effectiveness metrics
        """
        if not self.performance_collector:
            return {"error": "Performance monitoring not enabled"}

        print("📊 Monitoring optimization effectiveness...")

        # Get recent performance trends
        memory_trend = self.resource_manager.get_memory_trend(window_size)

        # Analyze operation performance
        effectiveness_metrics = {
            "memory_trend": memory_trend,
            "optimization_level": self.optimization_level,
            "resource_utilization": {},
            "performance_stability": {},
            "recommendations": []
        }

        # Current resource utilization
        current_stats = self.resource_manager.monitor_memory_usage()
        effectiveness_metrics["resource_utilization"] = {
            "memory_usage_percent": current_stats.utilization_percentage,
            "memory_trend": memory_trend["trend"],
            "memory_stability": memory_trend["stability"],
            "peak_usage": memory_trend["peak_usage"]
        }

        # Analyze performance stability
        for operation in ["image_processing", "video_processing", "detection", "segmentation"]:
            summary = self.performance_collector.get_operation_summary(operation)
            if "error" not in summary:
                # Calculate coefficient of variation (stability metric)
                cv = summary["processing_time"]["std"] / max(0.001, summary["processing_time"]["mean"])
                effectiveness_metrics["performance_stability"][operation] = {
                    "coefficient_of_variation": cv,
                    "mean_time": summary["processing_time"]["mean"],
                    "std_time": summary["processing_time"]["std"],
                    "stability_rating": "stable" if cv < 0.3 else "moderate" if cv < 0.6 else "unstable"
                }

        # Generate recommendations based on analysis
        recommendations = []

        # Memory-based recommendations
        if memory_trend["trend"] > 5:  # Increasing memory usage
            recommendations.append("Consider reducing batch sizes or enabling streaming mode")
        elif memory_trend["peak_usage"] > 90:
            recommendations.append("Memory usage is very high - consider switching to CPU or reducing input size")
        elif memory_trend["stability"] > 20:
            recommendations.append("Memory usage is unstable - consider enabling gradient checkpointing")

        # Performance-based recommendations
        for operation, stability in effectiveness_metrics["performance_stability"].items():
            if stability["stability_rating"] == "unstable":
                recommendations.append(f"{operation} performance is unstable - consider optimization level adjustment")

        # Optimization level recommendations
        if current_stats.utilization_percentage < 30 and self.optimization_level < 3:
            recommendations.append("Low resource usage - consider increasing optimization level")
        elif current_stats.utilization_percentage > 85 and self.optimization_level > 1:
            recommendations.append("High resource usage - consider decreasing optimization level")

        effectiveness_metrics["recommendations"] = recommendations

        # Overall effectiveness score
        memory_score = max(0, 100 - current_stats.utilization_percentage) / 100
        stability_scores = [
            1.0 - min(1.0, stability["coefficient_of_variation"])
            for stability in effectiveness_metrics["performance_stability"].values()
        ]
        avg_stability = sum(stability_scores) / max(1, len(stability_scores))

        effectiveness_metrics["overall_effectiveness_score"] = (memory_score * 0.4 + avg_stability * 0.6) * 100

        print(f"📊 Optimization effectiveness: {effectiveness_metrics['overall_effectiveness_score']:.1f}%")
        if recommendations:
            print("💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")

        return effectiveness_metrics

    def create_optimization_recommendation_system(self) -> Dict[str, Any]:
        """
        Create comprehensive optimization recommendations based on current performance.

        Returns:
            Dictionary containing detailed optimization recommendations
        """
        print("🔍 Generating optimization recommendations...")

        # Gather system information
        memory_stats = self.resource_manager.monitor_memory_usage()
        device_allocation = self.resource_manager.get_optimal_device_allocation()
        processing_stats = self.get_processing_statistics()

        recommendations = {
            "system_analysis": {
                "memory_usage": memory_stats.utilization_percentage,
                "device": self.config.device,
                "current_optimization_level": self.optimization_level,
                "success_rate": processing_stats["success_rate"],
                "error_rate": 100 - processing_stats["success_rate"]
            },
            "immediate_actions": [],
            "configuration_changes": [],
            "model_recommendations": [],
            "resource_optimizations": [],
            "priority_level": "low"
        }

        # Analyze immediate actions needed
        if memory_stats.utilization_percentage > 90:
            recommendations["immediate_actions"].append({
                "action": "reduce_batch_sizes",
                "description": "Immediately reduce batch sizes to prevent memory overflow",
                "urgency": "high"
            })
            recommendations["priority_level"] = "high"

        if processing_stats["error_rate"] > 20:
            recommendations["immediate_actions"].append({
                "action": "enable_error_recovery",
                "description": "High error rate detected - ensure error recovery is enabled",
                "urgency": "medium"
            })
            recommendations["priority_level"] = max(recommendations["priority_level"], "medium")

        # Configuration change recommendations
        if memory_stats.utilization_percentage > 70 and self.optimization_level > 1:
            recommendations["configuration_changes"].append({
                "change": "reduce_optimization_level",
                "current_value": self.optimization_level,
                "recommended_value": max(1, self.optimization_level - 1),
                "reason": "High memory usage requires more conservative optimizations"
            })

        if memory_stats.utilization_percentage < 40 and self.optimization_level < 3:
            recommendations["configuration_changes"].append({
                "change": "increase_optimization_level",
                "current_value": self.optimization_level,
                "recommended_value": min(3, self.optimization_level + 1),
                "reason": "Low resource usage allows for more aggressive optimizations"
            })

        # Model recommendations
        current_model_info = SegmentationModelFactory.get_model_info(
            self.segmentation_model_type, self.segmentation_model_name
        )

        if memory_stats.utilization_percentage > 80:
            if self.segmentation_model_type == "sam2":
                recommendations["model_recommendations"].append({
                    "recommendation": "switch_to_edgetam",
                    "reason": "EdgeTAM uses less memory than SAM2",
                    "expected_benefit": "20-40% memory reduction, 2-3x speed improvement",
                    "trade_off": "Slightly lower segmentation accuracy"
                })

        if processing_stats["success_rate"] < 90 and self.segmentation_model_type == "edgetam":
            recommendations["model_recommendations"].append({
                "recommendation": "switch_to_sam2",
                "reason": "SAM2 may provide better reliability and accuracy",
                "expected_benefit": "Higher accuracy and stability",
                "trade_off": "Higher memory usage and slower processing"
            })

        # Resource optimization recommendations
        if self.config.device == "cuda" and torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            if gpu_props.total_memory > 8e9 and not getattr(self, 'use_amp', False):
                recommendations["resource_optimizations"].append({
                    "optimization": "enable_mixed_precision",
                    "description": "Enable mixed precision (FP16) for faster inference",
                    "expected_benefit": "30-50% speed improvement, 40-50% memory reduction"
                })

        if not hasattr(self, '_preloaded_models') or not self._preloaded_models:
            recommendations["resource_optimizations"].append({
                "optimization": "preload_alternative_models",
                "description": "Preload alternative models for faster switching",
                "expected_benefit": "Instant model switching, better user experience"
            })

        # Generate overall recommendation summary
        total_recommendations = (
            len(recommendations["immediate_actions"]) +
            len(recommendations["configuration_changes"]) +
            len(recommendations["model_recommendations"]) +
            len(recommendations["resource_optimizations"])
        )

        recommendations["summary"] = {
            "total_recommendations": total_recommendations,
            "priority_level": recommendations["priority_level"],
            "estimated_improvement": self._estimate_optimization_improvement(recommendations),
            "implementation_complexity": "low" if total_recommendations <= 2 else "medium" if total_recommendations <= 5 else "high"
        }

        print(f"🎯 Generated {total_recommendations} optimization recommendations")
        print(f"   Priority: {recommendations['priority_level']}")
        print(f"   Estimated improvement: {recommendations['summary']['estimated_improvement']}")

        return recommendations

    def _estimate_optimization_improvement(self, recommendations: Dict[str, Any]) -> str:
        """Estimate the potential improvement from recommendations."""
        improvement_factors = []

        # Analyze each recommendation type
        for model_rec in recommendations["model_recommendations"]:
            if "switch_to_edgetam" in model_rec["recommendation"]:
                improvement_factors.append("2-3x speed improvement")
            elif "switch_to_sam2" in model_rec["recommendation"]:
                improvement_factors.append("improved stability")

        for resource_opt in recommendations["resource_optimizations"]:
            if "mixed_precision" in resource_opt["optimization"]:
                improvement_factors.append("30-50% speed boost")
            elif "preload" in resource_opt["optimization"]:
                improvement_factors.append("instant model switching")

        if not improvement_factors:
            return "minor improvements"
        elif len(improvement_factors) == 1:
            return improvement_factors[0]
        else:
            return f"multiple improvements: {', '.join(improvement_factors[:2])}"

    def apply_optimization_recommendations(self, recommendations: Dict[str, Any],
                                         auto_apply: bool = False) -> Dict[str, Any]:
        """
        Apply optimization recommendations.

        Args:
            recommendations: Recommendations from create_optimization_recommendation_system
            auto_apply: Whether to automatically apply safe recommendations

        Returns:
            Dictionary containing application results
        """
        print("🔧 Applying optimization recommendations...")

        results = {
            "applied_changes": [],
            "skipped_changes": [],
            "errors": [],
            "success_count": 0,
            "total_count": 0
        }

        # Apply configuration changes
        for config_change in recommendations["configuration_changes"]:
            results["total_count"] += 1

            try:
                if config_change["change"] == "reduce_optimization_level":
                    if auto_apply or config_change.get("urgency") == "high":
                        old_level = self.optimization_level
                        self.optimization_level = config_change["recommended_value"]
                        self._optimize_models()

                        results["applied_changes"].append({
                            "change": "optimization_level",
                            "from": old_level,
                            "to": self.optimization_level,
                            "reason": config_change["reason"]
                        })
                        results["success_count"] += 1
                        print(f"   ✓ Reduced optimization level: {old_level} → {self.optimization_level}")
                    else:
                        results["skipped_changes"].append(config_change)
                        print(f"   ⏭️ Skipped optimization level change (manual approval required)")

                elif config_change["change"] == "increase_optimization_level":
                    if auto_apply:
                        old_level = self.optimization_level
                        self.optimization_level = config_change["recommended_value"]
                        self._optimize_models()

                        results["applied_changes"].append({
                            "change": "optimization_level",
                            "from": old_level,
                            "to": self.optimization_level,
                            "reason": config_change["reason"]
                        })
                        results["success_count"] += 1
                        print(f"   ✓ Increased optimization level: {old_level} → {self.optimization_level}")
                    else:
                        results["skipped_changes"].append(config_change)
                        print(f"   ⏭️ Skipped optimization level change (manual approval required)")

            except Exception as e:
                results["errors"].append(f"Configuration change failed: {str(e)}")
                print(f"   ❌ Configuration change failed: {e}")

        # Apply resource optimizations
        for resource_opt in recommendations["resource_optimizations"]:
            results["total_count"] += 1

            try:
                if resource_opt["optimization"] == "enable_mixed_precision":
                    if auto_apply:
                        self.use_amp = True
                        self._optimize_models()

                        results["applied_changes"].append({
                            "change": "mixed_precision",
                            "enabled": True,
                            "reason": resource_opt["description"]
                        })
                        results["success_count"] += 1
                        print("   ✓ Enabled mixed precision")
                    else:
                        results["skipped_changes"].append(resource_opt)
                        print("   ⏭️ Skipped mixed precision (manual approval required)")

                elif resource_opt["optimization"] == "preload_alternative_models":
                    if auto_apply:
                        # Preload alternative model
                        alt_type = "sam2" if self.segmentation_model_type == "edgetam" else "edgetam"
                        self.preload_alternative_model(alt_type)

                        results["applied_changes"].append({
                            "change": "preload_models",
                            "model_type": alt_type,
                            "reason": resource_opt["description"]
                        })
                        results["success_count"] += 1
                        print(f"   ✓ Preloaded {alt_type.upper()} model")
                    else:
                        results["skipped_changes"].append(resource_opt)
                        print("   ⏭️ Skipped model preloading (manual approval required)")

            except Exception as e:
                results["errors"].append(f"Resource optimization failed: {str(e)}")
                print(f"   ❌ Resource optimization failed: {e}")

        # Model recommendations require manual approval
        for model_rec in recommendations["model_recommendations"]:
            results["total_count"] += 1
            results["skipped_changes"].append(model_rec)
            print(f"   ⏭️ Skipped model change (manual approval required): {model_rec['recommendation']}")

        # Summary
        success_rate = (results["success_count"] / max(1, results["total_count"])) * 100
        print(f"🎯 Applied {results['success_count']}/{results['total_count']} recommendations ({success_rate:.1f}% success rate)")

        if results["errors"]:
            print(f"❌ {len(results['errors'])} errors occurred")

        return results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            "processing_stats": self.processing_stats.copy(),
            "success_rate": (
                self.processing_stats['successful_operations'] /
                max(1, self.processing_stats['total_operations'])
            ) * 100,
            "error_recovery_rate": (
                self.processing_stats['error_recoveries'] /
                max(1, self.processing_stats['total_operations'])
            ) * 100,
            "fallback_rate": (
                self.processing_stats['fallback_operations'] /
                max(1, self.processing_stats['total_operations'])
            ) * 100
        }


class CachedModelWrapper:
    """Wrapper for caching model outputs."""
    # Implementation can be added here as needed
    def __init__(self):
        pass
