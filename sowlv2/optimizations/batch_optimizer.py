"""
Intelligent batch processing for optimal GPU utilization.
"""
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import torch


@dataclass
class BatchConfig:
    """Dynamic batch configuration based on available resources."""
    detection_batch_size: int
    segmentation_batch_size: int
    frame_batch_size: int
    use_mixed_precision: bool


class IntelligentBatchOptimizer:
    """Dynamically optimizes batch sizes based on GPU memory and model characteristics."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.profiling_results: Dict[str, float] = {}

    def profile_and_optimize(self,
                           test_image_size: Tuple[int, int],
                           num_prompts: int) -> BatchConfig:
        """Profile models and determine optimal batch sizes."""
        if self.device == "cpu":
            return BatchConfig(
                detection_batch_size=1,
                segmentation_batch_size=1,
                frame_batch_size=1,
                use_mixed_precision=False
            )

        # Get GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        available_memory = (total_memory -
                          torch.cuda.memory_allocated() / 1e9)

        # Estimate memory requirements
        pixels_per_image = test_image_size[0] * test_image_size[1]
        base_memory_per_image = pixels_per_image * 4 * 3 / 1e9  # RGB float32

        # Detection: OWLv2 typically needs ~2GB for base model + image memory
        detection_memory_per_batch = 2.0 + base_memory_per_image * num_prompts
        detection_batch_size = max(1, int(available_memory * 0.3 /
                                         detection_memory_per_batch))

        # Segmentation: SAM2 needs ~4GB for base model + more for processing
        segmentation_memory_per_image = 4.0 + base_memory_per_image * 2
        segmentation_batch_size = max(1, int(available_memory * 0.4 /
                                            segmentation_memory_per_image))

        # Frame processing: Consider V-JEPA2 if enabled
        frame_memory_per_batch = base_memory_per_image * 16  # V-JEPA2 processes clips
        frame_batch_size = max(1, int(available_memory * 0.3 / frame_memory_per_batch))

        # Use mixed precision if GPU supports it
        use_mixed_precision = torch.cuda.get_device_capability()[0] >= 7

        return BatchConfig(
            detection_batch_size=min(detection_batch_size, 8),  # Cap at 8
            segmentation_batch_size=min(segmentation_batch_size, 4),  # Cap at 4
            frame_batch_size=min(frame_batch_size, 16),  # Cap at 16
            use_mixed_precision=use_mixed_precision
        )

    def adaptive_batch_processing(self,
                                items: List[Any],
                                process_func,
                                initial_batch_size: int,
                                *args, **kwargs) -> List[Any]:
        """Process items with adaptive batch size based on memory pressure."""
        results = []
        current_batch_size = initial_batch_size
        i = 0

        while i < len(items):
            batch_end = min(i + current_batch_size, len(items))
            batch = items[i:batch_end]

            try:
                # Try processing batch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                batch_results = process_func(batch, *args, **kwargs)
                results.extend(batch_results)

                # Increase batch size if successful and memory allows
                if torch.cuda.is_available():
                    memory_used = (torch.cuda.memory_allocated() /
                                 torch.cuda.get_device_properties(0).total_memory)
                    if memory_used < 0.7:  # Less than 70% memory used
                        current_batch_size = min(current_batch_size + 1,
                                               initial_batch_size * 2)

                i = batch_end

            except torch.cuda.OutOfMemoryError:
                # Reduce batch size and retry
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(f"Reducing batch size to {current_batch_size} due to memory pressure")

                if current_batch_size == 1 and len(batch) == 1:
                    # Single item still fails, skip it
                    print(f"Skipping item {i} due to memory constraints")
                    i += 1

        return results 