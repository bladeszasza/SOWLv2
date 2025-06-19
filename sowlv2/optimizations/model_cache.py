"""
Intelligent model caching and memory management for SOWLv2 pipeline.
"""
import gc
from typing import Dict, Any

import torch


class IntelligentModelCache:
    """Manages model loading and memory for optimal performance."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.loaded_models: Dict[str, Any] = {}
        self.model_usage_count: Dict[str, int] = {}
        self.memory_threshold = 0.8  # 80% GPU memory threshold

    def load_model_lazy(self, model_name: str, loader_func, *args, **kwargs):
        """Load model only when needed, with memory management."""
        if model_name in self.loaded_models:
            self.model_usage_count[model_name] += 1
            return self.loaded_models[model_name]

        # Check memory before loading
        if self.device == "cuda" and torch.cuda.is_available():
            self._check_and_free_memory()

        # Load model
        model = loader_func(*args, **kwargs)
        self.loaded_models[model_name] = model
        self.model_usage_count[model_name] = 1

        return model

    def _check_and_free_memory(self):
        """Free memory if usage is too high."""
        if not torch.cuda.is_available():
            return

        memory_used = (torch.cuda.memory_allocated() /
                      torch.cuda.get_device_properties(0).total_memory)

        if memory_used > self.memory_threshold:
            # Free least used models
            sorted_models = sorted(
                self.model_usage_count.items(),
                key=lambda x: x[1]
            )

            for model_name, _ in sorted_models[:1]:  # Free one model at a time
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]
                    del self.model_usage_count[model_name]
                    gc.collect()
                    torch.cuda.empty_cache()
                    break

    def optimize_for_video_batch(self, num_frames: int, models_needed: list):
        """Pre-allocate memory and optimize for batch processing."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return

        # Estimate memory needed
        estimated_memory_per_frame = 0.1  # GB, adjust based on your models
        total_memory_needed = num_frames * estimated_memory_per_frame

        # Free memory if needed
        available_memory = (torch.cuda.get_device_properties(0).total_memory -
                          torch.cuda.memory_allocated()) / 1e9  # GB

        if total_memory_needed > available_memory * 0.8:
            # Free all non-essential models
            essential_models = set(models_needed)
            models_to_free = [m for m in self.loaded_models if m not in essential_models]

            for model_name in models_to_free:
                del self.loaded_models[model_name]
                if model_name in self.model_usage_count:
                    del self.model_usage_count[model_name]

            gc.collect()
            torch.cuda.empty_cache() 