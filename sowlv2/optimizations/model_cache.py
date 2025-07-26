"""
Intelligent model caching and memory management for SOWLv2 pipeline.
Enhanced with LRU eviction, priority loading, and comprehensive statistics.
"""
import gc
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum

import torch


class ModelPriority(Enum):
    """Model loading priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_models: int
    loaded_models: int
    cache_hits: int
    cache_misses: int
    evictions: int
    total_memory_used: float  # GB
    hit_rate: float
    average_load_time: float


@dataclass
class ModelInfo:
    """Information about a cached model."""
    model: Any
    load_time: float
    last_accessed: float
    access_count: int
    priority: ModelPriority
    memory_usage: float  # GB
    loader_func: Callable
    loader_args: tuple
    loader_kwargs: dict


class IntelligentModelCache:
    """Enhanced model cache with LRU eviction and priority-based loading."""

    def __init__(self, device: str = "cuda", max_models: int = 5, memory_limit: Optional[float] = None):
        self.device = device
        self.max_models = max_models
        self.memory_limit = memory_limit  # GB
        self.memory_threshold = 0.8  # 80% memory threshold for eviction
        
        # Enhanced cache storage with LRU ordering
        self.loaded_models: OrderedDict[str, ModelInfo] = OrderedDict()
        
        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.load_times: List[float] = []
        
        # Priority queues for preloading
        self.preload_queue: Dict[ModelPriority, List[str]] = {
            priority: [] for priority in ModelPriority
        }

    def load_model_lazy(self, model_name: str, loader_func, *args, **kwargs):
        """Load model only when needed, with memory management."""
        return self.load_model_with_priority(model_name, ModelPriority.NORMAL, loader_func, *args, **kwargs)
    
    def load_model_with_priority(self, model_name: str, priority: ModelPriority, 
                               loader_func: Callable, *args, **kwargs) -> Any:
        """
        Load model with specified priority, implementing LRU eviction.
        
        Args:
            model_name: Unique identifier for the model
            priority: Loading priority level
            loader_func: Function to load the model
            *args, **kwargs: Arguments for loader function
            
        Returns:
            Loaded model instance
        """
        current_time = time.time()
        
        # Check if model is already loaded
        if model_name in self.loaded_models:
            model_info = self.loaded_models[model_name]
            model_info.last_accessed = current_time
            model_info.access_count += 1
            model_info.priority = max(model_info.priority, priority)  # Upgrade priority if higher
            
            # Move to end (most recently used)
            self.loaded_models.move_to_end(model_name)
            self.cache_hits += 1
            
            return model_info.model
        
        # Cache miss - need to load model
        self.cache_misses += 1
        
        # Check memory and evict if necessary
        self._ensure_memory_available(priority)
        
        # Load the model
        start_time = time.time()
        try:
            model = loader_func(*args, **kwargs)
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            
            # Estimate model memory usage
            memory_usage = self._estimate_model_memory(model)
            
            # Create model info
            model_info = ModelInfo(
                model=model,
                load_time=load_time,
                last_accessed=current_time,
                access_count=1,
                priority=priority,
                memory_usage=memory_usage,
                loader_func=loader_func,
                loader_args=args,
                loader_kwargs=kwargs
            )
            
            # Add to cache
            self.loaded_models[model_name] = model_info
            
            # Enforce cache size limits
            self._enforce_cache_limits()
            
            return model
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            raise

    def implement_lru_eviction(self, memory_threshold: float = None) -> int:
        """
        Implement LRU eviction policy to free memory.
        
        Args:
            memory_threshold: Memory threshold to trigger eviction (0-1)
            
        Returns:
            Number of models evicted
        """
        if memory_threshold is None:
            memory_threshold = self.memory_threshold
            
        evicted_count = 0
        
        if not torch.cuda.is_available() and self.device == "cuda":
            return evicted_count
            
        # Check current memory usage
        if self.device == "cuda":
            current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        else:
            # For CPU, use estimated memory from model sizes
            current_memory = sum(info.memory_usage for info in self.loaded_models.values())
            if self.memory_limit:
                current_memory = current_memory / self.memory_limit
            else:
                current_memory = 0  # Can't determine without limit
        
        # Evict models if memory usage is too high
        while (current_memory > memory_threshold and 
               len(self.loaded_models) > 0):
            
            # Find least recently used model with lowest priority
            lru_model = None
            lru_key = None
            
            # Iterate from least recently used (beginning of OrderedDict)
            for model_name, model_info in self.loaded_models.items():
                if lru_model is None or model_info.priority.value <= lru_model.priority.value:
                    # Don't evict CRITICAL priority models unless absolutely necessary
                    if model_info.priority != ModelPriority.CRITICAL or len(self.loaded_models) > self.max_models:
                        lru_model = model_info
                        lru_key = model_name
                        break
            
            if lru_key is None:
                break  # No models can be evicted
                
            # Evict the model
            del self.loaded_models[lru_key]
            evicted_count += 1
            self.evictions += 1
            
            # Clean up memory
            del lru_model.model
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            else:
                current_memory = sum(info.memory_usage for info in self.loaded_models.values())
                if self.memory_limit:
                    current_memory = current_memory / self.memory_limit
                    
            print(f"Evicted model {lru_key} (LRU policy). Memory usage: {current_memory:.1%}")
            
        return evicted_count

    def preload_models_for_batch(self, model_specs: List[tuple], priority: ModelPriority = ModelPriority.HIGH):
        """
        Preload models for batch processing optimization.
        
        Args:
            model_specs: List of (model_name, loader_func, args, kwargs) tuples
            priority: Priority level for preloaded models
        """
        print(f"Preloading {len(model_specs)} models for batch processing...")
        
        # Ensure we have enough memory for all models
        self._ensure_memory_available(priority, len(model_specs))
        
        for model_name, loader_func, args, kwargs in model_specs:
            if model_name not in self.loaded_models:
                try:
                    self.load_model_with_priority(model_name, priority, loader_func, *args, **kwargs)
                    print(f"Preloaded model: {model_name}")
                except Exception as e:
                    print(f"Failed to preload model {model_name}: {e}")
                    
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
            # Mark essential models with high priority
            essential_models = set(models_needed)
            for model_name, model_info in self.loaded_models.items():
                if model_name in essential_models:
                    model_info.priority = ModelPriority.HIGH
                else:
                    model_info.priority = ModelPriority.LOW
                    
            # Trigger LRU eviction to free non-essential models
            self.implement_lru_eviction(0.6)  # More aggressive eviction for batch processing
            
    def get_cache_statistics(self) -> CacheStats:
        """
        Get comprehensive cache performance statistics.
        
        Returns:
            CacheStats: Current cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        total_memory = sum(info.memory_usage for info in self.loaded_models.values())
        
        avg_load_time = sum(self.load_times) / len(self.load_times) if self.load_times else 0.0
        
        return CacheStats(
            total_models=len(self.loaded_models),
            loaded_models=len(self.loaded_models),
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            evictions=self.evictions,
            total_memory_used=total_memory,
            hit_rate=hit_rate,
            average_load_time=avg_load_time
        )
        
    def _ensure_memory_available(self, priority: ModelPriority, models_to_load: int = 1):
        """Ensure sufficient memory is available for loading new models."""
        # Implement LRU eviction if memory is tight
        if len(self.loaded_models) + models_to_load > self.max_models:
            models_to_evict = len(self.loaded_models) + models_to_load - self.max_models
            self.implement_lru_eviction()
            
        # Check memory threshold
        if self.device == "cuda" and torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if memory_usage > self.memory_threshold:
                self.implement_lru_eviction()
                
    def _enforce_cache_limits(self):
        """Enforce maximum cache size limits."""
        while len(self.loaded_models) > self.max_models:
            # Remove least recently used model
            lru_key = next(iter(self.loaded_models))  # First item is LRU
            del self.loaded_models[lru_key]
            self.evictions += 1
            
    def _estimate_model_memory(self, model) -> float:
        """
        Estimate memory usage of a model in GB.
        
        Args:
            model: Model instance
            
        Returns:
            Estimated memory usage in GB
        """
        if hasattr(model, 'parameters'):
            # PyTorch model
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / 1e9
        else:
            # Fallback estimate
            return 1.0  # 1GB default estimate
            
    def clear_cache(self):
        """Clear all cached models."""
        self.loaded_models.clear()
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a cached model."""
        return self.loaded_models.get(model_name)
        
    def list_cached_models(self) -> List[str]:
        """Get list of currently cached model names."""
        return list(self.loaded_models.keys())
        
    def set_memory_limit(self, limit_gb: float):
        """Set memory limit for the cache."""
        self.memory_limit = limit_gb
        # Trigger eviction if current usage exceeds new limit
        self.implement_lru_eviction()
