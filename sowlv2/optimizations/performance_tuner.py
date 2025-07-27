"""
Automatic performance tuning system for SOWLv2 pipeline.
Analyzes system capabilities and optimizes parameters for maximum performance.
"""
import time
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import psutil
import numpy as np
from PIL import Image

from sowlv2.optimizations.resource_manager import AdvancedResourceManager
from sowlv2.optimizations.batch_optimizer import IntelligentBatchOptimizer, OptimizationLevel
from sowlv2.models.model_factory import SegmentationModelFactory


@dataclass
class SystemProfile:
    """System hardware profile for optimization."""
    gpu_name: str
    gpu_memory_gb: float
    gpu_compute_capability: Tuple[int, int]
    cpu_cores: int
    system_memory_gb: float
    supports_mixed_precision: bool
    estimated_performance_tier: str  # "low", "medium", "high", "ultra"


@dataclass
class OptimizedParameters:
    """Optimized parameters for the pipeline."""
    batch_sizes: Dict[str, int]
    memory_settings: Dict[str, Any]
    processing_settings: Dict[str, Any]
    model_settings: Dict[str, Any]
    performance_tier: str
    estimated_speedup: float


class PerformanceTuner:
    """Automatic performance tuning system."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.resource_manager = AdvancedResourceManager(device)
        self.batch_optimizer = IntelligentBatchOptimizer(device)
        
        # Performance benchmarks for different tiers
        self.performance_tiers = {
            "low": {"memory_gb": 4, "compute_score": 100},
            "medium": {"memory_gb": 8, "compute_score": 300},
            "high": {"memory_gb": 12, "compute_score": 600},
            "ultra": {"memory_gb": 16, "compute_score": 1000}
        }
    
    def profile_system(self) -> SystemProfile:
        """Profile system hardware capabilities."""
        if self.device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_gb = props.total_memory / 1e9
            gpu_compute_capability = (props.major, props.minor)
            supports_mixed_precision = props.major >= 7
            
            # Estimate performance tier based on GPU specs
            compute_score = (
                props.multi_processor_count * 
                (props.major * 100 + props.minor * 10) * 
                (gpu_memory_gb / 8.0)
            )
            
        else:
            gpu_name = "CPU"
            gpu_memory_gb = 0
            gpu_compute_capability = (0, 0)
            supports_mixed_precision = False
            compute_score = 50  # Low performance for CPU
        
        # Determine performance tier
        if compute_score >= self.performance_tiers["ultra"]["compute_score"]:
            tier = "ultra"
        elif compute_score >= self.performance_tiers["high"]["compute_score"]:
            tier = "high"
        elif compute_score >= self.performance_tiers["medium"]["compute_score"]:
            tier = "medium"
        else:
            tier = "low"
        
        return SystemProfile(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_compute_capability=gpu_compute_capability,
            cpu_cores=psutil.cpu_count(),
            system_memory_gb=psutil.virtual_memory().total / 1e9,
            supports_mixed_precision=supports_mixed_precision,
            estimated_performance_tier=tier
        )
    
    def benchmark_operations(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, float]:
        """Benchmark key operations to determine optimal parameters."""
        if image_sizes is None:
            image_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        
        benchmarks = {}
        
        for h, w in image_sizes:
            size_key = f"{h}x{w}"
            
            # Benchmark tensor operations
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                start_time = time.time()
                
                # Simulate typical pipeline operations
                x = torch.randn(1, 3, h, w, device=self.device)
                
                # Convolution (detection-like operation)
                conv_weight = torch.randn(64, 3, 3, 3, device=self.device)
                y = torch.conv2d(x, conv_weight, padding=1)
                
                # Activation and pooling
                y = torch.relu(y)
                y = torch.max_pool2d(y, 2)
                
                # Upsampling (segmentation-like operation)
                y = torch.nn.functional.interpolate(y, size=(h, w), mode='bilinear')
                
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                
            else:
                # CPU benchmark
                start_time = time.time()
                x = torch.randn(1, 3, h, w)
                y = torch.conv2d(x, torch.randn(64, 3, 3, 3), padding=1)
                y = torch.relu(y)
                elapsed = time.time() - start_time
                memory_used = 0.1  # Estimate
            
            benchmarks[size_key] = {
                "processing_time": elapsed,
                "memory_usage": memory_used,
                "throughput": 1.0 / elapsed if elapsed > 0 else 0
            }
        
        return benchmarks
    
    def optimize_batch_sizes(self, system_profile: SystemProfile, 
                           benchmarks: Dict[str, float]) -> Dict[str, int]:
        """Optimize batch sizes based on system profile and benchmarks."""
        # Base batch sizes by performance tier
        base_batches = {
            "low": {"detection": 1, "segmentation": 1, "frame": 2},
            "medium": {"detection": 4, "segmentation": 2, "frame": 8},
            "high": {"detection": 8, "segmentation": 4, "frame": 16},
            "ultra": {"detection": 16, "segmentation": 8, "frame": 32}
        }
        
        tier = system_profile.estimated_performance_tier
        batch_sizes = base_batches[tier].copy()
        
        # Adjust based on available memory
        memory_factor = min(2.0, system_profile.gpu_memory_gb / 8.0)
        
        # Adjust based on benchmark performance
        if "1024x1024" in benchmarks:
            benchmark = benchmarks["1024x1024"]
            if benchmark["processing_time"] > 0.5:  # Slow processing
                memory_factor *= 0.7
            elif benchmark["processing_time"] < 0.1:  # Fast processing
                memory_factor *= 1.3
        
        # Apply memory factor
        for key in batch_sizes:
            batch_sizes[key] = max(1, int(batch_sizes[key] * memory_factor))
        
        return batch_sizes
    
    def optimize_memory_settings(self, system_profile: SystemProfile) -> Dict[str, Any]:
        """Optimize memory-related settings."""
        settings = {}
        
        # Memory limit (leave some headroom)
        if system_profile.gpu_memory_gb > 0:
            settings["memory_limit"] = system_profile.gpu_memory_gb * 0.9
        else:
            settings["memory_limit"] = None
        
        # Streaming settings
        if system_profile.gpu_memory_gb < 6:
            settings["streaming_chunk_size"] = 50
            settings["enable_streaming_mode"] = True
        elif system_profile.gpu_memory_gb < 12:
            settings["streaming_chunk_size"] = 100
            settings["enable_streaming_mode"] = False
        else:
            settings["streaming_chunk_size"] = 200
            settings["enable_streaming_mode"] = False
        
        # Cache settings
        cache_memory = min(4.0, system_profile.gpu_memory_gb * 0.3)
        settings["cache_size_limit"] = cache_memory
        
        # Memory monitoring
        settings["memory_monitoring"] = True
        settings["auto_memory_adjustment"] = True
        
        return settings
    
    def optimize_processing_settings(self, system_profile: SystemProfile) -> Dict[str, Any]:
        """Optimize processing-related settings."""
        settings = {}
        
        # Mixed precision
        settings["enable_mixed_precision"] = system_profile.supports_mixed_precision
        
        # Parallel processing
        if system_profile.estimated_performance_tier in ["high", "ultra"]:
            settings["max_workers"] = min(6, system_profile.cpu_cores)
            settings["parallel_prompts"] = True
            settings["parallel_frames"] = True
        else:
            settings["max_workers"] = min(4, system_profile.cpu_cores)
            settings["parallel_prompts"] = True
            settings["parallel_frames"] = False
        
        # Optimization level
        tier_to_level = {
            "low": 1,
            "medium": 2,
            "high": 2,
            "ultra": 3
        }
        settings["optimization_level"] = tier_to_level[system_profile.estimated_performance_tier]
        
        # V-JEPA2 settings
        if system_profile.estimated_performance_tier in ["high", "ultra"]:
            settings["vjepa2_frames_per_clip"] = 16
            settings["temporal_detection_frames"] = 5
        else:
            settings["vjepa2_frames_per_clip"] = 8
            settings["temporal_detection_frames"] = 3
        
        return settings
    
    def optimize_model_settings(self, system_profile: SystemProfile) -> Dict[str, Any]:
        """Optimize model selection and settings."""
        settings = {}
        
        # Model selection based on performance tier
        if system_profile.estimated_performance_tier in ["high", "ultra"]:
            settings["edgetam"] = True
            settings["edgetam_model"] = "facebook/edgetam-base"
            settings["edgetam_optimization_level"] = 2
            settings["sam_model"] = "facebook/sam2.1-hiera-small"  # Fallback
        elif system_profile.estimated_performance_tier == "medium":
            settings["edgetam"] = True
            settings["edgetam_model"] = "facebook/edgetam-small"
            settings["edgetam_optimization_level"] = 3
            settings["sam_model"] = "facebook/sam2.1-hiera-tiny"
        else:  # low performance
            settings["edgetam"] = False
            settings["sam_model"] = "facebook/sam2.1-hiera-tiny"
        
        # Detection settings
        if system_profile.estimated_performance_tier in ["high", "ultra"]:
            settings["threshold"] = 0.15
            settings["fps"] = 30
        else:
            settings["threshold"] = 0.25
            settings["fps"] = 15
        
        return settings
    
    def auto_tune(self, target_image_size: Tuple[int, int] = (1024, 1024)) -> OptimizedParameters:
        """Automatically tune all parameters for optimal performance."""
        self.logger.info("Starting automatic performance tuning...")
        
        # Profile system
        system_profile = self.profile_system()
        self.logger.info(f"System profile: {system_profile.estimated_performance_tier} tier, "
                        f"{system_profile.gpu_memory_gb:.1f}GB GPU memory")
        
        # Run benchmarks
        benchmarks = self.benchmark_operations([target_image_size])
        
        # Optimize different parameter categories
        batch_sizes = self.optimize_batch_sizes(system_profile, benchmarks)
        memory_settings = self.optimize_memory_settings(system_profile)
        processing_settings = self.optimize_processing_settings(system_profile)
        model_settings = self.optimize_model_settings(system_profile)
        
        # Estimate performance improvement
        tier_speedups = {"low": 1.2, "medium": 1.8, "high": 2.5, "ultra": 3.2}
        estimated_speedup = tier_speedups[system_profile.estimated_performance_tier]
        
        optimized_params = OptimizedParameters(
            batch_sizes=batch_sizes,
            memory_settings=memory_settings,
            processing_settings=processing_settings,
            model_settings=model_settings,
            performance_tier=system_profile.estimated_performance_tier,
            estimated_speedup=estimated_speedup
        )
        
        self.logger.info(f"Performance tuning complete. Estimated speedup: {estimated_speedup:.1f}x")
        
        return optimized_params
    
    def generate_optimized_config(self, optimized_params: OptimizedParameters, 
                                output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate optimized configuration file."""
        config = {
            "# Auto-generated optimized configuration": None,
            "# Performance tier": optimized_params.performance_tier,
            "# Estimated speedup": f"{optimized_params.estimated_speedup:.1f}x",
            
            # Basic settings
            "device": self.device,
            
            # Model settings
            **optimized_params.model_settings,
            
            # Batch settings
            "batch-size": optimized_params.batch_sizes["detection"],
            
            # Memory settings
            **optimized_params.memory_settings,
            
            # Processing settings
            **optimized_params.processing_settings,
            
            # Batch optimization
            "batch-optimization": {
                "adaptive-batch-size": True,
                "max-batch-size": optimized_params.batch_sizes["detection"] * 2,
                "min-batch-size": 1,
                "memory-based-adjustment": True,
                "model-specific-tuning": True
            },
            
            # Output settings (optimized for performance)
            "merged": True,
            "binary": False,
            "overlay": True,
            "individual_masks": False,
            "confidence_maps": False,
            
            # Error handling
            "error-handling": {
                "continue-on-error": True,
                "max-consecutive-errors": 3,
                "retry-attempts": 2,
                "fallback-enabled": True
            }
        }
        
        # Remove None values (comments)
        config = {k: v for k, v in config.items() if v is not None}
        
        if output_path:
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Optimized configuration saved to {output_path}")
        
        return config
    
    def save_tuning_report(self, system_profile: SystemProfile, 
                          optimized_params: OptimizedParameters,
                          output_path: str = "performance_tuning_report.json"):
        """Save detailed tuning report."""
        report = {
            "timestamp": time.time(),
            "system_profile": asdict(system_profile),
            "optimized_parameters": asdict(optimized_params),
            "recommendations": self._generate_recommendations(system_profile, optimized_params)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Tuning report saved to {output_path}")
    
    def _generate_recommendations(self, system_profile: SystemProfile, 
                                optimized_params: OptimizedParameters) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if system_profile.gpu_memory_gb < 6:
            recommendations.append("Consider upgrading GPU memory for better performance")
        
        if not system_profile.supports_mixed_precision:
            recommendations.append("Upgrade to a newer GPU for mixed precision support")
        
        if system_profile.estimated_performance_tier == "low":
            recommendations.append("Enable streaming mode for large videos")
            recommendations.append("Use smaller batch sizes to avoid memory issues")
        
        if system_profile.cpu_cores < 4:
            recommendations.append("Consider upgrading CPU for better parallel processing")
        
        return recommendations


def main():
    """Main function for standalone performance tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-tune SOWLv2 performance parameters")
    parser.add_argument("--device", default="cuda", help="Device to optimize for")
    parser.add_argument("--output-config", help="Output path for optimized config")
    parser.add_argument("--output-report", default="tuning_report.json", 
                       help="Output path for tuning report")
    parser.add_argument("--image-size", nargs=2, type=int, default=[1024, 1024],
                       help="Target image size for optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tuning
    tuner = PerformanceTuner(args.device)
    
    # Profile and optimize
    system_profile = tuner.profile_system()
    optimized_params = tuner.auto_tune(tuple(args.image_size))
    
    # Generate outputs
    if args.output_config:
        tuner.generate_optimized_config(optimized_params, args.output_config)
    
    tuner.save_tuning_report(system_profile, optimized_params, args.output_report)
    
    print(f"Performance tuning complete!")
    print(f"Performance tier: {system_profile.estimated_performance_tier}")
    print(f"Estimated speedup: {optimized_params.estimated_speedup:.1f}x")


if __name__ == "__main__":
    main()