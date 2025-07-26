"""
Performance monitoring and metrics collection system for SOWLv2 pipeline.
Provides comprehensive timing, memory, and GPU utilization tracking.
"""
import time
import uuid
import psutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation or model."""
    processing_time: float  # seconds
    memory_peak_usage: float  # GB
    gpu_utilization: float  # percentage
    throughput_fps: float  # frames per second
    model_loading_time: float  # seconds
    cpu_utilization: float  # percentage
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComparisonReport:
    """Comparative analysis between two models or configurations."""
    sam2_metrics: PerformanceMetrics
    edgetam_metrics: PerformanceMetrics
    speed_improvement: float  # percentage improvement
    memory_savings: float  # percentage savings
    quality_comparison: Optional[Dict[str, float]] = None
    recommendation: str = ""


@dataclass
class TimingContext:
    """Context for timing measurements."""
    operation: str
    start_time: float
    start_memory: float
    start_gpu_memory: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceCollector:
    """Comprehensive performance metrics collection and analysis."""
    
    def __init__(self, device: str = "cuda", enable_gpu_monitoring: bool = True):
        """
        Initialize the performance collector.
        
        Args:
            device: Primary device being monitored
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.device = device
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Active timing contexts
        self._active_timers: Dict[str, TimingContext] = {}
        
        # Collected metrics
        self.operation_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.model_metrics: Dict[str, PerformanceMetrics] = {}
        
        # System monitoring
        self._baseline_cpu_percent = psutil.cpu_percent(interval=None)
        if self.enable_gpu_monitoring:
            self._baseline_gpu_memory = torch.cuda.memory_allocated() / 1e9
            
        # Performance history
        self.performance_history: List[Dict[str, Any]] = []
        
    def start_timing(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation being timed
            metadata: Additional context information
            
        Returns:
            str: Timer ID for ending the timing
        """
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        
        # Get baseline measurements
        start_memory = psutil.virtual_memory().used / 1e9  # GB
        start_gpu_memory = 0.0
        
        if self.enable_gpu_monitoring:
            torch.cuda.synchronize()  # Ensure all operations are complete
            start_gpu_memory = torch.cuda.memory_allocated() / 1e9
            
        context = TimingContext(
            operation=operation,
            start_time=time.perf_counter(),
            start_memory=start_memory,
            start_gpu_memory=start_gpu_memory,
            metadata=metadata or {}
        )
        
        self._active_timers[timer_id] = context
        return timer_id
        
    def end_timing(self, timer_id: str) -> PerformanceMetrics:
        """
        End timing for an operation and calculate metrics.
        
        Args:
            timer_id: Timer ID returned by start_timing
            
        Returns:
            PerformanceMetrics: Collected performance metrics
        """
        if timer_id not in self._active_timers:
            raise ValueError(f"Timer ID {timer_id} not found in active timers")
            
        context = self._active_timers.pop(timer_id)
        
        # Calculate timing
        end_time = time.perf_counter()
        processing_time = end_time - context.start_time
        
        # Calculate memory usage
        end_memory = psutil.virtual_memory().used / 1e9
        memory_peak_usage = end_memory - context.start_memory
        
        # Calculate GPU metrics
        gpu_utilization = 0.0
        if self.enable_gpu_monitoring:
            torch.cuda.synchronize()
            end_gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_memory_used = end_gpu_memory - context.start_gpu_memory
            memory_peak_usage = max(memory_peak_usage, gpu_memory_used)
            
            # Estimate GPU utilization (simplified)
            if processing_time > 0:
                gpu_utilization = min(100.0, (gpu_memory_used / processing_time) * 10)
                
        # Calculate CPU utilization
        cpu_utilization = psutil.cpu_percent(interval=None)
        
        # Calculate throughput if frame count is available
        throughput_fps = 0.0
        if 'frame_count' in context.metadata and processing_time > 0:
            throughput_fps = context.metadata['frame_count'] / processing_time
            
        # Model loading time (if available)
        model_loading_time = context.metadata.get('model_loading_time', 0.0)
        
        metrics = PerformanceMetrics(
            processing_time=processing_time,
            memory_peak_usage=memory_peak_usage,
            gpu_utilization=gpu_utilization,
            throughput_fps=throughput_fps,
            model_loading_time=model_loading_time,
            cpu_utilization=cpu_utilization
        )
        
        # Store metrics
        self.operation_metrics[context.operation].append(metrics)
        
        return metrics
        
    def record_memory_usage(self, stage: str) -> Dict[str, float]:
        """
        Record current memory usage for a specific stage.
        
        Args:
            stage: Processing stage name
            
        Returns:
            Dict containing memory usage statistics
        """
        # System memory
        system_memory = psutil.virtual_memory()
        memory_stats = {
            'stage': stage,
            'system_memory_used_gb': system_memory.used / 1e9,
            'system_memory_percent': system_memory.percent,
            'system_memory_available_gb': system_memory.available / 1e9,
            'timestamp': time.time()
        }
        
        # GPU memory if available
        if self.enable_gpu_monitoring:
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            memory_stats.update({
                'gpu_memory_allocated_gb': gpu_memory_allocated,
                'gpu_memory_reserved_gb': gpu_memory_reserved,
                'gpu_memory_total_gb': gpu_memory_total,
                'gpu_memory_percent': (gpu_memory_allocated / gpu_memory_total) * 100
            })
            
        # Store in history
        self.performance_history.append({
            'type': 'memory_usage',
            'data': memory_stats
        })
        
        return memory_stats
        
    def record_gpu_utilization(self, stage: str) -> Dict[str, float]:
        """
        Record GPU utilization metrics for a specific stage.
        
        Args:
            stage: Processing stage name
            
        Returns:
            Dict containing GPU utilization statistics
        """
        gpu_stats = {
            'stage': stage,
            'timestamp': time.time(),
            'gpu_available': self.enable_gpu_monitoring
        }
        
        if self.enable_gpu_monitoring:
            # Memory utilization
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_utilization = (memory_allocated / memory_total) * 100
            
            # Device properties
            device_props = torch.cuda.get_device_properties(0)
            
            gpu_stats.update({
                'memory_utilization_percent': memory_utilization,
                'memory_allocated_gb': memory_allocated,
                'memory_total_gb': memory_total,
                'device_name': device_props.name,
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multiprocessor_count': device_props.multi_processor_count
            })
            
            # Try to get additional GPU metrics if nvidia-ml-py is available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_stats['gpu_utilization_percent'] = utilization.gpu
                gpu_stats['memory_utilization_percent'] = utilization.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_stats['temperature_celsius'] = temp
                
                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                gpu_stats['power_usage_watts'] = power
                
            except ImportError:
                # pynvml not available, use basic metrics
                gpu_stats['gpu_utilization_percent'] = 0.0
                gpu_stats['note'] = 'Install nvidia-ml-py for detailed GPU metrics'
                
        # Store in history
        self.performance_history.append({
            'type': 'gpu_utilization',
            'data': gpu_stats
        })
        
        return gpu_stats
        
    def compare_models(self, sam2_metrics: PerformanceMetrics, 
                      edgetam_metrics: PerformanceMetrics,
                      quality_scores: Optional[Dict[str, Tuple[float, float]]] = None) -> ComparisonReport:
        """
        Compare performance metrics between SAM2 and EdgeTAM models.
        
        Args:
            sam2_metrics: Performance metrics for SAM2
            edgetam_metrics: Performance metrics for EdgeTAM
            quality_scores: Optional quality comparison scores (metric_name: (sam2_score, edgetam_score))
            
        Returns:
            ComparisonReport: Detailed comparison analysis
        """
        # Calculate speed improvement (positive = EdgeTAM is faster)
        if sam2_metrics.processing_time > 0:
            speed_improvement = ((sam2_metrics.processing_time - edgetam_metrics.processing_time) / 
                               sam2_metrics.processing_time) * 100
        else:
            speed_improvement = 0.0
            
        # Calculate memory savings (positive = EdgeTAM uses less memory)
        if sam2_metrics.memory_peak_usage > 0:
            memory_savings = ((sam2_metrics.memory_peak_usage - edgetam_metrics.memory_peak_usage) / 
                            sam2_metrics.memory_peak_usage) * 100
        else:
            memory_savings = 0.0
            
        # Process quality comparison if provided
        quality_comparison = None
        if quality_scores:
            quality_comparison = {}
            for metric, (sam2_score, edgetam_score) in quality_scores.items():
                if sam2_score > 0:
                    quality_diff = ((edgetam_score - sam2_score) / sam2_score) * 100
                    quality_comparison[metric] = quality_diff
                    
        # Generate recommendation
        recommendation = self._generate_model_recommendation(
            speed_improvement, memory_savings, quality_comparison
        )
        
        report = ComparisonReport(
            sam2_metrics=sam2_metrics,
            edgetam_metrics=edgetam_metrics,
            speed_improvement=speed_improvement,
            memory_savings=memory_savings,
            quality_comparison=quality_comparison,
            recommendation=recommendation
        )
        
        # Store comparison in history
        self.performance_history.append({
            'type': 'model_comparison',
            'data': {
                'speed_improvement': speed_improvement,
                'memory_savings': memory_savings,
                'quality_comparison': quality_comparison,
                'recommendation': recommendation,
                'timestamp': time.time()
            }
        })
        
        return report
        
    def _generate_model_recommendation(self, speed_improvement: float, 
                                     memory_savings: float,
                                     quality_comparison: Optional[Dict[str, float]]) -> str:
        """Generate a recommendation based on performance comparison."""
        recommendations = []
        
        # Speed analysis
        if speed_improvement > 20:
            recommendations.append("EdgeTAM provides significant speed improvement")
        elif speed_improvement > 5:
            recommendations.append("EdgeTAM is moderately faster")
        elif speed_improvement < -10:
            recommendations.append("SAM2 is significantly faster")
            
        # Memory analysis
        if memory_savings > 15:
            recommendations.append("EdgeTAM uses significantly less memory")
        elif memory_savings > 5:
            recommendations.append("EdgeTAM is more memory efficient")
        elif memory_savings < -15:
            recommendations.append("SAM2 is more memory efficient")
            
        # Quality analysis
        if quality_comparison:
            avg_quality_diff = np.mean(list(quality_comparison.values()))
            if avg_quality_diff > 5:
                recommendations.append("EdgeTAM provides better quality")
            elif avg_quality_diff < -5:
                recommendations.append("SAM2 provides better quality")
            else:
                recommendations.append("Quality is comparable between models")
                
        # Overall recommendation
        if speed_improvement > 10 and memory_savings > 0:
            overall = "Recommend EdgeTAM for performance-critical applications"
        elif speed_improvement < -5 and memory_savings < -5:
            overall = "Recommend SAM2 for this use case"
        else:
            overall = "Both models are suitable - choose based on specific requirements"
            
        if recommendations:
            return f"{overall}. {'. '.join(recommendations)}."
        else:
            return overall
            
    def get_operation_summary(self, operation: str) -> Dict[str, Any]:
        """
        Get summary statistics for a specific operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dict containing summary statistics
        """
        if operation not in self.operation_metrics:
            return {"error": f"No metrics found for operation: {operation}"}
            
        metrics_list = self.operation_metrics[operation]
        if not metrics_list:
            return {"error": f"No metrics recorded for operation: {operation}"}
            
        # Calculate statistics
        processing_times = [m.processing_time for m in metrics_list]
        memory_usages = [m.memory_peak_usage for m in metrics_list]
        gpu_utilizations = [m.gpu_utilization for m in metrics_list]
        throughputs = [m.throughput_fps for m in metrics_list if m.throughput_fps > 0]
        
        summary = {
            'operation': operation,
            'total_runs': len(metrics_list),
            'processing_time': {
                'mean': np.mean(processing_times),
                'std': np.std(processing_times),
                'min': np.min(processing_times),
                'max': np.max(processing_times),
                'median': np.median(processing_times)
            },
            'memory_usage': {
                'mean': np.mean(memory_usages),
                'std': np.std(memory_usages),
                'min': np.min(memory_usages),
                'max': np.max(memory_usages),
                'median': np.median(memory_usages)
            },
            'gpu_utilization': {
                'mean': np.mean(gpu_utilizations),
                'std': np.std(gpu_utilizations),
                'min': np.min(gpu_utilizations),
                'max': np.max(gpu_utilizations),
                'median': np.median(gpu_utilizations)
            }
        }
        
        if throughputs:
            summary['throughput'] = {
                'mean': np.mean(throughputs),
                'std': np.std(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs),
                'median': np.median(throughputs)
            }
            
        return summary
        
    def clear_metrics(self, operation: Optional[str] = None):
        """
        Clear collected metrics.
        
        Args:
            operation: Specific operation to clear, or None to clear all
        """
        if operation:
            if operation in self.operation_metrics:
                self.operation_metrics[operation].clear()
        else:
            self.operation_metrics.clear()
            self.model_metrics.clear()
            self.performance_history.clear()
            
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all collected metrics for external analysis.
        
        Returns:
            Dict containing all metrics and performance data
        """
        return {
            'operation_metrics': {
                op: [
                    {
                        'processing_time': m.processing_time,
                        'memory_peak_usage': m.memory_peak_usage,
                        'gpu_utilization': m.gpu_utilization,
                        'throughput_fps': m.throughput_fps,
                        'model_loading_time': m.model_loading_time,
                        'cpu_utilization': m.cpu_utilization,
                        'timestamp': m.timestamp.isoformat()
                    }
                    for m in metrics_list
                ]
                for op, metrics_list in self.operation_metrics.items()
            },
            'model_metrics': {
                model: {
                    'processing_time': m.processing_time,
                    'memory_peak_usage': m.memory_peak_usage,
                    'gpu_utilization': m.gpu_utilization,
                    'throughput_fps': m.throughput_fps,
                    'model_loading_time': m.model_loading_time,
                    'cpu_utilization': m.cpu_utilization,
                    'timestamp': m.timestamp.isoformat()
                }
                for model, m in self.model_metrics.items()
            },
            'performance_history': self.performance_history,
            'device': self.device,
            'gpu_monitoring_enabled': self.enable_gpu_monitoring,
            'export_timestamp': datetime.now().isoformat()
        }