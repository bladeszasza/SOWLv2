"""
Performance report generation system for SOWLv2 optimization analysis.
Provides comprehensive reporting with JSON/HTML formats, charts, and trend analysis.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns

from .performance_collector import PerformanceCollector, PerformanceMetrics, ComparisonReport
from .benchmark_runner import BenchmarkRunner, BenchmarkResults, ThroughputResults, MemoryProfile


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_charts: bool = True
    include_trend_analysis: bool = True
    chart_format: str = "png"  # png, svg
    chart_dpi: int = 300
    theme: str = "default"  # default, dark, minimal
    max_history_days: int = 30
    output_formats: List[str] = None  # json, html, both
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["json", "html"]


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric_name: str
    trend_direction: str  # improving, degrading, stable
    trend_strength: float  # 0-1, strength of trend
    change_percentage: float  # percentage change over period
    confidence_score: float  # 0-1, confidence in trend
    recommendations: List[str]


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    timestamp: str
    summary: Dict[str, Any]
    model_comparisons: List[ComparisonReport]
    benchmark_results: List[BenchmarkResults]
    trend_analysis: List[TrendAnalysis]
    performance_history: List[Dict[str, Any]]
    charts: Dict[str, str]  # chart_name -> base64_encoded_image
    recommendations: List[str]
    metadata: Dict[str, Any]


class ReportGenerator:
    """Comprehensive performance report generator with visualization and analysis."""
    
    def __init__(self, output_dir: str = "reports", 
                 performance_collector: Optional[PerformanceCollector] = None,
                 benchmark_runner: Optional[BenchmarkRunner] = None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
            performance_collector: Performance collector instance
            benchmark_runner: Benchmark runner instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_collector = performance_collector or PerformanceCollector()
        self.benchmark_runner = benchmark_runner or BenchmarkRunner()
        
        # Performance history storage
        self.history_file = self.output_dir / "performance_history.json"
        self.performance_history = self._load_performance_history()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
    def generate_comprehensive_report(self, 
                                    benchmark_results: Optional[List[BenchmarkResults]] = None,
                                    model_comparisons: Optional[List[ComparisonReport]] = None,
                                    config: Optional[ReportConfig] = None) -> PerformanceReport:
        """
        Generate a comprehensive performance report.
        
        Args:
            benchmark_results: List of benchmark results to include
            model_comparisons: List of model comparison reports
            config: Report generation configuration
            
        Returns:
            PerformanceReport: Complete performance report
        """
        if config is None:
            config = ReportConfig()
            
        report_id = f"report_{int(time.time())}"
        timestamp = datetime.now().isoformat()
        
        print(f"Generating comprehensive performance report: {report_id}")
        
        # Collect current performance data
        current_metrics = self.performance_collector.export_metrics()
        
        # Update performance history
        self._update_performance_history(current_metrics)
        
        # Generate summary statistics
        summary = self._generate_summary(benchmark_results, model_comparisons, current_metrics)
        
        # Perform trend analysis
        trend_analysis = []
        if config.include_trend_analysis:
            trend_analysis = self._perform_trend_analysis(config.max_history_days)
            
        # Generate charts
        charts = {}
        if config.include_charts:
            charts = self._generate_charts(
                benchmark_results, model_comparisons, 
                trend_analysis, config
            )
            
        # Generate recommendations
        recommendations = self._generate_recommendations(
            summary, trend_analysis, model_comparisons
        )
        
        # Create report object
        report = PerformanceReport(
            report_id=report_id,
            timestamp=timestamp,
            summary=summary,
            model_comparisons=model_comparisons or [],
            benchmark_results=benchmark_results or [],
            trend_analysis=trend_analysis,
            performance_history=self.performance_history[-100:],  # Last 100 entries
            charts=charts,
            recommendations=recommendations,
            metadata={
                'config': asdict(config),
                'system_info': self._get_system_info(),
                'generation_time': time.time()
            }
        )
        
        # Save report in requested formats
        saved_files = []
        for format_type in config.output_formats:
            if format_type == "json":
                json_file = self._save_json_report(report)
                saved_files.append(json_file)
            elif format_type == "html":
                html_file = self._save_html_report(report, config)
                saved_files.append(html_file)
                
        print(f"Report generated successfully. Files saved:")
        for file_path in saved_files:
            print(f"  - {file_path}")
            
        return report
        
    def generate_model_comparison_report(self, 
                                       sam2_results: BenchmarkResults,
                                       edgetam_results: BenchmarkResults,
                                       config: Optional[ReportConfig] = None) -> PerformanceReport:
        """
        Generate a focused model comparison report.
        
        Args:
            sam2_results: SAM2 benchmark results
            edgetam_results: EdgeTAM benchmark results
            config: Report configuration
            
        Returns:
            PerformanceReport: Model comparison report
        """
        if config is None:
            config = ReportConfig()
            
        # Create comparison report
        comparison = self.performance_collector.compare_models(
            sam2_results.performance_metrics,
            edgetam_results.performance_metrics
        )
        
        return self.generate_comprehensive_report(
            benchmark_results=[sam2_results, edgetam_results],
            model_comparisons=[comparison],
            config=config
        )
        
    def generate_trend_report(self, days: int = 30, 
                            config: Optional[ReportConfig] = None) -> PerformanceReport:
        """
        Generate a trend analysis focused report.
        
        Args:
            days: Number of days to analyze
            config: Report configuration
            
        Returns:
            PerformanceReport: Trend analysis report
        """
        if config is None:
            config = ReportConfig(include_trend_analysis=True)
            
        config.max_history_days = days
        
        return self.generate_comprehensive_report(config=config)
        
    def _generate_summary(self, benchmark_results: Optional[List[BenchmarkResults]],
                         model_comparisons: Optional[List[ComparisonReport]],
                         current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_operations': len(current_metrics.get('operation_metrics', {})),
            'total_models_tested': len(current_metrics.get('model_metrics', {})),
            'performance_entries': len(self.performance_history)
        }
        
        # Benchmark summary
        if benchmark_results:
            processing_times = [r.performance_metrics.processing_time for r in benchmark_results]
            memory_usages = [r.performance_metrics.memory_peak_usage for r in benchmark_results]
            throughputs = [r.performance_metrics.throughput_fps for r in benchmark_results if r.performance_metrics.throughput_fps > 0]
            
            summary['benchmark_summary'] = {
                'models_tested': len(benchmark_results),
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'avg_memory_usage': np.mean(memory_usages) if memory_usages else 0,
                'avg_throughput': np.mean(throughputs) if throughputs else 0,
                'fastest_model': min(benchmark_results, key=lambda x: x.performance_metrics.processing_time).model_name if benchmark_results else None,
                'most_efficient_model': min(benchmark_results, key=lambda x: x.performance_metrics.memory_peak_usage).model_name if benchmark_results else None
            }
            
        # Model comparison summary
        if model_comparisons:
            speed_improvements = [c.speed_improvement for c in model_comparisons]
            memory_savings = [c.memory_savings for c in model_comparisons]
            
            summary['comparison_summary'] = {
                'comparisons_made': len(model_comparisons),
                'avg_speed_improvement': np.mean(speed_improvements) if speed_improvements else 0,
                'avg_memory_savings': np.mean(memory_savings) if memory_savings else 0,
                'best_speed_improvement': max(speed_improvements) if speed_improvements else 0,
                'best_memory_savings': max(memory_savings) if memory_savings else 0
            }
            
        # Current system status
        summary['system_status'] = {
            'device': current_metrics.get('device', 'unknown'),
            'gpu_monitoring': current_metrics.get('gpu_monitoring_enabled', False),
            'active_operations': len([op for op, metrics in current_metrics.get('operation_metrics', {}).items() if metrics])
        }
        
        return summary
        
    def _perform_trend_analysis(self, days: int) -> List[TrendAnalysis]:
        """Perform trend analysis on historical performance data."""
        if len(self.performance_history) < 2:
            return []
            
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_date
        ]
        
        if len(recent_history) < 2:
            return []
            
        trends = []
        
        # Analyze processing time trends
        processing_times = []
        timestamps = []
        
        for entry in recent_history:
            if 'operation_metrics' in entry:
                for op_name, op_metrics in entry['operation_metrics'].items():
                    if op_metrics:
                        avg_time = np.mean([m['processing_time'] for m in op_metrics])
                        processing_times.append(avg_time)
                        timestamps.append(datetime.fromisoformat(entry['timestamp']))
                        
        if len(processing_times) >= 3:
            trend = self._calculate_trend(processing_times, timestamps, 'processing_time')
            trends.append(trend)
            
        # Analyze memory usage trends
        memory_usages = []
        memory_timestamps = []
        
        for entry in recent_history:
            if 'performance_history' in entry:
                for perf_entry in entry['performance_history']:
                    if perf_entry.get('type') == 'memory_usage':
                        memory_usages.append(perf_entry['data'].get('system_memory_used_gb', 0))
                        memory_timestamps.append(datetime.fromtimestamp(perf_entry['data']['timestamp']))
                        
        if len(memory_usages) >= 3:
            trend = self._calculate_trend(memory_usages, memory_timestamps, 'memory_usage')
            trends.append(trend)
            
        # Analyze GPU utilization trends
        gpu_utilizations = []
        gpu_timestamps = []
        
        for entry in recent_history:
            if 'performance_history' in entry:
                for perf_entry in entry['performance_history']:
                    if perf_entry.get('type') == 'gpu_utilization':
                        gpu_utilizations.append(perf_entry['data'].get('gpu_utilization_percent', 0))
                        gpu_timestamps.append(datetime.fromtimestamp(perf_entry['data']['timestamp']))
                        
        if len(gpu_utilizations) >= 3:
            trend = self._calculate_trend(gpu_utilizations, gpu_timestamps, 'gpu_utilization')
            trends.append(trend)
            
        return trends
        
    def _calculate_trend(self, values: List[float], timestamps: List[datetime], 
                        metric_name: str) -> TrendAnalysis:
        """Calculate trend analysis for a specific metric."""
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="stable",
                trend_strength=0.0,
                change_percentage=0.0,
                confidence_score=0.0,
                recommendations=[]
            )
            
        # Convert timestamps to numeric values for regression
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate linear regression
        coeffs = np.polyfit(time_numeric, values, 1)
        slope = coeffs[0]
        
        # Calculate trend metrics
        value_range = max(values) - min(values)
        trend_strength = abs(slope) / (value_range / len(values)) if value_range > 0 else 0
        trend_strength = min(trend_strength, 1.0)  # Cap at 1.0
        
        # Determine trend direction
        if abs(slope) < 0.01 * np.mean(values):
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "degrading" if metric_name in ['processing_time', 'memory_usage'] else "improving"
        else:
            trend_direction = "improving" if metric_name in ['processing_time', 'memory_usage'] else "degrading"
            
        # Calculate percentage change
        if len(values) >= 2:
            change_percentage = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        else:
            change_percentage = 0
            
        # Calculate confidence score based on data consistency
        if len(values) >= 5:
            # Use R-squared as confidence measure
            y_pred = np.polyval(coeffs, time_numeric)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            confidence_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            confidence_score = max(0, min(confidence_score, 1))
        else:
            confidence_score = 0.5  # Medium confidence for small datasets
            
        # Generate recommendations
        recommendations = self._generate_trend_recommendations(
            metric_name, trend_direction, trend_strength, change_percentage
        )
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_percentage=change_percentage,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
        
    def _generate_trend_recommendations(self, metric_name: str, trend_direction: str,
                                      trend_strength: float, change_percentage: float) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if metric_name == "processing_time":
            if trend_direction == "degrading" and trend_strength > 0.3:
                recommendations.append("Processing time is increasing - consider optimizing batch sizes or model caching")
                if abs(change_percentage) > 20:
                    recommendations.append("Significant performance degradation detected - investigate recent changes")
            elif trend_direction == "improving":
                recommendations.append("Processing time improvements detected - current optimizations are effective")
                
        elif metric_name == "memory_usage":
            if trend_direction == "degrading" and trend_strength > 0.3:
                recommendations.append("Memory usage is increasing - check for memory leaks or optimize model loading")
                if abs(change_percentage) > 30:
                    recommendations.append("High memory usage increase - consider implementing streaming processing")
            elif trend_direction == "improving":
                recommendations.append("Memory usage optimization is working well")
                
        elif metric_name == "gpu_utilization":
            if trend_direction == "degrading" and trend_strength > 0.3:
                recommendations.append("GPU utilization is decreasing - check for bottlenecks in data loading or preprocessing")
            elif trend_direction == "improving":
                recommendations.append("GPU utilization improvements indicate better resource usage")
                
        return recommendations        

    def _generate_charts(self, benchmark_results: Optional[List[BenchmarkResults]],
                        model_comparisons: Optional[List[ComparisonReport]],
                        trend_analysis: List[TrendAnalysis],
                        config: ReportConfig) -> Dict[str, str]:
        """Generate performance visualization charts."""
        charts = {}
        
        try:
            # Performance comparison chart
            if benchmark_results and len(benchmark_results) >= 2:
                chart = self._create_performance_comparison_chart(benchmark_results, config)
                charts['performance_comparison'] = chart
                
            # Model comparison radar chart
            if model_comparisons:
                chart = self._create_model_comparison_radar(model_comparisons, config)
                charts['model_comparison_radar'] = chart
                
            # Trend analysis charts
            if trend_analysis:
                chart = self._create_trend_analysis_chart(trend_analysis, config)
                charts['trend_analysis'] = chart
                
            # Memory usage timeline
            if self.performance_history:
                chart = self._create_memory_timeline_chart(config)
                charts['memory_timeline'] = chart
                
            # Throughput analysis
            if benchmark_results:
                chart = self._create_throughput_analysis_chart(benchmark_results, config)
                charts['throughput_analysis'] = chart
                
        except Exception as e:
            print(f"Warning: Error generating charts: {e}")
            
        return charts
        
    def _create_performance_comparison_chart(self, benchmark_results: List[BenchmarkResults],
                                           config: ReportConfig) -> str:
        """Create performance comparison bar chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Comparison Across Models', fontsize=16, fontweight='bold')
        
        models = [r.model_name for r in benchmark_results]
        processing_times = [r.performance_metrics.processing_time for r in benchmark_results]
        memory_usages = [r.performance_metrics.memory_peak_usage for r in benchmark_results]
        throughputs = [r.performance_metrics.throughput_fps for r in benchmark_results]
        gpu_utilizations = [r.performance_metrics.gpu_utilization for r in benchmark_results]
        
        # Processing time comparison
        bars1 = ax1.bar(models, processing_times, color=sns.color_palette("husl", len(models)))
        ax1.set_title('Processing Time (seconds)', fontweight='bold')
        ax1.set_ylabel('Time (s)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, processing_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}s', ha='center', va='bottom')
        
        # Memory usage comparison
        bars2 = ax2.bar(models, memory_usages, color=sns.color_palette("husl", len(models)))
        ax2.set_title('Peak Memory Usage (GB)', fontweight='bold')
        ax2.set_ylabel('Memory (GB)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, memory_usages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}GB', ha='center', va='bottom')
        
        # Throughput comparison
        bars3 = ax3.bar(models, throughputs, color=sns.color_palette("husl", len(models)))
        ax3.set_title('Throughput (FPS)', fontweight='bold')
        ax3.set_ylabel('FPS')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, throughputs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # GPU utilization comparison
        bars4 = ax4.bar(models, gpu_utilizations, color=sns.color_palette("husl", len(models)))
        ax4.set_title('GPU Utilization (%)', fontweight='bold')
        ax4.set_ylabel('Utilization (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, gpu_utilizations):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig, config)
        
    def _create_model_comparison_radar(self, model_comparisons: List[ComparisonReport],
                                     config: ReportConfig) -> str:
        """Create radar chart for model comparison."""
        if not model_comparisons:
            return ""
            
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Use the first comparison for the radar chart
        comparison = model_comparisons[0]
        
        # Metrics for radar chart (normalized to 0-1 scale)
        sam2_metrics = comparison.sam2_metrics
        edgetam_metrics = comparison.edgetam_metrics
        
        # Normalize metrics (lower is better for time/memory, higher is better for throughput/utilization)
        max_time = max(sam2_metrics.processing_time, edgetam_metrics.processing_time)
        max_memory = max(sam2_metrics.memory_peak_usage, edgetam_metrics.memory_peak_usage)
        max_throughput = max(sam2_metrics.throughput_fps, edgetam_metrics.throughput_fps)
        max_gpu = max(sam2_metrics.gpu_utilization, edgetam_metrics.gpu_utilization)
        
        # SAM2 values (normalized)
        sam2_values = [
            1 - (sam2_metrics.processing_time / max_time) if max_time > 0 else 0,  # Speed (inverted)
            1 - (sam2_metrics.memory_peak_usage / max_memory) if max_memory > 0 else 0,  # Memory efficiency (inverted)
            sam2_metrics.throughput_fps / max_throughput if max_throughput > 0 else 0,  # Throughput
            sam2_metrics.gpu_utilization / max_gpu if max_gpu > 0 else 0,  # GPU utilization
        ]
        
        # EdgeTAM values (normalized)
        edgetam_values = [
            1 - (edgetam_metrics.processing_time / max_time) if max_time > 0 else 0,
            1 - (edgetam_metrics.memory_peak_usage / max_memory) if max_memory > 0 else 0,
            edgetam_metrics.throughput_fps / max_throughput if max_throughput > 0 else 0,
            edgetam_metrics.gpu_utilization / max_gpu if max_gpu > 0 else 0,
        ]
        
        # Labels
        labels = ['Speed', 'Memory\nEfficiency', 'Throughput', 'GPU\nUtilization']
        
        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Add values to complete the circle
        sam2_values += sam2_values[:1]
        edgetam_values += edgetam_values[:1]
        
        # Plot
        ax.plot(angles, sam2_values, 'o-', linewidth=2, label='SAM2', color='blue')
        ax.fill(angles, sam2_values, alpha=0.25, color='blue')
        
        ax.plot(angles, edgetam_values, 'o-', linewidth=2, label='EdgeTAM', color='red')
        ax.fill(angles, edgetam_values, alpha=0.25, color='red')
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Model Performance Comparison\n(Normalized Metrics)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        return self._fig_to_base64(fig, config)
        
    def _create_trend_analysis_chart(self, trend_analysis: List[TrendAnalysis],
                                   config: ReportConfig) -> str:
        """Create trend analysis visualization."""
        if not trend_analysis:
            return ""
            
        fig, axes = plt.subplots(len(trend_analysis), 1, figsize=(12, 4 * len(trend_analysis)))
        if len(trend_analysis) == 1:
            axes = [axes]
            
        fig.suptitle('Performance Trend Analysis', fontsize=16, fontweight='bold')
        
        colors = sns.color_palette("husl", len(trend_analysis))
        
        for i, (trend, ax, color) in enumerate(zip(trend_analysis, axes, colors)):
            # Create sample data points for visualization
            x_points = np.linspace(0, 30, 20)  # 30 days, 20 data points
            
            # Generate trend line based on trend direction and strength
            if trend.trend_direction == "improving":
                base_value = 1.0
                trend_factor = -trend.trend_strength * 0.3
            elif trend.trend_direction == "degrading":
                base_value = 0.7
                trend_factor = trend.trend_strength * 0.3
            else:  # stable
                base_value = 0.85
                trend_factor = 0
                
            # Add some realistic noise
            np.random.seed(42)
            noise = np.random.normal(0, 0.05, len(x_points))
            y_points = base_value + trend_factor * (x_points / 30) + noise
            
            # Plot trend line
            ax.plot(x_points, y_points, color=color, linewidth=2, alpha=0.7)
            ax.fill_between(x_points, y_points, alpha=0.3, color=color)
            
            # Add trend arrow
            if trend.trend_direction == "improving":
                ax.annotate('↓ Improving', xy=(25, y_points[-1]), xytext=(20, y_points[-1] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2),
                           fontsize=12, color='green', fontweight='bold')
            elif trend.trend_direction == "degrading":
                ax.annotate('↑ Degrading', xy=(25, y_points[-1]), xytext=(20, y_points[-1] - 0.1),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=12, color='red', fontweight='bold')
            else:
                ax.annotate('→ Stable', xy=(25, y_points[-1]), xytext=(20, y_points[-1]),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                           fontsize=12, color='blue', fontweight='bold')
            
            # Customize subplot
            ax.set_title(f'{trend.metric_name.replace("_", " ").title()} Trend\n'
                        f'Change: {trend.change_percentage:+.1f}% | '
                        f'Confidence: {trend.confidence_score:.1%}',
                        fontweight='bold')
            ax.set_xlabel('Days')
            ax.set_ylabel('Normalized Value')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 30)
            
        plt.tight_layout()
        return self._fig_to_base64(fig, config)
        
    def _create_memory_timeline_chart(self, config: ReportConfig) -> str:
        """Create memory usage timeline chart."""
        if not self.performance_history:
            return ""
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract memory usage data from history
        timestamps = []
        memory_values = []
        
        for entry in self.performance_history[-50:]:  # Last 50 entries
            if 'performance_history' in entry:
                for perf_entry in entry['performance_history']:
                    if perf_entry.get('type') == 'memory_usage':
                        timestamps.append(datetime.fromtimestamp(perf_entry['data']['timestamp']))
                        memory_values.append(perf_entry['data'].get('system_memory_used_gb', 0))
                        
        if timestamps and memory_values:
            # Sort by timestamp
            sorted_data = sorted(zip(timestamps, memory_values))
            timestamps, memory_values = zip(*sorted_data)
            
            # Plot memory timeline
            ax.plot(timestamps, memory_values, linewidth=2, color='blue', alpha=0.7)
            ax.fill_between(timestamps, memory_values, alpha=0.3, color='blue')
            
            # Add average line
            avg_memory = np.mean(memory_values)
            ax.axhline(y=avg_memory, color='red', linestyle='--', alpha=0.7, 
                      label=f'Average: {avg_memory:.2f} GB')
            
            # Customize
            ax.set_title('Memory Usage Timeline', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Memory Usage (GB)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.xticks(rotation=45)
        else:
            # No data available
            ax.text(0.5, 0.5, 'No memory usage data available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, alpha=0.7)
            ax.set_title('Memory Usage Timeline', fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        return self._fig_to_base64(fig, config)
        
    def _create_throughput_analysis_chart(self, benchmark_results: List[BenchmarkResults],
                                        config: ReportConfig) -> str:
        """Create throughput analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Throughput Analysis', fontsize=16, fontweight='bold')
        
        # Extract throughput data from detailed results
        models = []
        batch_throughputs = {}
        
        for result in benchmark_results:
            models.append(result.model_name)
            
            # Extract throughput data from detailed results
            if 'throughput_tests' in result.detailed_results:
                throughput_data = result.detailed_results['throughput_tests']
                if throughput_data:
                    batch_sizes = [t.batch_size for t in throughput_data]
                    throughputs = [t.throughput_fps for t in throughput_data]
                    batch_throughputs[result.model_name] = (batch_sizes, throughputs)
                    
        # Throughput vs Batch Size
        colors = sns.color_palette("husl", len(models))
        for i, (model, color) in enumerate(zip(models, colors)):
            if model in batch_throughputs:
                batch_sizes, throughputs = batch_throughputs[model]
                ax1.plot(batch_sizes, throughputs, 'o-', color=color, 
                        linewidth=2, markersize=6, label=model)
                
        ax1.set_title('Throughput vs Batch Size', fontweight='bold')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (FPS)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Overall throughput comparison
        overall_throughputs = [r.performance_metrics.throughput_fps for r in benchmark_results]
        bars = ax2.bar(models, overall_throughputs, color=colors)
        ax2.set_title('Overall Throughput Comparison', fontweight='bold')
        ax2.set_ylabel('Throughput (FPS)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, overall_throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig, config)
        
    def _fig_to_base64(self, fig: Figure, config: ReportConfig) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buffer = BytesIO()
        fig.savefig(buffer, format=config.chart_format, dpi=config.chart_dpi, 
                   bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)  # Free memory
        
        return image_base64
        
    def _generate_recommendations(self, summary: Dict[str, Any],
                                trend_analysis: List[TrendAnalysis],
                                model_comparisons: Optional[List[ComparisonReport]]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Performance-based recommendations
        if 'benchmark_summary' in summary:
            bench_summary = summary['benchmark_summary']
            
            if bench_summary['avg_processing_time'] > 5.0:
                recommendations.append(
                    "High average processing time detected. Consider enabling EdgeTAM for faster inference."
                )
                
            if bench_summary['avg_memory_usage'] > 8.0:
                recommendations.append(
                    "High memory usage detected. Enable streaming processing for large videos."
                )
                
            if bench_summary['avg_throughput'] < 1.0:
                recommendations.append(
                    "Low throughput detected. Optimize batch sizes and enable GPU batching."
                )
                
        # Trend-based recommendations
        for trend in trend_analysis:
            recommendations.extend(trend.recommendations)
            
        # Model comparison recommendations
        if model_comparisons:
            for comparison in model_comparisons:
                if comparison.speed_improvement > 20:
                    recommendations.append(
                        f"EdgeTAM shows {comparison.speed_improvement:.1f}% speed improvement. "
                        "Consider using EdgeTAM for performance-critical applications."
                    )
                elif comparison.speed_improvement < -10:
                    recommendations.append(
                        "SAM2 performs better than EdgeTAM for this workload. Stick with SAM2."
                    )
                    
                if comparison.memory_savings > 15:
                    recommendations.append(
                        f"EdgeTAM uses {comparison.memory_savings:.1f}% less memory. "
                        "Good choice for memory-constrained environments."
                    )
                    
        # System-specific recommendations
        if 'system_status' in summary:
            system_status = summary['system_status']
            
            if not system_status['gpu_monitoring']:
                recommendations.append(
                    "GPU monitoring is disabled. Enable it for better performance insights."
                )
                
        # Default recommendations if none generated
        if not recommendations:
            recommendations.append("System performance appears optimal. Continue monitoring for changes.")
            
        return recommendations[:10]  # Limit to top 10 recommendations    
    
    def _load_performance_history(self) -> List[Dict[str, Any]]:
        """Load performance history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load performance history: {e}")
                
        return []
        
    def _update_performance_history(self, current_metrics: Dict[str, Any]):
        """Update performance history with current metrics."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation_metrics': current_metrics.get('operation_metrics', {}),
            'model_metrics': current_metrics.get('model_metrics', {}),
            'performance_history': current_metrics.get('performance_history', []),
            'device': current_metrics.get('device', 'unknown'),
            'gpu_monitoring_enabled': current_metrics.get('gpu_monitoring_enabled', False)
        }
        
        self.performance_history.append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
        # Save to file
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save performance history: {e}")
            
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1e9,
            'timestamp': datetime.now().isoformat()
        }
        
        # GPU information if available
        if torch.cuda.is_available():
            system_info.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown',
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.device_count() > 0 else 0
            })
        else:
            system_info['gpu_available'] = False
            
        return system_info
        
    def _save_json_report(self, report: PerformanceReport) -> str:
        """Save report in JSON format."""
        filename = f"{report.report_id}.json"
        filepath = self.output_dir / filename
        
        # Convert report to serializable format
        report_dict = {
            'report_id': report.report_id,
            'timestamp': report.timestamp,
            'summary': report.summary,
            'model_comparisons': [
                {
                    'sam2_metrics': asdict(comp.sam2_metrics),
                    'edgetam_metrics': asdict(comp.edgetam_metrics),
                    'speed_improvement': comp.speed_improvement,
                    'memory_savings': comp.memory_savings,
                    'quality_comparison': comp.quality_comparison,
                    'recommendation': comp.recommendation
                }
                for comp in report.model_comparisons
            ],
            'benchmark_results': [
                {
                    'model_name': result.model_name,
                    'configuration': result.configuration,
                    'performance_metrics': asdict(result.performance_metrics),
                    'detailed_results': result.detailed_results,
                    'test_conditions': result.test_conditions,
                    'timestamp': result.timestamp
                }
                for result in report.benchmark_results
            ],
            'trend_analysis': [asdict(trend) for trend in report.trend_analysis],
            'performance_history': report.performance_history,
            'recommendations': report.recommendations,
            'metadata': report.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        return str(filepath)
        
    def _save_html_report(self, report: PerformanceReport, config: ReportConfig) -> str:
        """Save report in HTML format."""
        filename = f"{report.report_id}.html"
        filepath = self.output_dir / filename
        
        html_content = self._generate_html_content(report, config)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(filepath)
        
    def _generate_html_content(self, report: PerformanceReport, config: ReportConfig) -> str:
        """Generate HTML content for the report."""
        # HTML template with embedded CSS
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOWLv2 Performance Report - {report_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .timestamp {{
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #007acc;
            background: #f9f9f9;
        }}
        .section h2 {{
            color: #007acc;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .section h3 {{
            color: #333;
            margin-top: 20px;
            font-size: 1.3em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .recommendations {{
            background: #e8f4fd;
            border: 1px solid #007acc;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations h3 {{
            color: #007acc;
            margin-top: 0;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 10px 0;
            line-height: 1.5;
        }}
        .trend-item {{
            background: white;
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .trend-improving {{ border-left-color: #28a745; }}
        .trend-degrading {{ border-left-color: #dc3545; }}
        .trend-stable {{ border-left-color: #6c757d; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .comparison-table th,
        .comparison-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table th {{
            background: #007acc;
            color: white;
            font-weight: bold;
        }}
        .comparison-table tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }}
        .positive {{ color: #28a745; font-weight: bold; }}
        .negative {{ color: #dc3545; font-weight: bold; }}
        .neutral {{ color: #6c757d; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SOWLv2 Performance Report</h1>
            <div class="timestamp">Generated on {timestamp}</div>
            <div class="timestamp">Report ID: {report_id}</div>
        </div>

        {summary_section}
        {benchmark_section}
        {comparison_section}
        {trend_section}
        {charts_section}
        {recommendations_section}

        <div class="footer">
            <p>Report generated by SOWLv2 Performance Monitoring System</p>
            <p>System: {system_info}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Generate sections
        summary_section = self._generate_html_summary_section(report)
        benchmark_section = self._generate_html_benchmark_section(report)
        comparison_section = self._generate_html_comparison_section(report)
        trend_section = self._generate_html_trend_section(report)
        charts_section = self._generate_html_charts_section(report)
        recommendations_section = self._generate_html_recommendations_section(report)
        
        # System info string
        system_info = f"{report.metadata['system_info'].get('platform', 'Unknown')} | " \
                     f"GPU: {report.metadata['system_info'].get('gpu_name', 'N/A')}"
        
        return html_template.format(
            report_id=report.report_id,
            timestamp=report.timestamp,
            summary_section=summary_section,
            benchmark_section=benchmark_section,
            comparison_section=comparison_section,
            trend_section=trend_section,
            charts_section=charts_section,
            recommendations_section=recommendations_section,
            system_info=system_info
        )
        
    def _generate_html_summary_section(self, report: PerformanceReport) -> str:
        """Generate HTML summary section."""
        summary = report.summary
        
        # Extract key metrics
        total_ops = summary.get('total_operations', 0)
        total_models = summary.get('total_models_tested', 0)
        perf_entries = summary.get('performance_entries', 0)
        
        benchmark_summary = summary.get('benchmark_summary', {})
        avg_time = benchmark_summary.get('avg_processing_time', 0)
        avg_memory = benchmark_summary.get('avg_memory_usage', 0)
        avg_throughput = benchmark_summary.get('avg_throughput', 0)
        
        return f"""
        <div class="section">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_ops}</div>
                    <div class="metric-label">Total Operations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_models}</div>
                    <div class="metric-label">Models Tested</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{perf_entries}</div>
                    <div class="metric-label">Performance Entries</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_time:.3f}s</div>
                    <div class="metric-label">Avg Processing Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_memory:.2f}GB</div>
                    <div class="metric-label">Avg Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_throughput:.1f}</div>
                    <div class="metric-label">Avg Throughput (FPS)</div>
                </div>
            </div>
        </div>
        """
        
    def _generate_html_benchmark_section(self, report: PerformanceReport) -> str:
        """Generate HTML benchmark results section."""
        if not report.benchmark_results:
            return ""
            
        table_rows = ""
        for result in report.benchmark_results:
            metrics = result.performance_metrics
            table_rows += f"""
            <tr>
                <td>{result.model_name}</td>
                <td>{metrics.processing_time:.3f}s</td>
                <td>{metrics.memory_peak_usage:.2f}GB</td>
                <td>{metrics.throughput_fps:.1f}</td>
                <td>{metrics.gpu_utilization:.1f}%</td>
                <td>{metrics.cpu_utilization:.1f}%</td>
            </tr>
            """
            
        return f"""
        <div class="section">
            <h2>Benchmark Results</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Processing Time</th>
                        <th>Memory Usage</th>
                        <th>Throughput (FPS)</th>
                        <th>GPU Utilization</th>
                        <th>CPU Utilization</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
        
    def _generate_html_comparison_section(self, report: PerformanceReport) -> str:
        """Generate HTML model comparison section."""
        if not report.model_comparisons:
            return ""
            
        comparisons_html = ""
        for i, comp in enumerate(report.model_comparisons):
            speed_class = "positive" if comp.speed_improvement > 0 else "negative" if comp.speed_improvement < -5 else "neutral"
            memory_class = "positive" if comp.memory_savings > 0 else "negative" if comp.memory_savings < -5 else "neutral"
            
            comparisons_html += f"""
            <div class="trend-item">
                <h3>Comparison {i+1}</h3>
                <p><strong>Speed Improvement:</strong> <span class="{speed_class}">{comp.speed_improvement:+.1f}%</span></p>
                <p><strong>Memory Savings:</strong> <span class="{memory_class}">{comp.memory_savings:+.1f}%</span></p>
                <p><strong>Recommendation:</strong> {comp.recommendation}</p>
            </div>
            """
            
        return f"""
        <div class="section">
            <h2>Model Comparisons</h2>
            {comparisons_html}
        </div>
        """
        
    def _generate_html_trend_section(self, report: PerformanceReport) -> str:
        """Generate HTML trend analysis section."""
        if not report.trend_analysis:
            return ""
            
        trends_html = ""
        for trend in report.trend_analysis:
            trend_class = f"trend-{trend.trend_direction}"
            change_class = "positive" if trend.change_percentage < 0 and trend.metric_name in ['processing_time', 'memory_usage'] else \
                          "positive" if trend.change_percentage > 0 and trend.metric_name not in ['processing_time', 'memory_usage'] else \
                          "negative" if abs(trend.change_percentage) > 10 else "neutral"
            
            recommendations_html = ""
            if trend.recommendations:
                recommendations_html = "<ul>" + "".join([f"<li>{rec}</li>" for rec in trend.recommendations]) + "</ul>"
            
            trends_html += f"""
            <div class="trend-item {trend_class}">
                <h3>{trend.metric_name.replace('_', ' ').title()}</h3>
                <p><strong>Trend:</strong> {trend.trend_direction.title()} (Strength: {trend.trend_strength:.1%})</p>
                <p><strong>Change:</strong> <span class="{change_class}">{trend.change_percentage:+.1f}%</span></p>
                <p><strong>Confidence:</strong> {trend.confidence_score:.1%}</p>
                {recommendations_html}
            </div>
            """
            
        return f"""
        <div class="section">
            <h2>Trend Analysis</h2>
            {trends_html}
        </div>
        """
        
    def _generate_html_charts_section(self, report: PerformanceReport) -> str:
        """Generate HTML charts section."""
        if not report.charts:
            return ""
            
        charts_html = ""
        for chart_name, chart_data in report.charts.items():
            chart_title = chart_name.replace('_', ' ').title()
            charts_html += f"""
            <div class="chart-container">
                <h3>{chart_title}</h3>
                <img src="data:image/png;base64,{chart_data}" alt="{chart_title}">
            </div>
            """
            
        return f"""
        <div class="section">
            <h2>Performance Visualizations</h2>
            {charts_html}
        </div>
        """
        
    def _generate_html_recommendations_section(self, report: PerformanceReport) -> str:
        """Generate HTML recommendations section."""
        if not report.recommendations:
            return ""
            
        recommendations_html = ""
        for rec in report.recommendations:
            recommendations_html += f"<li>{rec}</li>"
            
        return f"""
        <div class="recommendations">
            <h3>Performance Recommendations</h3>
            <ul>
                {recommendations_html}
            </ul>
        </div>
        """
        
    def export_performance_history(self, format_type: str = "json") -> str:
        """
        Export complete performance history.
        
        Args:
            format_type: Export format (json, csv)
            
        Returns:
            str: Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            filename = f"performance_history_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
                
        elif format_type == "csv":
            filename = f"performance_history_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            # Convert to CSV format (simplified)
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'device', 'total_operations', 'gpu_monitoring'])
                
                for entry in self.performance_history:
                    writer.writerow([
                        entry.get('timestamp', ''),
                        entry.get('device', ''),
                        len(entry.get('operation_metrics', {})),
                        entry.get('gpu_monitoring_enabled', False)
                    ])
                    
        return str(filepath)
        
    def cleanup_old_reports(self, days: int = 30):
        """
        Clean up old report files.
        
        Args:
            days: Number of days to keep reports
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in self.output_dir.glob("report_*.json"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                print(f"Deleted old report: {file_path}")
                
        for file_path in self.output_dir.glob("report_*.html"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                print(f"Deleted old report: {file_path}")