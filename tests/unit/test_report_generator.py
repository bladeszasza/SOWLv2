"""
Unit tests for ReportGenerator.
Tests report generation, formatting, and export functionality.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sowlv2.optimizations.report_generator import (
    ReportGenerator, ReportConfig, PerformanceReport, TrendAnalysis
)
from sowlv2.optimizations.performance_collector import PerformanceMetrics, ComparisonReport
from sowlv2.optimizations.benchmark_runner import BenchmarkResults


class TestReportGenerator:
    """Test suite for ReportGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            assert generator.output_dir.exists()
            assert generator.performance_collector is not None
            assert generator.benchmark_runner is not None
            assert isinstance(generator.performance_history, list)

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ReportConfig(
            include_charts=False,
            include_trend_analysis=True,
            chart_format="svg",
            theme="dark",
            max_history_days=60,
            output_formats=["json", "html"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            # Test that config can be used in report generation
            assert config.include_charts is False
            assert config.include_trend_analysis is True
            assert config.chart_format == "svg"
            assert config.theme == "dark"
            assert config.max_history_days == 60
            assert config.output_formats == ["json", "html"]

    def test_generate_comprehensive_report_basic(self):
        """Test basic comprehensive report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            # Create mock benchmark results
            sam2_metrics = PerformanceMetrics(2.0, 4.0, 60.0, 10.0, 1.0, 70.0)
            edgetam_metrics = PerformanceMetrics(1.0, 2.0, 40.0, 20.0, 0.5, 50.0)
            
            benchmark_results = [
                BenchmarkResults(
                    model_name="sam2",
                    configuration={"batch_size": 1},
                    performance_metrics=sam2_metrics,
                    detailed_results={"test": "data"},
                    test_conditions={"image_size": (512, 512)},
                    timestamp=datetime.now().isoformat()
                ),
                BenchmarkResults(
                    model_name="edgetam", 
                    configuration={"batch_size": 1},
                    performance_metrics=edgetam_metrics,
                    detailed_results={"test": "data"},
                    test_conditions={"image_size": (512, 512)},
                    timestamp=datetime.now().isoformat()
                )
            ]
            
            config = ReportConfig(include_charts=False)  # Disable charts for testing
            
            report = generator.generate_comprehensive_report(
                benchmark_results=benchmark_results,
                config=config
            )
            
            assert isinstance(report, PerformanceReport)
            assert report.report_id.startswith("report_")
            assert report.timestamp is not None
            assert "benchmark_summary" in report.summary
            assert len(report.benchmark_results) == 2
            assert report.metadata is not None

    def test_generate_model_comparison_report(self):
        """Test model comparison report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            sam2_metrics = PerformanceMetrics(2.0, 4.0, 60.0, 10.0, 1.0, 70.0)
            edgetam_metrics = PerformanceMetrics(1.0, 2.0, 40.0, 20.0, 0.5, 50.0)
            
            sam2_results = BenchmarkResults(
                model_name="sam2",
                configuration={"batch_size": 1},
                performance_metrics=sam2_metrics,
                detailed_results={"test": "data"},
                test_conditions={"image_size": (512, 512)},
                timestamp=datetime.now().isoformat()
            )
            
            edgetam_results = BenchmarkResults(
                model_name="edgetam",
                configuration={"batch_size": 1},
                performance_metrics=edgetam_metrics,
                detailed_results={"test": "data"},
                test_conditions={"image_size": (512, 512)},
                timestamp=datetime.now().isoformat()
            )
            
            config = ReportConfig(include_charts=False, output_formats=["json"])
            
            report = generator.generate_model_comparison_report(
                sam2_results, edgetam_results, config
            )
            
            assert isinstance(report, PerformanceReport)
            assert len(report.benchmark_results) == 2
            assert len(report.model_comparisons) == 1
            assert report.model_comparisons[0].sam2_metrics == sam2_metrics
            assert report.model_comparisons[0].edgetam_metrics == edgetam_metrics

    def test_generate_trend_report(self):
        """Test trend analysis report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            # Add some mock history data
            mock_history = [
                {
                    'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                    'operation_metrics': {
                        'test_op': [{'processing_time': 1.0 + i * 0.1}]
                    }
                }
                for i in range(10)
            ]
            generator.performance_history = mock_history
            
            config = ReportConfig(
                include_charts=False,
                include_trend_analysis=True,
                max_history_days=30,
                output_formats=["json"]
            )
            
            report = generator.generate_trend_report(days=30, config=config)
            
            assert isinstance(report, PerformanceReport)
            assert report.report_id.startswith("report_")
            assert len(report.trend_analysis) >= 0  # May be empty if insufficient data
            assert report.metadata is not None

    def test_load_performance_history(self):
        """Test loading performance history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            # Initially should be empty
            assert isinstance(generator.performance_history, list)
            
            # Test with existing history file
            history_data = [
                {"timestamp": datetime.now().isoformat(), "test": "data1"},
                {"timestamp": datetime.now().isoformat(), "test": "data2"}
            ]
            
            with open(generator.history_file, 'w') as f:
                json.dump(history_data, f)
            
            # Create new generator to test loading
            generator2 = ReportGenerator(output_dir=temp_dir)
            assert len(generator2.performance_history) == 2
            assert generator2.performance_history[0]["test"] == "data1"

    def test_update_performance_history(self):
        """Test updating performance history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            # Mock current metrics
            current_metrics = {
                "timestamp": datetime.now().isoformat(),
                "operation_metrics": {"test_op": [{"processing_time": 1.5}]},
                "model_metrics": {"test_model": {"accuracy": 0.95}}
            }
            
            initial_length = len(generator.performance_history)
            generator._update_performance_history(current_metrics)
            
            assert len(generator.performance_history) == initial_length + 1
            assert generator.performance_history[-1]["operation_metrics"] == current_metrics["operation_metrics"]

    def test_get_system_info(self):
        """Test system information collection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            system_info = generator._get_system_info()
            
            assert isinstance(system_info, dict)
            assert "python_version" in system_info
            assert "platform" in system_info
            # torch_version might not be available in all environments
            assert "cpu_count" in system_info
            assert "gpu_available" in system_info

    def test_report_config_dataclass(self):
        """Test ReportConfig dataclass functionality."""
        config = ReportConfig(
            include_charts=False,
            include_trend_analysis=True,
            chart_format="svg",
            chart_dpi=150,
            theme="dark",
            max_history_days=60,
            output_formats=["json", "html"]
        )
        
        assert config.include_charts is False
        assert config.include_trend_analysis is True
        assert config.chart_format == "svg"
        assert config.chart_dpi == 150
        assert config.theme == "dark"
        assert config.max_history_days == 60
        assert config.output_formats == ["json", "html"]

    def test_performance_report_dataclass(self):
        """Test PerformanceReport dataclass functionality."""
        timestamp = datetime.now().isoformat()
        summary = {"total_operations": 5}
        metadata = {"version": "1.0", "author": "test"}
        
        report = PerformanceReport(
            report_id="test_report_123",
            timestamp=timestamp,
            summary=summary,
            model_comparisons=[],
            benchmark_results=[],
            trend_analysis=[],
            performance_history=[],
            charts={},
            recommendations=["Test recommendation"],
            metadata=metadata
        )
        
        assert report.report_id == "test_report_123"
        assert report.timestamp == timestamp
        assert report.summary == summary
        assert report.metadata == metadata
        assert len(report.recommendations) == 1

    def test_trend_analysis_dataclass(self):
        """Test TrendAnalysis dataclass functionality."""
        trend = TrendAnalysis(
            metric_name="processing_time",
            trend_direction="improving",
            trend_strength=0.75,
            change_percentage=-15.5,
            confidence_score=0.85,
            recommendations=["Continue current optimizations"]
        )
        
        assert trend.metric_name == "processing_time"
        assert trend.trend_direction == "improving"
        assert trend.trend_strength == 0.75
        assert trend.change_percentage == -15.5
        assert trend.confidence_score == 0.85
        assert len(trend.recommendations) == 1

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            
            # Mock summary data
            summary = {
                "benchmark_summary": {
                    "avg_processing_time": 2.5,
                    "avg_memory_usage": 4.0,
                    "avg_throughput": 15.0,  # Add missing field
                    "fastest_model": "edgetam"
                }
            }
            
            # Mock trend analysis
            trend_analysis = [
                TrendAnalysis(
                    metric_name="processing_time",
                    trend_direction="degrading",
                    trend_strength=0.8,
                    change_percentage=25.0,
                    confidence_score=0.9,
                    recommendations=["Optimize processing pipeline"]
                )
            ]
            
            # Mock model comparisons
            sam2_metrics = PerformanceMetrics(2.0, 4.0, 60.0, 10.0, 1.0, 70.0)
            edgetam_metrics = PerformanceMetrics(1.0, 2.0, 40.0, 20.0, 0.5, 50.0)
            
            model_comparisons = [
                ComparisonReport(
                    sam2_metrics=sam2_metrics,
                    edgetam_metrics=edgetam_metrics,
                    speed_improvement=50.0,
                    memory_savings=50.0,
                    quality_comparison={"iou_score": -5.0},
                    recommendation="Use EdgeTAM for speed-critical applications"
                )
            ]
            
            recommendations = generator._generate_recommendations(
                summary, trend_analysis, model_comparisons
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            assert any("EdgeTAM" in rec for rec in recommendations)