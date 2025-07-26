"""
Performance regression tests for the optimized SOWLv2 pipeline.
Tests performance improvements and prevents performance regressions.
"""
import os
import time
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from sowlv2.data.config import PipelineBaseData
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.optimizations.parallel_processor import ParallelConfig


class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests for the optimized pipeline."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.test_device = "cpu"
        from sowlv2.data.config import PipelineConfig
        
        pipeline_config = PipelineConfig(merged=True, binary=True, overlay=True)
        self.config = PipelineBaseData(
            owl_model="google/owlv2-base-patch16-ensemble",
            sam_model="facebook/sam2.1-hiera-small",
            threshold=0.1,
            fps=30,
            device=self.test_device,
            pipeline_config=pipeline_config
        )
        self.parallel_config = ParallelConfig(max_workers=2)
        
        # Create test images of different sizes
        self.small_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        self.medium_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        self.large_image = Image.fromarray(
            np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        )
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Save test images
        self.small_image_path = os.path.join(self.temp_dir, "small.png")
        self.medium_image_path = os.path.join(self.temp_dir, "medium.png")
        self.large_image_path = os.path.join(self.temp_dir, "large.png")
        
        self.small_image.save(self.small_image_path)
        self.medium_image.save(self.medium_image_path)
        self.large_image.save(self.large_image_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_pipeline(self, model_type="sam2", enable_monitoring=True, opt_level=1):
        """Create a mock pipeline for testing."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            return OptimizedSOWLv2Pipeline(
                config=self.config,
                parallel_config=self.parallel_config,
                segmentation_model_type=model_type,
                enable_performance_monitoring=enable_monitoring,
                optimization_level=opt_level
            )
    
    def test_initialization_performance(self):
        """Test pipeline initialization performance."""
        start_time = time.time()
        
        pipeline = self._create_mock_pipeline()
        
        init_time = time.time() - start_time
        
        # Initialization should be fast (< 1 second for mocked components)
        self.assertLess(init_time, 1.0, "Pipeline initialization took too long")
        
        # Verify all components are initialized
        self.assertIsNotNone(pipeline.resource_manager)
        self.assertIsNotNone(pipeline.error_recovery)
        self.assertIsNotNone(pipeline.performance_collector)
        self.assertIsNotNone(pipeline.content_analyzer)
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization effectiveness."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True, opt_level=2)
        
        # Test memory monitoring
        memory_stats = pipeline.resource_manager.monitor_memory_usage()
        
        # Memory usage should be reasonable
        self.assertGreaterEqual(memory_stats.utilization_percentage, 0)
        self.assertLessEqual(memory_stats.utilization_percentage, 100)
        
        # Test batch size optimization
        batch_config = pipeline.resource_manager.optimize_batch_sizes(50.0)
        
        # Batch sizes should be reasonable
        self.assertGreater(batch_config.detection_batch_size, 0)
        self.assertLessEqual(batch_config.detection_batch_size, 16)
        self.assertGreater(batch_config.segmentation_batch_size, 0)
        self.assertLessEqual(batch_config.segmentation_batch_size, 8)
    
    def test_model_switching_performance(self):
        """Test model switching performance."""
        pipeline = self._create_mock_pipeline(model_type="sam2")
        
        # Measure model switching time
        start_time = time.time()
        
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline.switch_segmentation_model("edgetam", "facebook/edgetam-base")
        
        switch_time = time.time() - start_time
        
        # Model switching should be fast (< 2 seconds for mocked components)
        self.assertLess(switch_time, 2.0, "Model switching took too long")
        
        # Verify switch was successful
        self.assertEqual(pipeline.segmentation_model_type, "edgetam")
    
    def test_performance_monitoring_overhead(self):
        """Test performance monitoring overhead."""
        # Test with monitoring disabled
        pipeline_no_monitoring = self._create_mock_pipeline(enable_monitoring=False)
        
        # Test with monitoring enabled
        pipeline_with_monitoring = self._create_mock_pipeline(enable_monitoring=True)
        
        # Both should initialize successfully
        self.assertIsNone(pipeline_no_monitoring.performance_collector)
        self.assertIsNotNone(pipeline_with_monitoring.performance_collector)
        
        # Performance monitoring should not significantly impact initialization
        # (This is a basic check since we're using mocks)
        self.assertIsNotNone(pipeline_with_monitoring.resource_manager)
    
    def test_optimization_level_performance_impact(self):
        """Test performance impact of different optimization levels."""
        optimization_levels = [1, 2, 3]
        pipelines = []
        
        for level in optimization_levels:
            pipeline = self._create_mock_pipeline(opt_level=level)
            pipelines.append(pipeline)
            
            # Verify optimization level is set
            self.assertEqual(pipeline.optimization_level, level)
        
        # All optimization levels should initialize successfully
        self.assertEqual(len(pipelines), 3)
    
    def test_resource_cleanup_effectiveness(self):
        """Test resource cleanup effectiveness."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True)
        
        # Get initial memory stats
        initial_stats = pipeline.resource_manager.monitor_memory_usage()
        
        # Perform cleanup
        pipeline.resource_manager.cleanup_resources(force=True)
        
        # Get post-cleanup stats
        post_cleanup_stats = pipeline.resource_manager.monitor_memory_usage()
        
        # Cleanup should not increase memory usage
        self.assertLessEqual(
            post_cleanup_stats.utilization_percentage,
            initial_stats.utilization_percentage + 5.0  # Allow small variance
        )
    
    def test_error_recovery_performance(self):
        """Test error recovery performance."""
        pipeline = self._create_mock_pipeline()
        
        # Test retry logic performance
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        start_time = time.time()
        
        result = pipeline.error_recovery.implement_retry_logic(
            operation=failing_operation,
            max_retries=3,
            base_delay=0.01,  # Very short delay for testing
            operation_name="test_operation"
        )
        
        retry_time = time.time() - start_time
        
        # Retry logic should succeed
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
        
        # Should complete reasonably quickly
        self.assertLess(retry_time, 1.0, "Retry logic took too long")
    
    def test_batch_processing_scalability(self):
        """Test batch processing scalability."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Test batch configuration
            batch_config = pipeline.resource_manager.optimize_batch_sizes(
                current_usage=30.0,  # Low usage
                image_size=(512, 512),
                num_prompts=batch_size
            )
            
            # Batch sizes should scale appropriately
            self.assertGreater(batch_config.detection_batch_size, 0)
            self.assertGreater(batch_config.segmentation_batch_size, 0)
            
            # Larger prompts should not cause excessive batch size reduction
            if batch_size <= 4:
                self.assertGreaterEqual(batch_config.detection_batch_size, 1)
    
    def test_streaming_mode_activation_performance(self):
        """Test streaming mode activation performance."""
        pipeline = self._create_mock_pipeline()
        
        # Test streaming decision for different video sizes
        video_sizes = [100, 500, 1000, 2000, 5000]
        
        for video_size in video_sizes:
            start_time = time.time()
            
            should_stream = pipeline.resource_manager.should_enable_streaming(
                video_frames=video_size,
                frame_size=(1024, 1024)
            )
            
            decision_time = time.time() - start_time
            
            # Decision should be fast
            self.assertLess(decision_time, 0.1, "Streaming decision took too long")
            
            # Large videos should enable streaming
            if video_size > 1000:
                self.assertTrue(should_stream, f"Streaming should be enabled for {video_size} frames")
    
    def test_optimization_recommendation_performance(self):
        """Test optimization recommendation generation performance."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True)
        
        start_time = time.time()
        
        recommendations = pipeline.create_optimization_recommendation_system()
        
        recommendation_time = time.time() - start_time
        
        # Recommendation generation should be fast
        self.assertLess(recommendation_time, 1.0, "Recommendation generation took too long")
        
        # Should return valid recommendations
        self.assertIn("system_analysis", recommendations)
        self.assertIn("immediate_actions", recommendations)
        self.assertIn("priority_level", recommendations)
    
    def test_performance_report_generation_speed(self):
        """Test performance report generation speed."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True)
        
        # Add some mock performance data
        mock_metrics = Mock()
        mock_metrics.processing_time = 1.0
        mock_metrics.memory_peak_usage = 2.0
        mock_metrics.gpu_utilization = 50.0
        mock_metrics.throughput_fps = 10.0
        
        pipeline.performance_collector.operation_metrics["test_operation"].append(mock_metrics)
        
        start_time = time.time()
        
        report = pipeline.get_performance_report()
        
        report_time = time.time() - start_time
        
        # Report generation should be fast
        self.assertLess(report_time, 0.5, "Performance report generation took too long")
        
        # Should return valid report
        self.assertIn("pipeline_stats", report)
        self.assertIn("resource_status", report)
        self.assertIn("timestamp", report)
    
    def test_concurrent_operation_performance(self):
        """Test performance under concurrent operations."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True)
        
        # Simulate concurrent memory monitoring
        start_time = time.time()
        
        results = []
        for _ in range(10):
            memory_stats = pipeline.resource_manager.monitor_memory_usage()
            results.append(memory_stats)
        
        concurrent_time = time.time() - start_time
        
        # Concurrent monitoring should be efficient
        self.assertLess(concurrent_time, 1.0, "Concurrent monitoring took too long")
        
        # All results should be valid
        self.assertEqual(len(results), 10)
        for stats in results:
            self.assertGreaterEqual(stats.utilization_percentage, 0)
    
    def test_memory_trend_analysis_performance(self):
        """Test memory trend analysis performance."""
        pipeline = self._create_mock_pipeline(enable_monitoring=True)
        
        # Add mock memory history
        for i in range(50):
            mock_stats = Mock()
            mock_stats.utilization_percentage = 50.0 + i * 0.5
            mock_stats.allocated_memory = 2.0 + i * 0.01
            mock_stats.free_memory = 6.0 - i * 0.01
            pipeline.resource_manager.memory_history.append(mock_stats)
        
        start_time = time.time()
        
        trend = pipeline.resource_manager.get_memory_trend(window_size=20)
        
        trend_time = time.time() - start_time
        
        # Trend analysis should be fast
        self.assertLess(trend_time, 0.1, "Memory trend analysis took too long")
        
        # Should return valid trend data
        self.assertIn("trend", trend)
        self.assertIn("stability", trend)
        self.assertIn("peak_usage", trend)
    
    def test_model_validation_performance(self):
        """Test model validation performance."""
        start_time = time.time()
        
        pipeline = self._create_mock_pipeline()
        validation_result = pipeline.validate_model_switching()
        
        validation_time = time.time() - start_time
        
        # Validation should complete reasonably quickly
        self.assertLess(validation_time, 5.0, "Model validation took too long")
        
        # Should return validation results
        self.assertIn("overall_success", validation_result)
        self.assertIn("errors", validation_result)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for key operations."""
    
    def setUp(self):
        """Set up benchmark environment."""
        self.test_device = "cpu"
        from sowlv2.data.config import PipelineConfig
        
        pipeline_config = PipelineConfig(merged=True, binary=True, overlay=True)
        self.config = PipelineBaseData(
            owl_model="google/owlv2-base-patch16-ensemble",
            sam_model="facebook/sam2.1-hiera-small",
            threshold=0.1,
            fps=30,
            device=self.test_device,
            pipeline_config=pipeline_config
        )
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up benchmark environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization_benchmark(self):
        """Benchmark pipeline initialization time."""
        times = []
        
        for _ in range(5):
            start_time = time.time()
            
            with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
                mock_model = Mock()
                mock_create.return_value = mock_model
                
                pipeline = OptimizedSOWLv2Pipeline(
                    config=self.config,
                    enable_performance_monitoring=True,
                    optimization_level=2
                )
            
            init_time = time.time() - start_time
            times.append(init_time)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"Pipeline initialization - Avg: {avg_time:.3f}s, Max: {max_time:.3f}s")
        
        # Benchmark thresholds
        self.assertLess(avg_time, 0.5, f"Average initialization time too high: {avg_time:.3f}s")
        self.assertLess(max_time, 1.0, f"Maximum initialization time too high: {max_time:.3f}s")
    
    def test_model_switching_benchmark(self):
        """Benchmark model switching time."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_edgetam_model = Mock()
            mock_create.side_effect = [mock_sam2_model, mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=False
            )
            
            # Benchmark switching to EdgeTAM
            start_time = time.time()
            pipeline.switch_segmentation_model("edgetam")
            switch_to_edgetam_time = time.time() - start_time
            
            # Benchmark switching back to SAM2
            start_time = time.time()
            pipeline.switch_segmentation_model("sam2")
            switch_to_sam2_time = time.time() - start_time
            
            avg_switch_time = (switch_to_edgetam_time + switch_to_sam2_time) / 2
            
            print(f"Model switching - EdgeTAM: {switch_to_edgetam_time:.3f}s, "
                  f"SAM2: {switch_to_sam2_time:.3f}s, Avg: {avg_switch_time:.3f}s")
            
            # Benchmark thresholds
            self.assertLess(avg_switch_time, 1.0, f"Average switching time too high: {avg_switch_time:.3f}s")
    
    def test_resource_monitoring_benchmark(self):
        """Benchmark resource monitoring performance."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True
            )
            
            # Benchmark memory monitoring
            times = []
            for _ in range(100):
                start_time = time.time()
                pipeline.resource_manager.monitor_memory_usage()
                monitor_time = time.time() - start_time
                times.append(monitor_time)
            
            avg_monitor_time = sum(times) / len(times)
            max_monitor_time = max(times)
            
            print(f"Memory monitoring - Avg: {avg_monitor_time:.6f}s, Max: {max_monitor_time:.6f}s")
            
            # Benchmark thresholds
            self.assertLess(avg_monitor_time, 0.01, f"Average monitoring time too high: {avg_monitor_time:.6f}s")
            self.assertLess(max_monitor_time, 0.05, f"Maximum monitoring time too high: {max_monitor_time:.6f}s")


if __name__ == '__main__':
    # Run with verbose output to see benchmark results
    unittest.main(verbosity=2)