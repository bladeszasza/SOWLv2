"""
Comprehensive integration tests for the optimized SOWLv2 pipeline.
Tests EdgeTAM integration, resource management, performance monitoring, and error recovery.
"""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import torch

from sowlv2.data.config import PipelineBaseData
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.optimizations.parallel_processor import ParallelConfig


class TestOptimizedPipelineIntegration(unittest.TestCase):
    """Integration tests for the optimized pipeline with all new components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_device = "cpu"  # Use CPU for consistent testing
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
        
        # Create test image
        self.test_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
        self.test_image.save(self.test_image_path)
        
        self.test_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization_with_sam2(self):
        """Test pipeline initialization with SAM2 model."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                parallel_config=self.parallel_config,
                segmentation_model_type="sam2",
                segmentation_model_name="facebook/sam2.1-hiera-small",
                enable_performance_monitoring=True,
                optimization_level=1
            )
            
            self.assertEqual(pipeline.segmentation_model_type, "sam2")
            self.assertEqual(pipeline.segmentation_model_name, "facebook/sam2.1-hiera-small")
            self.assertEqual(pipeline.optimization_level, 1)
            self.assertIsNotNone(pipeline.performance_collector)
            self.assertIsNotNone(pipeline.resource_manager)
            self.assertIsNotNone(pipeline.error_recovery)
    
    def test_pipeline_initialization_with_edgetam(self):
        """Test pipeline initialization with EdgeTAM model."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                parallel_config=self.parallel_config,
                segmentation_model_type="edgetam",
                segmentation_model_name="facebook/edgetam-base",
                enable_performance_monitoring=True,
                optimization_level=2
            )
            
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
            self.assertEqual(pipeline.segmentation_model_name, "facebook/edgetam-base")
            self.assertEqual(pipeline.optimization_level, 2)
    
    def test_model_fallback_mechanism(self):
        """Test automatic fallback from EdgeTAM to SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            # First call (EdgeTAM) fails, second call (SAM2 fallback) succeeds
            mock_sam2_model = Mock()
            mock_create.side_effect = [
                Exception("EdgeTAM model not available"),
                mock_sam2_model
            ]
            
            with patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2') as mock_fallback:
                mock_fallback.return_value = mock_sam2_model
                
                pipeline = OptimizedSOWLv2Pipeline(
                    config=self.config,
                    segmentation_model_type="edgetam",
                    enable_performance_monitoring=False
                )
                
                # Should have fallen back to SAM2
                self.assertEqual(pipeline.processing_stats['fallback_operations'], 1)
    
    @patch('sowlv2.optimizations.optimized_pipeline.ParallelDetectionProcessor')
    @patch('sowlv2.optimizations.optimized_pipeline.ParallelSegmentationProcessor')
    def test_image_processing_with_performance_monitoring(self, mock_seg_proc, mock_det_proc):
        """Test image processing with performance monitoring enabled."""
        # Mock processors
        mock_detection_result = Mock()
        mock_detection_result.detections = [
            {
                'core_prompt': 'test object',
                'box': [100, 100, 200, 200],
                'confidence': 0.8
            }
        ]
        
        mock_det_proc.return_value.detect_multiple_prompts_parallel.return_value = [mock_detection_result]
        
        mock_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_seg_proc.return_value.segment_detections_parallel.return_value = [
            (mock_detection_result.detections[0], mock_mask)
        ]
        
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=1
            )
            
            # Mock additional methods
            with patch.object(pipeline, '_filter_outputs_by_flags'), \
                 patch.object(pipeline, '_get_color_for_prompt', return_value=(255, 0, 0)), \
                 patch('sowlv2.image_pipeline.create_and_save_merged_overlay'), \
                 patch('sowlv2.utils.filesystem_utils.remove_empty_folders'):
                
                # Process image
                pipeline.process_image(self.test_image_path, "test object", self.test_output_dir)
                
                # Verify performance monitoring was used
                self.assertGreater(pipeline.processing_stats['total_operations'], 0)
                self.assertGreater(pipeline.processing_stats['successful_operations'], 0)
    
    def test_model_switching_functionality(self):
        """Test runtime model switching."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_edgetam_model = Mock()
            
            # Return different models for different calls
            mock_create.side_effect = [mock_sam2_model, mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=False
            )
            
            # Initial model should be SAM2
            self.assertEqual(pipeline.segmentation_model_type, "sam2")
            
            # Switch to EdgeTAM
            pipeline.switch_segmentation_model("edgetam", "facebook/edgetam-base")
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
            self.assertEqual(pipeline.segmentation_model_name, "facebook/edgetam-base")
            
            # Switch back to SAM2
            pipeline.switch_segmentation_model("sam2", "facebook/sam2.1-hiera-small")
            self.assertEqual(pipeline.segmentation_model_type, "sam2")
            self.assertEqual(pipeline.segmentation_model_name, "facebook/sam2.1-hiera-small")
    
    def test_resource_management_integration(self):
        """Test resource management integration."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=2
            )
            
            # Test memory monitoring
            memory_stats = pipeline.resource_manager.monitor_memory_usage()
            self.assertIsNotNone(memory_stats)
            self.assertGreaterEqual(memory_stats.utilization_percentage, 0)
            
            # Test batch optimization
            batch_config = pipeline.resource_manager.optimize_batch_sizes(50.0)
            self.assertIsNotNone(batch_config)
            self.assertGreater(batch_config.detection_batch_size, 0)
            self.assertGreater(batch_config.segmentation_batch_size, 0)
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=False
            )
            
            # Test model loading error recovery
            recovery_result = pipeline.error_recovery.handle_model_loading_error(
                model_name="test_model",
                error=Exception("Test error"),
                fallback_callback=lambda: Mock()
            )
            
            self.assertIn("success", recovery_result)
            self.assertIn("user_message", recovery_result)
            
            # Test memory overflow handling
            memory_result = pipeline.error_recovery.handle_memory_overflow(
                current_batch_size=4,
                memory_usage_gb=8.0,
                available_memory_gb=2.0
            )
            
            self.assertIn("new_batch_size", memory_result)
            self.assertLessEqual(memory_result["new_batch_size"], 4)
    
    def test_optimization_level_selection(self):
        """Test automatic optimization level selection."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                optimization_level=1
            )
            
            # Test auto-selection for different use cases
            level_realtime = pipeline.auto_select_optimization_level("realtime")
            self.assertEqual(level_realtime, 3)
            
            level_memory = pipeline.auto_select_optimization_level("memory_constrained")
            self.assertEqual(level_memory, 1)
            
            level_batch = pipeline.auto_select_optimization_level("batch")
            self.assertGreaterEqual(level_batch, 2)
    
    def test_performance_comparison(self):
        """Test model performance comparison."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=1
            )
            
            # Mock the process_image method to avoid actual processing
            with patch.object(pipeline, 'process_image') as mock_process:
                mock_process.return_value = None
                
                # Mock performance metrics
                mock_metrics = Mock()
                mock_metrics.processing_time = 1.0
                mock_metrics.memory_peak_usage = 2.0
                mock_metrics.gpu_utilization = 50.0
                mock_metrics.throughput_fps = 10.0
                
                with patch.object(pipeline.performance_collector, 'end_timing', return_value=mock_metrics):
                    comparison_result = pipeline.compare_model_performance(
                        self.test_image_path, "test object"
                    )
                    
                    self.assertIn("comparison", comparison_result)
                    self.assertIn("current_model_metrics", comparison_result)
                    self.assertIn("alternative_model_metrics", comparison_result)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation system."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=1
            )
            
            # Generate recommendations
            recommendations = pipeline.create_optimization_recommendation_system()
            
            self.assertIn("system_analysis", recommendations)
            self.assertIn("immediate_actions", recommendations)
            self.assertIn("configuration_changes", recommendations)
            self.assertIn("model_recommendations", recommendations)
            self.assertIn("resource_optimizations", recommendations)
            self.assertIn("priority_level", recommendations)
    
    def test_model_preloading_and_switching(self):
        """Test model preloading and fast switching."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_edgetam_model = Mock()
            
            # Return different models for different calls
            mock_create.side_effect = [mock_sam2_model, mock_edgetam_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=False
            )
            
            # Preload EdgeTAM model
            pipeline.preload_alternative_model("edgetam", "facebook/edgetam-base")
            
            # Verify model was preloaded
            self.assertTrue(hasattr(pipeline, '_preloaded_models'))
            self.assertIn("edgetam_facebook/edgetam-base", pipeline._preloaded_models)
            
            # Switch to preloaded model
            pipeline.switch_to_preloaded_model("edgetam", "facebook/edgetam-base")
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
    
    def test_model_switching_validation(self):
        """Test model switching validation."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=False
            )
            
            # Mock the process_image method
            with patch.object(pipeline, 'process_image') as mock_process:
                mock_process.return_value = None
                
                validation_result = pipeline.validate_model_switching()
                
                self.assertIn("switch_to_edgetam", validation_result)
                self.assertIn("switch_to_sam2", validation_result)
                self.assertIn("switch_back", validation_result)
                self.assertIn("overall_success", validation_result)
    
    def test_streaming_mode_activation(self):
        """Test streaming mode activation for large videos."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True
            )
            
            # Test streaming mode decision
            should_stream = pipeline.resource_manager.should_enable_streaming(
                video_frames=2000,  # Large number of frames
                frame_size=(1920, 1080)
            )
            
            self.assertTrue(should_stream)
            
            # Test streaming configuration
            streaming_config = pipeline.resource_manager.enable_streaming_mode(2000)
            self.assertIsNotNone(streaming_config)
            self.assertGreater(streaming_config.chunk_size, 0)
            self.assertGreaterEqual(streaming_config.overlap_frames, 0)
    
    def test_content_analysis_integration(self):
        """Test content analysis integration."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True
            )
            
            # Mock content analysis
            mock_analysis = {
                'content_type': 'dynamic',
                'frame_count': 100,
                'frame_size': (1024, 1024),
                'motion_level': 'medium'
            }
            
            with patch.object(pipeline.content_analyzer, 'analyze_video_content', return_value=mock_analysis):
                # Test that content analysis is used in video processing decision
                with patch.object(pipeline, '_process_video_optimized_standard') as mock_standard:
                    mock_standard.return_value = None
                    
                    pipeline.process_video("dummy_video.mp4", "test object", self.test_output_dir)
                    
                    # Verify content analysis was called
                    pipeline.content_analyzer.analyze_video_content.assert_called_once()
    
    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=2
            )
            
            # Generate performance report
            report = pipeline.get_performance_report()
            
            self.assertIn("pipeline_stats", report)
            self.assertIn("resource_status", report)
            self.assertIn("model_info", report)
            self.assertIn("optimization_config", report)
            self.assertIn("timestamp", report)
            
            # Verify model info
            self.assertEqual(report["model_info"]["segmentation_model"]["type"], pipeline.segmentation_model_type)
            self.assertEqual(report["optimization_config"]["optimization_level"], 2)


class TestPipelineStressTests(unittest.TestCase):
    """Stress tests for resource management and error recovery."""
    
    def setUp(self):
        """Set up stress test environment."""
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
        self.test_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up stress test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_high_memory_usage_handling(self):
        """Test pipeline behavior under high memory usage."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=1
            )
            
            # Simulate high memory usage
            with patch.object(pipeline.resource_manager, 'monitor_memory_usage') as mock_memory:
                mock_stats = Mock()
                mock_stats.utilization_percentage = 95.0  # Very high usage
                mock_stats.allocated_memory = 7.5
                mock_stats.free_memory = 0.5
                mock_stats.total_memory = 8.0
                mock_memory.return_value = mock_stats
                
                # Test batch size optimization under high memory
                batch_config = pipeline.resource_manager.optimize_batch_sizes(95.0)
                
                # Should use CPU fallback mode
                from sowlv2.optimizations.resource_manager import ProcessingMode
                self.assertEqual(batch_config.processing_mode, ProcessingMode.CPU_FALLBACK)
                self.assertEqual(batch_config.detection_batch_size, 1)
                self.assertEqual(batch_config.segmentation_batch_size, 1)
    
    def test_multiple_error_recovery_scenarios(self):
        """Test multiple consecutive error recovery scenarios."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True
            )
            
            # Test multiple memory overflow scenarios
            for i in range(3):
                recovery_result = pipeline.error_recovery.handle_memory_overflow(
                    current_batch_size=4 - i,
                    memory_usage_gb=8.0 + i,
                    available_memory_gb=2.0 - i * 0.5
                )
                
                self.assertIn("new_batch_size", recovery_result)
                self.assertLessEqual(recovery_result["new_batch_size"], 4 - i)
    
    def test_rapid_model_switching(self):
        """Test rapid model switching for stability."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_edgetam_model = Mock()
            
            # Alternate between models
            mock_create.side_effect = [mock_sam2_model, mock_edgetam_model] * 10
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=False
            )
            
            # Rapidly switch between models
            for i in range(5):
                target_type = "edgetam" if i % 2 == 0 else "sam2"
                pipeline.switch_segmentation_model(target_type)
                self.assertEqual(pipeline.segmentation_model_type, target_type)
    
    def test_optimization_effectiveness_monitoring(self):
        """Test optimization effectiveness monitoring over time."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                enable_performance_monitoring=True,
                optimization_level=2
            )
            
            # Simulate multiple operations with varying performance
            mock_metrics = []
            for i in range(10):
                metrics = Mock()
                metrics.processing_time = 1.0 + i * 0.1  # Gradually increasing time
                metrics.memory_peak_usage = 2.0 + i * 0.05
                metrics.gpu_utilization = 50.0 - i * 2
                metrics.throughput_fps = 10.0 - i * 0.5
                mock_metrics.append(metrics)
            
            # Add metrics to performance collector
            for i, metrics in enumerate(mock_metrics):
                pipeline.performance_collector.operation_metrics["test_operation"].append(metrics)
            
            # Monitor effectiveness
            effectiveness = pipeline.monitor_optimization_effectiveness(window_size=5)
            
            self.assertIn("optimization_level", effectiveness)
            self.assertIn("resource_utilization", effectiveness)
            self.assertIn("performance_stability", effectiveness)
            self.assertIn("recommendations", effectiveness)
            self.assertIn("overall_effectiveness_score", effectiveness)


if __name__ == '__main__':
    unittest.main()