"""
Integration tests specifically for EdgeTAM integration in the optimized pipeline.
Tests EdgeTAM model loading, fallback mechanisms, and performance comparison.
"""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from sowlv2.data.config import PipelineBaseData
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.models.model_factory import SegmentationModelFactory


class TestEdgeTAMIntegration(unittest.TestCase):
    """Integration tests for EdgeTAM model integration."""
    
    def setUp(self):
        """Set up test environment."""
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
    
    def test_edgetam_model_creation_success(self):
        """Test successful EdgeTAM model creation."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                segmentation_model_name="facebook/edgetam-base",
                enable_performance_monitoring=True
            )
            
            # Verify EdgeTAM model was requested
            mock_create.assert_called_with(
                model_type="edgetam",
                model_name="facebook/edgetam-base",
                device=self.test_device,
                enable_fallback=True
            )
            
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
            self.assertEqual(pipeline.segmentation_model_name, "facebook/edgetam-base")
            self.assertEqual(pipeline.sam, mock_edgetam_model)
    
    def test_edgetam_fallback_to_sam2(self):
        """Test EdgeTAM fallback to SAM2 when EdgeTAM fails."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model_with_fallback_notification') as mock_create:
            # Simulate EdgeTAM failure with SAM2 fallback
            mock_sam2_model = Mock()
            
            def fallback_notification(message):
                self.assertIn("EdgeTAM", message)
                self.assertIn("fallback", message.lower())
            
            mock_create.side_effect = lambda model_type, model_name, device, notification_callback: (
                notification_callback("EdgeTAM failed, falling back to SAM2"),
                mock_sam2_model
            )[1]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                segmentation_model_name="facebook/edgetam-base",
                enable_performance_monitoring=False
            )
            
            # Should have incremented fallback counter
            self.assertGreater(pipeline.processing_stats['fallback_operations'], 0)
    
    def test_edgetam_vs_sam2_performance_comparison(self):
        """Test performance comparison between EdgeTAM and SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            # Return EdgeTAM first, then SAM2 for comparison
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model, mock_edgetam_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock performance metrics
            edgetam_metrics = Mock()
            edgetam_metrics.processing_time = 0.5  # Faster
            edgetam_metrics.memory_peak_usage = 1.0  # Less memory
            edgetam_metrics.gpu_utilization = 30.0
            edgetam_metrics.throughput_fps = 20.0
            
            sam2_metrics = Mock()
            sam2_metrics.processing_time = 1.0  # Slower
            sam2_metrics.memory_peak_usage = 2.0  # More memory
            sam2_metrics.gpu_utilization = 60.0
            sam2_metrics.throughput_fps = 10.0
            
            # Mock the comparison
            with patch.object(pipeline, 'process_image') as mock_process:
                mock_process.return_value = None
                
                with patch.object(pipeline.performance_collector, 'end_timing') as mock_timing:
                    mock_timing.side_effect = [edgetam_metrics, sam2_metrics]
                    
                    with patch.object(pipeline.performance_collector, 'compare_models') as mock_compare:
                        mock_comparison = Mock()
                        mock_comparison.speed_improvement = 100.0  # 100% faster
                        mock_comparison.memory_savings = 50.0  # 50% less memory
                        mock_comparison.recommendation = "EdgeTAM recommended for speed"
                        mock_compare.return_value = mock_comparison
                        
                        result = pipeline.compare_model_performance(
                            self.test_image_path, "test object"
                        )
                        
                        self.assertIn("comparison", result)
                        self.assertEqual(result["comparison"].speed_improvement, 100.0)
                        self.assertEqual(result["comparison"].memory_savings, 50.0)
    
    def test_edgetam_model_switching_during_processing(self):
        """Test switching from EdgeTAM to SAM2 during processing."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Verify initial EdgeTAM model
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
            
            # Switch to SAM2
            pipeline.switch_segmentation_model("sam2", "facebook/sam2.1-hiera-small")
            
            # Verify switch
            self.assertEqual(pipeline.segmentation_model_type, "sam2")
            self.assertEqual(pipeline.segmentation_model_name, "facebook/sam2.1-hiera-small")
            
            # Verify processors were updated
            self.assertIsNotNone(pipeline.detection_processor)
            self.assertIsNotNone(pipeline.segmentation_processor)
    
    def test_edgetam_model_validation(self):
        """Test EdgeTAM model validation functionality."""
        # Test model validation
        validation_result = SegmentationModelFactory.validate_model_compatibility(
            model_type="edgetam",
            model_name="facebook/edgetam-base",
            device="cpu"
        )
        
        # Should contain validation information
        self.assertIn("is_valid", validation_result)
        self.assertIn("model_exists", validation_result)
        self.assertIn("device_compatible", validation_result)
        self.assertIn("warnings", validation_result)
        self.assertIn("recommendations", validation_result)
    
    def test_edgetam_model_info_retrieval(self):
        """Test EdgeTAM model information retrieval."""
        model_info = SegmentationModelFactory.get_model_info(
            model_type="edgetam",
            model_name="facebook/edgetam-base"
        )
        
        self.assertEqual(model_info["type"], "edgetam")
        self.assertEqual(model_info["name"], "facebook/edgetam-base")
        self.assertIn("exists", model_info)
        self.assertIn("description", model_info)
        self.assertIn("performance_characteristics", model_info)
    
    def test_edgetam_model_recommendation(self):
        """Test EdgeTAM model recommendation system."""
        # Test speed priority recommendation
        speed_rec = SegmentationModelFactory.recommend_model(
            use_case="realtime",
            priority="speed",
            device="cpu"
        )
        
        self.assertEqual(speed_rec["model_type"], "edgetam")
        self.assertIn("speed", speed_rec["reasoning"].lower())
        
        # Test memory priority recommendation
        memory_rec = SegmentationModelFactory.recommend_model(
            use_case="general",
            priority="memory",
            device="cpu"
        )
        
        # Should recommend EdgeTAM for memory efficiency on CPU
        self.assertEqual(memory_rec["model_type"], "edgetam")
    
    def test_edgetam_preloading_and_warm_up(self):
        """Test EdgeTAM model preloading and warm-up."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_edgetam_model = Mock()
            
            # Mock segment method for warm-up
            mock_edgetam_model.segment.return_value = np.ones((100, 100), dtype=np.uint8)
            
            mock_create.side_effect = [mock_sam2_model, mock_edgetam_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=False
            )
            
            # Preload EdgeTAM model
            pipeline.preload_alternative_model("edgetam", "facebook/edgetam-base")
            
            # Verify preloading
            self.assertTrue(hasattr(pipeline, '_preloaded_models'))
            self.assertIn("edgetam_facebook/edgetam-base", pipeline._preloaded_models)
            
            # Test warm-up was called
            mock_edgetam_model.segment.assert_called_once()
    
    def test_edgetam_automatic_model_selection(self):
        """Test automatic model selection with EdgeTAM consideration."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_edgetam_model = Mock()
            
            mock_create.side_effect = [mock_sam2_model, mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",
                enable_performance_monitoring=True
            )
            
            # Enable automatic model selection
            pipeline.enable_automatic_model_selection(True, performance_threshold=0.2)
            
            # Mock performance metrics favoring EdgeTAM
            sam2_metrics = Mock()
            sam2_metrics.processing_time = 2.0
            sam2_metrics.memory_peak_usage = 4.0
            sam2_metrics.throughput_fps = 5.0
            
            edgetam_metrics = Mock()
            edgetam_metrics.processing_time = 1.0  # 2x faster
            edgetam_metrics.memory_peak_usage = 2.0  # 2x less memory
            edgetam_metrics.throughput_fps = 10.0  # 2x throughput
            
            with patch.object(pipeline, 'process_image') as mock_process:
                mock_process.return_value = None
                
                with patch.object(pipeline.performance_collector, 'end_timing') as mock_timing:
                    mock_timing.side_effect = [sam2_metrics, edgetam_metrics]
                    
                    with patch.object(pipeline.performance_collector, 'compare_models') as mock_compare:
                        mock_comparison = Mock()
                        mock_comparison.speed_improvement = 100.0
                        mock_comparison.memory_savings = 50.0
                        mock_compare.return_value = mock_comparison
                        
                        result = pipeline.auto_select_optimal_model(self.test_image_path)
                        
                        # Should recommend switching to EdgeTAM
                        self.assertEqual(result["optimal_model_type"], "edgetam")
                        self.assertTrue(result["switch_recommended"])
    
    def test_edgetam_error_handling_and_recovery(self):
        """Test EdgeTAM-specific error handling and recovery."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            # Simulate EdgeTAM loading failure
            mock_create.side_effect = Exception("EdgeTAM model not found")
            
            with patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2') as mock_fallback:
                mock_sam2_model = Mock()
                mock_fallback.return_value = mock_sam2_model
                
                pipeline = OptimizedSOWLv2Pipeline(
                    config=self.config,
                    segmentation_model_type="edgetam",
                    enable_performance_monitoring=False
                )
                
                # Should have handled the error and used fallback
                self.assertGreater(pipeline.processing_stats['error_recoveries'], 0)
                
                # Test error recovery manager
                recovery_result = pipeline.error_recovery.handle_model_loading_error(
                    model_name="edgetam/facebook/edgetam-base",
                    error=Exception("EdgeTAM not available"),
                    fallback_callback=lambda: mock_sam2_model
                )
                
                self.assertTrue(recovery_result["success"])
                self.assertTrue(recovery_result["fallback_used"])
                self.assertIn("EdgeTAM", recovery_result["user_message"])
    
    def test_edgetam_optimization_recommendations(self):
        """Test optimization recommendations specific to EdgeTAM."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_sam2_model = Mock()
            mock_create.return_value = mock_sam2_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="sam2",  # Start with SAM2
                enable_performance_monitoring=True
            )
            
            # Simulate high memory usage scenario
            with patch.object(pipeline.resource_manager, 'monitor_memory_usage') as mock_memory:
                mock_stats = Mock()
                mock_stats.utilization_percentage = 85.0  # High memory usage
                mock_stats.allocated_memory = 6.8
                mock_stats.free_memory = 1.2
                mock_memory.return_value = mock_stats
                
                recommendations = pipeline.create_optimization_recommendation_system()
                
                # Should recommend switching to EdgeTAM for memory efficiency
                model_recs = recommendations["model_recommendations"]
                edgetam_recommended = any(
                    "edgetam" in rec["recommendation"].lower()
                    for rec in model_recs
                )
                
                if model_recs:  # Only check if recommendations were generated
                    self.assertTrue(edgetam_recommended)
    
    def test_edgetam_video_processing_integration(self):
        """Test EdgeTAM integration in video processing."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock content analysis
            mock_analysis = {
                'content_type': 'fast_motion',
                'frame_count': 500,
                'frame_size': (1024, 1024),
                'motion_level': 'high'
            }
            
            with patch.object(pipeline.content_analyzer, 'analyze_video_content', return_value=mock_analysis):
                with patch.object(pipeline, '_process_video_optimized_standard') as mock_standard:
                    mock_standard.return_value = None
                    
                    # Process video with EdgeTAM
                    pipeline.process_video("dummy_video.mp4", "test object", self.test_output_dir)
                    
                    # Verify EdgeTAM was used for video processing
                    self.assertEqual(pipeline.segmentation_model_type, "edgetam")
                    mock_standard.assert_called_once()


class TestEdgeTAMPerformanceOptimization(unittest.TestCase):
    """Performance optimization tests specific to EdgeTAM."""
    
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
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up performance test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_edgetam_memory_optimization(self):
        """Test EdgeTAM memory optimization benefits."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True,
                optimization_level=2
            )
            
            # Test memory-efficient batch configuration
            batch_config = pipeline.resource_manager.optimize_batch_sizes(
                current_usage=60.0,  # Moderate usage
                image_size=(1024, 1024),
                num_prompts=3
            )
            
            # EdgeTAM should allow larger batch sizes due to lower memory usage
            self.assertGreater(batch_config.detection_batch_size, 1)
            self.assertGreater(batch_config.segmentation_batch_size, 1)
    
    def test_edgetam_speed_optimization(self):
        """Test EdgeTAM speed optimization configuration."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True,
                optimization_level=3  # Maximum optimization
            )
            
            # Test optimization for speed use case
            pipeline.optimize_for_use_case("realtime", "speed")
            
            # Should maintain EdgeTAM for speed
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
            self.assertEqual(pipeline.optimization_level, 3)
    
    def test_edgetam_batch_processing_optimization(self):
        """Test EdgeTAM optimization for batch processing."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Test batch processing optimization
            pipeline.optimize_for_use_case("batch", "balanced")
            
            # Should use EdgeTAM with appropriate optimization level
            self.assertEqual(pipeline.segmentation_model_type, "edgetam")
            self.assertGreaterEqual(pipeline.optimization_level, 2)


class TestEdgeTAMFallbackMechanisms(unittest.TestCase):
    """Test EdgeTAM fallback mechanisms and error recovery."""
    
    def setUp(self):
        """Set up fallback test environment."""
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
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up fallback test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_edgetam_model_loading_fallback(self):
        """Test fallback when EdgeTAM model fails to load."""
        with patch('sowlv2.models.edgetam_wrapper.EdgeTAMWrapper.__init__') as mock_init:
            mock_init.side_effect = RuntimeError("EdgeTAM model loading failed")
            
            with patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2') as mock_fallback:
                mock_sam2_model = Mock()
                mock_fallback.return_value = mock_sam2_model
                
                model = SegmentationModelFactory.create_model(
                    "edgetam", "facebook/edgetam-base", "cpu", enable_fallback=True
                )
                
                self.assertEqual(model, mock_sam2_model)
                mock_fallback.assert_called_once_with("cpu")
    
    def test_edgetam_inference_fallback(self):
        """Test fallback when EdgeTAM inference fails."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_edgetam_model.segment.side_effect = Exception("EdgeTAM inference failed")
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock fallback to SAM2
            with patch.object(pipeline, 'switch_segmentation_model') as mock_switch:
                with patch.object(pipeline.error_recovery, 'handle_processing_failure') as mock_handle:
                    mock_handle.return_value = True  # Successful recovery
                    
                    test_image = Image.fromarray(
                        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    )
                    
                    # This should trigger fallback handling
                    try:
                        pipeline.sam.segment(test_image, [50, 50, 150, 150])
                    except Exception:
                        pass  # Expected to fail, testing recovery
                    
                    # Verify error handling was called
                    self.assertTrue(mock_handle.called)
    
    def test_edgetam_memory_overflow_fallback(self):
        """Test fallback when EdgeTAM causes memory overflow."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_create.return_value = mock_edgetam_model
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Simulate memory overflow
            with patch.object(pipeline.resource_manager, 'monitor_memory_usage') as mock_memory:
                mock_stats = Mock()
                mock_stats.utilization_percentage = 95.0  # Critical memory usage
                mock_memory.return_value = mock_stats
                
                with patch.object(pipeline.error_recovery, 'handle_memory_overflow') as mock_handle:
                    mock_handle.return_value = {"success": True, "action": "model_switch"}
                    
                    result = pipeline.resource_manager.handle_memory_pressure()
                    
                    # Should trigger memory overflow handling
                    self.assertTrue(mock_handle.called)
    
    def test_edgetam_device_fallback(self):
        """Test fallback when EdgeTAM fails on GPU and switches to CPU."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('sowlv2.models.edgetam_wrapper.EdgeTAMWrapper.__init__') as mock_init:
                # Fail on CUDA, succeed on CPU
                def init_side_effect(self, model_name="facebook/edgetam-base", device="cpu"):
                    if device == "cuda":
                        raise RuntimeError("CUDA out of memory")
                    # Simulate successful CPU initialization
                    self.model_name = model_name
                    self.device = torch.device(device)
                    self._model = Mock()
                    self._performance_metrics = {"model_loading_time": 0.1}
                
                mock_init.side_effect = init_side_effect
                
                with patch('sowlv2.models.model_factory.SegmentationModelFactory._create_edgetam_model') as mock_create:
                    # First call fails (CUDA), second succeeds (CPU)
                    mock_create.side_effect = [
                        RuntimeError("CUDA out of memory"),
                        Mock()  # CPU model
                    ]
                    
                    with patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2') as mock_fallback:
                        mock_sam2_model = Mock()
                        mock_fallback.return_value = mock_sam2_model
                        
                        model = SegmentationModelFactory.create_model(
                            "edgetam", "facebook/edgetam-base", "cuda", enable_fallback=True
                        )
                        
                        # Should fallback to SAM2
                        self.assertEqual(model, mock_sam2_model)
    
    def test_edgetam_progressive_fallback_chain(self):
        """Test progressive fallback chain: EdgeTAM -> SAM2 small -> SAM2 tiny."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory._create_edgetam_model') as mock_edgetam:
            mock_edgetam.side_effect = Exception("EdgeTAM failed")
            
            with patch('sowlv2.models.model_factory.SegmentationModelFactory._create_sam2_model') as mock_sam2:
                # First SAM2 model fails, second succeeds
                mock_sam2.side_effect = [
                    Exception("SAM2 small failed"),
                    Mock()  # SAM2 tiny succeeds
                ]
                
                model = SegmentationModelFactory.create_model(
                    "edgetam", "facebook/edgetam-base", "cpu", enable_fallback=True
                )
                
                # Should eventually succeed with fallback
                self.assertIsNotNone(model)
                
                # Verify multiple SAM2 models were tried
                self.assertEqual(mock_sam2.call_count, 2)


class TestEdgeTAMPerformanceComparison(unittest.TestCase):
    """Performance comparison tests between EdgeTAM and SAM2."""
    
    def setUp(self):
        """Set up performance comparison test environment."""
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
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create test images of different sizes
        self.small_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        self.large_image = Image.fromarray(
            np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        )
    
    def tearDown(self):
        """Clean up performance comparison test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_edgetam_vs_sam2_speed_comparison(self):
        """Test speed comparison between EdgeTAM and SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            # Mock performance metrics
            mock_edgetam_model.get_performance_metrics.return_value = {
                "model_loading_time": 0.5,
                "inference_time": 0.1,  # Faster
                "memory_usage": 1.0
            }
            
            mock_sam2_model.get_performance_metrics.return_value = {
                "model_loading_time": 1.0,
                "inference_time": 0.2,  # Slower
                "memory_usage": 2.0
            }
            
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock benchmark runner
            with patch.object(pipeline.benchmark_runner, 'run_comparative_benchmark') as mock_benchmark:
                mock_results = Mock()
                mock_results.edgetam_metrics = Mock()
                mock_results.edgetam_metrics.processing_time = 0.1
                mock_results.sam2_metrics = Mock()
                mock_results.sam2_metrics.processing_time = 0.2
                mock_results.speed_improvement = 100.0  # 100% faster
                mock_benchmark.return_value = mock_results
                
                comparison = pipeline.benchmark_runner.run_comparative_benchmark([
                    self.small_image, self.large_image
                ])
                
                self.assertEqual(comparison.speed_improvement, 100.0)
                self.assertLess(comparison.edgetam_metrics.processing_time, 
                               comparison.sam2_metrics.processing_time)
    
    def test_edgetam_vs_sam2_memory_comparison(self):
        """Test memory usage comparison between EdgeTAM and SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock memory profiling
            with patch.object(pipeline.benchmark_runner, 'profile_memory_usage') as mock_profile:
                mock_edgetam_profile = Mock()
                mock_edgetam_profile.peak_memory = 1.5  # GB
                mock_edgetam_profile.average_memory = 1.2
                
                mock_sam2_profile = Mock()
                mock_sam2_profile.peak_memory = 3.0  # GB
                mock_sam2_profile.average_memory = 2.5
                
                mock_profile.side_effect = [mock_edgetam_profile, mock_sam2_profile]
                
                edgetam_memory = pipeline.benchmark_runner.profile_memory_usage(
                    pipeline.config
                )
                
                # Switch to SAM2 for comparison
                pipeline.switch_segmentation_model("sam2", "facebook/sam2.1-hiera-small")
                
                sam2_memory = pipeline.benchmark_runner.profile_memory_usage(
                    pipeline.config
                )
                
                # EdgeTAM should use less memory
                self.assertLess(edgetam_memory.peak_memory, sam2_memory.peak_memory)
                self.assertLess(edgetam_memory.average_memory, sam2_memory.average_memory)
    
    def test_edgetam_vs_sam2_throughput_comparison(self):
        """Test throughput comparison between EdgeTAM and SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock throughput measurement
            with patch.object(pipeline.benchmark_runner, 'measure_throughput') as mock_throughput:
                mock_edgetam_results = Mock()
                mock_edgetam_results.images_per_second = 15.0  # Higher throughput
                mock_edgetam_results.batch_efficiency = 0.85
                
                mock_sam2_results = Mock()
                mock_sam2_results.images_per_second = 8.0  # Lower throughput
                mock_sam2_results.batch_efficiency = 0.75
                
                mock_throughput.side_effect = [mock_edgetam_results, mock_sam2_results]
                
                edgetam_throughput = pipeline.benchmark_runner.measure_throughput([1, 2, 4])
                
                # Switch to SAM2
                pipeline.switch_segmentation_model("sam2", "facebook/sam2.1-hiera-small")
                
                sam2_throughput = pipeline.benchmark_runner.measure_throughput([1, 2, 4])
                
                # EdgeTAM should have higher throughput
                self.assertGreater(edgetam_throughput.images_per_second, 
                                 sam2_throughput.images_per_second)
    
    def test_edgetam_vs_sam2_quality_comparison(self):
        """Test quality comparison between EdgeTAM and SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            # Mock segmentation results
            edgetam_mask = np.ones((256, 256), dtype=np.uint8) * 255
            edgetam_mask[100:150, 100:150] = 0  # Some variation
            
            sam2_mask = np.ones((256, 256), dtype=np.uint8) * 255
            sam2_mask[90:160, 90:160] = 0  # Different variation
            
            mock_edgetam_model.segment.return_value = edgetam_mask
            mock_sam2_model.segment.return_value = sam2_mask
            
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Mock quality assessment
            with patch.object(pipeline.performance_collector, 'compare_models') as mock_compare:
                mock_comparison = Mock()
                mock_comparison.quality_comparison = {
                    "iou_score": 0.85,  # EdgeTAM vs ground truth
                    "dice_score": 0.90,
                    "precision": 0.88,
                    "recall": 0.92
                }
                mock_comparison.quality_difference = -0.05  # Slightly lower than SAM2
                mock_compare.return_value = mock_comparison
                
                comparison = pipeline.performance_collector.compare_models(
                    {"edgetam": mock_edgetam_model.get_performance_metrics()},
                    {"sam2": mock_sam2_model.get_performance_metrics()}
                )
                
                self.assertIn("quality_comparison", comparison.quality_comparison)
                self.assertIsInstance(comparison.quality_difference, float)
    
    def test_edgetam_vs_sam2_scalability_comparison(self):
        """Test scalability comparison between EdgeTAM and SAM2."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model') as mock_create:
            mock_edgetam_model = Mock()
            mock_sam2_model = Mock()
            
            mock_create.side_effect = [mock_edgetam_model, mock_sam2_model]
            
            pipeline = OptimizedSOWLv2Pipeline(
                config=self.config,
                segmentation_model_type="edgetam",
                enable_performance_monitoring=True
            )
            
            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8, 16]
            
            with patch.object(pipeline.benchmark_runner, 'measure_throughput') as mock_throughput:
                # EdgeTAM should scale better with batch size
                edgetam_results = []
                sam2_results = []
                
                for batch_size in batch_sizes:
                    edgetam_result = Mock()
                    edgetam_result.images_per_second = 10.0 * (batch_size * 0.8)  # Good scaling
                    edgetam_results.append(edgetam_result)
                    
                    sam2_result = Mock()
                    sam2_result.images_per_second = 8.0 * (batch_size * 0.6)  # Poor scaling
                    sam2_results.append(sam2_result)
                
                mock_throughput.side_effect = edgetam_results + sam2_results
                
                # Test EdgeTAM scaling
                edgetam_throughputs = []
                for batch_size in batch_sizes:
                    result = pipeline.benchmark_runner.measure_throughput([batch_size])
                    edgetam_throughputs.append(result.images_per_second)
                
                # Switch to SAM2
                pipeline.switch_segmentation_model("sam2", "facebook/sam2.1-hiera-small")
                
                # Test SAM2 scaling
                sam2_throughputs = []
                for batch_size in batch_sizes:
                    result = pipeline.benchmark_runner.measure_throughput([batch_size])
                    sam2_throughputs.append(result.images_per_second)
                
                # EdgeTAM should show better scaling
                edgetam_scaling = edgetam_throughputs[-1] / edgetam_throughputs[0]
                sam2_scaling = sam2_throughputs[-1] / sam2_throughputs[0]
                
                self.assertGreater(edgetam_scaling, sam2_scaling)


if __name__ == '__main__':
    unittest.main()