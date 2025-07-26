"""
Unit tests for EdgeTAM wrapper functionality.
Tests EdgeTAMWrapper class methods, error handling, and performance metrics.
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import torch
import tempfile
import os

from sowlv2.models.edgetam_wrapper import EdgeTAMWrapper, _EDGETAM_MODELS
from sowlv2.utils.pipeline_utils import CUDA, CPU


class TestEdgeTAMWrapper:
    """Test suite for EdgeTAMWrapper class."""
    
    def test_init_valid_model(self):
        """Test EdgeTAMWrapper initialization with valid model."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        assert wrapper.model_name == "facebook/edgetam-base"
        assert wrapper.device == torch.device(CPU)
        assert wrapper._model is not None
        assert "model_loading_time" in wrapper._performance_metrics
        assert wrapper._performance_metrics["model_loading_time"] > 0
    
    def test_init_invalid_model(self):
        """Test EdgeTAMWrapper initialization with invalid model name."""
        with pytest.raises(ValueError) as exc_info:
            EdgeTAMWrapper(model_name="invalid/model", device=CPU)
        
        assert "Unsupported EdgeTAM model" in str(exc_info.value)
        assert "invalid/model" in str(exc_info.value)
        assert "Available models" in str(exc_info.value)
    
    def test_init_cuda_device(self):
        """Test EdgeTAMWrapper initialization with CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-small", device=CUDA)
            assert wrapper.device == torch.device(CUDA)
    
    def test_init_default_parameters(self):
        """Test EdgeTAMWrapper initialization with default parameters."""
        wrapper = EdgeTAMWrapper()
        
        assert wrapper.model_name == "facebook/edgetam-base"
        assert wrapper.device == torch.device(CPU)
        assert wrapper._model is not None
    
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_load_model_success(self, mock_logger):
        """Test successful model loading."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        mock_logger.info.assert_called()
        assert any("Loading EdgeTAM model" in str(call) for call in mock_logger.info.call_args_list)
        assert any("loaded successfully" in str(call) for call in mock_logger.info.call_args_list)
    
    @patch('sowlv2.models.edgetam_wrapper.EdgeTAMWrapper._create_mock_model')
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_load_model_failure(self, mock_logger, mock_create_model):
        """Test model loading failure handling."""
        mock_create_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        assert "EdgeTAM model loading failed" in str(exc_info.value)
        mock_logger.error.assert_called()
    
    def test_segment_valid_input(self):
        """Test segmentation with valid input."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')
        box_xyxy = [50, 50, 150, 150]
        
        mask = wrapper.segment(test_image, box_xyxy)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == (224, 224)
        assert wrapper._performance_metrics["inference_time"] > 0
    
    def test_segment_invalid_box_coordinates(self):
        """Test segmentation with invalid box coordinates."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test with wrong number of coordinates
        with pytest.raises(ValueError):
            wrapper.segment(test_image, [50, 50, 150])  # Only 3 coordinates
        
        # Test with invalid box (x2 <= x1)
        mask = wrapper.segment(test_image, [150, 50, 50, 150])
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == 0)  # Should return empty mask
    
    def test_segment_out_of_bounds_box(self):
        """Test segmentation with out-of-bounds box coordinates."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        test_image = Image.new('RGB', (100, 100), color='red')
        box_xyxy = [-10, -10, 200, 200]  # Extends beyond image bounds
        
        mask = wrapper.segment(test_image, box_xyxy)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)
        # Box should be clipped to image bounds
    
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_segment_processing_error(self, mock_logger):
        """Test segmentation error handling."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Mock model to raise exception
        wrapper._model.predict_mask = Mock(side_effect=Exception("Processing failed"))
        
        test_image = Image.new('RGB', (224, 224), color='red')
        box_xyxy = [50, 50, 150, 150]
        
        mask = wrapper.segment(test_image, box_xyxy)
        
        # Should return empty mask on error
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == 0)
        mock_logger.error.assert_called()
    
    def test_init_state_valid_directory(self):
        """Test video state initialization with valid directory."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            state = wrapper.init_state(temp_dir)
            
            assert isinstance(state, dict)
            assert state["frames_dir"] == temp_dir
            assert state["initialized"] is True
            assert "objects" in state
    
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_init_state_error_handling(self, mock_logger):
        """Test video state initialization error handling."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Test with invalid directory path
        with pytest.raises(RuntimeError):
            wrapper.init_state("/nonexistent/directory")
        
        mock_logger.error.assert_called()
    
    def test_add_new_box_valid_input(self):
        """Test adding new box to video state."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        state = {"initialized": True, "objects": {}}
        frame_idx = 5
        box = [100, 100, 200, 200]
        obj_idx = 1
        
        wrapper.add_new_box(state, frame_idx, box, obj_idx)
        
        assert obj_idx in state["objects"]
        assert state["objects"][obj_idx]["frame_idx"] == frame_idx
        assert state["objects"][obj_idx]["box"] == box
        assert state["objects"][obj_idx]["active"] is True
    
    def test_add_new_box_invalid_state(self):
        """Test adding box with invalid state."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Test with uninitialized state
        invalid_state = {"initialized": False}
        
        with pytest.raises(RuntimeError):
            wrapper.add_new_box(invalid_state, 0, [0, 0, 100, 100], 1)
        
        # Test with None state
        with pytest.raises(RuntimeError):
            wrapper.add_new_box(None, 0, [0, 0, 100, 100], 1)
    
    def test_add_new_box_invalid_box(self):
        """Test adding box with invalid coordinates."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        state = {"initialized": True, "objects": {}}
        
        with pytest.raises(RuntimeError):
            wrapper.add_new_box(state, 0, [100, 100, 200], 1)  # Only 3 coordinates
    
    def test_propagate_in_video_valid_state(self):
        """Test video propagation with valid state."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        state = {
            "initialized": True,
            "objects": {
                1: {"frame_idx": 0, "box": [100, 100, 200, 200], "active": True},
                2: {"frame_idx": 0, "box": [300, 300, 400, 400], "active": True}
            }
        }
        
        results = list(wrapper.propagate_in_video(state))
        
        assert len(results) == 10  # Mock returns 10 frames
        for frame_idx, frame_results in results:
            assert isinstance(frame_idx, int)
            assert isinstance(frame_results, dict)
            assert 1 in frame_results
            assert 2 in frame_results
            assert isinstance(frame_results[1], np.ndarray)
            assert isinstance(frame_results[2], np.ndarray)
    
    def test_propagate_in_video_invalid_state(self):
        """Test video propagation with invalid state."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        invalid_state = {"initialized": False}
        
        with pytest.raises(RuntimeError):
            list(wrapper.propagate_in_video(invalid_state))
    
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_propagate_in_video_error_handling(self, mock_logger):
        """Test video propagation error handling."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Create state that will cause error in propagation
        state = {"initialized": True, "objects": None}  # Invalid objects
        
        with pytest.raises(RuntimeError):
            list(wrapper.propagate_in_video(state))
        
        mock_logger.error.assert_called()
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Perform some operations to populate metrics
        test_image = Image.new('RGB', (224, 224), color='red')
        wrapper.segment(test_image, [50, 50, 150, 150])
        
        metrics = wrapper.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "model_loading_time" in metrics
        assert "inference_time" in metrics
        assert "memory_usage" in metrics
        assert metrics["model_loading_time"] > 0
        assert metrics["inference_time"] > 0
        
        # Ensure it returns a copy, not the original
        metrics["test_key"] = "test_value"
        original_metrics = wrapper.get_performance_metrics()
        assert "test_key" not in original_metrics
    
    @patch('torch.cuda.empty_cache')
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_cleanup_success(self, mock_logger, mock_empty_cache):
        """Test successful resource cleanup."""
        with patch('torch.cuda.is_available', return_value=True):
            wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CUDA)
            
            wrapper.cleanup()
            
            assert wrapper._model is None
            mock_empty_cache.assert_called_once()
            mock_logger.info.assert_called()
    
    @patch('sowlv2.models.edgetam_wrapper.logger')
    def test_cleanup_error_handling(self, mock_logger):
        """Test cleanup error handling."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Mock del to raise exception
        with patch('builtins.delattr', side_effect=Exception("Cleanup failed")):
            wrapper.cleanup()
            
            mock_logger.warning.assert_called()
    
    def test_mock_model_functionality(self):
        """Test the mock model used for simulation."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        mock_model = wrapper._model
        
        # Test mock model methods
        assert hasattr(mock_model, 'to')
        assert hasattr(mock_model, 'predict_mask')
        
        # Test device assignment
        mock_model.to(torch.device(CUDA))
        assert mock_model.device == torch.device(CUDA)
        
        # Test mask prediction
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = mock_model.predict_mask(test_image, [0, 0, 50, 50])
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)
    
    def test_available_models_constant(self):
        """Test that available models constant is properly defined."""
        assert isinstance(_EDGETAM_MODELS, dict)
        assert len(_EDGETAM_MODELS) > 0
        
        for model_name, config in _EDGETAM_MODELS.items():
            assert isinstance(model_name, str)
            assert isinstance(config, dict)
            assert "checkpoint" in config
            assert "config" in config
    
    def test_performance_metrics_initialization(self):
        """Test that performance metrics are properly initialized."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        metrics = wrapper._performance_metrics
        assert "model_loading_time" in metrics
        assert "inference_time" in metrics
        assert "memory_usage" in metrics
        
        # All metrics should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
    
    def test_device_handling(self):
        """Test proper device handling in different scenarios."""
        # Test CPU device
        wrapper_cpu = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        assert wrapper_cpu.device == torch.device(CPU)
        
        # Test CUDA device (mocked)
        with patch('torch.cuda.is_available', return_value=True):
            wrapper_cuda = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CUDA)
            assert wrapper_cuda.device == torch.device(CUDA)
    
    def test_error_recovery_in_segment(self):
        """Test error recovery mechanisms in segment method."""
        wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=CPU)
        
        # Test with corrupted image data
        test_image = Image.new('RGB', (0, 0))  # Empty image
        box_xyxy = [0, 0, 10, 10]
        
        mask = wrapper.segment(test_image, box_xyxy)
        
        # Should handle gracefully and return empty mask
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (0, 0) or np.all(mask == 0)
    
    @pytest.mark.parametrize("model_name", list(_EDGETAM_MODELS.keys()))
    def test_all_available_models(self, model_name):
        """Test initialization with all available EdgeTAM models."""
        wrapper = EdgeTAMWrapper(model_name=model_name, device=CPU)
        
        assert wrapper.model_name == model_name
        assert wrapper._model is not None
        assert wrapper._performance_metrics["model_loading_time"] > 0
    
    @pytest.mark.parametrize("device", [CPU, CUDA])
    def test_device_compatibility(self, device):
        """Test device compatibility for different devices."""
        if device == CUDA:
            with patch('torch.cuda.is_available', return_value=True):
                wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=device)
        else:
            wrapper = EdgeTAMWrapper(model_name="facebook/edgetam-base", device=device)
        
        assert wrapper.device == torch.device(device)
        
        # Test that model operations work on the specified device
        test_image = Image.new('RGB', (224, 224), color='red')
        mask = wrapper.segment(test_image, [50, 50, 150, 150])
        assert isinstance(mask, np.ndarray)