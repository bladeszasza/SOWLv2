"""
Unit tests for SegmentationModelFactory.
Tests model creation, validation, fallback mechanisms, and recommendations.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from sowlv2.models.model_factory import SegmentationModelFactory
from sowlv2.utils.pipeline_utils import CUDA, CPU


class TestSegmentationModelFactory:
    """Test suite for SegmentationModelFactory class."""
    
    def test_supported_model_types(self):
        """Test that supported model types are correctly defined."""
        expected_types = ["sam2", "edgetam"]
        assert SegmentationModelFactory.SUPPORTED_MODEL_TYPES == expected_types
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_sam2_model')
    def test_create_sam2_model_success(self, mock_create_sam2):
        """Test successful SAM2 model creation."""
        mock_model = Mock()
        mock_create_sam2.return_value = mock_model
        
        result = SegmentationModelFactory.create_model(
            "sam2", "facebook/sam2.1-hiera-small", CPU
        )
        
        assert result == mock_model
        mock_create_sam2.assert_called_once_with("facebook/sam2.1-hiera-small", CPU)
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_edgetam_model')
    def test_create_edgetam_model_success(self, mock_create_edgetam):
        """Test successful EdgeTAM model creation."""
        mock_model = Mock()
        mock_create_edgetam.return_value = mock_model
        
        result = SegmentationModelFactory.create_model(
            "edgetam", "facebook/edgetam-base", CPU
        )
        
        assert result == mock_model
        mock_create_edgetam.assert_called_once_with("facebook/edgetam-base", CPU)
    
    def test_create_model_invalid_type(self):
        """Test model creation with invalid model type."""
        with pytest.raises(ValueError) as exc_info:
            SegmentationModelFactory.create_model(
                "invalid_type", "some_model", CPU
            )
        
        assert "Unsupported model type: invalid_type" in str(exc_info.value)
        assert "Supported types:" in str(exc_info.value)
    
    @patch('sowlv2.models.model_factory.logger')
    def test_create_model_invalid_device_warning(self, mock_logger):
        """Test warning for invalid device specification."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory._create_sam2_model'):
            SegmentationModelFactory.create_model(
                "sam2", "facebook/sam2.1-hiera-small", "invalid_device"
            )
        
        mock_logger.warning.assert_called_with(
            "Unknown device 'invalid_device', defaulting to CPU"
        )
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_edgetam_model')
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2')
    def test_edgetam_fallback_enabled(self, mock_fallback, mock_create_edgetam):
        """Test EdgeTAM fallback when model creation fails."""
        mock_create_edgetam.side_effect = Exception("EdgeTAM failed")
        mock_fallback_model = Mock()
        mock_fallback.return_value = mock_fallback_model
        
        result = SegmentationModelFactory.create_model(
            "edgetam", "facebook/edgetam-base", CPU, enable_fallback=True
        )
        
        assert result == mock_fallback_model
        mock_fallback.assert_called_once_with(CPU)
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_edgetam_model')
    def test_edgetam_fallback_disabled(self, mock_create_edgetam):
        """Test EdgeTAM failure without fallback."""
        mock_create_edgetam.side_effect = Exception("EdgeTAM failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            SegmentationModelFactory.create_model(
                "edgetam", "facebook/edgetam-base", CPU, enable_fallback=False
            )
        
        assert "Model creation failed" in str(exc_info.value)
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_sam2_model')
    def test_sam2_failure_no_fallback(self, mock_create_sam2):
        """Test SAM2 failure (no fallback available)."""
        mock_create_sam2.side_effect = Exception("SAM2 failed")
        
        with pytest.raises(Exception) as exc_info:
            SegmentationModelFactory.create_model(
                "sam2", "facebook/sam2.1-hiera-small", CPU
            )
        
        assert "SAM2 failed" in str(exc_info.value)
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_sam2_model')
    @patch('sowlv2.models.model_factory.logger')
    def test_fallback_to_sam2_success(self, mock_logger, mock_create_sam2):
        """Test successful fallback to SAM2."""
        mock_model = Mock()
        mock_create_sam2.return_value = mock_model
        
        result = SegmentationModelFactory._fallback_to_sam2(CPU)
        
        assert result == mock_model
        mock_logger.info.assert_called()
        assert any("Successfully created SAM2 fallback" in str(call) 
                  for call in mock_logger.info.call_args_list)
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._create_sam2_model')
    @patch('sowlv2.models.model_factory.logger')
    def test_fallback_to_sam2_all_fail(self, mock_logger, mock_create_sam2):
        """Test fallback failure when all SAM2 models fail."""
        mock_create_sam2.side_effect = Exception("All SAM2 models failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            SegmentationModelFactory._fallback_to_sam2(CPU)
        
        assert "All SAM2 fallback models failed to load" in str(exc_info.value)
        mock_logger.warning.assert_called()
    
    @patch('sowlv2.models.model_factory._get_sam2_models')
    @patch('sowlv2.models.sam2_wrapper.SAM2Wrapper')
    def test_create_sam2_model_success(self, mock_sam2_wrapper, mock_get_sam2_models):
        """Test successful SAM2 model creation."""
        mock_get_sam2_models.return_value = {
            "facebook/sam2.1-hiera-small": ("checkpoint", "config", "video_config")
        }
        mock_instance = Mock()
        mock_sam2_wrapper.return_value = mock_instance
        
        result = SegmentationModelFactory._create_sam2_model(
            "facebook/sam2.1-hiera-small", CPU
        )
        
        assert result == mock_instance
        mock_sam2_wrapper.assert_called_once_with(
            model_name="facebook/sam2.1-hiera-small", device=CPU
        )
    
    @patch('sowlv2.models.model_factory._get_sam2_models')
    def test_create_sam2_model_invalid_name(self, mock_get_sam2_models):
        """Test SAM2 model creation with invalid model name."""
        mock_get_sam2_models.return_value = {
            "facebook/sam2.1-hiera-small": ("checkpoint", "config", "video_config")
        }
        
        with pytest.raises(ValueError) as exc_info:
            SegmentationModelFactory._create_sam2_model("invalid/model", CPU)
        
        assert "Unsupported SAM2 model: invalid/model" in str(exc_info.value)
        assert "Available SAM2 models:" in str(exc_info.value)
    
    def test_create_sam2_model_import_error(self):
        """Test SAM2 model creation with import error."""
        with patch('sowlv2.models.model_factory._get_sam2_models', return_value={}):
            with patch('builtins.__import__', side_effect=ImportError("SAM2 not available")):
                with pytest.raises(RuntimeError) as exc_info:
                    SegmentationModelFactory._create_sam2_model(
                        "facebook/sam2.1-hiera-small", CPU
                    )
                
                assert "SAM2 dependencies not available" in str(exc_info.value)
    
    @patch('sowlv2.models.model_factory._get_edgetam_models')
    @patch('sowlv2.models.edgetam_wrapper.EdgeTAMWrapper')
    def test_create_edgetam_model_success(self, mock_edgetam_wrapper, mock_get_edgetam_models):
        """Test successful EdgeTAM model creation."""
        mock_get_edgetam_models.return_value = {
            "facebook/edgetam-base": {"checkpoint": "model.pt", "config": "config.yaml"}
        }
        mock_instance = Mock()
        mock_edgetam_wrapper.return_value = mock_instance
        
        result = SegmentationModelFactory._create_edgetam_model(
            "facebook/edgetam-base", CPU
        )
        
        assert result == mock_instance
        mock_edgetam_wrapper.assert_called_once_with(
            model_name="facebook/edgetam-base", device=CPU
        )
    
    @patch('sowlv2.models.model_factory._get_edgetam_models')
    def test_create_edgetam_model_invalid_name(self, mock_get_edgetam_models):
        """Test EdgeTAM model creation with invalid model name."""
        mock_get_edgetam_models.return_value = {
            "facebook/edgetam-base": {"checkpoint": "model.pt", "config": "config.yaml"}
        }
        
        with pytest.raises(ValueError) as exc_info:
            SegmentationModelFactory._create_edgetam_model("invalid/model", CPU)
        
        assert "Unsupported EdgeTAM model: invalid/model" in str(exc_info.value)
        assert "Available EdgeTAM models:" in str(exc_info.value)
    
    def test_create_edgetam_model_import_error(self):
        """Test EdgeTAM model creation with import error."""
        with patch('sowlv2.models.model_factory._get_edgetam_models', return_value={}):
            with patch('builtins.__import__', side_effect=ImportError("EdgeTAM not available")):
                with pytest.raises(RuntimeError) as exc_info:
                    SegmentationModelFactory._create_edgetam_model(
                        "facebook/edgetam-base", CPU
                    )
                
                assert "EdgeTAM dependencies not available" in str(exc_info.value)
    
    @patch('sowlv2.models.model_factory._get_sam2_models')
    @patch('sowlv2.models.model_factory._get_edgetam_models')
    def test_get_available_models(self, mock_get_edgetam_models, mock_get_sam2_models):
        """Test getting available models."""
        mock_get_sam2_models.return_value = {
            "facebook/sam2.1-hiera-small": ("checkpoint", "config", "video_config"),
            "facebook/sam2.1-hiera-base": ("checkpoint", "config", "video_config")
        }
        mock_get_edgetam_models.return_value = {
            "facebook/edgetam-base": {"checkpoint": "model.pt", "config": "config.yaml"},
            "facebook/edgetam-small": {"checkpoint": "model.pt", "config": "config.yaml"}
        }
        
        result = SegmentationModelFactory.get_available_models()
        
        expected = {
            "sam2": ["facebook/sam2.1-hiera-small", "facebook/sam2.1-hiera-base"],
            "edgetam": ["facebook/edgetam-base", "facebook/edgetam-small"]
        }
        assert result == expected
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.get_available_models')
    def test_validate_model_compatibility_valid(self, mock_get_available_models):
        """Test model compatibility validation for valid model."""
        mock_get_available_models.return_value = {
            "sam2": ["facebook/sam2.1-hiera-small"],
            "edgetam": ["facebook/edgetam-base"]
        }
        
        with patch('torch.cuda.is_available', return_value=True):
            result = SegmentationModelFactory.validate_model_compatibility(
                "sam2", "facebook/sam2.1-hiera-small", CUDA
            )
        
        assert result["is_valid"] is True
        assert result["model_exists"] is True
        assert result["device_compatible"] is True
        assert len(result["warnings"]) == 0
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.get_available_models')
    def test_validate_model_compatibility_invalid_type(self, mock_get_available_models):
        """Test model compatibility validation for invalid type."""
        result = SegmentationModelFactory.validate_model_compatibility(
            "invalid_type", "some_model", CPU
        )
        
        assert result["is_valid"] is False
        assert "Unsupported model type" in result["warnings"][0]
        assert "Use one of:" in result["recommendations"][0]
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.get_available_models')
    def test_validate_model_compatibility_invalid_model(self, mock_get_available_models):
        """Test model compatibility validation for invalid model name."""
        mock_get_available_models.return_value = {
            "sam2": ["facebook/sam2.1-hiera-small"],
            "edgetam": ["facebook/edgetam-base"]
        }
        
        result = SegmentationModelFactory.validate_model_compatibility(
            "sam2", "invalid/model", CPU
        )
        
        assert result["is_valid"] is False
        assert result["model_exists"] is False
        assert "Model 'invalid/model' not found" in result["warnings"][0]
        assert "Available sam2 models:" in result["recommendations"][0]
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.get_available_models')
    def test_validate_model_compatibility_cuda_unavailable(self, mock_get_available_models):
        """Test model compatibility validation when CUDA is unavailable."""
        mock_get_available_models.return_value = {
            "sam2": ["facebook/sam2.1-hiera-small"],
            "edgetam": ["facebook/edgetam-base"]
        }
        
        with patch('torch.cuda.is_available', return_value=False):
            result = SegmentationModelFactory.validate_model_compatibility(
                "sam2", "facebook/sam2.1-hiera-small", CUDA
            )
        
        assert result["is_valid"] is False
        assert result["device_compatible"] is False
        assert "CUDA requested but not available" in result["warnings"][0]
        assert "Use CPU device or install CUDA support" in result["recommendations"][0]
    
    @patch('sowlv2.models.model_factory._get_sam2_models')
    @patch('sowlv2.models.model_factory._get_edgetam_models')
    def test_get_model_info_sam2(self, mock_get_edgetam_models, mock_get_sam2_models):
        """Test getting SAM2 model information."""
        mock_get_sam2_models.return_value = {
            "facebook/sam2.1-hiera-small": ("checkpoint", "config", "video_config")
        }
        mock_get_edgetam_models.return_value = {}
        
        result = SegmentationModelFactory.get_model_info(
            "sam2", "facebook/sam2.1-hiera-small"
        )
        
        assert result["type"] == "sam2"
        assert result["name"] == "facebook/sam2.1-hiera-small"
        assert result["exists"] is True
        assert "checkpoint" in result["config"]
        assert "SAM2" in result["description"]
        assert result["performance_characteristics"]["accuracy"] == "high"
    
    @patch('sowlv2.models.model_factory._get_sam2_models')
    @patch('sowlv2.models.model_factory._get_edgetam_models')
    def test_get_model_info_edgetam(self, mock_get_edgetam_models, mock_get_sam2_models):
        """Test getting EdgeTAM model information."""
        mock_get_sam2_models.return_value = {}
        mock_get_edgetam_models.return_value = {
            "facebook/edgetam-base": {"checkpoint": "model.pt", "config": "config.yaml"}
        }
        
        result = SegmentationModelFactory.get_model_info(
            "edgetam", "facebook/edgetam-base"
        )
        
        assert result["type"] == "edgetam"
        assert result["name"] == "facebook/edgetam-base"
        assert result["exists"] is True
        assert "checkpoint" in result["config"]
        assert "EdgeTAM" in result["description"]
        assert result["performance_characteristics"]["speed"] == "high"
    
    def test_get_model_info_nonexistent(self):
        """Test getting information for non-existent model."""
        with patch('sowlv2.models.model_factory._get_sam2_models', return_value={}):
            with patch('sowlv2.models.model_factory._get_edgetam_models', return_value={}):
                result = SegmentationModelFactory.get_model_info(
                    "sam2", "nonexistent/model"
                )
        
        assert result["exists"] is False
        assert result["config"] == {}
    
    def test_recommend_model_speed_priority(self):
        """Test model recommendation with speed priority."""
        result = SegmentationModelFactory.recommend_model(
            use_case="general", priority="speed", device=CPU
        )
        
        assert result["model_type"] == "edgetam"
        assert "speed" in result["reasoning"].lower()
    
    def test_recommend_model_accuracy_priority(self):
        """Test model recommendation with accuracy priority."""
        result = SegmentationModelFactory.recommend_model(
            use_case="general", priority="accuracy", device=CPU
        )
        
        assert result["model_type"] == "sam2"
        assert "accuracy" in result["reasoning"].lower()
    
    def test_recommend_model_memory_priority_cpu(self):
        """Test model recommendation with memory priority on CPU."""
        result = SegmentationModelFactory.recommend_model(
            use_case="general", priority="memory", device=CPU
        )
        
        assert result["model_type"] == "edgetam"
        assert "memory" in result["reasoning"].lower()
    
    def test_recommend_model_memory_priority_cuda(self):
        """Test model recommendation with memory priority on CUDA."""
        result = SegmentationModelFactory.recommend_model(
            use_case="general", priority="memory", device=CUDA
        )
        
        assert result["model_type"] == "sam2"
        assert "tiny" in result["model_name"]
        assert "memory" in result["reasoning"].lower()
    
    def test_recommend_model_video_use_case(self):
        """Test model recommendation for video use case."""
        result = SegmentationModelFactory.recommend_model(
            use_case="video", priority="balanced", device=CPU
        )
        
        assert result["model_type"] == "sam2"
        assert "video" in result["reasoning"].lower()
    
    def test_recommend_model_realtime_use_case(self):
        """Test model recommendation for real-time use case."""
        result = SegmentationModelFactory.recommend_model(
            use_case="realtime", priority="balanced", device=CPU
        )
        
        assert result["model_type"] == "edgetam"
        assert "real-time" in result["reasoning"].lower()
    
    def test_recommend_model_batch_use_case_cuda(self):
        """Test model recommendation for batch processing on CUDA."""
        result = SegmentationModelFactory.recommend_model(
            use_case="batch", priority="balanced", device=CUDA
        )
        
        assert result["model_type"] == "sam2"
        assert "batch" in result["reasoning"].lower()
    
    def test_recommend_model_batch_use_case_cpu(self):
        """Test model recommendation for batch processing on CPU."""
        result = SegmentationModelFactory.recommend_model(
            use_case="batch", priority="balanced", device=CPU
        )
        
        assert result["model_type"] == "edgetam"
        assert "batch" in result["reasoning"].lower()
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model')
    def test_create_model_with_fallback_notification_success(self, mock_create_model):
        """Test successful model creation with notification callback."""
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        result = SegmentationModelFactory.create_model_with_fallback_notification(
            "sam2", "facebook/sam2.1-hiera-small", CPU
        )
        
        assert result == mock_model
        mock_create_model.assert_called_once_with(
            "sam2", "facebook/sam2.1-hiera-small", CPU, enable_fallback=False
        )
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model')
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2')
    def test_create_model_with_fallback_notification_edgetam_fallback(
        self, mock_fallback, mock_create_model
    ):
        """Test EdgeTAM fallback with notification callback."""
        mock_create_model.side_effect = Exception("EdgeTAM failed")
        mock_fallback_model = Mock()
        mock_fallback.return_value = mock_fallback_model
        
        notification_messages = []
        def notification_callback(message):
            notification_messages.append(message)
        
        result = SegmentationModelFactory.create_model_with_fallback_notification(
            "edgetam", "facebook/edgetam-base", CPU, notification_callback
        )
        
        assert result == mock_fallback_model
        assert len(notification_messages) == 1
        assert "EdgeTAM model" in notification_messages[0]
        assert "Falling back to SAM2" in notification_messages[0]
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model')
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2')
    @patch('builtins.print')
    def test_create_model_with_fallback_notification_no_callback(
        self, mock_print, mock_fallback, mock_create_model
    ):
        """Test EdgeTAM fallback without notification callback."""
        mock_create_model.side_effect = Exception("EdgeTAM failed")
        mock_fallback_model = Mock()
        mock_fallback.return_value = mock_fallback_model
        
        result = SegmentationModelFactory.create_model_with_fallback_notification(
            "edgetam", "facebook/edgetam-base", CPU
        )
        
        assert result == mock_fallback_model
        mock_print.assert_called_once()
        assert "WARNING:" in mock_print.call_args[0][0]
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model')
    @patch('sowlv2.models.model_factory.SegmentationModelFactory._fallback_to_sam2')
    def test_create_model_with_fallback_notification_both_fail(
        self, mock_fallback, mock_create_model
    ):
        """Test when both EdgeTAM and SAM2 fallback fail."""
        mock_create_model.side_effect = Exception("EdgeTAM failed")
        mock_fallback.side_effect = Exception("SAM2 fallback failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            SegmentationModelFactory.create_model_with_fallback_notification(
                "edgetam", "facebook/edgetam-base", CPU
            )
        
        assert "Both EdgeTAM and SAM2 fallback failed" in str(exc_info.value)
    
    @patch('sowlv2.models.model_factory.SegmentationModelFactory.create_model')
    def test_create_model_with_fallback_notification_sam2_fail(self, mock_create_model):
        """Test SAM2 failure (no fallback available)."""
        mock_create_model.side_effect = Exception("SAM2 failed")
        
        with pytest.raises(Exception) as exc_info:
            SegmentationModelFactory.create_model_with_fallback_notification(
                "sam2", "facebook/sam2.1-hiera-small", CPU
            )
        
        assert "SAM2 failed" in str(exc_info.value)
    
    @pytest.mark.parametrize("model_type,expected_recommendation", [
        ("edgetam", "EdgeTAM provides faster inference"),
        ("sam2", "SAM2 provides higher accuracy")
    ])
    def test_validate_model_compatibility_recommendations(
        self, model_type, expected_recommendation
    ):
        """Test that validation provides appropriate recommendations."""
        with patch('sowlv2.models.model_factory.SegmentationModelFactory.get_available_models') as mock_get_models:
            mock_get_models.return_value = {
                "sam2": ["facebook/sam2.1-hiera-small"],
                "edgetam": ["facebook/edgetam-base"]
            }
            
            result = SegmentationModelFactory.validate_model_compatibility(
                model_type, 
                "facebook/edgetam-base" if model_type == "edgetam" else "facebook/sam2.1-hiera-small",
                CPU
            )
        
        assert result["is_valid"] is True
        assert any(expected_recommendation in rec for rec in result["recommendations"])