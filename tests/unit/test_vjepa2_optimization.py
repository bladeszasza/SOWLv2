"""
Unit tests for V-JEPA2 optimization functionality.
Tests enhanced importance scoring, content analysis, and batch processing optimization.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from PIL import Image

from sowlv2.optimizations.vjepa2_optimization import (
    VJepa2VideoOptimizer, ContentType
)
from sowlv2.data.config import PipelineBaseData, PipelineConfig


class TestVJepa2VideoOptimizer:
    """Test suite for VJepa2VideoOptimizer class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        
        optimizer = VJepa2VideoOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.model_name == "facebook/vjepa2-vitl-fpc16-256-ssv2"
        assert optimizer.frames_per_clip == 16
        assert optimizer.device == "cpu"
        assert optimizer._model is None
        assert optimizer._processor is None
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cuda",
            pipeline_config=PipelineConfig()
        )
        
        optimizer = VJepa2VideoOptimizer(
            config,
            model_name="custom/vjepa2-model",
            frames_per_clip=8,
            device="cpu"
        )
        
        assert optimizer.model_name == "custom/vjepa2-model"
        assert optimizer.frames_per_clip == 8
        assert optimizer.device == "cpu"  # Override config device
    
    def test_load_models_success(self):
        """Test successful model loading."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('sowlv2.optimizations.vjepa2_optimization.AutoModelForVideoClassification') as mock_model_class:
            with patch('sowlv2.optimizations.vjepa2_optimization.AutoVideoProcessor') as mock_processor_class:
                mock_model_class.from_pretrained.return_value.to.return_value = mock_model
                mock_processor_class.from_pretrained.return_value = mock_processor
                
                optimizer._load_models()
                
                assert optimizer._model == mock_model
                assert optimizer._processor == mock_processor
                mock_model.eval.assert_called_once()
    
    def test_load_models_import_error(self):
        """Test model loading with import error."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        with patch('builtins.__import__', side_effect=ImportError("transformers not available")):
            with pytest.raises(ImportError) as exc_info:
                optimizer._load_models()
            
            assert "transformers library required" in str(exc_info.value)
    
    def test_load_models_general_error(self):
        """Test model loading with general error."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        with patch('sowlv2.optimizations.vjepa2_optimization.AutoModelForVideoClassification') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model loading failed")
            
            optimizer._load_models()
            
            assert optimizer._model is None
            assert optimizer._processor is None
    
    def test_is_available_true(self):
        """Test is_available property when model is available."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        with patch.object(optimizer, '_load_models'):
            optimizer._model = Mock()
            
            assert optimizer.is_available is True
    
    def test_is_available_false(self):
        """Test is_available property when model is not available."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        with patch.object(optimizer, '_load_models', side_effect=Exception("Failed")):
            assert optimizer.is_available is False
    
    def test_extract_video_features_success(self):
        """Test successful video feature extraction."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        # Create test frames
        frames = [Image.new('RGB', (224, 224), color=(i*50, 0, 0)) for i in range(3)]
        
        mock_model = Mock()
        mock_processor = Mock()
        mock_outputs = Mock()
        mock_features = torch.randn(1, 16, 768)  # Mock feature tensor
        mock_outputs.last_hidden_state = mock_features
        
        optimizer._model = mock_model
        optimizer._processor = mock_processor
        
        mock_processor.return_value = {"input_ids": torch.randn(1, 16, 3, 224, 224)}
        mock_model.return_value = mock_outputs
        
        with patch('torch.no_grad'):
            features = optimizer.extract_video_features(frames)
        
        assert features is not None
        assert torch.equal(features, mock_features)
        mock_processor.assert_called_once()
        mock_model.assert_called_once()
    
    def test_extract_video_features_unavailable(self):
        """Test video feature extraction when model is unavailable."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (224, 224)) for _ in range(3)]
        
        with patch.object(optimizer, 'is_available', False):
            features = optimizer.extract_video_features(frames)
        
        assert features is None
    
    def test_get_temporal_importance_scores_success(self):
        """Test successful temporal importance scoring."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (224, 224)) for _ in range(5)]
        mock_features = torch.randn(1, 5, 768)
        
        with patch.object(optimizer, 'extract_video_features', return_value=mock_features):
            scores = optimizer.get_temporal_importance_scores(frames)
        
        assert scores is not None
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)
    
    def test_get_temporal_importance_scores_unavailable(self):
        """Test temporal importance scoring when features unavailable."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (224, 224)) for _ in range(5)]
        
        with patch.object(optimizer, 'extract_video_features', return_value=None):
            scores = optimizer.get_temporal_importance_scores(frames)
        
        assert scores is None
    
    def test_analyze_content_type_static(self):
        """Test content type analysis for static content."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        # Create static frames (very similar)
        frames = [Image.new('RGB', (100, 100), color=(100, 100, 100)) for _ in range(5)]
        
        with patch('cv2.calcOpticalFlowPyrLK') as mock_flow:
            mock_flow.return_value = (np.array([[0.1, 0.1], [0.1, 0.1]]), None)
            
            content_type = optimizer.analyze_content_type(frames)
        
        assert content_type == ContentType.STATIC
    
    def test_analyze_content_type_fast_motion(self):
        """Test content type analysis for fast motion content."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        # Create frames with different colors (simulating motion)
        frames = [Image.new('RGB', (100, 100), color=(i*50, 0, 0)) for i in range(5)]
        
        with patch('cv2.calcOpticalFlowPyrLK') as mock_flow:
            # High motion vectors
            mock_flow.return_value = (np.array([[15.0, 15.0], [20.0, 10.0]]), None)
            
            content_type = optimizer.analyze_content_type(frames)
        
        assert content_type == ContentType.FAST_MOTION
    
    def test_analyze_content_type_insufficient_frames(self):
        """Test content type analysis with insufficient frames."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (100, 100))]  # Only one frame
        
        content_type = optimizer.analyze_content_type(frames)
        
        assert content_type == ContentType.STATIC
    
    def test_get_adaptive_scoring_weights(self):
        """Test adaptive scoring weights for different content types."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        # Test all content types
        for content_type in ContentType:
            weights = optimizer.get_adaptive_scoring_weights(content_type)
            
            assert isinstance(weights, dict)
            assert 'feature_weight' in weights
            assert 'motion_weight' in weights
            assert 'edge_weight' in weights
            assert 'temporal_consistency_weight' in weights
            
            # Weights should be positive
            assert all(w >= 0 for w in weights.values())
    
    def test_calculate_advanced_motion_scores(self):
        """Test advanced motion score calculation."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (100, 100), color=(i*30, 0, 0)) for i in range(4)]
        
        with patch('cv2.calcOpticalFlowPyrLK') as mock_flow:
            mock_flow.return_value = (np.array([[5.0, 5.0], [3.0, 7.0]]), None)
            
            motion_scores = optimizer.calculate_advanced_motion_scores(frames)
        
        assert len(motion_scores) == 4
        assert motion_scores[0] == 0.0  # First frame has no motion
        assert all(score >= 0 for score in motion_scores)
    
    def test_calculate_temporal_consistency_scores(self):
        """Test temporal consistency score calculation."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (50, 50), color=(i*20, 0, 0)) for i in range(6)]
        
        consistency_scores = optimizer.calculate_temporal_consistency_scores(frames)
        
        assert len(consistency_scores) == 6
        assert all(0 <= score <= 1 for score in consistency_scores)
    
    def test_get_motion_aware_importance_scores_success(self):
        """Test motion-aware importance scoring."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (100, 100), color=(i*30, 0, 0)) for i in range(5)]
        
        with patch.object(optimizer, 'get_temporal_importance_scores') as mock_temporal:
            mock_temporal.return_value = [0.8, 0.6, 0.9, 0.4, 0.7]
            
            with patch.object(optimizer, 'analyze_content_type') as mock_analyze:
                mock_analyze.return_value = ContentType.DYNAMIC
                
                with patch.object(optimizer, 'calculate_advanced_motion_scores') as mock_motion:
                    mock_motion.return_value = [0.0, 5.0, 8.0, 3.0, 6.0]
                    
                    with patch.object(optimizer, 'calculate_temporal_consistency_scores') as mock_consistency:
                        mock_consistency.return_value = [1.0, 0.8, 0.9, 0.7, 0.8]
                        
                        with patch('cv2.Canny') as mock_canny:
                            mock_canny.return_value = np.ones((100, 100), dtype=np.uint8) * 255
                            
                            scores = optimizer.get_motion_aware_importance_scores(frames)
        
        assert scores is not None
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)
    
    def test_get_motion_aware_importance_scores_unavailable(self):
        """Test motion-aware importance scoring when features unavailable."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (100, 100)) for _ in range(5)]
        
        with patch.object(optimizer, 'get_temporal_importance_scores', return_value=None):
            scores = optimizer.get_motion_aware_importance_scores(frames)
        
        assert scores is None
    
    def test_get_adaptive_frame_spacing_static(self):
        """Test adaptive frame spacing for static content."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (50, 50)) for _ in range(20)]
        
        with patch.object(optimizer, 'analyze_content_type', return_value=ContentType.STATIC):
            indices = optimizer.get_adaptive_frame_spacing(frames, target_frames=5)
        
        assert len(indices) == 5
        assert indices == sorted(indices)  # Should be sorted
        assert all(0 <= idx < 20 for idx in indices)
    
    def test_get_adaptive_frame_spacing_fast_motion(self):
        """Test adaptive frame spacing for fast motion content."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (50, 50)) for _ in range(15)]
        
        with patch.object(optimizer, 'analyze_content_type', return_value=ContentType.FAST_MOTION):
            with patch.object(optimizer, 'get_motion_aware_importance_scores') as mock_scores:
                mock_scores.return_value = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.9, 0.1, 0.8, 0.3, 0.7, 0.2]
                
                indices = optimizer.get_adaptive_frame_spacing(frames, target_frames=5)
        
        assert len(indices) == 5
        assert indices == sorted(indices)
    
    def test_optimize_frame_selection_unavailable(self):
        """Test frame selection when V-JEPA2 is unavailable."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (50, 50)) for _ in range(20)]
        
        with patch.object(optimizer, 'is_available', False):
            indices = optimizer.optimize_frame_selection(frames, target_frames=5)
        
        assert len(indices) == 5
        assert indices == sorted(indices)
        # Should use uniform sampling
        expected_step = 20 // 5
        assert indices[0] == 0
        assert indices[1] == expected_step
    
    def test_optimize_frame_selection_with_adaptive_spacing(self):
        """Test frame selection with adaptive spacing enabled."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (50, 50)) for _ in range(15)]
        
        with patch.object(optimizer, 'is_available', True):
            with patch.object(optimizer, 'get_adaptive_frame_spacing') as mock_spacing:
                mock_spacing.return_value = [0, 3, 6, 9, 12]
                
                indices = optimizer.optimize_frame_selection(frames, target_frames=5, use_adaptive_spacing=True)
        
        assert indices == [0, 3, 6, 9, 12]
        mock_spacing.assert_called_once_with(frames, 5)
    
    def test_optimize_frame_selection_with_importance_scores(self):
        """Test frame selection using importance scores."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (50, 50)) for _ in range(10)]
        
        with patch.object(optimizer, 'is_available', True):
            with patch.object(optimizer, 'get_motion_aware_importance_scores') as mock_scores:
                mock_scores.return_value = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.9]
                
                indices = optimizer.optimize_frame_selection(frames, target_frames=3, use_adaptive_spacing=False)
        
        assert len(indices) == 3
        assert indices == sorted(indices)
        # Should select frames with highest scores while maintaining diversity
    
    def test_calculate_content_similarity(self):
        """Test content similarity calculation."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        # Create similar feature tensors
        features1 = torch.randn(1, 10, 768)
        features2 = features1 + torch.randn(1, 10, 768) * 0.1  # Similar but with noise
        
        similarity = optimizer.calculate_content_similarity(features1, features2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be similar
    
    def test_calculate_content_similarity_none_features(self):
        """Test content similarity with None features."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        similarity = optimizer.calculate_content_similarity(None, torch.randn(1, 10, 768))
        assert similarity == 0.0
        
        similarity = optimizer.calculate_content_similarity(torch.randn(1, 10, 768), None)
        assert similarity == 0.0
    
    def test_group_similar_content(self):
        """Test grouping similar content clips."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        # Create test clips with features
        base_features = torch.randn(1, 10, 768)
        video_clips = [
            ([Image.new('RGB', (50, 50))], base_features),
            ([Image.new('RGB', (50, 50))], base_features + torch.randn(1, 10, 768) * 0.1),  # Similar
            ([Image.new('RGB', (50, 50))], torch.randn(1, 10, 768)),  # Different
            ([Image.new('RGB', (50, 50))], base_features + torch.randn(1, 10, 768) * 0.05)  # Very similar
        ]
        
        with patch.object(optimizer, 'calculate_content_similarity') as mock_similarity:
            # Mock similarity scores
            mock_similarity.side_effect = [0.9, 0.3, 0.95, 0.2, 0.85]
            
            groups = optimizer.group_similar_content(video_clips, similarity_threshold=0.8)
        
        assert len(groups) >= 1
        assert all(isinstance(group, list) for group in groups)
    
    def test_group_similar_content_empty(self):
        """Test grouping with empty video clips."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        groups = optimizer.group_similar_content([])
        assert groups == []
    
    def test_create_feature_cache(self):
        """Test feature cache creation."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        features1 = torch.randn(1, 10, 768)
        features2 = torch.randn(1, 10, 768)
        
        video_clips = [
            ([Image.new('RGB', (50, 50), color=(100, 0, 0))], features1),
            ([Image.new('RGB', (50, 50), color=(0, 100, 0))], features2),
            ([Image.new('RGB', (50, 50), color=(100, 0, 0))], features1)  # Same signature as first
        ]
        
        with patch.object(optimizer, '_create_content_signature') as mock_signature:
            mock_signature.side_effect = ["sig1", "sig2", "sig1"]
            
            cache = optimizer.create_feature_cache(video_clips)
        
        assert isinstance(cache, dict)
        assert len(cache) == 2  # Two unique signatures
        assert "sig1" in cache
        assert "sig2" in cache
    
    def test_create_content_signature(self):
        """Test content signature creation."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        frames = [Image.new('RGB', (100, 100), color=(100, 50, 25)) for _ in range(5)]
        
        with patch('cv2.Canny') as mock_canny:
            mock_canny.return_value = np.ones((100, 100), dtype=np.uint8) * 128
            
            signature = optimizer._create_content_signature(frames)
        
        assert isinstance(signature, str)
        assert len(signature) > 0
        assert "_" in signature  # Should contain separators
    
    def test_create_content_signature_empty(self):
        """Test content signature creation with empty frames."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        signature = optimizer._create_content_signature([])
        assert signature == "empty"
    
    def test_batch_process_similar_content_unavailable(self):
        """Test batch processing when V-JEPA2 is unavailable."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        video_batches = [
            [Image.new('RGB', (50, 50)) for _ in range(3)],
            [Image.new('RGB', (50, 50)) for _ in range(3)]
        ]
        
        with patch.object(optimizer, 'is_available', False):
            results = optimizer.batch_process_similar_content(video_batches)
        
        assert len(results) == 2
        assert all(result is None for result in results)
    
    def test_batch_process_similar_content_with_reuse(self):
        """Test batch processing with feature reuse enabled."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        video_batches = [
            [Image.new('RGB', (50, 50)) for _ in range(16)],  # One clip
            [Image.new('RGB', (50, 50)) for _ in range(16)]   # One clip
        ]
        
        mock_features = torch.randn(1, 16, 768)
        
        with patch.object(optimizer, 'is_available', True):
            with patch.object(optimizer, 'group_similar_content') as mock_group:
                mock_group.return_value = [[0, 1]]  # Both clips are similar
                
                with patch.object(optimizer, 'extract_video_features') as mock_extract:
                    mock_extract.return_value = mock_features
                    
                    results = optimizer.batch_process_similar_content(
                        video_batches, enable_feature_reuse=True
                    )
        
        assert len(results) == 2
        assert all(result is not None for result in results)
        # Should only call extract_video_features once due to reuse
        mock_extract.assert_called_once()
    
    def test_create_clips_from_frames(self):
        """Test creating clips from frames."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        optimizer.frames_per_clip = 5
        
        frames = [Image.new('RGB', (50, 50)) for _ in range(12)]
        
        clips = optimizer._create_clips_from_frames(frames)
        
        assert len(clips) == 3  # 12 frames / 5 frames per clip = 2.4 -> 3 clips
        assert len(clips[0][0]) == 5  # First clip has 5 frames
        assert len(clips[1][0]) == 5  # Second clip has 5 frames
        assert len(clips[2][0]) == 2  # Third clip has 2 frames
        assert all(clip[1] is None for clip in clips)  # Features are None initially
    
    @pytest.mark.parametrize("content_type,expected_weights", [
        (ContentType.STATIC, {'feature_weight': 0.8, 'motion_weight': 0.1}),
        (ContentType.DYNAMIC, {'feature_weight': 0.5, 'motion_weight': 0.4}),
        (ContentType.FAST_MOTION, {'feature_weight': 0.3, 'motion_weight': 0.6}),
        (ContentType.MIXED, {'feature_weight': 0.4, 'motion_weight': 0.4})
    ])
    def test_adaptive_scoring_weights_values(self, content_type, expected_weights):
        """Test that adaptive scoring weights return expected values."""
        config = PipelineBaseData(
            owl_model="test", sam_model="test", threshold=0.1, fps=24, device="cpu",
            pipeline_config=PipelineConfig()
        )
        optimizer = VJepa2VideoOptimizer(config)
        
        weights = optimizer.get_adaptive_scoring_weights(content_type)
        
        for key, expected_value in expected_weights.items():
            assert weights[key] == expected_value
    
    def test_content_type_enum_values(self):
        """Test ContentType enum values."""
        assert ContentType.STATIC.value == "static"
        assert ContentType.DYNAMIC.value == "dynamic"
        assert ContentType.FAST_MOTION.value == "fast_motion"
        assert ContentType.MIXED.value == "mixed"