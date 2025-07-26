"""
Unit tests for ContentAnalyzer.
Tests video content analysis, optimization profiles, and parameter tuning.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from sowlv2.optimizations.content_analyzer import (
    ContentAnalyzer, ContentAnalysis, OptimizationProfile, ContentType
)


class TestContentAnalyzer:
    """Test suite for ContentAnalyzer class."""
    
    def test_init_creates_optimization_profiles(self):
        """Test that initialization creates optimization profiles."""
        analyzer = ContentAnalyzer()
        
        assert hasattr(analyzer, 'optimization_profiles')
        assert isinstance(analyzer.optimization_profiles, dict)
        assert len(analyzer.optimization_profiles) == 4  # All ContentType values
        
        for content_type in ContentType:
            assert content_type in analyzer.optimization_profiles
            profile = analyzer.optimization_profiles[content_type]
            assert isinstance(profile, OptimizationProfile)
            assert profile.content_type == content_type
    
    def test_create_optimization_profiles_values(self):
        """Test optimization profile values are correctly set."""
        analyzer = ContentAnalyzer()
        
        # Test static content profile
        static_profile = analyzer.optimization_profiles[ContentType.STATIC]
        assert static_profile.frame_sampling_rate == 0.1
        assert static_profile.batch_size_multiplier == 2.0
        assert static_profile.motion_threshold == 2.0
        assert static_profile.streaming_chunk_size == 200
        
        # Test fast motion profile
        fast_motion_profile = analyzer.optimization_profiles[ContentType.FAST_MOTION]
        assert fast_motion_profile.frame_sampling_rate == 0.5
        assert fast_motion_profile.batch_size_multiplier == 0.7
        assert fast_motion_profile.motion_threshold == 10.0
        assert fast_motion_profile.streaming_chunk_size == 50
    
    def test_analyze_video_content_insufficient_frames(self):
        """Test video content analysis with insufficient frames."""
        analyzer = ContentAnalyzer()
        
        frames = [Image.new('RGB', (100, 100))]  # Only one frame
        
        analysis = analyzer.analyze_video_content(frames)
        
        assert isinstance(analysis, ContentAnalysis)
        assert analysis.content_type == ContentType.DYNAMIC  # Default
        assert 'average_motion' in analysis.motion_characteristics
        assert 'average_edge_density' in analysis.scene_complexity
        assert 'temporal_consistency' in analysis.temporal_characteristics
    
    def test_analyze_video_content_success(self):
        """Test successful video content analysis."""
        analyzer = ContentAnalyzer()
        
        # Create test frames with different colors
        frames = [
            Image.new('RGB', (100, 100), color=(i*30, 0, 0))
            for i in range(5)
        ]
        
        with patch.object(analyzer, '_analyze_motion_characteristics') as mock_motion:
            with patch.object(analyzer, '_analyze_scene_complexity') as mock_scene:
                with patch.object(analyzer, '_analyze_temporal_characteristics') as mock_temporal:
                    with patch.object(analyzer, '_classify_content_type') as mock_classify:
                        with patch.object(analyzer, '_generate_optimization_recommendations') as mock_recommend:
                            
                            # Mock return values
                            mock_motion.return_value = {'average_motion': 5.0}
                            mock_scene.return_value = {'average_edge_density': 0.2}
                            mock_temporal.return_value = {'temporal_consistency': 0.7}
                            mock_classify.return_value = ContentType.DYNAMIC
                            mock_recommend.return_value = {'frame_sampling_rate': 0.3}
                            
                            analysis = analyzer.analyze_video_content(frames)
        
        assert analysis.content_type == ContentType.DYNAMIC
        assert analysis.motion_characteristics == {'average_motion': 5.0}
        assert analysis.scene_complexity == {'average_edge_density': 0.2}
        assert analysis.temporal_characteristics == {'temporal_consistency': 0.7}
        assert analysis.optimization_recommendations == {'frame_sampling_rate': 0.3}
    
    def test_analyze_motion_characteristics(self):
        """Test motion characteristics analysis."""
        analyzer = ContentAnalyzer()
        
        frames = [
            Image.new('RGB', (100, 100), color=(i*50, 0, 0))
            for i in range(4)
        ]
        
        with patch('cv2.goodFeaturesToTrack') as mock_corners:
            with patch('cv2.calcOpticalFlowPyrLK') as mock_flow:
                # Mock corner detection
                mock_corners.return_value = np.array([[[10, 10]], [[20, 20]], [[30, 30]]], dtype=np.float32)
                
                # Mock optical flow
                mock_flow.return_value = (
                    np.array([[[15, 15]], [[25, 25]], [[35, 35]]], dtype=np.float32),  # New positions
                    np.array([[1], [1], [1]], dtype=np.uint8),  # Status (all good)
                    None  # Error
                )
                
                motion_chars = analyzer._analyze_motion_characteristics(frames)
        
        assert isinstance(motion_chars, dict)
        assert 'average_motion' in motion_chars
        assert 'motion_variance' in motion_chars
        assert 'max_motion' in motion_chars
        assert 'motion_consistency' in motion_chars
        assert 'motion_acceleration' in motion_chars
        
        # All values should be non-negative
        assert all(v >= 0 for v in motion_chars.values())
    
    def test_analyze_motion_characteristics_no_corners(self):
        """Test motion analysis when no corners are detected."""
        analyzer = ContentAnalyzer()
        
        frames = [Image.new('RGB', (100, 100)) for _ in range(3)]
        
        with patch('cv2.goodFeaturesToTrack', return_value=None):
            motion_chars = analyzer._analyze_motion_characteristics(frames)
        
        assert motion_chars['average_motion'] == 0.0
    
    def test_analyze_motion_characteristics_error_handling(self):
        """Test motion analysis error handling."""
        analyzer = ContentAnalyzer()
        
        frames = [Image.new('RGB', (100, 100)) for _ in range(3)]
        
        with patch('cv2.goodFeaturesToTrack', side_effect=Exception("OpenCV error")):
            motion_chars = analyzer._analyze_motion_characteristics(frames)
        
        # Should handle errors gracefully
        assert isinstance(motion_chars, dict)
        assert 'average_motion' in motion_chars
    
    def test_analyze_scene_complexity(self):
        """Test scene complexity analysis."""
        analyzer = ContentAnalyzer()
        
        frames = [
            Image.new('RGB', (100, 100), color=(i*40, i*30, i*20))
            for i in range(3)
        ]
        
        with patch('cv2.Canny') as mock_canny:
            with patch('cv2.Sobel') as mock_sobel:
                with patch('cv2.calcHist') as mock_hist:
                    # Mock edge detection
                    mock_canny.return_value = np.ones((100, 100), dtype=np.uint8) * 128
                    
                    # Mock gradient calculation
                    mock_sobel.return_value = np.ones((100, 100)) * 10
                    
                    # Mock histogram calculation
                    mock_hist.return_value = np.ones((256, 1)) * 10
                    
                    scene_chars = analyzer._analyze_scene_complexity(frames)
        
        assert isinstance(scene_chars, dict)
        assert 'average_edge_density' in scene_chars
        assert 'edge_density_variance' in scene_chars
        assert 'average_texture_complexity' in scene_chars
        assert 'average_color_diversity' in scene_chars
        assert 'average_contrast' in scene_chars
        assert 'contrast_variance' in scene_chars
        
        # All values should be non-negative
        assert all(v >= 0 for v in scene_chars.values())
    
    def test_analyze_scene_complexity_error_handling(self):
        """Test scene complexity analysis error handling."""
        analyzer = ContentAnalyzer()
        
        frames = [Image.new('RGB', (100, 100)) for _ in range(2)]
        
        with patch('cv2.Canny', side_effect=Exception("Edge detection failed")):
            scene_chars = analyzer._analyze_scene_complexity(frames)
        
        # Should handle errors gracefully
        assert isinstance(scene_chars, dict)
        assert 'average_edge_density' in scene_chars
    
    def test_analyze_temporal_characteristics(self):
        """Test temporal characteristics analysis."""
        analyzer = ContentAnalyzer()
        
        frames = [
            Image.new('RGB', (100, 100), color=(i*60, 0, 0))
            for i in range(4)
        ]
        
        temporal_chars = analyzer._analyze_temporal_characteristics(frames)
        
        assert isinstance(temporal_chars, dict)
        assert 'average_frame_difference' in temporal_chars
        assert 'frame_difference_variance' in temporal_chars
        assert 'scene_change_rate' in temporal_chars
        assert 'temporal_consistency' in temporal_chars
        assert 'temporal_stability' in temporal_chars
        
        # Frame differences should be positive for different colored frames
        assert temporal_chars['average_frame_difference'] > 0
    
    def test_classify_content_type_static(self):
        """Test content type classification for static content."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {
            'average_motion': 1.0,  # Low motion
            'motion_variance': 5.0  # Low variance
        }
        scene_chars = {'average_edge_density': 0.1}
        temporal_chars = {
            'scene_change_rate': 0.05,  # Low scene change rate
            'temporal_consistency': 0.9
        }
        
        content_type = analyzer._classify_content_type(motion_chars, scene_chars, temporal_chars)
        
        assert content_type == ContentType.STATIC
    
    def test_classify_content_type_fast_motion(self):
        """Test content type classification for fast motion content."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {
            'average_motion': 20.0,  # High motion
            'motion_variance': 150.0  # High variance
        }
        scene_chars = {'average_edge_density': 0.3}
        temporal_chars = {
            'scene_change_rate': 0.4,  # High scene change rate
            'temporal_consistency': 0.3
        }
        
        content_type = analyzer._classify_content_type(motion_chars, scene_chars, temporal_chars)
        
        assert content_type == ContentType.FAST_MOTION
    
    def test_classify_content_type_mixed(self):
        """Test content type classification for mixed content."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {
            'average_motion': 8.0,  # Moderate motion
            'motion_variance': 40.0  # High variance
        }
        scene_chars = {'average_edge_density': 0.2}
        temporal_chars = {
            'scene_change_rate': 0.2,  # Moderate scene change rate
            'temporal_consistency': 0.5
        }
        
        content_type = analyzer._classify_content_type(motion_chars, scene_chars, temporal_chars)
        
        assert content_type == ContentType.MIXED
    
    def test_classify_content_type_dynamic(self):
        """Test content type classification for dynamic content."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {
            'average_motion': 7.0,  # Moderate motion
            'motion_variance': 15.0  # Low variance
        }
        scene_chars = {'average_edge_density': 0.2}
        temporal_chars = {
            'scene_change_rate': 0.08,  # Low scene change rate
            'temporal_consistency': 0.7
        }
        
        content_type = analyzer._classify_content_type(motion_chars, scene_chars, temporal_chars)
        
        assert content_type == ContentType.DYNAMIC
    
    def test_generate_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {'average_motion': 10.0}
        scene_chars = {'average_edge_density': 0.25}
        temporal_chars = {'temporal_consistency': 0.6, 'scene_change_rate': 0.05}
        
        recommendations = analyzer._generate_optimization_recommendations(
            ContentType.DYNAMIC, motion_chars, scene_chars, temporal_chars
        )
        
        assert isinstance(recommendations, dict)
        assert 'frame_sampling_rate' in recommendations
        assert 'batch_size_multiplier' in recommendations
        assert 'motion_threshold' in recommendations
        assert 'consistency_weight' in recommendations
        assert 'use_motion_prediction' in recommendations
        assert 'enable_scene_change_detection' in recommendations
        
        # Should recommend motion prediction for motion > 5.0
        assert recommendations['use_motion_prediction'] is True
    
    def test_generate_optimization_recommendations_high_motion(self):
        """Test recommendations for high motion content."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {'average_motion': 25.0, 'motion_variance': 80.0}
        scene_chars = {'average_edge_density': 0.15, 'contrast_variance': 500.0}
        temporal_chars = {'temporal_consistency': 0.4, 'scene_change_rate': 0.05}
        
        recommendations = analyzer._generate_optimization_recommendations(
            ContentType.FAST_MOTION, motion_chars, scene_chars, temporal_chars
        )
        
        # High motion should increase frame sampling rate
        base_rate = analyzer.optimization_profiles[ContentType.FAST_MOTION].frame_sampling_rate
        assert recommendations['frame_sampling_rate'] > base_rate
        
        # Should enable temporal smoothing for high variance
        assert recommendations['use_temporal_smoothing'] is True
    
    def test_generate_optimization_recommendations_low_motion(self):
        """Test recommendations for low motion content."""
        analyzer = ContentAnalyzer()
        
        motion_chars = {'average_motion': 0.5, 'motion_variance': 2.0}
        scene_chars = {'average_edge_density': 0.05, 'contrast_variance': 100.0}
        temporal_chars = {'temporal_consistency': 0.9, 'scene_change_rate': 0.02}
        
        recommendations = analyzer._generate_optimization_recommendations(
            ContentType.STATIC, motion_chars, scene_chars, temporal_chars
        )
        
        # Low motion should decrease frame sampling rate
        base_rate = analyzer.optimization_profiles[ContentType.STATIC].frame_sampling_rate
        assert recommendations['frame_sampling_rate'] < base_rate
        
        # Should not use motion prediction for low motion
        assert recommendations['use_motion_prediction'] is False
    
    def test_get_optimization_profile(self):
        """Test getting optimization profile for content type."""
        analyzer = ContentAnalyzer()
        
        profile = analyzer.get_optimization_profile(ContentType.DYNAMIC)
        
        assert isinstance(profile, OptimizationProfile)
        assert profile.content_type == ContentType.DYNAMIC
        assert profile.name == "Dynamic Content"
    
    def test_tune_parameters_for_content(self):
        """Test parameter tuning based on content analysis."""
        analyzer = ContentAnalyzer()
        
        base_params = {
            'batch_size': 4,
            'frame_sampling_rate': 0.2,
            'motion_threshold': 5.0,
            'consistency_weight': 0.5
        }
        
        # Create mock content analysis
        content_analysis = ContentAnalysis(
            content_type=ContentType.FAST_MOTION,
            motion_characteristics={'average_motion': 15.0},
            scene_complexity={'average_edge_density': 0.3},
            temporal_characteristics={'temporal_consistency': 0.4},
            optimization_recommendations={
                'batch_size_multiplier': 0.7,
                'frame_sampling_rate': 0.5,
                'motion_threshold': 10.0,
                'consistency_weight': 0.7,
                'use_motion_prediction': True,
                'enable_scene_change_detection': False,
                'recommended_detection_interval': 3
            }
        )
        
        tuned_params = analyzer.tune_parameters_for_content(base_params, content_analysis)
        
        assert tuned_params['batch_size'] == int(4 * 0.7)  # Applied multiplier
        assert tuned_params['frame_sampling_rate'] == 0.5  # Updated
        assert tuned_params['motion_threshold'] == 10.0  # Updated
        assert tuned_params['consistency_weight'] == 0.7  # Updated
        assert tuned_params['use_motion_prediction'] is True  # Added
        assert tuned_params['detection_interval'] == 3  # Added
    
    def test_create_content_report(self):
        """Test content analysis report creation."""
        analyzer = ContentAnalyzer()
        
        content_analysis = ContentAnalysis(
            content_type=ContentType.DYNAMIC,
            motion_characteristics={'average_motion': 8.0},
            scene_complexity={'average_edge_density': 0.25},
            temporal_characteristics={'temporal_consistency': 0.7},
            optimization_recommendations={'frame_sampling_rate': 0.3}
        )
        
        report = analyzer.create_content_report(content_analysis)
        
        assert isinstance(report, dict)
        assert report['content_type'] == 'dynamic'
        assert 'analysis_summary' in report
        assert 'detailed_metrics' in report
        assert 'optimization_recommendations' in report
        assert 'processing_suggestions' in report
        
        # Check analysis summary
        summary = report['analysis_summary']
        assert 'motion_level' in summary
        assert 'scene_complexity' in summary
        assert 'temporal_stability' in summary
    
    def test_categorize_motion_level(self):
        """Test motion level categorization."""
        analyzer = ContentAnalyzer()
        
        assert analyzer._categorize_motion_level(1.0) == "Very Low"
        assert analyzer._categorize_motion_level(3.0) == "Low"
        assert analyzer._categorize_motion_level(7.0) == "Moderate"
        assert analyzer._categorize_motion_level(15.0) == "High"
        assert analyzer._categorize_motion_level(25.0) == "Very High"
    
    def test_categorize_scene_complexity(self):
        """Test scene complexity categorization."""
        analyzer = ContentAnalyzer()
        
        assert analyzer._categorize_scene_complexity(0.05) == "Simple"
        assert analyzer._categorize_scene_complexity(0.15) == "Moderate"
        assert analyzer._categorize_scene_complexity(0.25) == "Complex"
        assert analyzer._categorize_scene_complexity(0.35) == "Very Complex"
    
    def test_categorize_temporal_stability(self):
        """Test temporal stability categorization."""
        analyzer = ContentAnalyzer()
        
        assert analyzer._categorize_temporal_stability(0.9) == "Very Stable"
        assert analyzer._categorize_temporal_stability(0.7) == "Stable"
        assert analyzer._categorize_temporal_stability(0.5) == "Moderate"
        assert analyzer._categorize_temporal_stability(0.3) == "Unstable"
        assert analyzer._categorize_temporal_stability(0.1) == "Very Unstable"
    
    def test_generate_processing_suggestions(self):
        """Test processing suggestions generation."""
        analyzer = ContentAnalyzer()
        
        content_analysis = ContentAnalysis(
            content_type=ContentType.FAST_MOTION,
            motion_characteristics={'average_motion': 20.0},
            scene_complexity={'average_edge_density': 0.35},
            temporal_characteristics={'temporal_consistency': 0.3},
            optimization_recommendations={}
        )
        
        suggestions = analyzer._generate_processing_suggestions(content_analysis)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)
        
        # Should suggest higher frame sampling for fast motion
        assert any("higher frame sampling" in suggestion for suggestion in suggestions)
        
        # Should suggest reducing batch size for complex scenes
        assert any("Reduce batch size" in suggestion for suggestion in suggestions)
        
        # Should suggest temporal smoothing for low consistency
        assert any("temporal smoothing" in suggestion for suggestion in suggestions)
    
    def test_create_default_analysis(self):
        """Test default analysis creation."""
        analyzer = ContentAnalyzer()
        
        default_analysis = analyzer._create_default_analysis()
        
        assert isinstance(default_analysis, ContentAnalysis)
        assert default_analysis.content_type == ContentType.DYNAMIC
        assert 'average_motion' in default_analysis.motion_characteristics
        assert 'average_edge_density' in default_analysis.scene_complexity
        assert 'temporal_consistency' in default_analysis.temporal_characteristics
        assert isinstance(default_analysis.optimization_recommendations, dict)


class TestOptimizationProfile:
    """Test suite for OptimizationProfile dataclass."""
    
    def test_optimization_profile_creation(self):
        """Test OptimizationProfile creation."""
        profile = OptimizationProfile(
            name="Test Profile",
            content_type=ContentType.DYNAMIC,
            frame_sampling_rate=0.3,
            batch_size_multiplier=1.0,
            motion_threshold=5.0,
            consistency_weight=0.5,
            feature_cache_size=100,
            parallel_processing=True,
            streaming_chunk_size=100
        )
        
        assert profile.name == "Test Profile"
        assert profile.content_type == ContentType.DYNAMIC
        assert profile.frame_sampling_rate == 0.3
        assert profile.batch_size_multiplier == 1.0
        assert profile.motion_threshold == 5.0
        assert profile.consistency_weight == 0.5
        assert profile.feature_cache_size == 100
        assert profile.parallel_processing is True
        assert profile.streaming_chunk_size == 100


class TestContentAnalysis:
    """Test suite for ContentAnalysis dataclass."""
    
    def test_content_analysis_creation(self):
        """Test ContentAnalysis creation."""
        motion_chars = {'average_motion': 5.0}
        scene_chars = {'average_edge_density': 0.2}
        temporal_chars = {'temporal_consistency': 0.7}
        recommendations = {'frame_sampling_rate': 0.3}
        
        analysis = ContentAnalysis(
            content_type=ContentType.DYNAMIC,
            motion_characteristics=motion_chars,
            scene_complexity=scene_chars,
            temporal_characteristics=temporal_chars,
            optimization_recommendations=recommendations
        )
        
        assert analysis.content_type == ContentType.DYNAMIC
        assert analysis.motion_characteristics == motion_chars
        assert analysis.scene_complexity == scene_chars
        assert analysis.temporal_characteristics == temporal_chars
        assert analysis.optimization_recommendations == recommendations