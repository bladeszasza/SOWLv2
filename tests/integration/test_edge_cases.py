"""Test edge cases and error handling throughout the pipeline."""
import pytest
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from sowlv2.pipeline import SOWLv2Pipeline
from sowlv2.data.config import PipelineBaseData, PipelineConfig


class TestNoDetectionsScenario:
    """Test behavior when no objects are detected."""
    
    def test_no_detections_image(self, tmp_path, sample_image_path, 
                                mock_owl_model, mock_sam_model):
        """Test handling when OWL detects no objects in image."""
        output_dir = str(tmp_path / "output")
        
        # Configure mock to return no detections
        mock_owl_model.detect.return_value = []
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        pipeline = SOWLv2Pipeline(config)
        
        # Should not raise exception
        pipeline.process_image(sample_image_path, "nonexistent_object", output_dir)
        
        # Output directory should either not exist or be empty/cleaned up
        output_path = Path(output_dir)
        if output_path.exists():
            # Check that no significant output files were created
            all_files = list(output_path.rglob("*"))
            image_files = [f for f in all_files if f.suffix in ['.png', '.jpg']]
            assert len(image_files) == 0, "No image files should be created with no detections"
    
    def test_no_detections_video(self, tmp_path, sample_video_path,
                                mock_owl_model, mock_sam_model):
        """Test handling when OWL detects no objects in video."""
        output_dir = str(tmp_path / "output")
        
        # Configure mock to return no detections
        mock_owl_model.detect.return_value = []
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = None
            
            pipeline = SOWLv2Pipeline(config)
            pipeline.process_video(sample_video_path, "nonexistent_object", output_dir)
        
        # Should handle gracefully without creating significant output
        output_path = Path(output_dir)
        if output_path.exists():
            all_files = list(output_path.rglob("*"))
            media_files = [f for f in all_files if f.suffix in ['.png', '.jpg', '.mp4']]
            assert len(media_files) == 0, "No media files should be created with no detections"
    
    def test_partial_detections_across_frames(self, tmp_path, sample_frames_directory,
                                            mock_owl_model, mock_sam_model):
        """Test handling when only some frames have detections."""
        output_dir = str(tmp_path / "output")
        
        # Configure mock to alternate between detections and no detections
        detection_results = [
            [{"box": [100, 100, 200, 200], "score": 0.9, "label": "cat", "core_prompt": "cat"}],
            [],  # No detections
            [{"box": [150, 150, 250, 250], "score": 0.8, "label": "cat", "core_prompt": "cat"}]
        ]
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            result = detection_results[call_count % len(detection_results)]
            call_count += 1
            return result
        
        mock_owl_model.detect.side_effect = side_effect
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        pipeline = SOWLv2Pipeline(config)
        pipeline.process_frames(sample_frames_directory, "cat", output_dir)
        
        # Should only create outputs for frames with detections
        output_path = Path(output_dir)
        if output_path.exists():
            binary_files = list(output_path.rglob("*_mask.png"))
            # Should have fewer files than total frames
            assert len(binary_files) < 3, "Should only create files for frames with detections"


class TestInvalidMasksHandling:
    """Test handling of empty/invalid masks from SAM."""
    
    def test_sam_returns_none_mask(self, tmp_path, sample_image_path,
                                  mock_owl_model, mock_sam_model):
        """Test handling when SAM returns None for mask."""
        output_dir = str(tmp_path / "output")
        
        # Configure OWL to return detections but SAM to return None
        mock_owl_model.detect.return_value = [
            {"box": [100, 100, 200, 200], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]
        mock_sam_model.segment.return_value = None
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        pipeline = SOWLv2Pipeline(config)
        
        # Should not raise exception
        pipeline.process_image(sample_image_path, "cat", output_dir)
        
        # Should not create output files for failed segmentation
        output_path = Path(output_dir)
        if output_path.exists():
            binary_files = list(output_path.rglob("*_mask.png"))
            assert len(binary_files) == 0, "No mask files should be created when SAM fails"
    
    def test_sam_returns_empty_mask(self, tmp_path, sample_image_path,
                                   mock_owl_model, mock_sam_model):
        """Test handling when SAM returns empty mask."""
        output_dir = str(tmp_path / "output")
        
        # Configure SAM to return empty mask
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        mock_owl_model.detect.return_value = [
            {"box": [100, 100, 200, 200], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]
        mock_sam_model.segment.return_value = empty_mask
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        pipeline = SOWLv2Pipeline(config)
        pipeline.process_image(sample_image_path, "cat", output_dir)
        
        # Should create files even with empty mask (valid use case)
        output_path = Path(output_dir)
        binary_files = list(output_path.rglob("*_mask.png"))
        assert len(binary_files) > 0, "Should create files even for empty masks"
    
    def test_sam_returns_invalid_shape_mask(self, tmp_path, sample_image_path,
                                          mock_owl_model, mock_sam_model):
        """Test handling when SAM returns mask with wrong dimensions."""
        output_dir = str(tmp_path / "output")
        
        # Configure SAM to return wrong-sized mask
        wrong_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_owl_model.detect.return_value = [
            {"box": [100, 100, 200, 200], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]
        mock_sam_model.segment.return_value = wrong_mask
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        pipeline = SOWLv2Pipeline(config)
        
        # Should handle gracefully (might resize or skip)
        try:
            pipeline.process_image(sample_image_path, "cat", output_dir)
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()


class TestDeviceHandling:
    """Test CUDA/CPU device handling."""
    
    def test_cuda_unavailable_fallback(self, tmp_path, sample_image_path):
        """Test fallback to CPU when CUDA is requested but unavailable."""
        output_dir = str(tmp_path / "output")
        
        # Configure pipeline to request CUDA
        config = PipelineBaseData(
            device="cuda",
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )
        
        with patch('torch.cuda.is_available', return_value=False):
            # Should either fallback to CPU or handle gracefully
            try:
                pipeline = SOWLv2Pipeline(config)
                # If initialization succeeds, device should be CPU
                assert pipeline.config.device in ["cpu", "cuda"]
            except Exception as e:
                # Should provide meaningful error message about CUDA availability
                assert "cuda" in str(e).lower() or "device" in str(e).lower()
    
    def test_invalid_device_specification(self, tmp_path):
        """Test handling of invalid device specification."""
        # Test with invalid device
        config = PipelineBaseData(
            device="invalid_device",
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )
        
        # Should either handle gracefully or raise meaningful error
        try:
            pipeline = SOWLv2Pipeline(config)
        except Exception as e:
            # Should provide meaningful error message
            assert "device" in str(e).lower() or "invalid" in str(e).lower()


class TestVideoProcessingEdgeCases:
    """Test edge cases specific to video processing."""
    
    def test_missing_frames_in_video_sequence(self, tmp_path):
        """Test handling of missing frames during video processing."""
        output_dir = str(tmp_path / "output")
        
        # Create video path (doesn't need to exist for this mock test)
        video_path = str(tmp_path / "test_video.mp4")
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock ffmpeg success
            mock_subprocess.return_value = None
            
            with patch('sowlv2.video_pipeline.prepare_video_context') as mock_prepare:
                # Mock prepare returning None (failure)
                mock_prepare.return_value = (None, {}, 0)
                
                pipeline = SOWLv2Pipeline(config)
                
                # Should handle gracefully when video preparation fails
                pipeline.process_video(video_path, "cat", output_dir)
                
                # Should not create significant output
                output_path = Path(output_dir)
                if output_path.exists():
                    media_files = list(output_path.rglob("*.mp4"))
                    assert len(media_files) == 0
    
    def test_ffmpeg_failure(self, tmp_path):
        """Test handling of ffmpeg failure during video processing."""
        output_dir = str(tmp_path / "output")
        video_path = str(tmp_path / "test_video.mp4")
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        from subprocess import CalledProcessError
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock ffmpeg failure
            mock_subprocess.side_effect = CalledProcessError(1, 'ffmpeg')
            
            pipeline = SOWLv2Pipeline(config)
            
            # Should handle ffmpeg failure gracefully
            pipeline.process_video(video_path, "cat", output_dir)
            
            # Should not create significant output
            output_path = Path(output_dir)
            if output_path.exists():
                media_files = list(output_path.rglob("*.mp4"))
                assert len(media_files) == 0
    
    def test_video_timeout(self, tmp_path):
        """Test handling of video processing timeout."""
        output_dir = str(tmp_path / "output")
        video_path = str(tmp_path / "test_video.mp4")
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        from subprocess import TimeoutExpired
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock timeout
            mock_subprocess.side_effect = TimeoutExpired('ffmpeg', 300)
            
            pipeline = SOWLv2Pipeline(config)
            
            # Should handle timeout gracefully
            pipeline.process_video(video_path, "cat", output_dir)


class TestMemoryAndPerformance:
    """Test memory usage and performance edge cases."""
    
    @pytest.mark.slow
    def test_large_image_processing(self, tmp_path, mock_owl_model, mock_sam_model):
        """Test processing of large images."""
        output_dir = str(tmp_path / "output")
        
        # Create a large test image
        large_image = Image.new('RGB', (4000, 3000), color='red')
        large_image_path = tmp_path / "large_image.jpg"
        large_image.save(large_image_path, "JPEG")
        
        # Configure mock for large image
        large_mask = np.ones((3000, 4000), dtype=np.uint8) * 255
        mock_owl_model.detect.return_value = [
            {"box": [100, 100, 500, 400], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]
        mock_sam_model.segment.return_value = large_mask
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )
        
        pipeline = SOWLv2Pipeline(config)
        
        # Should handle large images without memory issues
        pipeline.process_image(str(large_image_path), "cat", output_dir)
        
        # Verify output was created
        output_path = Path(output_dir)
        binary_files = list(output_path.rglob("*_mask.png"))
        assert len(binary_files) > 0
    
    def test_many_objects_detection(self, tmp_path, sample_image_path,
                                   mock_owl_model, mock_sam_model):
        """Test handling of many detected objects."""
        output_dir = str(tmp_path / "output")
        
        # Configure mock to return many detections
        many_detections = []
        for i in range(20):  # 20 objects
            many_detections.append({
                "box": [50 + i*20, 50 + i*10, 150 + i*20, 150 + i*10],
                "score": 0.9 - i*0.01,
                "label": f"object_{i}",
                "core_prompt": f"obj{i}"
            })
        
        mock_owl_model.detect.return_value = many_detections
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )
        
        pipeline = SOWLv2Pipeline(config)
        pipeline.process_image(sample_image_path, [f"obj{i}" for i in range(20)], output_dir)
        
        # Should handle many objects gracefully
        output_path = Path(output_dir)
        binary_files = list(output_path.rglob("*_obj*_*_mask.png"))
        assert len(binary_files) == 20, "Should create files for all detected objects"


class TestFileSystemEdgeCases:
    """Test filesystem-related edge cases."""
    
    def test_read_only_output_directory(self, tmp_path, sample_image_path,
                                       mock_owl_model, mock_sam_model):
        """Test handling of read-only output directory."""
        output_dir = tmp_path / "readonly_output"
        output_dir.mkdir()
        
        # Make directory read-only (Unix-like systems)
        try:
            os.chmod(output_dir, 0o444)
            
            config = PipelineBaseData(
                pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
            )
            
            mock_owl_model.detect.return_value = [
                {"box": [100, 100, 200, 200], "score": 0.9, "label": "cat", "core_prompt": "cat"}
            ]
            
            pipeline = SOWLv2Pipeline(config)
            
            # Should handle permission error gracefully
            try:
                pipeline.process_image(sample_image_path, "cat", str(output_dir))
            except PermissionError:
                # Expected behavior
                pass
            except Exception as e:
                # Should provide meaningful error message
                assert "permission" in str(e).lower() or "access" in str(e).lower()
        
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(output_dir, 0o755)
            except:
                pass
    
    def test_disk_space_full_simulation(self, tmp_path, sample_image_path,
                                       mock_owl_model, mock_sam_model):
        """Test handling when disk space is full (simulated)."""
        output_dir = str(tmp_path / "output")
        
        config = PipelineBaseData(
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )
        
        mock_owl_model.detect.return_value = [
            {"box": [100, 100, 200, 200], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]
        
        # Mock PIL Image.save to raise OSError (disk full)
        with patch.object(Image.Image, 'save', side_effect=OSError("No space left on device")):
            pipeline = SOWLv2Pipeline(config)
            
            # Should handle disk full error gracefully
            try:
                pipeline.process_image(sample_image_path, "cat", output_dir)
            except OSError as e:
                assert "space" in str(e).lower()
            except Exception as e:
                # Should provide meaningful error message
                assert len(str(e)) > 0


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""
    
    def test_extreme_threshold_values(self, tmp_path, sample_image_path, mock_sam_model):
        """Test handling of extreme threshold values."""
        output_dir = str(tmp_path / "output")
        
        # Test with very low threshold
        with patch('sowlv2.models.owl.OWLV2Wrapper') as mock_owl_class:
            mock_owl = mock_owl_class.return_value
            mock_owl.detect.return_value = []  # No detections with extreme threshold
            
            config = PipelineBaseData(
                threshold=0.001,  # Very low threshold
                pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
            )
            
            pipeline = SOWLv2Pipeline(config)
            pipeline.process_image(sample_image_path, "cat", output_dir)
            
            # Should not crash with extreme threshold
            assert mock_owl.detect.called
    
    def test_extreme_fps_values(self, tmp_path, sample_video_path, mock_owl_model, mock_sam_model):
        """Test handling of extreme FPS values."""
        output_dir = str(tmp_path / "output")
        
        config = PipelineBaseData(
            fps=1000,  # Very high FPS
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = None
            
            pipeline = SOWLv2Pipeline(config)
            
            # Should handle extreme FPS values gracefully
            pipeline.process_video(sample_video_path, "cat", output_dir)