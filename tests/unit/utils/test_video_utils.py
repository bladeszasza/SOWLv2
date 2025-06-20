"""Tests for video_utils module."""
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from sowlv2.utils import video_utils
from sowlv2.utils.path_config import FilePattern


class TestGenerateVideosForObject:
    """Test the _generate_videos_for_object function."""

    def test_individual_object_with_passed_prompt(self):
        """Test video generation for individual object with passed prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test mask file with prompt in filename
            mask_file = "000001_obj1_cat_mask.png"
            mask_path = os.path.join(temp_dir, mask_file)
            
            # Create the file (empty for test)
            with open(mask_path, 'w') as f:
                f.write('')
            
            files = {
                "mask": [mask_path],
                "overlay": []
            }
            
            video_dirs = {
                "binary": temp_dir,
                "overlay": temp_dir
            }
            
            flags = {"binary": True, "overlay": False}
            
            with patch('sowlv2.utils.video_utils.images_to_video') as mock_images_to_video:
                video_utils._generate_videos_for_object(
                    obj_id="obj1",
                    files=files,
                    video_dirs=video_dirs,
                    flags=flags,
                    fps=30,
                    prompt="person"  # Pass explicit prompt
                )
                
                # Verify the correct filename was used (with passed prompt)
                expected_filename = "obj1_person_mask.mp4"
                expected_path = os.path.join(temp_dir, expected_filename)
                mock_images_to_video.assert_called_once_with([mask_path], expected_path, 30)

    def test_individual_object_fallback_when_no_prompt_extracted(self):
        """Test video generation falls back to simple naming when prompt extraction fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test mask file with non-standard filename (no prompt)
            mask_file = "invalid_filename.png"
            mask_path = os.path.join(temp_dir, mask_file)
            
            # Create the file (empty for test)
            with open(mask_path, 'w') as f:
                f.write('')
            
            files = {
                "mask": [mask_path],
                "overlay": []
            }
            
            video_dirs = {
                "binary": temp_dir,
                "overlay": temp_dir
            }
            
            flags = {"binary": True, "overlay": False}
            
            with patch('sowlv2.utils.video_utils.images_to_video') as mock_images_to_video:
                video_utils._generate_videos_for_object(
                    obj_id="obj1",
                    files=files,
                    video_dirs=video_dirs,
                    flags=flags,
                    fps=30,
                    prompt=None  # No prompt passed
                )
                
                # Verify fallback filename was used
                expected_filename = "obj1_mask.mp4"
                expected_path = os.path.join(temp_dir, expected_filename)
                mock_images_to_video.assert_called_once_with([mask_path], expected_path, 30)

    def test_merged_object_uses_predefined_filename(self):
        """Test that merged objects use predefined filenames without prompt extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_file = "000001_merged_mask.png"
            mask_path = os.path.join(temp_dir, mask_file)
            
            # Create the file (empty for test)
            with open(mask_path, 'w') as f:
                f.write('')
            
            files = {
                "mask": [mask_path],
                "overlay": []
            }
            
            video_dirs = {
                "binary": temp_dir,
                "overlay": temp_dir
            }
            
            flags = {"binary": True, "overlay": False}
            
            with patch('sowlv2.utils.video_utils.images_to_video') as mock_images_to_video:
                video_utils._generate_videos_for_object(
                    obj_id="merged",
                    files=files,
                    video_dirs=video_dirs,
                    flags=flags,
                    fps=30,
                    prompt=None  # Merged objects don't need prompts
                )
                
                # Verify merged filename was used
                expected_filename = FilePattern.VIDEO_MERGED_MASK
                expected_path = os.path.join(temp_dir, expected_filename)
                mock_images_to_video.assert_called_once_with([mask_path], expected_path, 30)

    def test_overlay_video_generation_with_prompt(self):
        """Test overlay video generation with prompt extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test overlay file with prompt in filename
            overlay_file = "000001_obj2_dog_overlay.png"
            overlay_path = os.path.join(temp_dir, overlay_file)
            
            # Create the file (empty for test)
            with open(overlay_path, 'w') as f:
                f.write('')
            
            files = {
                "mask": [],
                "overlay": [overlay_path]
            }
            
            video_dirs = {
                "binary": temp_dir,
                "overlay": temp_dir
            }
            
            flags = {"binary": False, "overlay": True}
            
            with patch('sowlv2.utils.video_utils.images_to_video') as mock_images_to_video:
                video_utils._generate_videos_for_object(
                    obj_id="obj2",
                    files=files,
                    video_dirs=video_dirs,
                    flags=flags,
                    fps=30,
                    prompt="dog"  # Pass explicit prompt
                )
                
                # Verify the correct overlay filename was used (with passed prompt)
                expected_filename = "obj2_dog_overlay.mp4"
                expected_path = os.path.join(temp_dir, expected_filename)
                mock_images_to_video.assert_called_once_with([overlay_path], expected_path, 30)

    def test_generate_videos_with_prompt_details(self):
        """Test the main generate_videos function with prompt details."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            binary_frames_dir = os.path.join(temp_dir, "binary", "frames")
            os.makedirs(binary_frames_dir, exist_ok=True)
            
            # Create test mask file
            mask_file = "000001_obj1_person_mask.png"
            mask_path = os.path.join(binary_frames_dir, mask_file)
            with open(mask_path, 'w') as f:
                f.write('')
            
            # Prepare prompt details
            prompt_details = [
                {'sam_id': 1, 'core_prompt': 'person'},
                {'sam_id': 2, 'core_prompt': 'sun'}
            ]
            
            with patch('sowlv2.utils.video_utils.images_to_video') as mock_images_to_video:
                with patch('sowlv2.utils.video_utils._get_obj_files') as mock_get_obj_files:
                    # Mock the return value to simulate found files
                    mock_get_obj_files.return_value = {
                        "obj1": {"mask": [mask_path], "overlay": []}
                    }
                    
                    video_utils.generate_videos(
                        temp_dir=temp_dir,
                        fps=30,
                        binary=True,
                        overlay=False,
                        merged=False,
                        prompt_details=prompt_details
                    )
                    
                    # Verify the prompt was passed correctly
                    # The video should be generated with the correct prompt from prompt_details
                    mock_images_to_video.assert_called_once()
                    call_args = mock_images_to_video.call_args
                    generated_path = call_args[0][1]  # Second argument is the video path
                    assert "obj1_person_mask.mp4" in generated_path