"""Test path configuration and file naming patterns."""
import pytest
import os
import re
from pathlib import Path

from sowlv2.utils.path_config import (
    FilePattern, DirectoryStructure, FilePatternMatcher, OutputType
)


class TestFilePattern:
    """Test FilePattern class functionality."""
    
    def test_individual_mask_pattern(self):
        """Test individual mask file pattern formatting."""
        pattern = FilePattern.INDIVIDUAL_MASK
        
        # Test with typical values
        filename = pattern.format(frame_num="000001", obj_id="1", prompt="cat")
        assert filename == "000001_obj1_cat_mask.png"
        
        # Test with complex prompt
        filename = pattern.format(frame_num="000042", obj_id="5", prompt="red car")
        assert filename == "000042_obj5_red car_mask.png"
    
    def test_individual_overlay_pattern(self):
        """Test individual overlay file pattern formatting."""
        pattern = FilePattern.INDIVIDUAL_OVERLAY
        
        filename = pattern.format(frame_num="000001", obj_id="1", prompt="cat")
        assert filename == "000001_obj1_cat_overlay.png"
    
    def test_merged_patterns(self):
        """Test merged file patterns."""
        mask_pattern = FilePattern.MERGED_MASK
        overlay_pattern = FilePattern.MERGED_OVERLAY
        
        mask_filename = mask_pattern.format(frame_num="000001")
        overlay_filename = overlay_pattern.format(frame_num="000001")
        
        assert mask_filename == "000001_merged_mask.png"
        assert overlay_filename == "000001_merged_overlay.png"
    
    def test_video_patterns(self):
        """Test video file patterns."""
        mask_pattern = FilePattern.VIDEO_MASK
        overlay_pattern = FilePattern.VIDEO_OVERLAY
        
        mask_filename = mask_pattern.format(obj_id="1", prompt="cat")
        overlay_filename = overlay_pattern.format(obj_id="1", prompt="cat")
        
        assert mask_filename == "1_cat_mask.mp4"
        assert overlay_filename == "1_cat_overlay.mp4"
    
    def test_video_merged_patterns(self):
        """Test merged video file patterns."""
        assert FilePattern.VIDEO_MERGED_MASK == "merged_mask.mp4"
        assert FilePattern.VIDEO_MERGED_OVERLAY == "merged_overlay.mp4"
    
    def test_frame_number_formatting(self):
        """Test frame number formatting."""
        format_str = FilePattern.FRAME_NUM_FORMAT
        
        # Test with different numbers
        assert format_str.format(1) == "000001"
        assert format_str.format(42) == "000042"
        assert format_str.format(999999) == "999999"
    
    def test_get_individual_patterns(self):
        """Test get_individual_patterns class method."""
        patterns = FilePattern.get_individual_patterns()
        
        assert "mask" in patterns
        assert "overlay" in patterns
        assert patterns["mask"] == FilePattern.INDIVIDUAL_MASK
        assert patterns["overlay"] == FilePattern.INDIVIDUAL_OVERLAY
    
    def test_get_merged_patterns(self):
        """Test get_merged_patterns class method."""
        patterns = FilePattern.get_merged_patterns()
        
        assert "mask" in patterns
        assert "overlay" in patterns
        assert patterns["mask"] == FilePattern.MERGED_MASK
        assert patterns["overlay"] == FilePattern.MERGED_OVERLAY


class TestDirectoryStructure:
    """Test DirectoryStructure class functionality."""
    
    def test_directory_constants(self):
        """Test directory name constants."""
        assert DirectoryStructure.BINARY == "binary"
        assert DirectoryStructure.OVERLAY == "overlay"
        assert DirectoryStructure.VIDEO == "video"
        assert DirectoryStructure.FRAMES == "frames"
        assert DirectoryStructure.MERGED == "merged"
    
    def test_get_directory_map_without_video(self, tmp_path):
        """Test directory map generation without video directories."""
        base_dir = str(tmp_path)
        
        dirs = DirectoryStructure.get_directory_map(base_dir, include_video=False)
        
        expected_keys = [
            "binary", "binary_frames", "binary_merged",
            "overlay", "overlay_frames", "overlay_merged"
        ]
        
        for key in expected_keys:
            assert key in dirs
        
        # Video directories should not be included
        assert "video" not in dirs
        assert "video_binary" not in dirs
        assert "video_overlay" not in dirs
        
        # Check path formatting
        assert dirs["binary"] == os.path.join(base_dir, "binary")
        assert dirs["binary_frames"] == os.path.join(base_dir, "binary", "frames")
        assert dirs["binary_merged"] == os.path.join(base_dir, "binary", "merged")
    
    def test_get_directory_map_with_video(self, tmp_path):
        """Test directory map generation with video directories."""
        base_dir = str(tmp_path)
        
        dirs = DirectoryStructure.get_directory_map(base_dir, include_video=True)
        
        expected_keys = [
            "binary", "binary_frames", "binary_merged",
            "overlay", "overlay_frames", "overlay_merged",
            "video", "video_binary", "video_overlay"
        ]
        
        for key in expected_keys:
            assert key in dirs
        
        # Check video directory paths
        assert dirs["video"] == os.path.join(base_dir, "video")
        assert dirs["video_binary"] == os.path.join(base_dir, "video", "binary")
        assert dirs["video_overlay"] == os.path.join(base_dir, "video", "overlay")
    
    def test_get_base_directories(self):
        """Test get_base_directories class method."""
        base_dirs = DirectoryStructure.get_base_directories()
        
        expected = ["binary", "overlay", "video"]
        assert base_dirs == expected


class TestFilePatternMatcher:
    """Test FilePatternMatcher class functionality."""
    
    def test_individual_mask_pattern_matching(self):
        """Test individual mask pattern regex matching."""
        pattern = FilePatternMatcher.get_individual_mask_pattern()
        regex = re.compile(pattern)
        
        # Test valid filenames
        valid_names = [
            "000001_obj1_cat_mask.png",
            "000042_obj5_red car_mask.png",
            "123456_obj10_a photo of dog_mask.png"
        ]
        
        for name in valid_names:
            match = regex.match(name)
            assert match is not None, f"Should match: {name}"
            
            # Test capture groups
            frame_num, obj_num, prompt = match.groups()
            assert frame_num.isdigit()
            assert obj_num.isdigit()
            assert len(prompt) > 0
        
        # Test invalid filenames
        invalid_names = [
            "obj1_cat_mask.png",  # Missing frame number
            "000001_1_cat_mask.png",  # Missing "obj" prefix
            "000001_obj1_cat.png",  # Missing "_mask"
            "000001_obj1_cat_mask.jpg"  # Wrong extension
        ]
        
        for name in invalid_names:
            match = regex.match(name)
            assert match is None, f"Should not match: {name}"
    
    def test_simple_mask_pattern_matching(self):
        """Test simple mask pattern regex matching."""
        pattern = FilePatternMatcher.get_simple_mask_pattern()
        regex = re.compile(pattern)
        
        # Test valid filenames
        valid_names = [
            "000001_obj1_mask.png",
            "123456_obj42_mask.png"
        ]
        
        for name in valid_names:
            match = regex.match(name)
            assert match is not None, f"Should match: {name}"
            
            frame_num, obj_id = match.groups()
            assert frame_num.isdigit()
            assert obj_id.startswith("obj")
    
    def test_merged_mask_pattern_matching(self):
        """Test merged mask pattern regex matching."""
        pattern = FilePatternMatcher.get_merged_mask_pattern()
        regex = re.compile(pattern)
        
        # Test valid filenames
        valid_names = [
            "000001_merged_mask.png",
            "123456_merged_mask.png"
        ]
        
        for name in valid_names:
            match = regex.match(name)
            assert match is not None, f"Should match: {name}"
            
            frame_num = match.group(1)
            assert frame_num.isdigit()
        
        # Test invalid filenames
        invalid_names = [
            "merged_mask.png",  # Missing frame number
            "000001_mask.png",  # Missing "merged"
            "000001_merged_overlay.png"  # Wrong type
        ]
        
        for name in invalid_names:
            match = regex.match(name)
            assert match is None, f"Should not match: {name}"
    
    def test_merged_overlay_pattern_matching(self):
        """Test merged overlay pattern regex matching."""
        pattern = FilePatternMatcher.get_merged_overlay_pattern()
        regex = re.compile(pattern)
        
        # Test valid filenames
        valid_names = [
            "000001_merged_overlay.png",
            "999999_merged_overlay.png"
        ]
        
        for name in valid_names:
            match = regex.match(name)
            assert match is not None, f"Should match: {name}"
    
    def test_natural_sort_key(self):
        """Test natural sorting functionality."""
        # Test with filenames that should be sorted naturally
        filenames = [
            "000001_obj1_cat_mask.png",
            "000010_obj1_cat_mask.png", 
            "000002_obj1_cat_mask.png",
            "000020_obj1_cat_mask.png"
        ]
        
        # Sort using natural sort key
        sorted_names = sorted(filenames, key=FilePatternMatcher.natural_sort_key)
        
        expected_order = [
            "000001_obj1_cat_mask.png",
            "000002_obj1_cat_mask.png",
            "000010_obj1_cat_mask.png",
            "000020_obj1_cat_mask.png"
        ]
        
        assert sorted_names == expected_order
    
    def test_natural_sort_mixed_content(self):
        """Test natural sorting with mixed text and numbers."""
        items = [
            "obj10_cat",
            "obj2_dog", 
            "obj1_bird",
            "obj20_fish"
        ]
        
        sorted_items = sorted(items, key=FilePatternMatcher.natural_sort_key)
        
        expected = [
            "obj1_bird",
            "obj2_dog",
            "obj10_cat", 
            "obj20_fish"
        ]
        
        assert sorted_items == expected


class TestOutputType:
    """Test OutputType enum functionality."""
    
    def test_output_type_values(self):
        """Test OutputType enum values exist."""
        assert hasattr(OutputType, 'BINARY')
        assert hasattr(OutputType, 'OVERLAY')
        assert hasattr(OutputType, 'MERGED')
        
        # Test that they are different
        assert OutputType.BINARY != OutputType.OVERLAY
        assert OutputType.OVERLAY != OutputType.MERGED
        assert OutputType.BINARY != OutputType.MERGED


class TestIntegrationPatterns:
    """Test integration between pattern classes."""
    
    def test_pattern_and_matcher_consistency(self):
        """Test that FilePattern outputs match FilePatternMatcher patterns."""
        # Generate a test filename using FilePattern
        test_filename = FilePattern.INDIVIDUAL_MASK.format(
            frame_num="000001", obj_id="1", prompt="cat"
        )
        
        # Test that it matches the corresponding regex pattern
        pattern = FilePatternMatcher.get_individual_mask_pattern()
        regex = re.compile(pattern)
        
        match = regex.match(test_filename)
        assert match is not None
        
        # Verify captured groups
        frame_num, obj_num, prompt = match.groups()
        assert frame_num == "000001"
        assert obj_num == "1"
        assert prompt == "cat"
    
    def test_directory_structure_and_patterns(self, tmp_path):
        """Test compatibility between DirectoryStructure and FilePattern."""
        base_dir = str(tmp_path)
        dirs = DirectoryStructure.get_directory_map(base_dir, include_video=True)
        
        # Create a test file using patterns in appropriate directory
        test_filename = FilePattern.INDIVIDUAL_MASK.format(
            frame_num="000001", obj_id="1", prompt="cat"
        )
        
        test_path = os.path.join(dirs["binary_frames"], test_filename)
        
        # Verify path components
        assert "binary" in test_path
        assert "frames" in test_path
        assert test_filename in test_path
    
    def test_video_pattern_consistency(self):
        """Test consistency between video patterns and directory structure."""
        # Generate video filename
        video_filename = FilePattern.VIDEO_MASK.format(obj_id="1", prompt="cat")
        assert video_filename == "1_cat_mask.mp4"
        
        # Verify it would fit in video directory structure
        base_dir = "/test"
        dirs = DirectoryStructure.get_directory_map(base_dir, include_video=True)
        
        video_path = os.path.join(dirs["video_binary"], video_filename)
        assert video_path == "/test/video/binary/1_cat_mask.mp4"


class TestEdgeCases:
    """Test edge cases in path configuration."""
    
    def test_special_characters_in_prompts(self):
        """Test handling of special characters in prompts."""
        special_prompts = [
            "red car",
            "cat with hat",
            "dog-in-park",
            "bird (flying)",
            "fish & chips"
        ]
        
        for prompt in special_prompts:
            # Should be able to format without raising exceptions
            filename = FilePattern.INDIVIDUAL_MASK.format(
                frame_num="000001", obj_id="1", prompt=prompt
            )
            assert "000001_obj1_" in filename
            assert "_mask.png" in filename
    
    def test_very_long_prompts(self):
        """Test handling of very long prompts."""
        long_prompt = "a very long prompt that describes a specific object in great detail with many adjectives"
        
        filename = FilePattern.INDIVIDUAL_MASK.format(
            frame_num="000001", obj_id="1", prompt=long_prompt
        )
        
        # Should handle long prompts
        assert long_prompt in filename
    
    def test_numeric_prompts(self):
        """Test handling of numeric prompts."""
        numeric_prompt = "123"
        
        filename = FilePattern.INDIVIDUAL_MASK.format(
            frame_num="000001", obj_id="1", prompt=numeric_prompt
        )
        
        assert "obj1_123_mask.png" in filename
    
    def test_empty_base_directory(self):
        """Test behavior with empty base directory."""
        dirs = DirectoryStructure.get_directory_map("", include_video=True)
        
        # Should handle empty base directory
        assert dirs["binary"] == "binary"
        assert dirs["video_binary"] == os.path.join("video", "binary")
    
    def test_absolute_path_handling(self, tmp_path):
        """Test handling of absolute paths."""
        absolute_base = str(tmp_path.absolute())
        dirs = DirectoryStructure.get_directory_map(absolute_base, include_video=True)
        
        # All paths should be absolute
        for path in dirs.values():
            assert os.path.isabs(path)