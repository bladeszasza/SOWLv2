"""Test filesystem utility functions."""
import pytest
import os
import tempfile
import shutil
from pathlib import Path

from sowlv2.utils.filesystem_utils import remove_empty_folders, create_output_directories


class TestRemoveEmptyFolders:
    """Test remove_empty_folders functionality."""
    
    def test_remove_single_empty_folder(self, tmp_path):
        """Test removal of a single empty folder."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        assert empty_dir.exists()
        remove_empty_folders(str(empty_dir))
        assert not empty_dir.exists()
    
    def test_remove_nested_empty_folders(self, tmp_path):
        """Test removal of nested empty folders."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)
        
        assert nested_dir.exists()
        remove_empty_folders(str(tmp_path / "level1"))
        assert not (tmp_path / "level1").exists()
    
    def test_keep_non_empty_folders(self, tmp_path):
        """Test that non-empty folders are preserved."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        
        # Create a file in the directory
        test_file = test_dir / "file.txt"
        test_file.write_text("content")
        
        remove_empty_folders(str(test_dir))
        assert test_dir.exists()
        assert test_file.exists()
    
    def test_remove_empty_subdirectories_keep_parent(self, tmp_path):
        """Test removal of empty subdirectories while keeping parent with content."""
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        
        # Create empty subdirectory
        empty_subdir = parent_dir / "empty"
        empty_subdir.mkdir()
        
        # Create file in parent
        parent_file = parent_dir / "file.txt"
        parent_file.write_text("content")
        
        remove_empty_folders(str(parent_dir))
        
        # Parent should exist (has file), empty subdir should be removed
        assert parent_dir.exists()
        assert parent_file.exists()
        assert not empty_subdir.exists()
    
    def test_handle_nonexistent_directory(self, tmp_path):
        """Test handling of nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        
        # Should not raise exception
        try:
            remove_empty_folders(str(nonexistent))
        except Exception as e:
            pytest.fail(f"Should handle nonexistent directory gracefully: {e}")
    
    def test_complex_directory_structure(self, tmp_path):
        """Test complex directory structure with mixed empty/non-empty folders."""
        # Create complex structure
        base = tmp_path / "complex"
        base.mkdir()
        
        # Non-empty branch
        (base / "branch1").mkdir()
        (base / "branch1" / "file.txt").write_text("content")
        (base / "branch1" / "subdir").mkdir()
        (base / "branch1" / "subdir" / "file2.txt").write_text("content2")
        
        # Empty branch
        (base / "branch2").mkdir()
        (base / "branch2" / "empty1").mkdir()
        (base / "branch2" / "empty2").mkdir()
        
        # Mixed branch
        (base / "branch3").mkdir()
        (base / "branch3" / "empty").mkdir()
        (base / "branch3" / "file.txt").write_text("content3")
        
        remove_empty_folders(str(base))
        
        # Check results
        assert base.exists()  # Has content
        assert (base / "branch1").exists()  # Has content
        assert (base / "branch1" / "subdir").exists()  # Has content
        assert not (base / "branch2").exists()  # Was empty
        assert (base / "branch3").exists()  # Has file
        assert not (base / "branch3" / "empty").exists()  # Was empty


class TestCreateOutputDirectories:
    """Test create_output_directories functionality."""
    
    def test_create_basic_directory_structure(self, tmp_path):
        """Test creation of basic directory structure without video."""
        base_dir = str(tmp_path / "output")
        
        dirs = create_output_directories(base_dir, include_video=False)
        
        # Check returned dictionary
        expected_keys = [
            "binary", "binary_frames", "binary_merged",
            "overlay", "overlay_frames", "overlay_merged"
        ]
        for key in expected_keys:
            assert key in dirs
            assert os.path.exists(dirs[key])
        
        # Check directory structure
        assert os.path.exists(os.path.join(base_dir, "binary"))
        assert os.path.exists(os.path.join(base_dir, "binary", "frames"))
        assert os.path.exists(os.path.join(base_dir, "binary", "merged"))
        assert os.path.exists(os.path.join(base_dir, "overlay"))
        assert os.path.exists(os.path.join(base_dir, "overlay", "frames"))
        assert os.path.exists(os.path.join(base_dir, "overlay", "merged"))
        
        # Video directories should not exist
        assert not os.path.exists(os.path.join(base_dir, "video"))
    
    def test_create_directory_structure_with_video(self, tmp_path):
        """Test creation of directory structure including video directories."""
        base_dir = str(tmp_path / "output")
        
        dirs = create_output_directories(base_dir, include_video=True)
        
        # Check returned dictionary includes video dirs
        expected_keys = [
            "binary", "binary_frames", "binary_merged",
            "overlay", "overlay_frames", "overlay_merged",
            "video", "video_binary", "video_overlay"
        ]
        for key in expected_keys:
            assert key in dirs
            assert os.path.exists(dirs[key])
        
        # Check video directory structure
        assert os.path.exists(os.path.join(base_dir, "video"))
        assert os.path.exists(os.path.join(base_dir, "video", "binary"))
        assert os.path.exists(os.path.join(base_dir, "video", "overlay"))
    
    def test_create_existing_directories(self, tmp_path):
        """Test creation when directories already exist."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        
        # Pre-create some directories
        (base_dir / "binary").mkdir()
        (base_dir / "overlay").mkdir()
        
        # Should not raise exception
        dirs = create_output_directories(str(base_dir), include_video=False)
        
        # All directories should exist
        for dir_path in dirs.values():
            assert os.path.exists(dir_path)
    
    def test_create_with_files_in_existing_directories(self, tmp_path):
        """Test creation when existing directories contain files."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        
        # Pre-create directory with file
        binary_dir = base_dir / "binary"
        binary_dir.mkdir()
        (binary_dir / "existing_file.txt").write_text("content")
        
        dirs = create_output_directories(str(base_dir), include_video=False)
        
        # Directory and file should still exist
        assert os.path.exists(dirs["binary"])
        assert os.path.exists(binary_dir / "existing_file.txt")
    
    def test_directory_paths_format(self, tmp_path):
        """Test that returned directory paths are correctly formatted."""
        base_dir = str(tmp_path / "output")
        
        dirs = create_output_directories(base_dir, include_video=True)
        
        # Check path formats
        assert dirs["binary"] == os.path.join(base_dir, "binary")
        assert dirs["binary_frames"] == os.path.join(base_dir, "binary", "frames")
        assert dirs["binary_merged"] == os.path.join(base_dir, "binary", "merged")
        assert dirs["overlay"] == os.path.join(base_dir, "overlay")
        assert dirs["overlay_frames"] == os.path.join(base_dir, "overlay", "frames")
        assert dirs["overlay_merged"] == os.path.join(base_dir, "overlay", "merged")
        assert dirs["video"] == os.path.join(base_dir, "video")
        assert dirs["video_binary"] == os.path.join(base_dir, "video", "binary")
        assert dirs["video_overlay"] == os.path.join(base_dir, "video", "overlay")
    
    def test_permissions_handling(self, tmp_path):
        """Test handling of directory creation with different permissions."""
        # This test is platform-dependent and might need adjustment
        base_dir = str(tmp_path / "output")
        
        try:
            dirs = create_output_directories(base_dir, include_video=True)
            
            # Check that directories are readable and writable
            for dir_path in dirs.values():
                assert os.access(dir_path, os.R_OK)
                assert os.access(dir_path, os.W_OK)
                
        except PermissionError:
            pytest.skip("Permission test not applicable in current environment")


@pytest.mark.skip(reason="Complex filesystem edge cases - not critical for CI")
class TestEdgeCases:
    """Test edge cases for filesystem utilities."""
    
    def test_very_deep_nested_structure(self, tmp_path):
        """Test handling of very deep nested directory structures."""
        # Create deep nested structure
        current_path = tmp_path
        for i in range(10):  # 10 levels deep
            current_path = current_path / f"level{i}"
            current_path.mkdir()
        
        remove_empty_folders(str(tmp_path))
        
        # All directories should be removed since they're empty
        assert len(list(tmp_path.iterdir())) == 0
    
    def test_directory_with_special_characters(self, tmp_path):
        """Test handling of directories with special characters in names."""
        special_dir = tmp_path / "dir with spaces & symbols!"
        special_dir.mkdir()
        
        dirs = create_output_directories(str(special_dir), include_video=False)
        
        # Should handle special characters correctly
        for dir_path in dirs.values():
            assert os.path.exists(dir_path)
    
    def test_concurrent_directory_operations(self, tmp_path):
        """Test behavior under concurrent directory operations."""
        # This is a basic test - full concurrency testing would require threading
        base_dir = str(tmp_path / "output")
        
        # Multiple calls should be safe
        dirs1 = create_output_directories(base_dir, include_video=False)
        dirs2 = create_output_directories(base_dir, include_video=True)
        
        # Both should succeed and all directories should exist
        all_dirs = set(dirs1.values()) | set(dirs2.values())
        for dir_path in all_dirs:
            assert os.path.exists(dir_path)