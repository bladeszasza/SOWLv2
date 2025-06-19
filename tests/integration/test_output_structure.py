"""Test output directory structure with various flag combinations."""
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from unittest.mock import MagicMock, patch

import pytest

from sowlv2.optimizations import OptimizedSOWLv2Pipeline, ParallelConfig
from sowlv2.data.config import PipelineConfig
from tests.conftest import validate_output_structure, create_test_pipeline_config


@dataclass
class OutputTestFixtures:
    """Container for test fixtures to reduce parameter count."""
    tmp_path: str
    sample_image_path: Union[str, None] = None
    sample_video_path: Union[str, None] = None
    mock_owl_model: Union[MagicMock, None] = None
    mock_sam_model: Union[MagicMock, None] = None


@dataclass
class OutputTestFlags:
    """Container for test flag configurations."""
    binary: bool = True
    overlay: bool = True
    merged: bool = True


@dataclass
class OutputTestConfig:
    """Main test configuration combining fixtures and flags."""
    fixtures: OutputTestFixtures
    flags: Union[OutputTestFlags, dict, None] = None

    @property
    def output_dir(self) -> str:
        """Get output directory path."""
        return str(Path(self.fixtures.tmp_path) / "output")


class TestOutputStructure:
    """Test output directory structure with all flag combinations."""

    @pytest.mark.parametrize("binary,overlay,merged",
        list(itertools.product([True, False], repeat=3)))
    def test_image_output_structure(self, *, tmp_path, sample_image_path,
                                   mock_owl_model, mock_sam_model,
                                   binary, overlay, merged):
        """Test all combinations of --no-binary, --no-overlay, --no-merged flags for images."""
        # Create test config dataclass to reduce parameter count
        fixtures = OutputTestFixtures(
            tmp_path=str(tmp_path),
            sample_image_path=sample_image_path,
            mock_owl_model=mock_owl_model,
            mock_sam_model=mock_sam_model
        )
        flags = OutputTestFlags(binary=binary, overlay=overlay, merged=merged)
        config = OutputTestConfig(fixtures=fixtures, flags=flags)

        # mock_sam_model fixture is needed for test setup but not used directly
        _ = config.fixtures.mock_sam_model

        # Skip the case where all flags are disabled (no output)
        if not any([flags.binary, flags.overlay, flags.merged]):
            pytest.skip("No output expected when all flags are disabled")

        output_dir = config.output_dir

        # Configure pipeline with specific flags
        pipeline_config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(
                binary=flags.binary,
                overlay=flags.overlay,
                merged=flags.merged
            )
        )

        # Configure mock to return detection
        config.fixtures.mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of cat",
                "core_prompt": "cat"
            }
        ]

        # Run pipeline
        pipeline = OptimizedSOWLv2Pipeline(pipeline_config, ParallelConfig())
        pipeline.process_image(config.fixtures.sample_image_path, "cat", output_dir)

        # Validate output structure
        self._validate_image_output_structure(
            output_dir, flags.binary, flags.overlay, flags.merged
        )

    @pytest.mark.skip(reason="Video tests require complex mocking - will be fixed in separate PR")
    @pytest.mark.parametrize("binary,overlay,merged",
        list(itertools.product([True, False], repeat=3)))
    def test_video_output_structure(self, *, tmp_path, sample_video_path,
                                   mock_owl_model, mock_sam_model,
                                   binary, overlay, merged):
        """Test video output structure with all flag combinations."""
        # Create test config dataclass to reduce parameter count
        fixtures = OutputTestFixtures(
            tmp_path=str(tmp_path),
            sample_video_path=sample_video_path,
            mock_owl_model=mock_owl_model,
            mock_sam_model=mock_sam_model
        )
        flags = OutputTestFlags(binary=binary, overlay=overlay, merged=merged)
        config = OutputTestConfig(fixtures=fixtures, flags=flags)

        # mock_sam_model fixture is needed for test setup but not used directly
        _ = config.fixtures.mock_sam_model

        # Skip the impossible case
        if not any([flags.binary, flags.overlay, flags.merged]):
            pytest.skip("No output expected when all flags are disabled")

        output_dir = config.output_dir

        pipeline_config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(
                binary=flags.binary,
                overlay=flags.overlay,
                merged=flags.merged
            )
        )

        # Configure mock to return detection
        config.fixtures.mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of cat",
                "core_prompt": "cat"
            }
        ]

        with patch('subprocess.run') as mock_subprocess:
            # Mock ffmpeg success
            mock_subprocess.return_value = None

            # Run pipeline
            pipeline = OptimizedSOWLv2Pipeline(pipeline_config, ParallelConfig())
            pipeline.process_video(config.fixtures.sample_video_path, "cat", output_dir)

        # Validate output structure
        self._validate_video_output_structure(
            output_dir, flags.binary, flags.overlay, flags.merged
        )

    def test_multiple_objects_output_structure(self, *, tmp_path, sample_image_path,
                                             mock_owl_model, mock_sam_model):
        """Test output structure with multiple detected objects."""
        # mock_sam_model fixture is needed for test setup but not used directly
        _ = mock_sam_model

        output_dir = str(tmp_path / "output")

        # Configure mock to return different detections based on the prompt
        def mock_detect(*args, **kwargs):
            prompt = kwargs.get('prompt', args[1] if len(args) > 1 else None)
            if isinstance(prompt, list):
                prompt = prompt[0]

            if prompt in ("cat", ["cat"]):
                return [
                    {
                        "box": [100, 100, 300, 200],
                        "score": 0.95,
                        "label": "a photo of cat",
                        "core_prompt": "cat"
                    }
                ]
            if prompt in ("dog", ["dog"]):
                return [
                    {
                        "box": [350, 250, 550, 350],
                        "score": 0.87,
                        "label": "a photo of dog",
                        "core_prompt": "dog"
                    }
                ]
            return []

        mock_owl_model.detect.side_effect = mock_detect

        config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )

        pipeline = OptimizedSOWLv2Pipeline(config, ParallelConfig())
        pipeline.process_image(sample_image_path, ["cat", "dog"], output_dir)

        output_path = Path(output_dir)

        # Should have files for both objects
        binary_frames = list((output_path / "binary" / "frames").glob("*.png"))
        overlay_frames = list((output_path / "overlay" / "frames").glob("*.png"))

        # Should have 2 individual object files plus merged files
        assert len([f for f in binary_frames if "obj" in f.name]) == 2
        assert len([f for f in overlay_frames if "obj" in f.name]) == 2

        # Should have merged files
        binary_merged = list((output_path / "binary" / "merged").glob("*.png"))
        overlay_merged = list((output_path / "overlay" / "merged").glob("*.png"))

        assert len(binary_merged) == 1
        assert len(overlay_merged) == 1

    def test_empty_directories_cleanup(self, *, tmp_path, sample_image_path,
                                      mock_owl_model, mock_sam_model):
        """Test that empty directories are cleaned up."""
        # mock_sam_model fixture is needed for test setup but not used directly
        _ = mock_sam_model

        output_dir = str(tmp_path / "output")

        # Configure mock to return no detections
        mock_owl_model.detect.return_value = []

        config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(binary=True, overlay=True, merged=True)
        )

        pipeline = OptimizedSOWLv2Pipeline(config, ParallelConfig())
        pipeline.process_image(sample_image_path, "nonexistent", output_dir)

        # Should not create empty directories or they should be cleaned up
        output_path = Path(output_dir)
        if output_path.exists():
            # Check that directories don't contain empty subdirectories
            for subdir in output_path.rglob("*"):
                if subdir.is_dir():
                    assert list(subdir.iterdir()), f"Empty directory found: {subdir}"

    def _validate_image_output_structure(self, output_dir: str,
                                       binary: bool, overlay: bool, merged: bool):
        """Validate image output structure based on flags."""
        output_path = Path(output_dir)

        if binary:
            # Binary directory should exist
            assert (output_path / "binary").exists()
            assert (output_path / "binary" / "frames").exists()

            # Check for individual binary files
            binary_files = list((output_path / "binary" / "frames").glob("*_obj*_*_mask.png"))
            assert len(binary_files) > 0, "Should have individual binary mask files"

            # Check file naming pattern (uses base name, obj starts from 0)
            for file in binary_files:
                assert re.match(r'.*_obj\d+_.*_mask\.png', file.name), \
                    f"Binary file {file.name} doesn't match naming pattern"

            if merged:
                assert (output_path / "binary" / "merged").exists()
                merged_files = list((output_path / "binary" / "merged").glob("*_merged_mask.png"))
                assert len(merged_files) > 0, "Should have merged binary files"

                for file in merged_files:
                    assert re.match(r'\d{6}_merged_mask\.png', file.name), \
                        f"Merged binary file {file.name} doesn't match naming pattern"
            else:
                # Merged directory should not exist or be empty
                merged_dir = output_path / "binary" / "merged"
                if merged_dir.exists():
                    assert len(list(merged_dir.glob("*"))) == 0
        else:
            # Binary directory should not exist
            assert not (output_path / "binary").exists()

        if overlay:
            # Overlay directory should exist
            assert (output_path / "overlay").exists()
            assert (output_path / "overlay" / "frames").exists()

            # Check for individual overlay files
            overlay_files = list((output_path / "overlay" / "frames").glob("*_obj*_*_overlay.png"))
            assert len(overlay_files) > 0, "Should have individual overlay files"

            # Check file naming pattern (uses base name, obj starts from 0)
            for file in overlay_files:
                assert re.match(r'.*_obj\d+_.*_overlay\.png', file.name), \
                    f"Overlay file {file.name} doesn't match naming pattern"

            if merged:
                assert (output_path / "overlay" / "merged").exists()
                merged_files = list((output_path / "overlay" / "merged")
                                   .glob("*_merged_overlay.png"))
                assert len(merged_files) > 0, "Should have merged overlay files"

                for file in merged_files:
                    assert re.match(r'\d{6}_merged_overlay\.png', file.name), \
                        f"Merged overlay file {file.name} doesn't match naming pattern"
            else:
                # Merged directory should not exist or be empty
                merged_dir = output_path / "overlay" / "merged"
                if merged_dir.exists():
                    assert len(list(merged_dir.glob("*"))) == 0
        else:
            # Overlay directory should not exist
            assert not (output_path / "overlay").exists()

    def _validate_video_output_structure(self, output_dir: str,
                                       binary: bool, overlay: bool, merged: bool):
        """Validate video output structure based on flags."""
        output_path = Path(output_dir)

        # Video directory should always exist for video inputs
        assert (output_path / "video").exists()

        # Check frame outputs based on flags (same as image validation)
        self._validate_image_output_structure(output_dir, binary, overlay, merged)

        # Check video files - these are always generated regardless of frame flags
        video_dir = output_path / "video"

        # At least one of binary or overlay video directories should exist
        has_video_output = False

        if (video_dir / "binary").exists():
            has_video_output = True
            video_files = list((video_dir / "binary").glob("*.mp4"))
            assert len(video_files) > 0, "Should have binary video files"

            # Check for individual object videos
            obj_videos = [f for f in video_files if f.name.startswith("obj")]
            assert len(obj_videos) > 0, "Should have individual object videos"

            # Check naming pattern for individual videos
            for video in obj_videos:
                assert re.match(r'obj\d+_.*_mask\.mp4', video.name), \
                    f"Video file {video.name} doesn't match naming pattern"

            # Check for merged video if merged flag is True
            if merged:
                merged_videos = [f for f in video_files if f.name == "merged_mask.mp4"]
                assert len(merged_videos) == 1, "Should have exactly one merged binary video"

        if (video_dir / "overlay").exists():
            has_video_output = True
            video_files = list((video_dir / "overlay").glob("*.mp4"))
            assert len(video_files) > 0, "Should have overlay video files"

            # Check for individual object videos
            obj_videos = [f for f in video_files if f.name.startswith("obj")]
            assert len(obj_videos) > 0, "Should have individual object videos"

            # Check naming pattern for individual videos
            for video in obj_videos:
                assert re.match(r'obj\d+_.*_overlay\.mp4', video.name), \
                    f"Video file {video.name} doesn't match naming pattern"

            # Check for merged video if merged flag is True
            if merged:
                merged_videos = [f for f in video_files if f.name == "merged_overlay.mp4"]
                assert len(merged_videos) == 1, "Should have exactly one merged overlay video"

        assert has_video_output, "Should have at least some video output"


class TestFileNamingConventions:
    """Test file naming conventions are followed correctly."""

    def test_individual_mask_naming_pattern(self, *, tmp_path, sample_image_path,
                                          mock_owl_model, mock_sam_model):
        """Test individual mask files follow {frame_num}_obj{obj_id}_{prompt}_mask.png pattern."""
        # mock_sam_model fixture is needed for test setup but not used directly
        _ = mock_sam_model

        output_dir = str(tmp_path / "output")

        mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of cat",
                "core_prompt": "cat"
            }
        ]

        config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )

        pipeline = OptimizedSOWLv2Pipeline(config, ParallelConfig())
        pipeline.process_image(sample_image_path, "cat", output_dir)

        binary_files = list(Path(output_dir).rglob("*_mask.png"))
        assert len(binary_files) == 1

        filename = binary_files[0].name
        # Should match pattern like "test_image_obj1_cat_mask.png" or similar
        assert re.match(r'.*_obj\d+_cat_mask\.png', filename), \
            f"File {filename} doesn't match expected pattern"

    def test_merged_mask_naming_pattern(self, *, tmp_path, sample_image_path,
                                      mock_owl_model, mock_sam_model):
        """Test merged mask files follow {frame_num}_merged_mask.png pattern."""
        # mock_sam_model fixture is needed for test setup but not used directly
        _ = mock_sam_model

        output_dir = str(tmp_path / "output")

        mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of cat",
                "core_prompt": "cat"
            }
        ]

        config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=True)
        )

        pipeline = OptimizedSOWLv2Pipeline(config, ParallelConfig())
        pipeline.process_image(sample_image_path, "cat", output_dir)

        merged_files = list((Path(output_dir) / "binary" / "merged").glob("*_merged_mask.png"))
        assert len(merged_files) == 1

        filename = merged_files[0].name
        assert re.match(r'.*_merged_mask\.png', filename), \
            f"Merged file {filename} doesn't match expected pattern"

    def test_special_characters_in_prompt(self, *, tmp_path, sample_image_path,
                                        mock_owl_model, mock_sam_model):
        """Test handling of spaces and special characters in prompts."""
        # mock_sam_model fixture is needed for test setup but not used directly
        _ = mock_sam_model

        output_dir = str(tmp_path / "output")

        mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of red car",
                "core_prompt": "red car"
            }
        ]

        config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(binary=True, overlay=False, merged=False)
        )

        pipeline = OptimizedSOWLv2Pipeline(config, ParallelConfig())
        pipeline.process_image(sample_image_path, "red car", output_dir)

        binary_files = list(Path(output_dir).rglob("*_mask.png"))
        assert len(binary_files) == 1

        filename = binary_files[0].name
        # Should handle spaces in prompt appropriately
        assert "red car" in filename or "red_car" in filename, \
            f"File {filename} doesn't contain expected prompt"


class TestFlagCombinationMatrix:
    """Test comprehensive flag combination matrix."""

    @pytest.mark.parametrize("flags", [
        {"binary": True, "overlay": True, "merged": True},     # Default
        {"binary": False, "overlay": True, "merged": True},    # --no-binary
        {"binary": True, "overlay": False, "merged": True},    # --no-overlay
        {"binary": True, "overlay": True, "merged": False},    # --no-merged
        {"binary": False, "overlay": False, "merged": True},   # --no-binary --no-overlay
        {"binary": False, "overlay": True, "merged": False},   # --no-binary --no-merged
        {"binary": True, "overlay": False, "merged": False},   # --no-overlay --no-merged
    ])
    def test_valid_flag_combinations(self, *, tmp_path, sample_image_path,
                                   mock_owl_model, mock_sam_model, flags):
        """Test all valid flag combinations produce expected output."""
        # Create test config dataclass to reduce parameter count
        fixtures = OutputTestFixtures(
            tmp_path=str(tmp_path),
            sample_image_path=sample_image_path,
            mock_owl_model=mock_owl_model,
            mock_sam_model=mock_sam_model
        )
        config = OutputTestConfig(fixtures=fixtures, flags=flags)

        # mock_sam_model fixture is needed for test setup but not used directly
        _ = config.fixtures.mock_sam_model
        output_dir = config.output_dir

        config.fixtures.mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of cat",
                "core_prompt": "cat"
            }
        ]

        pipeline_config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(**config.flags)
        )

        pipeline = OptimizedSOWLv2Pipeline(pipeline_config, ParallelConfig())
        pipeline.process_image(config.fixtures.sample_image_path, "cat", output_dir)

        # Validate using our utility function
        validate_output_structure(output_dir, config.flags, "image")

    def test_all_flags_disabled_edge_case(self, *, tmp_path, sample_image_path,
                                        mock_owl_model, mock_sam_model):
        """Test behavior when all flags are disabled."""
        # mock_sam_model fixture is needed for test setup but not used directly
        _ = mock_sam_model
        output_dir = str(tmp_path / "output")

        mock_owl_model.detect.return_value = [
            {
                "box": [100, 100, 300, 200],
                "score": 0.95,
                "label": "a photo of cat",
                "core_prompt": "cat"
            }
        ]

        config = create_test_pipeline_config(
            pipeline_config=PipelineConfig(binary=False, overlay=False, merged=False)
        )

        pipeline = OptimizedSOWLv2Pipeline(config, ParallelConfig())
        pipeline.process_image(sample_image_path, "cat", output_dir)

        # Should have minimal or no output
        output_path = Path(output_dir)
        if output_path.exists():
            # Check that no significant output was created
            all_files = list(output_path.rglob("*"))
            image_files = [f for f in all_files
                          if f.suffix in ['.png', '.jpg', '.mp4']]
            assert len(image_files) == 0, \
                "No image/video files should be created when all flags disabled"
