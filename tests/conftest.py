"""Shared fixtures and configuration for SOWLv2 tests."""
import pytest
import tempfile
import shutil
import os
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from typing import Dict, List, Any
import cv2

from sowlv2.data.config import PipelineBaseData, PipelineConfig


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    # Create a simple test image with different colored regions
    img_array = np.zeros((480, 640, 3), dtype=np.uint8)
    # Red rectangle (simulating an object)
    img_array[100:200, 100:300] = [255, 0, 0]
    # Blue rectangle (simulating another object)
    img_array[250:350, 350:550] = [0, 0, 255]
    # Green background
    img_array[img_array.sum(axis=2) == 0] = [0, 255, 0]

    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def sample_image_path(tmp_path, sample_image):
    """Create a sample image file."""
    image_path = tmp_path / "test_image.jpg"
    sample_image.save(image_path, "JPEG")
    return str(image_path)


@pytest.fixture
def sample_detection():
    """Create a sample detection result."""
    return {
        "box": [100, 100, 300, 200],
        "score": 0.95,
        "label": "a photo of cat",
        "core_prompt": "cat"
    }


@pytest.fixture
def sample_detections():
    """Create multiple sample detection results."""
    return [
        {
            "box": [100, 100, 300, 200],
            "score": 0.95,
            "label": "a photo of cat",
            "core_prompt": "cat"
        },
        {
            "box": [350, 250, 550, 350],
            "score": 0.87,
            "label": "a photo of dog",
            "core_prompt": "dog"
        }
    ]


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:300] = 255  # White region for detected object
    return mask


@pytest.fixture
def sample_masks():
    """Create multiple sample binary masks."""
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[100:200, 100:300] = 255

    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[250:350, 350:550] = 255

    return [mask1, mask2]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def pipeline_config_default():
    """Create default pipeline configuration."""
    return PipelineConfig(
        merged=True,
        binary=True,
        overlay=True
    )


@pytest.fixture
def pipeline_base_data_default():
    """Create default pipeline base data."""
    return PipelineBaseData(
        owl_model="google/owlv2-base-patch16-ensemble",
        sam_model="facebook/sam2.1-hiera-small",
        threshold=0.1,
        fps=24,
        device="cpu",  # Use CPU for tests
        pipeline_config=PipelineConfig(merged=True, binary=True, overlay=True)
    )


@pytest.fixture
def mock_owl_model(mocker):
    """Mock OWLv2 model to avoid loading actual models."""
    # Patch at the pipeline import location
    mock = mocker.patch('sowlv2.pipeline.OWLV2Wrapper')
    instance = mock.return_value
    # Default return: single detection
    instance.detect.return_value = [
        {
            "box": [100, 100, 300, 200],
            "score": 0.95,
            "label": "a photo of cat",
            "core_prompt": "cat"
        }
    ]
    return instance


@pytest.fixture
def mock_sam_model(mocker, sample_mask):
    """Mock SAM2 model."""
    # Patch at the pipeline import location
    mock = mocker.patch('sowlv2.pipeline.SAM2Wrapper')
    instance = mock.return_value
    instance.segment.return_value = sample_mask
    instance.init_state.return_value = "mock_state"
    instance.add_new_box.return_value = None
    instance.propagate_in_video.return_value = [
        (0, np.array([1]), np.array([[[sample_mask]]])),
        (1, np.array([1]), np.array([[[sample_mask]]]))
    ]
    return instance


@pytest.fixture
def test_config_yaml(tmp_path):
    """Create a test YAML configuration file."""
    config_data = {
        'prompt': ['cat', 'dog'],
        'input': 'test_input.jpg',  # Add required input field
        'owl-model': 'google/owlv2-base-patch16-ensemble',
        'sam-model': 'facebook/sam2.1-hiera-small',
        'threshold': 0.15,
        'fps': 30,
        'device': 'cpu'
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    return str(config_path)


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video file for testing."""
    video_path = tmp_path / "test_video.mp4"

    # Create a simple video with colored frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 24.0, (640, 480))

    # Create 5 frames
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Different colored rectangle in each frame (simulating movement)
        x_offset = i * 20
        frame[100:200, 100+x_offset:200+x_offset] = [255, 0, 0]  # Red moving rectangle
        out.write(frame)

    out.release()
    return str(video_path)


@pytest.fixture
def sample_frames_directory(tmp_path, sample_image):
    """Create a directory with sample frame images."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create 3 sample frames
    for i in range(3):
        frame_path = frames_dir / f"{i+1:06d}.jpg"
        sample_image.save(frame_path, "JPEG")

    return str(frames_dir)


@pytest.fixture(params=[
    (True, True, True),   # All enabled
    (False, True, True),  # --no-binary
    (True, False, True),  # --no-overlay
    (True, True, False),  # --no-merged
    (False, False, True), # --no-binary --no-overlay
    (False, True, False), # --no-binary --no-merged
    (True, False, False), # --no-overlay --no-merged
    (False, False, False) # All disabled
])
def flag_combinations(request):
    """Parametrized fixture for all flag combinations."""
    binary, overlay, merged = request.param
    return {
        'binary': binary,
        'overlay': overlay,
        'merged': merged,
        'config': PipelineConfig(binary=binary, overlay=overlay, merged=merged)
    }


@pytest.fixture
def expected_output_structure():
    """Define expected output directory structure."""
    return {
        'base_dirs': ['binary', 'overlay', 'video'],
        'binary_subdirs': ['frames', 'merged'],
        'overlay_subdirs': ['frames', 'merged'],
        'video_subdirs': ['binary', 'overlay'],
        'individual_mask_pattern': r'\d{6}_obj\d+_.*_mask\.png',
        'individual_overlay_pattern': r'\d{6}_obj\d+_.*_overlay\.png',
        'merged_mask_pattern': r'\d{6}_merged_mask\.png',
        'merged_overlay_pattern': r'\d{6}_merged_overlay\.png',
        'video_mask_pattern': r'obj\d+_.*_mask\.mp4',
        'video_overlay_pattern': r'obj\d+_.*_overlay\.mp4',
        'merged_video_mask': 'merged_mask.mp4',
        'merged_video_overlay': 'merged_overlay.mp4'
    }


def create_test_frames(directory: str, count: int = 5):
    """Utility function to create test frame images."""
    os.makedirs(directory, exist_ok=True)

    for i in range(count):
        frame_path = os.path.join(directory, f"{i+1:06d}.jpg")
        # Create simple test image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img.save(frame_path, "JPEG")


def create_test_video(path: str, num_frames: int = 10, fps: int = 24):
    """Utility function to create a test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (640, 480))

    for i in range(num_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)

    out.release()


def create_test_pipeline_config(**kwargs):
    """Create a complete PipelineBaseData configuration for testing."""
    defaults = {
        "owl_model": "google/owlv2-base-patch16-ensemble",
        "sam_model": "facebook/sam2.1-hiera-small",
        "threshold": 0.1,
        "fps": 24,
        "device": "cpu",
        "pipeline_config": PipelineConfig(binary=True, overlay=True, merged=True)
    }
    defaults.update(kwargs)
    return PipelineBaseData(**defaults)


def validate_output_structure(output_dir: str, flags: Dict[str, bool], input_type: str = "image"):
    """Utility function to validate output directory structure."""
    output_path = Path(output_dir)

    # Check base directories based on flags
    if flags.get('binary', True):
        assert (output_path / 'binary').exists(), "Binary directory should exist"
        if flags.get('merged', True):
            assert (output_path / 'binary' / 'merged').exists(), "Binary merged directory should exist"
        assert (output_path / 'binary' / 'frames').exists(), "Binary frames directory should exist"
    else:
        assert not (output_path / 'binary').exists(), "Binary directory should not exist"

    if flags.get('overlay', True):
        assert (output_path / 'overlay').exists(), "Overlay directory should exist"
        if flags.get('merged', True):
            assert (output_path / 'overlay' / 'merged').exists(), "Overlay merged directory should exist"
        assert (output_path / 'overlay' / 'frames').exists(), "Overlay frames directory should exist"
    else:
        assert not (output_path / 'overlay').exists(), "Overlay directory should not exist"

    # Video directory should exist for video inputs regardless of flags
    if input_type == "video":
        assert (output_path / 'video').exists(), "Video directory should exist for video input"
