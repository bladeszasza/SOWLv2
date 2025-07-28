"""Integration tests for SOWLv2 optimization modules."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from sowlv2.optimizations.parallel_processor import (
    ParallelConfig, ParallelDetectionProcessor,
    ParallelSegmentationProcessor, ParallelIOProcessor,
    BatchDetectionResult
)

from sowlv2.optimizations.gpu_optimizations import GPUOptimizer
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.data.config import (PipelineBaseData, PipelineConfig)


class TestParallelProcessing:
    """Test parallel processing optimizations."""

    @pytest.fixture
    def mock_models(self):
        """Create mock OWL and SAM models."""
        mock_owl = MagicMock()
        mock_owl.device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_owl.detect.return_value = [
            {"box": [10, 10, 50, 50], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]

        mock_sam = MagicMock()
        mock_sam.segment_from_box.return_value = np.ones((100, 100), dtype=np.uint8) * 255

        return mock_owl, mock_sam

    def test_parallel_detection_multiple_prompts(self, mock_models, sample_image):
        """Test parallel detection with multiple prompts."""
        mock_owl, mock_sam = mock_models

        # Configure mock to return different results for different prompts
        def mock_detect(_, prompts_list):
            # prompts_list is a list containing a single prompt
            prompt = prompts_list[0] if prompts_list else ""
            return [{
                "box": [10, 10, 50, 50],
                "score": 0.9,
                "label": f"a photo of {prompt}",
                "core_prompt": prompt
            }]

        mock_owl.detect_objects.side_effect = mock_detect

        # Test parallel detection - patch ProcessPoolExecutor to use ThreadPoolExecutor for mocks
        config = ParallelConfig(max_workers=2)
        processor = ParallelDetectionProcessor(mock_owl, mock_sam, config)

        prompts = ["cat", "dog", "bird", "car"]

        # Run parallel detection
        results = processor.detect_multiple_prompts_parallel(
            sample_image, prompts, threshold=0.1
        )

        # Verify results
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, BatchDetectionResult)
            assert len(result.detections) > 0
            assert result.success_count >= 0
            assert result.error_count >= 0

    def test_parallel_segmentation(self, mock_models, sample_image):
        """Test parallel segmentation processing."""
        _, mock_sam = mock_models

        # Configure mock to return numpy array
        mock_sam.segment.return_value = np.ones((100, 100), dtype=np.uint8) * 255

        # Create test detections
        detections = [
            {"box": [10, 10, 50, 50], "score": 0.9, "core_prompt": "cat"},
            {"box": [60, 60, 100, 100], "score": 0.8, "core_prompt": "dog"},
            {"box": [110, 110, 150, 150], "score": 0.7, "core_prompt": "bird"},
        ]

        config = ParallelConfig(max_workers=4)
        processor = ParallelSegmentationProcessor(mock_sam, config)

        results = processor.segment_detections_parallel(sample_image, detections)

        # Verify results
        assert len(results) == len(detections)
        for (_, mask) in results:
            assert mask is not None
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (100, 100)  # Check expected shape

    def test_parallel_io_saving(self, tmp_path):
        """Test parallel I/O operations."""
        # Create test images
        save_tasks = []
        for i in range(10):
            img = Image.new('RGB', (100, 100), color=(i*20, 0, 0))
            filepath = tmp_path / f"test_{i}.png"
            save_tasks.append((str(filepath), img))

        config = ParallelConfig(max_workers=4)
        processor = ParallelIOProcessor(config)

        # Time the parallel saving
        start_time = time.time()
        processor.save_outputs_parallel(save_tasks)
        parallel_time = time.time() - start_time

        # Verify all files were saved
        for filepath, _ in save_tasks:
            assert Path(filepath).exists()

        print(f"Parallel I/O completed in {parallel_time:.3f}s")


class TestGPUOptimizations:
    """Test GPU-specific optimizations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_optimizer_initialization(self):
        """Test GPU optimizer initialization."""
        optimizer = GPUOptimizer(device="cuda")

        assert optimizer.device == "cuda"
        assert optimizer.use_amp is True
        assert optimizer.scaler is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_context(self):
        """Test mixed precision autocast context."""
        optimizer = GPUOptimizer(device="cuda")

        # Create a simple model
        model = torch.nn.Linear(10, 10).cuda()
        input_tensor = torch.randn(1, 10).cuda()

        # Test autocast
        with optimizer.autocast_context():
            output = model(input_tensor)
            # In autocast, computations should be in float16
            assert output.dtype == torch.float16

    def test_model_optimization(self):
        """Test model optimization for inference."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        optimizer = GPUOptimizer(device=device)

        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )

        # Optimize model
        optimized_model = optimizer.optimize_model_for_inference(model)

        # Verify optimizations
        assert not optimized_model.training
        for param in optimized_model.parameters():
            assert not param.requires_grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_inference(self):
        """Test batched inference."""
        optimizer = GPUOptimizer(device="cuda")

        # Create a simple model
        model = torch.nn.Linear(10, 5).cuda()
        model = optimizer.optimize_model_for_inference(model)

        # Create test inputs
        inputs = [torch.randn(10) for _ in range(8)]

        # Run batch inference
        outputs = optimizer.batch_inference(model, inputs, batch_size=4)

        assert len(outputs) == len(inputs)
        for output in outputs:
            assert output.shape == (5,)


class TestOptimizedPipeline:
    """Test the optimized pipeline integration."""

    @pytest.fixture
    def optimized_pipeline(self, mocker):
        """Create an optimized pipeline with mocked models."""        
        # Mock the model initialization - patch both import paths
        mocker.patch('sowlv2.models.OWLV2Wrapper')
        mocker.patch('sowlv2.models.SAM2Wrapper')
        mocker.patch('sowlv2.pipeline.OWLV2Wrapper')
        mocker.patch('sowlv2.pipeline.SAM2Wrapper')

        config = PipelineBaseData(
            owl_model="google/owlv2-base-patch16-ensemble",
            sam_model="facebook/sam2.1-hiera-small",
            threshold=0.1,
            fps=24,
            device="cuda" if torch.cuda.is_available() else "cpu",
            pipeline_config=PipelineConfig(
                binary=True,
                overlay=True,
                merged=True
            )
        )
        parallel_config = ParallelConfig(
            max_workers=2,
            detection_batch_size=4,
            segmentation_batch_size=2
        )

        pipeline = OptimizedSOWLv2Pipeline(config, parallel_config)

        # Configure mocks
        pipeline.owl.detect.return_value = [
            {"box": [10, 10, 50, 50], "score": 0.9, "label": "cat", "core_prompt": "cat"}
        ]
        pipeline.sam.segment.return_value = np.ones((100, 100), dtype=np.uint8) * 255

        return pipeline

    def test_optimized_image_processing(self, optimized_pipeline, sample_image_path, tmp_path):
        """Test optimized image processing."""
        output_dir = str(tmp_path / "output")

        # Process with multiple prompts
        prompts = ["cat", "dog", "person"]

        # Mock the parallel processors to avoid actual parallel execution in tests
        with patch.object(optimized_pipeline.detection_processor,
                         'detect_multiple_prompts_parallel') as mock_detect, \
             patch.object(optimized_pipeline.segmentation_processor,
                         'segment_detections_parallel') as mock_segment:

            # Return mock batch results
            mock_detect.return_value = [
                BatchDetectionResult(
                    detections=[{
                        "box": [10, 10, 50, 50],
                        "score": 0.9,
                        "label": f"a photo of {p}",
                        "core_prompt": p
                    }]
                )
                for i, p in enumerate(prompts)
            ]

            # Mock segmentation results with correct image size
            mock_segment.return_value = [
                ({
                    "box": [10, 10, 50, 50],
                    "score": 0.9,
                    "label": f"a photo of {p}",
                    "core_prompt": p
                }, np.ones((480, 640), dtype=np.uint8) * 255)  # Match image size
                for p in prompts
            ]

            # Process image
            optimized_pipeline.process_image(sample_image_path, prompts, output_dir)

            # Verify parallel detection was called
            mock_detect.assert_called_once()

            # Verify output structure
            output_path = Path(output_dir)
            assert output_path.exists()


class TestPerformanceBenchmark:
    """Benchmark tests to measure optimization improvements."""

    @pytest.mark.benchmark
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_detection_speedup(self, benchmark, mock_models, sample_image):
        """Benchmark parallel vs sequential detection."""
        mock_owl, mock_sam = mock_models

        # Configure mock to simulate processing time
        def mock_detect_with_delay(_, prompt, __):
            time.sleep(0.01)  # Simulate 10ms processing
            return [{
                "box": [10, 10, 50, 50],
                "score": 0.9,
                "label": f"a photo of {prompt}",
                "core_prompt": prompt
            }]

        mock_owl.detect.side_effect = mock_detect_with_delay

        prompts = ["cat", "dog", "bird", "car", "person"]

        # Benchmark parallel processing
        config = ParallelConfig()
        processor = ParallelDetectionProcessor(mock_owl, mock_sam, config)

        result = benchmark(
            processor.detect_multiple_prompts_parallel,
            sample_image, prompts, 0.1
        )

        assert len(result) == len(prompts)
