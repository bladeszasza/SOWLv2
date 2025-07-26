"""
Unit tests for StreamingVideoProcessor.
Tests chunked processing, overlap handling, and progressive loading.
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from sowlv2.optimizations.streaming_processor import (
    StreamingVideoProcessor, StreamingConfig, ChunkInfo, ProcessingResult
)


class TestStreamingVideoProcessor:
    """Test suite for StreamingVideoProcessor class."""
    
    def test_init_with_config(self):
        """Test initialization with streaming configuration."""
        config = StreamingConfig(
            chunk_size=100,
            overlap_frames=5,
            enable_progressive_loading=True,
            memory_threshold=0.8,
            auto_cleanup=True,
            temp_dir="/tmp/streaming"
        )
        
        with patch('os.makedirs') as mock_makedirs:
            processor = StreamingVideoProcessor(config)
            
            assert processor.config == config
            assert processor.chunk_cache == {}
            assert processor.processing_stats == {}
            assert processor.temp_files == []
            mock_makedirs.assert_called_once_with("/tmp/streaming", exist_ok=True)
    
    def test_init_without_temp_dir(self):
        """Test initialization without temp directory."""
        config = StreamingConfig(
            chunk_size=50,
            overlap_frames=3,
            enable_progressive_loading=False,
            memory_threshold=0.7,
            auto_cleanup=False
        )
        
        processor = StreamingVideoProcessor(config)
        
        assert processor.config.temp_dir is None
    
    def test_calculate_chunks_basic(self):
        """Test basic chunk calculation."""
        config = StreamingConfig(
            chunk_size=100,
            overlap_frames=10,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        chunks = processor._calculate_chunks(250)  # 250 total frames
        
        assert len(chunks) == 3  # 3 chunks for 250 frames with chunk_size=100
        
        # Check first chunk
        assert chunks[0].chunk_id == 0
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 100
        assert chunks[0].actual_frames == 100
        assert chunks[0].overlap_start == 0  # No overlap before first chunk
        assert chunks[0].overlap_end == 110  # 100 + 10 overlap
        
        # Check middle chunk
        assert chunks[1].chunk_id == 1
        assert chunks[1].start_frame == 100
        assert chunks[1].end_frame == 200
        assert chunks[1].actual_frames == 100
        assert chunks[1].overlap_start == 90  # 100 - 10 overlap
        assert chunks[1].overlap_end == 210  # 200 + 10 overlap
        
        # Check last chunk
        assert chunks[2].chunk_id == 2
        assert chunks[2].start_frame == 200
        assert chunks[2].end_frame == 250
        assert chunks[2].actual_frames == 50
        assert chunks[2].overlap_start == 190  # 200 - 10 overlap
        assert chunks[2].overlap_end == 250  # Limited by total frames
    
    def test_calculate_chunks_small_video(self):
        """Test chunk calculation for small video."""
        config = StreamingConfig(
            chunk_size=100,
            overlap_frames=5,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        chunks = processor._calculate_chunks(50)  # Small video
        
        assert len(chunks) == 1
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 50
        assert chunks[0].actual_frames == 50
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 50
    
    def test_load_frames_from_directory(self):
        """Test loading frames from directory."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            for i in range(5):
                img = Image.new('RGB', (100, 100), color=(i*50, 0, 0))
                img.save(os.path.join(temp_dir, f"{i:06d}.jpg"))
            
            chunk_info = ChunkInfo(
                chunk_id=0,
                start_frame=0,
                end_frame=3,
                actual_frames=3,
                overlap_start=0,
                overlap_end=5,
                memory_usage=0.0
            )
            
            frames = processor._load_frames_from_directory(temp_dir, chunk_info)
            
            assert len(frames) == 5  # All frames in overlap range
            assert all(isinstance(frame, Image.Image) for frame in frames)
            assert chunk_info.memory_usage > 0  # Memory usage calculated
    
    def test_load_frames_from_directory_missing_files(self):
        """Test loading frames with missing files."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only some images
            for i in [0, 2, 4]:  # Skip 1 and 3
                img = Image.new('RGB', (100, 100), color=(i*50, 0, 0))
                img.save(os.path.join(temp_dir, f"{i:06d}.jpg"))
            
            chunk_info = ChunkInfo(
                chunk_id=0,
                start_frame=0,
                end_frame=3,
                actual_frames=3,
                overlap_start=0,
                overlap_end=5,
                memory_usage=0.0
            )
            
            frames = processor._load_frames_from_directory(temp_dir, chunk_info)
            
            assert len(frames) == 3  # Only existing frames loaded
    
    def test_load_frames_from_video_not_implemented(self):
        """Test that video file loading raises NotImplementedError."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        chunk_info = ChunkInfo(0, 0, 10, 10, 0, 10, 0.0)
        
        with pytest.raises(NotImplementedError):
            processor._load_frames_from_video("test.mp4", chunk_info)
    
    def test_load_frames_from_generator_list(self):
        """Test loading frames from list-like generator."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Create test frames
        test_frames = [
            Image.new('RGB', (100, 100), color=(i*30, 0, 0))
            for i in range(10)
        ]
        
        chunk_info = ChunkInfo(
            chunk_id=0,
            start_frame=2,
            end_frame=6,
            actual_frames=4,
            overlap_start=0,
            overlap_end=8,
            memory_usage=0.0
        )
        
        frames = processor._load_frames_from_generator(test_frames, chunk_info)
        
        assert len(frames) == 8  # overlap_end - overlap_start
        assert all(isinstance(frame, Image.Image) for frame in frames)
    
    def test_load_frames_from_generator_iterator(self):
        """Test loading frames from iterator."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Create test frames generator
        def frame_generator():
            for i in range(10):
                yield Image.new('RGB', (100, 100), color=(i*30, 0, 0))
        
        chunk_info = ChunkInfo(
            chunk_id=0,
            start_frame=2,
            end_frame=6,
            actual_frames=4,
            overlap_start=0,
            overlap_end=8,
            memory_usage=0.0
        )
        
        frames = processor._load_frames_from_generator(frame_generator(), chunk_info)
        
        assert len(frames) == 8
        assert all(isinstance(frame, Image.Image) for frame in frames)
    
    def test_process_chunk(self):
        """Test processing a single chunk."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Create test frames
        frames = [
            Image.new('RGB', (100, 100), color=(i*30, 0, 0))
            for i in range(8)  # 2 overlap before + 4 main + 2 overlap after
        ]
        
        chunk_info = ChunkInfo(
            chunk_id=0,
            start_frame=2,
            end_frame=6,
            actual_frames=4,
            overlap_start=0,
            overlap_end=8,
            memory_usage=0.0
        )
        
        # Mock processing function
        def mock_process_func(frames_batch):
            return [f"result_{i}" for i in range(len(frames_batch))]
        
        with patch.object(processor, '_get_memory_usage', side_effect=[1.0, 1.5]):
            result = processor._process_chunk(frames, chunk_info, mock_process_func)
        
        assert isinstance(result, ProcessingResult)
        assert result.chunk_id == 0
        assert result.start_frame == 2
        assert result.end_frame == 6
        assert len(result.results) == 4  # Main results only
        assert len(result.overlap_results['before']) == 2  # Overlap before
        assert len(result.overlap_results['after']) == 2  # Overlap after
        assert result.processing_time > 0
        assert result.memory_peak == 0.5  # 1.5 - 1.0
    
    def test_process_chunk_no_overlap_after(self):
        """Test processing chunk with no overlap after."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Create test frames (no overlap after)
        frames = [
            Image.new('RGB', (100, 100), color=(i*30, 0, 0))
            for i in range(6)  # 2 overlap before + 4 main
        ]
        
        chunk_info = ChunkInfo(
            chunk_id=1,
            start_frame=2,
            end_frame=6,
            actual_frames=4,
            overlap_start=0,
            overlap_end=6,  # No overlap after
            memory_usage=0.0
        )
        
        def mock_process_func(frames_batch):
            return [f"result_{i}" for i in range(len(frames_batch))]
        
        with patch.object(processor, '_get_memory_usage', side_effect=[1.0, 1.2]):
            result = processor._process_chunk(frames, chunk_info, mock_process_func)
        
        assert len(result.results) == 4
        assert len(result.overlap_results['before']) == 2
        assert len(result.overlap_results['after']) == 0
    
    def test_process_chunk_error_handling(self):
        """Test error handling in chunk processing."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        frames = [Image.new('RGB', (100, 100)) for _ in range(4)]
        chunk_info = ChunkInfo(0, 0, 4, 4, 0, 4, 0.0)
        
        def failing_process_func(frames_batch):
            raise Exception("Processing failed")
        
        with pytest.raises(Exception):
            processor._process_chunk(frames, chunk_info, failing_process_func)
    
    def test_get_memory_usage_cuda(self):
        """Test memory usage calculation with CUDA."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=2e9):  # 2GB
                memory_usage = processor._get_memory_usage()
                assert memory_usage == 2.0
    
    def test_get_memory_usage_cpu_with_psutil(self):
        """Test memory usage calculation with psutil."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('psutil.Process') as mock_process:
                mock_process.return_value.memory_info.return_value.rss = 1.5e9  # 1.5GB
                
                memory_usage = processor._get_memory_usage()
                assert memory_usage == 1.5
    
    def test_get_memory_usage_fallback(self):
        """Test memory usage fallback when psutil not available."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('builtins.__import__', side_effect=ImportError):
                memory_usage = processor._get_memory_usage()
                assert memory_usage == 0.0
    
    def test_cleanup_chunk(self):
        """Test chunk cleanup."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        processor.chunk_cache[0] = [Image.new('RGB', (100, 100)) for _ in range(5)]
        
        with patch('gc.collect') as mock_gc:
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.empty_cache') as mock_empty_cache:
                    processor._cleanup_chunk(0)
                    
                    assert 0 not in processor.chunk_cache
                    mock_gc.assert_called_once()
                    mock_empty_cache.assert_called_once()
    
    def test_final_cleanup(self):
        """Test final cleanup of all resources."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Add some cached data and temp files
        processor.chunk_cache[0] = [Image.new('RGB', (100, 100))]
        processor.chunk_cache[1] = [Image.new('RGB', (100, 100))]
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            processor.temp_files.append(temp_file.name)
        
        with patch('gc.collect') as mock_gc:
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.empty_cache') as mock_empty_cache:
                    processor._final_cleanup()
                    
                    assert len(processor.chunk_cache) == 0
                    assert len(processor.temp_files) == 0
                    mock_gc.assert_called_once()
                    mock_empty_cache.assert_called_once()
    
    def test_merge_chunk_results_basic(self):
        """Test basic chunk result merging."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Create mock chunk results
        chunk_results = [
            ProcessingResult(
                chunk_id=0,
                start_frame=0,
                end_frame=10,
                results=["result_0", "result_1", "result_2"],
                overlap_results={'before': [], 'after': ["overlap_1", "overlap_2"]},
                processing_time=1.0,
                memory_peak=0.5
            ),
            ProcessingResult(
                chunk_id=1,
                start_frame=10,
                end_frame=20,
                results=["result_3", "result_4", "result_5"],
                overlap_results={'before': ["overlap_1", "overlap_2"], 'after': []},
                processing_time=1.2,
                memory_peak=0.6
            )
        ]
        
        merged = processor.merge_chunk_results(chunk_results)
        
        assert len(merged) == 6  # All results combined
        assert merged == ["result_0", "result_1", "result_2", "result_3", "result_4", "result_5"]
    
    def test_merge_chunk_results_with_merge_func(self):
        """Test chunk result merging with custom merge function."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        chunk_results = [
            ProcessingResult(
                chunk_id=0,
                start_frame=0,
                end_frame=10,
                results=["A", "B", "C"],
                overlap_results={'before': [], 'after': ["C", "D"]},
                processing_time=1.0,
                memory_peak=0.5
            ),
            ProcessingResult(
                chunk_id=1,
                start_frame=10,
                end_frame=20,
                results=["E", "F", "G"],
                overlap_results={'before': ["C", "D"], 'after': []},
                processing_time=1.2,
                memory_peak=0.6
            )
        ]
        
        def custom_merge_func(existing, overlap):
            # Simple merge that combines strings
            return [f"{e}+{o}" for e, o in zip(existing, overlap)]
        
        merged = processor.merge_chunk_results(chunk_results, custom_merge_func)
        
        assert len(merged) == 5  # 3 from first + 2 merged + 3 from second - 2 overlap
        assert "C+C" in merged  # Merged overlap result
        assert "D+D" in merged  # Merged overlap result
    
    def test_merge_chunk_results_empty(self):
        """Test merging empty chunk results."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        merged = processor.merge_chunk_results([])
        assert merged == []
    
    def test_get_processing_statistics(self):
        """Test processing statistics calculation."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        processor.processing_stats = {
            'chunk_0': 1.0,
            'chunk_1': 1.5,
            'chunk_2': 0.8
        }
        
        stats = processor.get_processing_statistics()
        
        assert stats['total_chunks'] == 3
        assert stats['average_chunk_time'] == (1.0 + 1.5 + 0.8) / 3
        assert stats['total_processing_time'] == 3.3
        assert 'memory_efficiency' in stats
    
    def test_get_processing_statistics_empty(self):
        """Test processing statistics with no data."""
        config = StreamingConfig(
            chunk_size=10,
            overlap_frames=2,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        stats = processor.get_processing_statistics()
        
        assert stats['total_chunks'] == 0
        assert stats['average_chunk_time'] == 0
        assert stats['total_processing_time'] == 0
    
    def test_should_use_streaming_large_video(self):
        """Test streaming recommendation for large video."""
        config = StreamingConfig(
            chunk_size=100,
            overlap_frames=5,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        should_stream = processor.should_use_streaming(
            total_frames=5000,
            frame_size=(1920, 1080),
            available_memory_gb=8.0
        )
        
        assert should_stream is True
    
    def test_should_use_streaming_small_video(self):
        """Test streaming recommendation for small video."""
        config = StreamingConfig(
            chunk_size=100,
            overlap_frames=5,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        should_stream = processor.should_use_streaming(
            total_frames=100,
            frame_size=(512, 512),
            available_memory_gb=16.0
        )
        
        assert should_stream is False
    
    def test_create_auto_config(self):
        """Test automatic configuration creation."""
        config = StreamingVideoProcessor.create_auto_config(
            total_frames=2000,
            available_memory_gb=8.0,
            target_memory_usage=0.7
        )
        
        assert isinstance(config, StreamingConfig)
        assert config.chunk_size > 0
        assert config.overlap_frames >= 0
        assert config.memory_threshold == 0.7
        assert config.auto_cleanup is True
    
    def test_create_auto_config_large_video(self):
        """Test automatic configuration for large video."""
        config = StreamingVideoProcessor.create_auto_config(
            total_frames=10000,
            available_memory_gb=4.0,
            target_memory_usage=0.6
        )
        
        assert config.enable_progressive_loading is True
        assert config.chunk_size < 10000
    
    def test_create_auto_config_small_video(self):
        """Test automatic configuration for small video."""
        config = StreamingVideoProcessor.create_auto_config(
            total_frames=500,
            available_memory_gb=16.0,
            target_memory_usage=0.8
        )
        
        assert config.enable_progressive_loading is False
        assert config.chunk_size >= 100  # Minimum chunk size
    
    def test_process_video_stream_integration(self):
        """Test complete video streaming process."""
        config = StreamingConfig(
            chunk_size=3,
            overlap_frames=1,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        # Create test frames
        test_frames = [
            Image.new('RGB', (50, 50), color=(i*30, 0, 0))
            for i in range(8)
        ]
        
        def mock_processing_func(frames_batch):
            return [f"processed_{i}" for i in range(len(frames_batch))]
        
        with patch.object(processor, '_get_memory_usage', return_value=1.0):
            results = list(processor.process_video_stream(
                test_frames, mock_processing_func, 8
            ))
        
        assert len(results) == 3  # 3 chunks for 8 frames with chunk_size=3
        assert all(isinstance(result, ProcessingResult) for result in results)
        assert results[0].chunk_id == 0
        assert results[1].chunk_id == 1
        assert results[2].chunk_id == 2
    
    def test_process_video_stream_with_directory(self):
        """Test video streaming with directory source."""
        config = StreamingConfig(
            chunk_size=2,
            overlap_frames=1,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=False
        )
        
        processor = StreamingVideoProcessor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            for i in range(4):
                img = Image.new('RGB', (50, 50), color=(i*60, 0, 0))
                img.save(os.path.join(temp_dir, f"{i:06d}.jpg"))
            
            def mock_processing_func(frames_batch):
                return [f"processed_{len(frames_batch)}"]
            
            with patch.object(processor, '_get_memory_usage', return_value=0.5):
                results = list(processor.process_video_stream(
                    temp_dir, mock_processing_func, 4
                ))
            
            assert len(results) == 2  # 2 chunks for 4 frames
            assert all(isinstance(result, ProcessingResult) for result in results)
    
    def test_process_video_stream_error_recovery(self):
        """Test error recovery in video streaming."""
        config = StreamingConfig(
            chunk_size=2,
            overlap_frames=0,
            enable_progressive_loading=False,
            memory_threshold=0.8,
            auto_cleanup=True
        )
        
        processor = StreamingVideoProcessor(config)
        
        test_frames = [Image.new('RGB', (50, 50)) for _ in range(4)]
        
        def failing_processing_func(frames_batch):
            if len(frames_batch) == 2:  # Fail on first chunk
                raise Exception("Processing failed")
            return ["success"]
        
        with patch.object(processor, '_get_memory_usage', return_value=0.5):
            results = list(processor.process_video_stream(
                test_frames, failing_processing_func, 4
            ))
        
        # Should continue processing despite first chunk failure
        assert len(results) == 1  # Only second chunk succeeded
        assert results[0].chunk_id == 1