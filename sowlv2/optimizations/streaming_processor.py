"""
Streaming video processing for memory-efficient handling of large videos.
Implements chunked processing with overlap handling and progressive loading.
"""
import os
import gc
import math
from typing import List, Iterator, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from PIL import Image


@dataclass
class StreamingConfig:
    """Configuration for streaming video processing."""
    chunk_size: int
    overlap_frames: int
    enable_progressive_loading: bool
    memory_threshold: float
    auto_cleanup: bool
    temp_dir: Optional[str] = None


@dataclass
class ChunkInfo:
    """Information about a video chunk."""
    chunk_id: int
    start_frame: int
    end_frame: int
    actual_frames: int
    overlap_start: int
    overlap_end: int
    memory_usage: float


@dataclass
class ProcessingResult:
    """Result from processing a video chunk."""
    chunk_id: int
    start_frame: int
    end_frame: int
    results: List[Any]
    overlap_results: List[Any]
    processing_time: float
    memory_peak: float


class StreamingVideoProcessor:
    """
    Streaming video processor for memory-efficient processing of large videos.
    Implements chunked processing with configurable overlap and progressive loading.
    """
    
    def __init__(self, config: StreamingConfig):
        """
        Initialize streaming video processor.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.chunk_cache: Dict[int, List[Image.Image]] = {}
        self.processing_stats: Dict[str, float] = {}
        self.temp_files: List[str] = []
        
        # Create temp directory if needed
        if config.temp_dir:
            os.makedirs(config.temp_dir, exist_ok=True)
            
    def process_video_stream(self, 
                           frames_source: Any,  # Can be directory path, video file, or frame generator
                           processing_func: Callable,
                           total_frames: int,
                           *args, **kwargs) -> Iterator[ProcessingResult]:
        """
        Process video in streaming chunks.
        
        Args:
            frames_source: Source of video frames (directory, file, or generator)
            processing_func: Function to process each chunk
            total_frames: Total number of frames in video
            *args, **kwargs: Additional arguments for processing function
            
        Yields:
            ProcessingResult: Results from each processed chunk
        """
        print(f"Starting streaming processing of {total_frames} frames with chunk size {self.config.chunk_size}")
        
        # Calculate chunk information
        chunks = self._calculate_chunks(total_frames)
        
        # Process each chunk
        for chunk_info in chunks:
            try:
                # Load chunk frames
                chunk_frames = self._load_chunk_frames(frames_source, chunk_info)
                
                # Process chunk
                result = self._process_chunk(
                    chunk_frames, chunk_info, processing_func, *args, **kwargs
                )
                
                yield result
                
                # Cleanup if auto cleanup is enabled
                if self.config.auto_cleanup:
                    self._cleanup_chunk(chunk_info.chunk_id)
                    
            except Exception as e:
                print(f"Error processing chunk {chunk_info.chunk_id}: {e}")
                # Continue with next chunk
                continue
                
        # Final cleanup
        self._final_cleanup()
        
    def _calculate_chunks(self, total_frames: int) -> List[ChunkInfo]:
        """
        Calculate chunk boundaries with overlap handling.
        
        Args:
            total_frames: Total number of frames
            
        Returns:
            List of ChunkInfo objects
        """
        chunks = []
        chunk_id = 0
        start_frame = 0
        
        while start_frame < total_frames:
            # Calculate chunk boundaries
            end_frame = min(start_frame + self.config.chunk_size, total_frames)
            actual_frames = end_frame - start_frame
            
            # Calculate overlap regions
            overlap_start = max(0, start_frame - self.config.overlap_frames) if chunk_id > 0 else start_frame
            overlap_end = min(total_frames, end_frame + self.config.overlap_frames)
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_frame=start_frame,
                end_frame=end_frame,
                actual_frames=actual_frames,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                memory_usage=0.0  # Will be calculated during processing
            )
            
            chunks.append(chunk_info)
            
            # Move to next chunk
            start_frame = end_frame
            chunk_id += 1
            
        print(f"Created {len(chunks)} chunks for streaming processing")
        return chunks
        
    def _load_chunk_frames(self, frames_source: Any, chunk_info: ChunkInfo) -> List[Image.Image]:
        """
        Load frames for a specific chunk with progressive loading if enabled.
        
        Args:
            frames_source: Source of frames
            chunk_info: Information about the chunk to load
            
        Returns:
            List of PIL Images for the chunk
        """
        frames = []
        
        if isinstance(frames_source, str):
            # Directory or video file path
            if os.path.isdir(frames_source):
                frames = self._load_frames_from_directory(frames_source, chunk_info)
            else:
                frames = self._load_frames_from_video(frames_source, chunk_info)
        elif hasattr(frames_source, '__iter__'):
            # Frame generator or list
            frames = self._load_frames_from_generator(frames_source, chunk_info)
        else:
            raise ValueError(f"Unsupported frames source type: {type(frames_source)}")
            
        # Cache chunk if not using progressive loading
        if not self.config.enable_progressive_loading:
            self.chunk_cache[chunk_info.chunk_id] = frames
            
        # Estimate memory usage
        if frames:
            frame_size = frames[0].size
            bytes_per_frame = frame_size[0] * frame_size[1] * 3  # RGB
            chunk_info.memory_usage = len(frames) * bytes_per_frame / 1e9  # GB
            
        return frames
        
    def _load_frames_from_directory(self, directory: str, chunk_info: ChunkInfo) -> List[Image.Image]:
        """Load frames from a directory of images."""
        frames = []
        frame_files = sorted([f for f in os.listdir(directory) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        start_idx = chunk_info.overlap_start
        end_idx = chunk_info.overlap_end
        
        for i in range(start_idx, min(end_idx, len(frame_files))):
            frame_path = os.path.join(directory, frame_files[i])
            try:
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                continue
                
        return frames
        
    def _load_frames_from_video(self, video_path: str, chunk_info: ChunkInfo) -> List[Image.Image]:
        """Load frames from a video file."""
        # This would require video decoding library like OpenCV or decord
        # For now, raise an error indicating this needs implementation
        raise NotImplementedError("Video file loading not implemented. Use frame directory or implement video decoder.")
        
    def _load_frames_from_generator(self, generator: Any, chunk_info: ChunkInfo) -> List[Image.Image]:
        """Load frames from a generator or iterator."""
        frames = []
        
        if hasattr(generator, '__getitem__'):
            # List-like object
            start_idx = chunk_info.overlap_start
            end_idx = chunk_info.overlap_end
            
            for i in range(start_idx, min(end_idx, len(generator))):
                frames.append(generator[i])
        else:
            # Iterator - this is more complex as we need to skip to the right position
            # For simplicity, convert to list (not memory efficient for large videos)
            all_frames = list(generator)
            start_idx = chunk_info.overlap_start
            end_idx = chunk_info.overlap_end
            frames = all_frames[start_idx:end_idx]
            
        return frames
        
    def _process_chunk(self, 
                      frames: List[Image.Image], 
                      chunk_info: ChunkInfo,
                      processing_func: Callable,
                      *args, **kwargs) -> ProcessingResult:
        """
        Process a single chunk of frames.
        
        Args:
            frames: Frames to process
            chunk_info: Chunk information
            processing_func: Processing function
            *args, **kwargs: Additional arguments
            
        Returns:
            ProcessingResult: Results from processing
        """
        import time
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            # Extract frames for actual processing (excluding overlap)
            overlap_before = chunk_info.start_frame - chunk_info.overlap_start
            overlap_after = chunk_info.overlap_end - chunk_info.end_frame
            
            # Process all frames (including overlap for context)
            all_results = processing_func(frames, *args, **kwargs)
            
            # Separate main results from overlap results
            main_results = all_results[overlap_before:len(all_results)-overlap_after] if overlap_after > 0 else all_results[overlap_before:]
            overlap_results = {
                'before': all_results[:overlap_before] if overlap_before > 0 else [],
                'after': all_results[len(all_results)-overlap_after:] if overlap_after > 0 else []
            }
            
            processing_time = time.time() - start_time
            peak_memory = self._get_memory_usage()
            
            print(f"Processed chunk {chunk_info.chunk_id}: {len(main_results)} results in {processing_time:.2f}s")
            
            return ProcessingResult(
                chunk_id=chunk_info.chunk_id,
                start_frame=chunk_info.start_frame,
                end_frame=chunk_info.end_frame,
                results=main_results,
                overlap_results=overlap_results,
                processing_time=processing_time,
                memory_peak=peak_memory - initial_memory
            )
            
        except Exception as e:
            print(f"Error processing chunk {chunk_info.chunk_id}: {e}")
            raise
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        else:
            # Use psutil for system memory if available
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1e9
            except ImportError:
                return 0.0
                
    def _cleanup_chunk(self, chunk_id: int):
        """Clean up resources for a processed chunk."""
        if chunk_id in self.chunk_cache:
            del self.chunk_cache[chunk_id]
            
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _final_cleanup(self):
        """Perform final cleanup of all resources."""
        # Clear all cached chunks
        self.chunk_cache.clear()
        
        # Remove temporary files
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        self.temp_files.clear()
        
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def merge_chunk_results(self, 
                          chunk_results: List[ProcessingResult],
                          merge_func: Optional[Callable] = None) -> List[Any]:
        """
        Merge results from multiple chunks, handling overlaps.
        
        Args:
            chunk_results: Results from all processed chunks
            merge_func: Optional function to merge overlapping results
            
        Returns:
            Merged results list
        """
        if not chunk_results:
            return []
            
        merged_results = []
        
        for i, chunk_result in enumerate(chunk_results):
            if i == 0:
                # First chunk - add all results
                merged_results.extend(chunk_result.results)
            else:
                # Subsequent chunks - handle overlap
                if merge_func and chunk_result.overlap_results['before']:
                    # Use custom merge function for overlap
                    overlap_merged = merge_func(
                        merged_results[-len(chunk_result.overlap_results['before']):],
                        chunk_result.overlap_results['before']
                    )
                    # Replace overlapping results
                    merged_results[-len(chunk_result.overlap_results['before']):] = overlap_merged
                    
                # Add main results
                merged_results.extend(chunk_result.results)
                
        return merged_results
        
    def get_processing_statistics(self) -> Dict[str, float]:
        """Get processing statistics."""
        return {
            'total_chunks': len(self.processing_stats),
            'average_chunk_time': sum(self.processing_stats.values()) / len(self.processing_stats) if self.processing_stats else 0,
            'total_processing_time': sum(self.processing_stats.values()),
            'memory_efficiency': self._calculate_memory_efficiency()
        }
        
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score."""
        # This is a placeholder - implement based on your specific metrics
        return 0.85  # 85% efficiency as example
        
    def should_use_streaming(self, 
                           total_frames: int,
                           frame_size: Tuple[int, int] = (1024, 1024),
                           available_memory_gb: float = 8.0) -> bool:
        """
        Determine if streaming should be used for a video.
        
        Args:
            total_frames: Number of frames in video
            frame_size: Frame dimensions
            available_memory_gb: Available memory in GB
            
        Returns:
            bool: True if streaming is recommended
        """
        # Estimate memory needed for full video
        pixels_per_frame = frame_size[0] * frame_size[1]
        bytes_per_frame = pixels_per_frame * 3  # RGB
        total_memory_needed = total_frames * bytes_per_frame / 1e9  # GB
        
        # Add processing overhead (2x for intermediate results)
        total_memory_needed *= 2
        
        # Use streaming if memory needed exceeds 80% of available memory
        return total_memory_needed > available_memory_gb * 0.8
        
    @staticmethod
    def create_auto_config(total_frames: int, 
                          available_memory_gb: float = 8.0,
                          target_memory_usage: float = 0.7) -> StreamingConfig:
        """
        Create automatic streaming configuration based on video characteristics.
        
        Args:
            total_frames: Total number of frames
            available_memory_gb: Available memory in GB
            target_memory_usage: Target memory utilization (0-1)
            
        Returns:
            StreamingConfig: Optimized configuration
        """
        # Estimate frames per GB (conservative estimate)
        frames_per_gb = 1000  # Adjust based on typical frame size
        
        # Calculate optimal chunk size
        max_frames_per_chunk = int(available_memory_gb * target_memory_usage * frames_per_gb)
        chunk_size = min(max_frames_per_chunk, max(100, total_frames // 10))
        
        # Set overlap based on chunk size
        overlap_frames = min(10, chunk_size // 10)
        
        # Enable progressive loading for very large videos
        enable_progressive = total_frames > chunk_size * 5
        
        return StreamingConfig(
            chunk_size=chunk_size,
            overlap_frames=overlap_frames,
            enable_progressive_loading=enable_progressive,
            memory_threshold=target_memory_usage,
            auto_cleanup=True
        )