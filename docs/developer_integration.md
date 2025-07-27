# Developer Integration Guide

## Overview

This guide provides detailed information for developers who want to integrate SOWLv2 with EdgeTAM into their applications, extend the functionality, or contribute to the project. It covers architecture, extension points, and best practices for development.

## Architecture Overview

### Core Components

SOWLv2 with EdgeTAM integration follows a modular architecture:

```
sowlv2/
├── models/                 # Model wrappers and factories
│   ├── edgetam_wrapper.py  # EdgeTAM integration
│   ├── sam2_wrapper.py     # SAM2 integration
│   └── model_factory.py    # Model creation and management
├── optimizations/          # Performance optimization modules
│   ├── resource_manager.py # Memory and resource management
│   ├── batch_optimizer.py  # Batch processing optimization
│   ├── streaming_processor.py # Large video streaming
│   ├── vjepa2_optimization.py # V-JEPA2 enhancements
│   └── optimized_pipeline.py # Main pipeline controller
├── utils/                  # Utility modules
│   ├── error_recovery.py   # Error handling and recovery
│   ├── enhanced_logger.py  # Advanced logging
│   └── pipeline_utils.py   # Common utilities
└── data/                   # Configuration and data structures
    └── config.py           # Configuration classes
```

### Design Patterns

The codebase follows several key design patterns:

1. **Factory Pattern**: Model creation through `SegmentationModelFactory`
2. **Strategy Pattern**: Different optimization strategies based on content/hardware
3. **Observer Pattern**: Performance monitoring and event handling
4. **Adapter Pattern**: Unified interface for different segmentation models
5. **Builder Pattern**: Configuration building and validation

## Integration Patterns

### Basic Integration

#### Simple Video Processing

```python
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.data.config import OptimizationConfig, EdgeTAMConfig

def process_video_simple(video_path, prompts, output_dir):
    """Simple video processing with EdgeTAM."""
    
    # Configure EdgeTAM
    edgetam_config = EdgeTAMConfig(
        model_name="facebook/edgetam-base",
        optimization_level=2
    )
    
    # Configure optimization
    opt_config = OptimizationConfig(
        enable_mixed_precision=True,
        memory_limit_gb=8.0
    )
    
    # Initialize pipeline
    pipeline = OptimizedSOWLv2Pipeline(
        edgetam_config=edgetam_config,
        optimization_config=opt_config
    )
    
    # Process video
    results = pipeline.process_video(
        video_path=video_path,
        prompts=prompts,
        output_dir=output_dir
    )
    
    return results
```

#### Batch Processing Integration

```python
from sowlv2.optimizations.batch_optimizer import IntelligentBatchOptimizer
from sowlv2.models.model_factory import SegmentationModelFactory
import concurrent.futures

class BatchVideoProcessor:
    """Batch video processing with intelligent optimization."""
    
    def __init__(self, model_type="edgetam", device="cuda"):
        self.model = SegmentationModelFactory.create_model(
            model_type=model_type,
            model_name=f"facebook/{model_type}-base",
            device=device
        )
        self.batch_optimizer = IntelligentBatchOptimizer(
            device=device,
            initial_batch_size=16
        )
    
    def process_video_batch(self, video_paths, prompts, max_workers=4):
        """Process multiple videos in parallel."""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for video_path in video_paths:
                future = executor.submit(
                    self._process_single_video,
                    video_path,
                    prompts
                )
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Video processing failed: {e}")
                    results.append(None)
            
            return results
    
    def _process_single_video(self, video_path, prompts):
        """Process a single video with optimization."""
        # Implementation details...
        pass
```

### Advanced Integration

#### Custom Model Integration

```python
from sowlv2.models.model_factory import SegmentationModelFactory
from abc import ABC, abstractmethod

class CustomSegmentationModel(ABC):
    """Abstract base class for custom segmentation models."""
    
    @abstractmethod
    def segment(self, image, box_xyxy):
        """Perform segmentation on image."""
        pass
    
    @abstractmethod
    def init_state(self, frames_dir):
        """Initialize video tracking state."""
        pass
    
    @abstractmethod
    def propagate_in_video(self, state):
        """Propagate tracking through video."""
        pass

class MyCustomModel(CustomSegmentationModel):
    """Example custom model implementation."""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load custom model."""
        # Custom model loading logic
        pass
    
    def segment(self, image, box_xyxy):
        """Custom segmentation implementation."""
        # Custom segmentation logic
        pass
    
    def init_state(self, frames_dir):
        """Custom video state initialization."""
        # Custom state initialization
        pass
    
    def propagate_in_video(self, state):
        """Custom video propagation."""
        # Custom propagation logic
        pass

# Register custom model with factory
def register_custom_model():
    """Register custom model with the factory."""
    
    def create_custom_model(model_name, device):
        return MyCustomModel(model_name, device)
    
    # Add to factory (this would require extending the factory)
    SegmentationModelFactory.register_model_type(
        "custom",
        create_custom_model
    )
```

#### Custom Optimization Strategy

```python
from sowlv2.optimizations.resource_manager import AdvancedResourceManager
from sowlv2.optimizations.batch_optimizer import IntelligentBatchOptimizer

class CustomOptimizationStrategy:
    """Custom optimization strategy for specific use cases."""
    
    def __init__(self, target_fps=30, quality_threshold=0.9):
        self.target_fps = target_fps
        self.quality_threshold = quality_threshold
        self.resource_manager = AdvancedResourceManager()
        self.batch_optimizer = IntelligentBatchOptimizer()
    
    def optimize_for_realtime(self, video_info):
        """Optimize configuration for real-time processing."""
        
        # Analyze video characteristics
        frame_rate = video_info.get('fps', 30)
        resolution = video_info.get('resolution', (1920, 1080))
        
        # Calculate required processing speed
        required_speed = frame_rate / self.target_fps
        
        # Adjust model selection based on requirements
        if required_speed > 2.0:
            model_config = {
                'model_type': 'edgetam',
                'model_name': 'facebook/edgetam-small',
                'optimization_level': 3
            }
        elif required_speed > 1.5:
            model_config = {
                'model_type': 'edgetam',
                'model_name': 'facebook/edgetam-base',
                'optimization_level': 2
            }
        else:
            model_config = {
                'model_type': 'sam2',
                'model_name': 'facebook/sam2.1-hiera-small',
                'optimization_level': 1
            }
        
        # Optimize batch configuration
        memory_stats = self.resource_manager.monitor_memory_usage()
        batch_config = self.batch_optimizer.optimize_batch_processing(
            memory_usage=memory_stats.utilization_percentage,
            target_fps=self.target_fps
        )
        
        return {
            'model_config': model_config,
            'batch_config': batch_config,
            'streaming_config': self._get_streaming_config(video_info)
        }
    
    def _get_streaming_config(self, video_info):
        """Get streaming configuration based on video info."""
        # Custom streaming configuration logic
        pass
```

## Extension Points

### Adding New Models

To add support for a new segmentation model:

1. **Create Model Wrapper**:

```python
# sowlv2/models/new_model_wrapper.py
class NewModelWrapper:
    """Wrapper for new segmentation model."""
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load the new model."""
        # Model loading implementation
        pass
    
    def segment(self, image, box_xyxy):
        """Segmentation interface compatible with existing models."""
        # Segmentation implementation
        pass
    
    # Implement other required methods...
```

2. **Register with Factory**:

```python
# sowlv2/models/model_factory.py
from .new_model_wrapper import NewModelWrapper

class SegmentationModelFactory:
    # ... existing code ...
    
    @staticmethod
    def create_model(model_type, model_name, device="cpu", enable_fallback=True):
        if model_type == "new_model":
            return NewModelWrapper(model_name, device)
        # ... existing model creation logic ...
```

3. **Add Configuration Support**:

```python
# sowlv2/data/config.py
@dataclass
class NewModelConfig:
    """Configuration for new model."""
    model_name: str = "default/new-model"
    custom_parameter: float = 1.0
    enable_feature: bool = True
```

### Adding New Optimizations

To add a new optimization strategy:

1. **Create Optimization Module**:

```python
# sowlv2/optimizations/new_optimization.py
class NewOptimizer:
    """New optimization strategy."""
    
    def __init__(self, config):
        self.config = config
    
    def optimize(self, input_data):
        """Apply optimization to input data."""
        # Optimization implementation
        pass
    
    def get_metrics(self):
        """Get optimization metrics."""
        # Metrics collection
        pass
```

2. **Integrate with Pipeline**:

```python
# sowlv2/optimizations/optimized_pipeline.py
from .new_optimization import NewOptimizer

class OptimizedSOWLv2Pipeline:
    def __init__(self, ..., new_optimizer_config=None):
        # ... existing initialization ...
        if new_optimizer_config:
            self.new_optimizer = NewOptimizer(new_optimizer_config)
    
    def _apply_optimizations(self, data):
        # ... existing optimizations ...
        if hasattr(self, 'new_optimizer'):
            data = self.new_optimizer.optimize(data)
        return data
```

### Adding New Monitoring Metrics

To add custom performance metrics:

1. **Extend Performance Collector**:

```python
# sowlv2/optimizations/performance_collector.py
class PerformanceCollector:
    def __init__(self):
        # ... existing initialization ...
        self.custom_metrics = {}
    
    def record_custom_metric(self, metric_name, value, timestamp=None):
        """Record custom performance metric."""
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = []
        
        self.custom_metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def get_custom_metric_summary(self, metric_name):
        """Get summary statistics for custom metric."""
        if metric_name not in self.custom_metrics:
            return None
        
        values = [m['value'] for m in self.custom_metrics[metric_name]]
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
```

2. **Use in Custom Code**:

```python
from sowlv2.optimizations.performance_collector import PerformanceCollector

collector = PerformanceCollector()

# Record custom metrics
collector.record_custom_metric("custom_processing_time", 0.5)
collector.record_custom_metric("custom_accuracy", 0.95)

# Get summaries
time_summary = collector.get_custom_metric_summary("custom_processing_time")
```

## Development Best Practices

### Code Organization

1. **Module Structure**: Follow the existing module structure
2. **Naming Conventions**: Use descriptive names following Python conventions
3. **Documentation**: Include comprehensive docstrings
4. **Type Hints**: Use type hints for all public APIs
5. **Error Handling**: Implement proper error handling and recovery

### Testing

#### Unit Testing

```python
# tests/unit/test_new_feature.py
import unittest
from unittest.mock import Mock, patch
from sowlv2.models.new_model_wrapper import NewModelWrapper

class TestNewModelWrapper(unittest.TestCase):
    """Test cases for new model wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = NewModelWrapper("test-model", "cpu")
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "test-model")
        self.assertEqual(self.model.device, "cpu")
    
    @patch('sowlv2.models.new_model_wrapper.load_model')
    def test_model_loading(self, mock_load):
        """Test model loading with mocking."""
        mock_load.return_value = Mock()
        model = NewModelWrapper("test-model", "cuda")
        mock_load.assert_called_once()
    
    def test_segmentation(self):
        """Test segmentation functionality."""
        # Test implementation
        pass

if __name__ == '__main__':
    unittest.main()
```

#### Integration Testing

```python
# tests/integration/test_new_integration.py
import unittest
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
from sowlv2.data.config import OptimizationConfig

class TestNewIntegration(unittest.TestCase):
    """Integration tests for new features."""
    
    def test_end_to_end_processing(self):
        """Test complete processing pipeline."""
        config = OptimizationConfig(optimization_level=2)
        pipeline = OptimizedSOWLv2Pipeline(optimization_config=config)
        
        # Test with sample data
        result = pipeline.process_video(
            video_path="test_data/sample_video.mp4",
            prompts=["person"],
            output_dir="test_output/"
        )
        
        self.assertIsNotNone(result)
        # Additional assertions...
```

### Performance Considerations

1. **Memory Management**: Always clean up resources properly
2. **GPU Utilization**: Optimize for maximum GPU utilization
3. **Batch Processing**: Use appropriate batch sizes
4. **Caching**: Implement intelligent caching strategies
5. **Profiling**: Profile code regularly to identify bottlenecks

### Error Handling

```python
from sowlv2.utils.error_recovery import ErrorRecoveryManager
import logging

logger = logging.getLogger(__name__)

class RobustProcessor:
    """Example of robust processing with error handling."""
    
    def __init__(self):
        self.error_recovery = ErrorRecoveryManager()
    
    def process_with_recovery(self, data):
        """Process data with comprehensive error handling."""
        
        try:
            return self._process_data(data)
        
        except MemoryError as e:
            logger.warning(f"Memory error: {e}")
            # Handle memory overflow
            new_config = self.error_recovery.handle_memory_overflow(
                self.current_config
            )
            return self._process_data(data, config=new_config)
        
        except Exception as e:
            logger.error(f"Processing error: {e}")
            # Implement retry logic
            return self.error_recovery.implement_retry_logic(
                operation=lambda: self._process_data(data),
                max_retries=3
            )
    
    def _process_data(self, data, config=None):
        """Internal data processing method."""
        # Processing implementation
        pass
```

## Contributing Guidelines

### Code Style

1. Follow PEP 8 style guidelines
2. Use Black for code formatting
3. Use isort for import sorting
4. Include type hints for all public APIs
5. Write comprehensive docstrings

### Pull Request Process

1. **Fork and Branch**: Create a feature branch from main
2. **Implement Changes**: Follow coding standards and best practices
3. **Add Tests**: Include unit and integration tests
4. **Update Documentation**: Update relevant documentation
5. **Submit PR**: Create pull request with detailed description

### Example Development Workflow

```bash
# 1. Fork and clone repository
git clone https://github.com/your-username/sowlv2.git
cd sowlv2

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Create feature branch
git checkout -b feature/new-optimization

# 5. Make changes and add tests
# ... implement your feature ...

# 6. Run tests
python -m pytest tests/

# 7. Format code
black sowlv2/
isort sowlv2/

# 8. Commit and push
git add .
git commit -m "Add new optimization feature"
git push origin feature/new-optimization

# 9. Create pull request
# ... create PR on GitHub ...
```

## Debugging and Profiling

### Debug Mode

```python
import logging
from sowlv2.utils.enhanced_logger import EnhancedErrorLogger

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = EnhancedErrorLogger()

# Enable performance profiling
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile a function call."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

### Memory Profiling

```python
from memory_profiler import profile
import tracemalloc

@profile
def memory_intensive_function():
    """Function with memory profiling."""
    # Function implementation
    pass

# Alternative: tracemalloc
def trace_memory_usage():
    """Trace memory usage during execution."""
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

## Deployment Considerations

### Docker Integration

```dockerfile
# Dockerfile for SOWLv2 application
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install SOWLv2
COPY . /app
WORKDIR /app
RUN pip3 install -e .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV SOWLV2_OPTIMIZATION_LEVEL=2

# Run application
CMD ["python3", "-m", "sowlv2.cli", "--config", "config.yaml"]
```

### Production Deployment

```python
# production_server.py
from flask import Flask, request, jsonify
from sowlv2.optimizations.optimized_pipeline import OptimizedSOWLv2Pipeline
import tempfile
import os

app = Flask(__name__)

# Initialize pipeline once
pipeline = OptimizedSOWLv2Pipeline(
    optimization_config=OptimizationConfig(optimization_level=2),
    edgetam_config=EdgeTAMConfig(model_name="facebook/edgetam-base")
)

@app.route('/process_video', methods=['POST'])
def process_video():
    """API endpoint for video processing."""
    
    try:
        # Get uploaded file
        video_file = request.files['video']
        prompts = request.form.get('prompts', '').split(',')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            
            # Process video
            results = pipeline.process_video(
                video_path=tmp_file.name,
                prompts=prompts,
                output_dir=tempfile.mkdtemp()
            )
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return jsonify({
                'status': 'success',
                'results': results
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

For more detailed information, see the [API Reference](api_reference.md) and [Troubleshooting Guide](troubleshooting.md).