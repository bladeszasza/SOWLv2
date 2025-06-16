"""
GPU-specific optimizations for SOWLv2 pipeline.
Includes mixed precision, memory management, and CUDA optimizations.
"""
import torch
import torch.cuda.amp as amp
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import gc


class GPUOptimizer:
    """Manages GPU optimizations for the pipeline."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize GPU optimizer."""
        self.device = device
        self.use_amp = torch.cuda.is_available() and device != "cpu"
        
        # Initialize AMP scaler for mixed precision
        self.scaler = amp.GradScaler() if self.use_amp else None
        
        # Memory management settings
        self.memory_fraction = 0.9  # Use 90% of available GPU memory
        self.enable_memory_efficient_attention = True
        
        # Apply initial optimizations
        self._setup_cuda_optimizations()
    
    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        if not torch.cuda.is_available():
            return
        
        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN autotuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.use_amp:
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                yield
        else:
            yield
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize a model for inference on GPU.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Move to GPU
        if torch.cuda.is_available() and self.device != "cpu":
            model = model.to(self.device)
            
            # Try to compile with torch.compile if available
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(
                        model, 
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    print(f"Successfully compiled model with torch.compile")
                except Exception as e:
                    print(f"Failed to compile model: {e}")
            
            # Enable memory efficient attention if available
            if self.enable_memory_efficient_attention:
                self._enable_memory_efficient_attention(model)
        
        return model
    
    def _enable_memory_efficient_attention(self, model: torch.nn.Module):
        """Enable memory efficient attention mechanisms."""
        # Check for specific attention implementations
        for module in model.modules():
            if hasattr(module, 'set_use_memory_efficient_attention'):
                module.set_use_memory_efficient_attention(True)
            elif hasattr(module, 'enable_xformers'):
                try:
                    module.enable_xformers()
                except Exception:
                    pass
    
    def batch_inference(
        self, 
        model: torch.nn.Module, 
        inputs: List[torch.Tensor], 
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        """
        Perform batched inference for better GPU utilization.
        
        Args:
            model: Model to run inference on
            inputs: List of input tensors
            batch_size: Batch size for processing
            
        Returns:
            List of output tensors
        """
        outputs = []
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                
                # Stack into batch tensor
                if len(batch) > 1:
                    batch_tensor = torch.stack(batch)
                else:
                    batch_tensor = batch[0].unsqueeze(0)
                
                # Move to device
                batch_tensor = batch_tensor.to(self.device)
                
                # Run inference with autocast
                with self.autocast_context():
                    batch_output = model(batch_tensor)
                
                # Collect outputs
                if isinstance(batch_output, torch.Tensor):
                    for j in range(batch_output.shape[0]):
                        outputs.append(batch_output[j])
                else:
                    outputs.extend(batch_output)
                
                # Clear intermediate tensors
                del batch_tensor
                if i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        return outputs
    
    @staticmethod
    def profile_gpu_memory():
        """Profile current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'free': (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_reserved()) / 1024**3        # GB
        }
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


class StreamedProcessing:
    """
    Implements CUDA streams for overlapping computation and data transfer.
    """
    
    def __init__(self, num_streams: int = 2):
        """Initialize CUDA streams."""
        self.num_streams = num_streams
        self.streams = []
        
        if torch.cuda.is_available():
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream())
    
    def process_with_streams(
        self, 
        process_func, 
        data_list: List[Any],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process data using CUDA streams for overlapping operations.
        
        Args:
            process_func: Function to process each data item
            data_list: List of data items to process
            *args, **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        if not self.streams:
            # No CUDA, process sequentially
            return [process_func(data, *args, **kwargs) for data in data_list]
        
        results = [None] * len(data_list)
        
        # Process data with streams
        for i, data in enumerate(data_list):
            stream_idx = i % self.num_streams
            
            with torch.cuda.stream(self.streams[stream_idx]):
                results[i] = process_func(data, *args, **kwargs)
        
        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()
        
        return results


class TensorRTOptimizer:
    """
    Optional TensorRT optimization for maximum inference speed.
    Requires torch_tensorrt to be installed.
    """
    
    @staticmethod
    def optimize_with_tensorrt(
        model: torch.nn.Module,
        example_inputs: torch.Tensor,
        fp16: bool = True
    ) -> Optional[torch.nn.Module]:
        """
        Optimize model with TensorRT.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example input tensor for tracing
            fp16: Whether to use FP16 precision
            
        Returns:
            TensorRT optimized model or None if failed
        """
        try:
            import torch_tensorrt
            
            # Trace the model
            model.eval()
            traced_model = torch.jit.trace(model, example_inputs)
            
            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                traced_model,
                inputs=[example_inputs],
                enabled_precisions={torch.float16} if fp16 else {torch.float32},
                workspace_size=1 << 30,  # 1GB workspace
                truncate_long_and_double=True
            )
            
            print("Successfully optimized model with TensorRT")
            return trt_model
            
        except ImportError:
            print("torch_tensorrt not installed. Skipping TensorRT optimization.")
            return None
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return None 