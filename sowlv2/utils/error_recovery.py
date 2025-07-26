"""
Error recovery utilities for SOWLv2 pipeline.
Provides fallback mechanisms and user notification systems.
"""
import logging
import time
import random
from typing import Callable, Optional, Any, Dict, Union, List
from functools import wraps
import torch
import psutil
import gc

logger = logging.getLogger(__name__)


class ModelFallbackManager:
    """
    Manages model fallback scenarios and user notifications.
    """
    
    @staticmethod
    def handle_model_loading_error(
        model_type: str,
        model_name: str,
        error: Exception,
        fallback_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Handle model loading errors with appropriate fallback strategies.
        
        Args:
            model_type: Type of model that failed
            model_name: Name of the model that failed
            error: The exception that occurred
            fallback_callback: Optional callback for fallback model creation
            
        Returns:
            Dictionary containing error handling results
        """
        result = {
            "success": False,
            "fallback_used": False,
            "fallback_model": None,
            "error_message": str(error),
            "user_message": ""
        }
        
        try:
            if model_type == "edgetam":
                # EdgeTAM specific fallback handling
                user_message = (
                    f"EdgeTAM model '{model_name}' failed to load.\n"
                    f"Error: {str(error)}\n"
                    "Attempting to fallback to SAM2 for segmentation.\n"
                    "Note: Processing may be slower but will continue."
                )
                
                result["user_message"] = user_message
                logger.warning(user_message)
                
                # Attempt fallback if callback provided
                if fallback_callback:
                    try:
                        fallback_model = fallback_callback()
                        result["success"] = True
                        result["fallback_used"] = True
                        result["fallback_model"] = fallback_model
                        
                        success_message = "Successfully fell back to SAM2 model."
                        result["user_message"] += f"\n{success_message}"
                        logger.info(success_message)
                        
                    except Exception as fallback_error:
                        fallback_error_msg = f"Fallback to SAM2 also failed: {str(fallback_error)}"
                        result["user_message"] += f"\n{fallback_error_msg}"
                        logger.error(fallback_error_msg)
                        
            elif model_type == "sam2":
                # SAM2 specific error handling (no fallback available)
                user_message = (
                    f"SAM2 model '{model_name}' failed to load.\n"
                    f"Error: {str(error)}\n"
                    "No fallback model available. Please check your configuration."
                )
                
                result["user_message"] = user_message
                logger.error(user_message)
                
        except Exception as handling_error:
            error_msg = f"Error in fallback handling: {str(handling_error)}"
            result["user_message"] = error_msg
            logger.error(error_msg)
        
        return result
    
    @staticmethod
    def log_model_selection_event(
        selected_model_type: str,
        selected_model_name: str,
        was_fallback: bool = False,
        original_model_type: Optional[str] = None,
        original_model_name: Optional[str] = None
    ):
        """
        Log model selection events for debugging and monitoring.
        
        Args:
            selected_model_type: Type of the selected model
            selected_model_name: Name of the selected model
            was_fallback: Whether this was a fallback selection
            original_model_type: Original model type if fallback occurred
            original_model_name: Original model name if fallback occurred
        """
        if was_fallback and original_model_type and original_model_name:
            log_message = (
                f"Model Selection (FALLBACK): "
                f"Original: {original_model_type}/{original_model_name} -> "
                f"Selected: {selected_model_type}/{selected_model_name}"
            )
            logger.warning(log_message)
        else:
            log_message = (
                f"Model Selection: {selected_model_type}/{selected_model_name}"
            )
            logger.info(log_message)


class UserNotificationSystem:
    """
    System for providing user-friendly notifications about errors and fallbacks.
    """
    
    @staticmethod
    def notify_fallback_scenario(
        original_model: str,
        fallback_model: str,
        reason: str,
        impact: str = "Processing may be slower but will continue"
    ):
        """
        Notify user about fallback scenario.
        
        Args:
            original_model: The model that failed
            fallback_model: The fallback model being used
            reason: Reason for the fallback
            impact: Impact description for the user
        """
        notification = (
            f"\n{'='*60}\n"
            f"MODEL FALLBACK NOTIFICATION\n"
            f"{'='*60}\n"
            f"Original Model: {original_model}\n"
            f"Fallback Model: {fallback_model}\n"
            f"Reason: {reason}\n"
            f"Impact: {impact}\n"
            f"{'='*60}\n"
        )
        
        print(notification)
        logger.warning(f"Fallback notification: {original_model} -> {fallback_model}")
    
    @staticmethod
    def notify_error_with_solution(
        error_type: str,
        error_message: str,
        suggested_solutions: list
    ):
        """
        Notify user about error with suggested solutions.
        
        Args:
            error_type: Type of error that occurred
            error_message: Detailed error message
            suggested_solutions: List of suggested solutions
        """
        notification = (
            f"\n{'='*60}\n"
            f"ERROR: {error_type}\n"
            f"{'='*60}\n"
            f"Details: {error_message}\n"
            f"\nSuggested Solutions:\n"
        )
        
        for i, solution in enumerate(suggested_solutions, 1):
            notification += f"{i}. {solution}\n"
        
        notification += f"{'='*60}\n"
        
        print(notification)
        logger.error(f"Error notification: {error_type} - {error_message}")


def with_fallback_handling(fallback_model_type: str = "sam2"):
    """
    Decorator for functions that create models with automatic fallback handling.
    
    Args:
        fallback_model_type: Type of model to fallback to
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {str(e)}")
                
                # Attempt fallback logic here if needed
                if "edgetam" in str(func.__name__).lower():
                    UserNotificationSystem.notify_fallback_scenario(
                        original_model="EdgeTAM",
                        fallback_model="SAM2",
                        reason=str(e),
                        impact="Processing will be slower but more accurate"
                    )
                
                raise e
        return wrapper
    return decorator


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager for SOWLv2 pipeline.
    Handles model loading errors, memory overflow, and processing failures.
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.retry_counts = {}
        self.fallback_history = []
    
    def handle_model_loading_error(
        self,
        model_name: str,
        error: Exception,
        fallback_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Handle model loading errors with fallback scenarios.
        
        Args:
            model_name: Name of the model that failed to load
            error: The exception that occurred during loading
            fallback_callback: Optional callback to create fallback model
            
        Returns:
            Dictionary containing recovery results and fallback model
        """
        self.logger.error(f"Model loading failed for {model_name}: {str(error)}")
        
        result = {
            "success": False,
            "fallback_used": False,
            "fallback_model": None,
            "error_message": str(error),
            "user_message": "",
            "recovery_action": "none"
        }
        
        try:
            # Determine fallback strategy based on model type
            if "edgetam" in model_name.lower():
                result["recovery_action"] = "fallback_to_sam2"
                user_message = (
                    f"⚠️  EdgeTAM model '{model_name}' failed to load.\n"
                    f"Error: {str(error)}\n"
                    f"🔄 Falling back to SAM2 for segmentation.\n"
                    f"📝 Note: Processing may be slower but will continue with higher accuracy."
                )
                
                if fallback_callback:
                    try:
                        fallback_model = fallback_callback()
                        result.update({
                            "success": True,
                            "fallback_used": True,
                            "fallback_model": fallback_model
                        })
                        user_message += "\n✅ Successfully loaded SAM2 as fallback."
                        self.fallback_history.append({
                            "from": model_name,
                            "to": "SAM2",
                            "reason": str(error),
                            "timestamp": time.time()
                        })
                    except Exception as fallback_error:
                        user_message += f"\n❌ Fallback to SAM2 also failed: {str(fallback_error)}"
                        self.logger.error(f"Fallback failed: {str(fallback_error)}")
                        
            elif "sam2" in model_name.lower():
                result["recovery_action"] = "no_fallback_available"
                user_message = (
                    f"❌ SAM2 model '{model_name}' failed to load.\n"
                    f"Error: {str(error)}\n"
                    f"🚫 No fallback segmentation model available.\n"
                    f"💡 Suggestions:\n"
                    f"   - Check internet connection for model download\n"
                    f"   - Verify sufficient disk space\n"
                    f"   - Try a different SAM2 model variant"
                )
                
            elif "vjepa" in model_name.lower():
                result["recovery_action"] = "disable_vjepa_optimization"
                user_message = (
                    f"⚠️  V-JEPA2 model '{model_name}' failed to load.\n"
                    f"Error: {str(error)}\n"
                    f"🔄 Disabling V-JEPA2 optimization, using uniform frame sampling.\n"
                    f"📝 Note: Frame selection will be less intelligent but processing will continue."
                )
                result["success"] = True  # Can continue without V-JEPA2
                
            else:
                result["recovery_action"] = "unknown_model_type"
                user_message = (
                    f"❌ Unknown model '{model_name}' failed to load.\n"
                    f"Error: {str(error)}\n"
                    f"🔍 Please check model name and configuration."
                )
            
            result["user_message"] = user_message
            self.logger.info(f"Recovery action for {model_name}: {result['recovery_action']}")
            
        except Exception as recovery_error:
            error_msg = f"Error during recovery handling: {str(recovery_error)}"
            result["user_message"] = error_msg
            self.logger.error(error_msg)
        
        return result
    
    def handle_memory_overflow(
        self,
        current_batch_size: int,
        memory_usage_gb: float,
        available_memory_gb: float
    ) -> Dict[str, Any]:
        """
        Handle memory overflow by adjusting batch sizes and clearing cache.
        
        Args:
            current_batch_size: Current batch size being used
            memory_usage_gb: Current memory usage in GB
            available_memory_gb: Available memory in GB
            
        Returns:
            Dictionary containing adjusted configuration
        """
        self.logger.warning(
            f"Memory overflow detected: {memory_usage_gb:.2f}GB used, "
            f"{available_memory_gb:.2f}GB available"
        )
        
        result = {
            "success": False,
            "new_batch_size": current_batch_size,
            "actions_taken": [],
            "memory_freed_gb": 0.0,
            "user_message": ""
        }
        
        try:
            initial_memory = self._get_memory_usage()
            
            # Step 1: Reduce batch size
            if current_batch_size > 1:
                new_batch_size = max(1, current_batch_size // 2)
                result["new_batch_size"] = new_batch_size
                result["actions_taken"].append(f"Reduced batch size: {current_batch_size} → {new_batch_size}")
                self.logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
            
            # Step 2: Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                result["actions_taken"].append("Cleared GPU cache")
                self.logger.info("Cleared GPU cache")
            
            # Step 3: Force garbage collection
            gc.collect()
            result["actions_taken"].append("Forced garbage collection")
            
            # Step 4: Check memory improvement
            final_memory = self._get_memory_usage()
            memory_freed = initial_memory - final_memory
            result["memory_freed_gb"] = memory_freed
            
            if memory_freed > 0:
                result["success"] = True
                result["user_message"] = (
                    f"🔧 Memory overflow handled successfully:\n"
                    f"   • Freed {memory_freed:.2f}GB of memory\n"
                    f"   • Actions taken: {', '.join(result['actions_taken'])}\n"
                    f"   • New batch size: {result['new_batch_size']}"
                )
            else:
                result["user_message"] = (
                    f"⚠️  Memory overflow handling completed but limited improvement:\n"
                    f"   • Actions taken: {', '.join(result['actions_taken'])}\n"
                    f"   • Consider reducing input size or using CPU processing"
                )
            
            self.logger.info(f"Memory recovery freed {memory_freed:.2f}GB")
            
        except Exception as recovery_error:
            error_msg = f"Error during memory overflow handling: {str(recovery_error)}"
            result["user_message"] = error_msg
            self.logger.error(error_msg)
        
        return result
    
    def handle_processing_failure(
        self,
        operation_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle processing failures with appropriate recovery strategies.
        
        Args:
            operation_name: Name of the operation that failed
            error: The exception that occurred
            context: Optional context information
            
        Returns:
            Dictionary containing recovery recommendations
        """
        self.logger.error(f"Processing failure in {operation_name}: {str(error)}")
        
        result = {
            "should_retry": False,
            "retry_delay": 0,
            "max_retries": 3,
            "recovery_suggestions": [],
            "user_message": "",
            "context": context or {}
        }
        
        try:
            error_type = type(error).__name__
            error_message = str(error).lower()
            
            # Analyze error type and provide specific recovery strategies
            if "cuda" in error_message or "gpu" in error_message:
                result.update({
                    "should_retry": True,
                    "retry_delay": 2,
                    "recovery_suggestions": [
                        "Clear GPU cache and retry",
                        "Reduce batch size",
                        "Switch to CPU processing",
                        "Check GPU memory availability"
                    ]
                })
                
            elif "memory" in error_message or "out of memory" in error_message:
                result.update({
                    "should_retry": True,
                    "retry_delay": 1,
                    "recovery_suggestions": [
                        "Reduce batch size",
                        "Enable streaming processing",
                        "Clear model cache",
                        "Use mixed precision training"
                    ]
                })
                
            elif "connection" in error_message or "network" in error_message:
                result.update({
                    "should_retry": True,
                    "retry_delay": 5,
                    "max_retries": 5,
                    "recovery_suggestions": [
                        "Check internet connection",
                        "Retry with exponential backoff",
                        "Use cached models if available",
                        "Switch to offline mode"
                    ]
                })
                
            elif "file" in error_message or "path" in error_message:
                result.update({
                    "should_retry": False,
                    "recovery_suggestions": [
                        "Check file path exists",
                        "Verify file permissions",
                        "Ensure sufficient disk space",
                        "Validate file format"
                    ]
                })
                
            else:
                result.update({
                    "should_retry": True,
                    "retry_delay": 1,
                    "recovery_suggestions": [
                        "Retry operation",
                        "Check system resources",
                        "Validate input parameters",
                        "Review error logs"
                    ]
                })
            
            # Create user-friendly message
            result["user_message"] = (
                f"❌ Processing failure in {operation_name}:\n"
                f"   Error: {error_type} - {str(error)}\n"
                f"   Retry recommended: {'Yes' if result['should_retry'] else 'No'}\n"
                f"   Suggestions:\n"
            )
            
            for i, suggestion in enumerate(result["recovery_suggestions"], 1):
                result["user_message"] += f"   {i}. {suggestion}\n"
            
            self.logger.info(f"Recovery strategy for {operation_name}: retry={result['should_retry']}")
            
        except Exception as recovery_error:
            error_msg = f"Error during processing failure handling: {str(recovery_error)}"
            result["user_message"] = error_msg
            self.logger.error(error_msg)
        
        return result
    
    def implement_retry_logic(
        self,
        operation: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        operation_name: str = "unknown"
    ) -> Any:
        """
        Implement retry logic with exponential backoff.
        
        Args:
            operation: The operation to retry
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            backoff_factor: Exponential backoff factor
            operation_name: Name of the operation for logging
            
        Returns:
            Result of the successful operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        retry_key = f"{operation_name}_{id(operation)}"
        
        if retry_key not in self.retry_counts:
            self.retry_counts[retry_key] = 0
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (backoff_factor ** (attempt - 1))
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter
                    
                    self.logger.info(
                        f"Retrying {operation_name} (attempt {attempt}/{max_retries}) "
                        f"after {total_delay:.2f}s delay"
                    )
                    time.sleep(total_delay)
                
                # Attempt the operation
                result = operation()
                
                # Success - reset retry count and return
                if retry_key in self.retry_counts:
                    del self.retry_counts[retry_key]
                
                if attempt > 0:
                    self.logger.info(f"Operation {operation_name} succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                self.retry_counts[retry_key] = attempt + 1
                
                if attempt < max_retries:
                    self.logger.warning(
                        f"Operation {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                    )
                else:
                    self.logger.error(
                        f"Operation {operation_name} failed after {max_retries + 1} attempts: {str(e)}"
                    )
        
        # All retries exhausted
        if retry_key in self.retry_counts:
            del self.retry_counts[retry_key]
        
        raise last_exception
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 ** 3)  # Convert to GB
        except Exception:
            return 0.0
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery operations."""
        return {
            "active_retries": len(self.retry_counts),
            "retry_counts": dict(self.retry_counts),
            "fallback_history": self.fallback_history,
            "total_fallbacks": len(self.fallback_history)
        }
    
    def reset_recovery_state(self):
        """Reset recovery state and statistics."""
        self.retry_counts.clear()
        self.fallback_history.clear()
        self.logger.info("Recovery state reset")


class GracefulDegradationManager:
    """
    Manages graceful degradation scenarios for the SOWLv2 pipeline.
    Provides fallback mechanisms and progressive quality reduction.
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.degradation_history = []
        self.current_degradation_level = 0
        self.notification_system = UserNotificationSystem()
    
    def handle_gpu_resource_exhaustion(
        self,
        current_device: str,
        operation_name: str
    ) -> Dict[str, Any]:
        """
        Handle GPU resource exhaustion by falling back to CPU processing.
        
        Args:
            current_device: Current device being used
            operation_name: Name of the operation that failed
            
        Returns:
            Dictionary containing fallback configuration
        """
        self.logger.warning(f"GPU resources exhausted for {operation_name}")
        
        result = {
            "success": False,
            "fallback_device": "cpu",
            "performance_impact": "significant_slowdown",
            "user_message": "",
            "degradation_actions": []
        }
        
        try:
            if current_device != "cpu":
                result.update({
                    "success": True,
                    "degradation_actions": ["device_fallback_to_cpu"]
                })
                
                # Record degradation event
                degradation_event = {
                    "type": "device_fallback",
                    "from_device": current_device,
                    "to_device": "cpu",
                    "operation": operation_name,
                    "timestamp": time.time(),
                    "level": 1
                }
                self.degradation_history.append(degradation_event)
                self.current_degradation_level = max(self.current_degradation_level, 1)
                
                result["user_message"] = (
                    f"🔄 GPU resources exhausted for {operation_name}\n"
                    f"   • Falling back to CPU processing\n"
                    f"   • Expected performance impact: 3-10x slower\n"
                    f"   • Processing will continue with same quality\n"
                    f"   • Consider reducing batch size or input resolution"
                )
                
                self.notification_system.notify_fallback_scenario(
                    original_model=f"GPU-{operation_name}",
                    fallback_model=f"CPU-{operation_name}",
                    reason="GPU memory exhausted",
                    impact="Processing will be significantly slower"
                )
                
                self.logger.info(f"Successfully configured CPU fallback for {operation_name}")
            else:
                result["user_message"] = (
                    f"❌ Already using CPU for {operation_name}\n"
                    f"   • No further device fallback available\n"
                    f"   • Consider reducing input size or batch size"
                )
                
        except Exception as e:
            error_msg = f"Error during GPU fallback handling: {str(e)}"
            result["user_message"] = error_msg
            self.logger.error(error_msg)
        
        return result
    
    def implement_progressive_quality_reduction(
        self,
        current_config: Dict[str, Any],
        memory_constraint_gb: float
    ) -> Dict[str, Any]:
        """
        Implement progressive quality reduction for memory-constrained scenarios.
        
        Args:
            current_config: Current processing configuration
            memory_constraint_gb: Memory constraint in GB
            
        Returns:
            Dictionary containing reduced quality configuration
        """
        self.logger.info(f"Implementing progressive quality reduction for {memory_constraint_gb}GB constraint")
        
        result = {
            "success": False,
            "new_config": current_config.copy(),
            "quality_reductions": [],
            "estimated_memory_savings": 0.0,
            "user_message": ""
        }
        
        try:
            config = result["new_config"]
            memory_savings = 0.0
            
            # Level 1: Reduce batch size
            if config.get("batch_size", 1) > 1:
                original_batch = config["batch_size"]
                config["batch_size"] = max(1, original_batch // 2)
                memory_savings += (original_batch - config["batch_size"]) * 0.5  # Estimate
                result["quality_reductions"].append(
                    f"Reduced batch size: {original_batch} → {config['batch_size']}"
                )
                self.current_degradation_level = max(self.current_degradation_level, 1)
            
            # Level 2: Reduce input resolution
            if memory_constraint_gb < 4.0 and config.get("input_resolution"):
                original_res = config["input_resolution"]
                if isinstance(original_res, (list, tuple)) and len(original_res) == 2:
                    new_res = [int(original_res[0] * 0.75), int(original_res[1] * 0.75)]
                    config["input_resolution"] = new_res
                    memory_savings += 1.5  # Estimate
                    result["quality_reductions"].append(
                        f"Reduced input resolution: {original_res} → {new_res}"
                    )
                    self.current_degradation_level = max(self.current_degradation_level, 2)
            
            # Level 3: Enable mixed precision
            if memory_constraint_gb < 6.0 and not config.get("mixed_precision", False):
                config["mixed_precision"] = True
                memory_savings += 2.0  # Estimate
                result["quality_reductions"].append("Enabled mixed precision (FP16)")
                self.current_degradation_level = max(self.current_degradation_level, 2)
            
            # Level 4: Reduce model precision/features
            if memory_constraint_gb < 3.0:
                if config.get("use_high_quality_features", True):
                    config["use_high_quality_features"] = False
                    memory_savings += 1.0
                    result["quality_reductions"].append("Disabled high-quality features")
                    self.current_degradation_level = max(self.current_degradation_level, 3)
                
                if config.get("enable_temporal_optimization", True):
                    config["enable_temporal_optimization"] = False
                    memory_savings += 0.5
                    result["quality_reductions"].append("Disabled temporal optimization")
            
            # Level 5: Enable streaming mode
            if memory_constraint_gb < 2.0 and not config.get("streaming_mode", False):
                config["streaming_mode"] = True
                config["streaming_chunk_size"] = min(50, config.get("streaming_chunk_size", 100))
                memory_savings += 3.0  # Significant savings
                result["quality_reductions"].append("Enabled streaming mode with small chunks")
                self.current_degradation_level = max(self.current_degradation_level, 4)
            
            result.update({
                "success": len(result["quality_reductions"]) > 0,
                "estimated_memory_savings": memory_savings
            })
            
            if result["success"]:
                # Record degradation event
                degradation_event = {
                    "type": "quality_reduction",
                    "reductions": result["quality_reductions"],
                    "memory_constraint": memory_constraint_gb,
                    "estimated_savings": memory_savings,
                    "timestamp": time.time(),
                    "level": self.current_degradation_level
                }
                self.degradation_history.append(degradation_event)
                
                result["user_message"] = (
                    f"🔧 Progressive quality reduction applied:\n"
                    f"   • Memory constraint: {memory_constraint_gb}GB\n"
                    f"   • Estimated memory savings: {memory_savings:.1f}GB\n"
                    f"   • Quality reductions applied:\n"
                )
                
                for i, reduction in enumerate(result["quality_reductions"], 1):
                    result["user_message"] += f"     {i}. {reduction}\n"
                
                result["user_message"] += (
                    f"   • Degradation level: {self.current_degradation_level}/4\n"
                    f"   • Processing will continue with reduced quality/speed"
                )
                
                self.logger.info(f"Applied {len(result['quality_reductions'])} quality reductions")
            else:
                result["user_message"] = (
                    f"⚠️  No quality reductions available for {memory_constraint_gb}GB constraint\n"
                    f"   • Current configuration already at minimum settings\n"
                    f"   • Consider using smaller input files or upgrading hardware"
                )
                
        except Exception as e:
            error_msg = f"Error during quality reduction: {str(e)}"
            result["user_message"] = error_msg
            self.logger.error(error_msg)
        
        return result
    
    def create_degradation_notification(
        self,
        degradation_type: str,
        details: Dict[str, Any],
        impact_description: str
    ):
        """
        Create user notification for degradation events.
        
        Args:
            degradation_type: Type of degradation that occurred
            details: Details about the degradation
            impact_description: Description of the impact on user experience
        """
        try:
            notification = (
                f"\n{'='*60}\n"
                f"GRACEFUL DEGRADATION NOTIFICATION\n"
                f"{'='*60}\n"
                f"Type: {degradation_type.replace('_', ' ').title()}\n"
                f"Level: {self.current_degradation_level}/4\n"
                f"Impact: {impact_description}\n"
                f"\nDetails:\n"
            )
            
            for key, value in details.items():
                notification += f"  • {key.replace('_', ' ').title()}: {value}\n"
            
            notification += (
                f"\nNote: Processing will continue with adjusted settings.\n"
                f"{'='*60}\n"
            )
            
            print(notification)
            self.logger.warning(f"Degradation notification: {degradation_type}")
            
        except Exception as e:
            self.logger.error(f"Error creating degradation notification: {str(e)}")
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status and history."""
        return {
            "current_level": self.current_degradation_level,
            "max_level": 4,
            "degradation_history": self.degradation_history,
            "total_degradations": len(self.degradation_history),
            "is_degraded": self.current_degradation_level > 0
        }
    
    def reset_degradation_state(self):
        """Reset degradation state to normal operation."""
        self.current_degradation_level = 0
        self.degradation_history.clear()
        self.logger.info("Degradation state reset to normal operation")
    
    def can_handle_further_degradation(self) -> bool:
        """Check if further degradation is possible."""
        return self.current_degradation_level < 4


class UserFriendlyErrorHandler:
    """
    User-friendly error handling system with comprehensive error messages and solutions.
    Provides error code classification and interactive troubleshooting guidance.
    """
    
    # Error code classification system
    ERROR_CODES = {
        "E001": "Model Loading Failure",
        "E002": "Memory Overflow",
        "E003": "GPU Resource Exhaustion", 
        "E004": "Network Connection Error",
        "E005": "File System Error",
        "E006": "Configuration Error",
        "E007": "Processing Pipeline Failure",
        "E008": "Dependency Missing",
        "E009": "Hardware Compatibility Issue",
        "E010": "Unknown Error"
    }
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.error_solutions_db = self._build_solutions_database()
        self.troubleshooting_guide = self._build_troubleshooting_guide()
    
    def handle_user_friendly_error(
        self,
        error: Exception,
        operation_name: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle errors with user-friendly messages and solutions.
        
        Args:
            error: The exception that occurred
            operation_name: Name of the operation that failed
            context: Optional context information
            
        Returns:
            Dictionary containing user-friendly error information
        """
        try:
            # Classify the error
            error_code = self._classify_error(error)
            error_category = self.ERROR_CODES.get(error_code, "Unknown Error")
            
            # Get solutions for this error type
            solutions = self._get_error_solutions(error_code, error, context)
            
            # Create user-friendly message
            user_message = self._create_user_friendly_message(
                error_code, error_category, error, operation_name, solutions
            )
            
            # Get troubleshooting steps
            troubleshooting_steps = self._get_troubleshooting_steps(error_code, error)
            
            result = {
                "error_code": error_code,
                "error_category": error_category,
                "user_message": user_message,
                "solutions": solutions,
                "troubleshooting_steps": troubleshooting_steps,
                "support_info": self._get_support_information(error_code),
                "quick_fixes": self._get_quick_fixes(error_code, error)
            }
            
            # Log the user-friendly error
            self.logger.error(f"User-friendly error [{error_code}]: {error_category} in {operation_name}")
            
            return result
            
        except Exception as handling_error:
            # Fallback error handling
            fallback_result = {
                "error_code": "E010",
                "error_category": "Unknown Error",
                "user_message": f"An unexpected error occurred: {str(error)}",
                "solutions": ["Check logs for more details", "Contact support"],
                "troubleshooting_steps": ["Review error message", "Check system resources"],
                "support_info": self._get_support_information("E010"),
                "quick_fixes": []
            }
            
            self.logger.error(f"Error in user-friendly error handling: {str(handling_error)}")
            return fallback_result
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error into predefined categories."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Model loading errors
        if any(keyword in error_message for keyword in ['model', 'checkpoint', 'weights', 'load']):
            if any(keyword in error_message for keyword in ['download', 'network', 'connection']):
                return "E004"  # Network error during model loading
            return "E001"  # Model loading failure
        
        # Memory errors
        if any(keyword in error_message for keyword in ['memory', 'out of memory', 'oom', 'allocation']):
            return "E002"  # Memory overflow
        
        # GPU errors
        if any(keyword in error_message for keyword in ['cuda', 'gpu', 'device', 'nvidia']):
            if 'memory' in error_message:
                return "E002"  # GPU memory overflow
            return "E003"  # GPU resource exhaustion
        
        # Network errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'ssl', 'http']):
            return "E004"  # Network connection error
        
        # File system errors
        if any(keyword in error_message for keyword in ['file', 'path', 'directory', 'permission', 'disk']):
            return "E005"  # File system error
        
        # Configuration errors
        if any(keyword in error_message for keyword in ['config', 'parameter', 'argument', 'invalid']):
            return "E006"  # Configuration error
        
        # Import/dependency errors
        if 'import' in error_type or 'module' in error_message:
            return "E008"  # Dependency missing
        
        # Hardware compatibility
        if any(keyword in error_message for keyword in ['unsupported', 'compatibility', 'version']):
            return "E009"  # Hardware compatibility issue
        
        # Processing pipeline errors
        if any(keyword in error_message for keyword in ['pipeline', 'processing', 'segmentation', 'detection']):
            return "E007"  # Processing pipeline failure
        
        return "E010"  # Unknown error
    
    def _get_error_solutions(
        self,
        error_code: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get specific solutions for the error code."""
        base_solutions = self.error_solutions_db.get(error_code, [])
        
        # Add context-specific solutions
        contextual_solutions = []
        error_message = str(error).lower()
        
        if error_code == "E001":  # Model loading failure
            if "edgetam" in error_message:
                contextual_solutions.append("Try using SAM2 instead with --no-edgetam flag")
            if "sam2" in error_message:
                contextual_solutions.append("Try using EdgeTAM instead with --edgetam flag")
            if "download" in error_message:
                contextual_solutions.append("Check internet connection and retry model download")
        
        elif error_code == "E002":  # Memory overflow
            if context and context.get("batch_size", 1) > 1:
                contextual_solutions.append(f"Reduce batch size from {context['batch_size']} to 1")
            if "gpu" in error_message:
                contextual_solutions.append("Switch to CPU processing with --device cpu")
        
        elif error_code == "E003":  # GPU resource exhaustion
            contextual_solutions.append("Use nvidia-smi to check GPU memory usage")
            contextual_solutions.append("Close other GPU-intensive applications")
        
        return base_solutions + contextual_solutions
    
    def _create_user_friendly_message(
        self,
        error_code: str,
        error_category: str,
        error: Exception,
        operation_name: str,
        solutions: List[str]
    ) -> str:
        """Create a comprehensive user-friendly error message."""
        
        # Error header with emoji and formatting
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              🚨 ERROR DETECTED 🚨                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 Error Code: {error_code}
🏷️  Category: {error_category}
⚙️  Operation: {operation_name}
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Error description
        description = f"""
📝 DESCRIPTION:
{self._get_error_description(error_code)}

❌ TECHNICAL ERROR:
{type(error).__name__}: {str(error)}

"""
        
        # Solutions section
        solutions_text = """
💡 RECOMMENDED SOLUTIONS:
"""
        for i, solution in enumerate(solutions[:5], 1):  # Limit to top 5 solutions
            solutions_text += f"   {i}. {solution}\n"
        
        # Quick actions
        quick_actions = f"""
⚡ QUICK ACTIONS:
   • Press Ctrl+C to stop current operation
   • Check system resources with Task Manager/Activity Monitor
   • Review the troubleshooting guide below
   • Contact support if problem persists

"""
        
        # Footer
        footer = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  💬 Need help? Include this error code when asking for support: {error_code}     ║
║  📚 Full troubleshooting guide: Use --help or check documentation           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".format(error_code=error_code)
        
        return header + description + solutions_text + quick_actions + footer
    
    def _get_error_description(self, error_code: str) -> str:
        """Get user-friendly description for error code."""
        descriptions = {
            "E001": "A model failed to load properly. This could be due to network issues, corrupted files, or incompatible model versions.",
            "E002": "The system ran out of memory while processing. This typically happens with large videos or high batch sizes.",
            "E003": "GPU resources are exhausted or unavailable. This may be due to insufficient GPU memory or driver issues.",
            "E004": "Network connection failed while downloading models or accessing remote resources.",
            "E005": "File system error occurred. This could be due to missing files, permission issues, or insufficient disk space.",
            "E006": "Configuration parameters are invalid or incompatible. Check your settings and command-line arguments.",
            "E007": "The processing pipeline encountered an error during video/image processing.",
            "E008": "Required dependencies are missing or incompatible. Check your Python environment and installed packages.",
            "E009": "Hardware compatibility issue detected. Your system may not support the requested features.",
            "E010": "An unexpected error occurred that doesn't fit into standard categories."
        }
        return descriptions.get(error_code, "An error occurred during processing.")
    
    def _get_troubleshooting_steps(self, error_code: str, error: Exception) -> List[str]:
        """Get step-by-step troubleshooting guide."""
        return self.troubleshooting_guide.get(error_code, [
            "Review the error message for specific details",
            "Check system resources (CPU, memory, disk space)",
            "Verify input files and parameters",
            "Try with default settings",
            "Contact support with error details"
        ])
    
    def _get_support_information(self, error_code: str) -> Dict[str, str]:
        """Get support information for the error."""
        return {
            "error_code": error_code,
            "documentation_url": "https://github.com/your-repo/sowlv2/docs/troubleshooting.md",
            "issue_template": f"Error Code: {error_code}\nOperation: [describe what you were doing]\nSystem: [OS, GPU, Python version]\nError Details: [paste full error message]",
            "support_email": "support@sowlv2.com",
            "community_forum": "https://github.com/your-repo/sowlv2/discussions"
        }
    
    def _get_quick_fixes(self, error_code: str, error: Exception) -> List[str]:
        """Get quick one-line fixes for common issues."""
        quick_fixes = {
            "E001": ["Retry with --no-edgetam flag", "Check internet connection"],
            "E002": ["Reduce batch size: --batch-size 1", "Enable streaming: --streaming"],
            "E003": ["Use CPU: --device cpu", "Close other GPU apps"],
            "E004": ["Check internet connection", "Use cached models"],
            "E005": ["Check file permissions", "Verify file paths"],
            "E006": ["Use default config", "Check parameter syntax"],
            "E007": ["Reduce input resolution", "Try different model"],
            "E008": ["pip install -r requirements.txt", "Check Python version"],
            "E009": ["Update drivers", "Check hardware compatibility"],
            "E010": ["Enable debug logging", "Contact support"]
        }
        return quick_fixes.get(error_code, ["Contact support"])
    
    def _build_solutions_database(self) -> Dict[str, List[str]]:
        """Build comprehensive solutions database."""
        return {
            "E001": [
                "Check internet connection for model downloads",
                "Verify sufficient disk space for model files",
                "Try alternative model variants (EdgeTAM vs SAM2)",
                "Clear model cache and re-download",
                "Check model file integrity",
                "Use offline mode if models are cached"
            ],
            "E002": [
                "Reduce batch size to 1 or smaller",
                "Enable streaming processing for large videos",
                "Use mixed precision (FP16) to reduce memory usage",
                "Clear GPU cache with torch.cuda.empty_cache()",
                "Reduce input resolution",
                "Close other memory-intensive applications"
            ],
            "E003": [
                "Check GPU memory with nvidia-smi",
                "Switch to CPU processing",
                "Reduce batch size and input resolution",
                "Update GPU drivers",
                "Close other GPU applications",
                "Use gradient checkpointing to save memory"
            ],
            "E004": [
                "Check internet connection stability",
                "Configure proxy settings if behind firewall",
                "Use cached models when available",
                "Retry with exponential backoff",
                "Switch to offline mode",
                "Check firewall and antivirus settings"
            ],
            "E005": [
                "Verify file paths exist and are accessible",
                "Check file permissions (read/write access)",
                "Ensure sufficient disk space",
                "Validate input file formats",
                "Check directory structure",
                "Run with administrator privileges if needed"
            ],
            "E006": [
                "Review configuration file syntax",
                "Use default configuration as baseline",
                "Validate parameter ranges and types",
                "Check command-line argument format",
                "Refer to configuration documentation",
                "Use configuration validation tools"
            ],
            "E007": [
                "Reduce input complexity (resolution, length)",
                "Try different model configurations",
                "Check input file format compatibility",
                "Enable debug logging for detailed errors",
                "Use fallback processing modes",
                "Validate input data integrity"
            ],
            "E008": [
                "Install missing dependencies: pip install -r requirements.txt",
                "Check Python version compatibility",
                "Update package versions",
                "Use virtual environment",
                "Check CUDA/PyTorch installation",
                "Verify system requirements"
            ],
            "E009": [
                "Check hardware requirements",
                "Update system drivers",
                "Verify CUDA compatibility",
                "Use CPU fallback mode",
                "Check operating system compatibility",
                "Update software to latest version"
            ],
            "E010": [
                "Enable debug logging for more details",
                "Check system resources and stability",
                "Try with minimal configuration",
                "Update to latest software version",
                "Contact support with full error details",
                "Check for known issues in documentation"
            ]
        }
    
    def _build_troubleshooting_guide(self) -> Dict[str, List[str]]:
        """Build step-by-step troubleshooting guide."""
        return {
            "E001": [
                "1. Check if you have internet connection",
                "2. Verify available disk space (need ~5GB for models)",
                "3. Try clearing model cache: rm -rf ~/.cache/huggingface",
                "4. Test with different model: --edgetam or --no-edgetam",
                "5. Check firewall/antivirus blocking downloads",
                "6. Try manual model download if automatic fails"
            ],
            "E002": [
                "1. Check current memory usage with Task Manager",
                "2. Reduce batch size: start with --batch-size 1",
                "3. Enable streaming: --streaming --chunk-size 50",
                "4. Use mixed precision: --mixed-precision",
                "5. Clear GPU cache: restart application",
                "6. Consider using CPU: --device cpu"
            ],
            "E003": [
                "1. Run nvidia-smi to check GPU status",
                "2. Close other GPU applications",
                "3. Restart GPU drivers if needed",
                "4. Try CPU processing: --device cpu",
                "5. Reduce memory usage with smaller batches",
                "6. Update CUDA drivers if outdated"
            ],
            "E004": [
                "1. Test internet connection in browser",
                "2. Check if behind corporate firewall",
                "3. Try different network (mobile hotspot)",
                "4. Configure proxy if needed",
                "5. Use cached models if available",
                "6. Contact IT support for network issues"
            ],
            "E005": [
                "1. Verify input file exists and is readable",
                "2. Check file permissions with ls -la (Linux/Mac)",
                "3. Ensure sufficient disk space",
                "4. Try different input file to isolate issue",
                "5. Check directory write permissions",
                "6. Run with elevated privileges if needed"
            ],
            "E006": [
                "1. Review command-line arguments for typos",
                "2. Check configuration file syntax",
                "3. Use --help to see valid options",
                "4. Try with default settings first",
                "5. Validate parameter ranges",
                "6. Check documentation for examples"
            ],
            "E007": [
                "1. Try with smaller/simpler input file",
                "2. Enable debug logging: --verbose",
                "3. Check input file format compatibility",
                "4. Try different model configuration",
                "5. Reduce processing complexity",
                "6. Check for corrupted input data"
            ],
            "E008": [
                "1. Check Python version: python --version",
                "2. Install requirements: pip install -r requirements.txt",
                "3. Update pip: pip install --upgrade pip",
                "4. Check virtual environment activation",
                "5. Verify CUDA installation if using GPU",
                "6. Reinstall problematic packages"
            ],
            "E009": [
                "1. Check system requirements in documentation",
                "2. Update GPU drivers",
                "3. Verify CUDA version compatibility",
                "4. Check operating system support",
                "5. Try CPU-only mode as fallback",
                "6. Consider hardware upgrade if needed"
            ],
            "E010": [
                "1. Enable maximum logging: --debug",
                "2. Check system stability and resources",
                "3. Try with minimal configuration",
                "4. Update software to latest version",
                "5. Search for similar issues online",
                "6. Contact support with full error details"
            ]
        }
    
    def create_interactive_error_resolution(self, error_code: str) -> str:
        """Create interactive error resolution guide."""
        try:
            guide = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔧 INTERACTIVE TROUBLESHOOTING GUIDE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Error Code: {error_code} - {self.ERROR_CODES.get(error_code, 'Unknown')}

Let's solve this step by step:

"""
            
            steps = self._get_troubleshooting_steps(error_code, None)
            for i, step in enumerate(steps, 1):
                guide += f"Step {i}: {step}\n"
                guide += f"   ✓ Completed? (If yes, continue to next step)\n"
                guide += f"   ❌ Still having issues? (Try the solutions below)\n\n"
            
            solutions = self.error_solutions_db.get(error_code, [])
            guide += "💡 Additional Solutions:\n"
            for i, solution in enumerate(solutions, 1):
                guide += f"   {i}. {solution}\n"
            
            guide += f"""
📞 Still need help?
   • Error Code: {error_code}
   • Support: {self._get_support_information(error_code)['support_email']}
   • Documentation: {self._get_support_information(error_code)['documentation_url']}

"""
            
            return guide
            
        except Exception as e:
            return f"Error creating interactive guide: {str(e)}"


class ErrorRecoveryLogger:
    """
    Enhanced logging for error recovery scenarios.
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    def log_fallback_attempt(
        self,
        original_model: str,
        fallback_model: str,
        error: Exception
    ):
        """Log fallback attempt with context."""
        self.logger.warning(
            f"Fallback attempt: {original_model} -> {fallback_model}. "
            f"Original error: {str(error)}"
        )
    
    def log_fallback_success(
        self,
        original_model: str,
        fallback_model: str,
        load_time: float
    ):
        """Log successful fallback."""
        self.logger.info(
            f"Fallback successful: {original_model} -> {fallback_model} "
            f"(loaded in {load_time:.2f}s)"
        )
    
    def log_fallback_failure(
        self,
        original_model: str,
        fallback_model: str,
        fallback_error: Exception
    ):
        """Log fallback failure."""
        self.logger.error(
            f"Fallback failed: {original_model} -> {fallback_model}. "
            f"Fallback error: {str(fallback_error)}"
        )
    
    def log_model_performance_context(
        self,
        model_name: str,
        performance_metrics: Dict[str, Any],
        error: Optional[Exception] = None
    ):
        """Log model performance context for debugging."""
        context_info = f"Model: {model_name}, Metrics: {performance_metrics}"
        
        if error:
            self.logger.error(f"Performance context (ERROR): {context_info}. Error: {str(error)}")
        else:
            self.logger.info(f"Performance context: {context_info}")