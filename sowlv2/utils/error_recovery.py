"""
Error recovery utilities for SOWLv2 pipeline.
Provides fallback mechanisms and user notification systems.
"""
import logging
from typing import Callable, Optional, Any, Dict
from functools import wraps

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