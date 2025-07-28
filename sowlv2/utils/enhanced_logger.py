"""
Enhanced error logging system for SOWLv2 pipeline.
Provides detailed error context, resource state logging, and debugging reports.
"""
import logging
import json
import time
import traceback
import sys
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import psutil
import torch


class EnhancedErrorLogger:
    """
    Enhanced error logger with performance context and resource state tracking.
    Provides comprehensive debugging information for SOWLv2 pipeline errors.
    """

    def __init__(self, logger_name: str = __name__, log_file: Optional[str] = None):
        self.logger = logging.getLogger(logger_name)
        self.log_file = log_file
        self.error_history = []
        self.performance_context_history = []
        self.resource_snapshots = []

        # Configure structured logging format
        self._setup_structured_logging()

    def _setup_structured_logging(self):
        """Setup structured logging with JSON format for better parsing."""
        try:
            # Create formatter for structured logs
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Add file handler if log file specified
            if self.log_file:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            # Ensure logger has appropriate level
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            self.logger.setLevel(logging.INFO)

        except Exception as e:
            print(f"Warning: Failed to setup structured logging: {str(e)}")

    def log_performance_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation_name: str = "unknown",
        severity: str = "ERROR"
    ):
        """
        Log detailed performance context when errors occur.

        Args:
            error: The exception that occurred
            context: Performance and operational context
            operation_name: Name of the operation that failed
            severity: Log severity level
        """
        try:
            # Create comprehensive context record
            performance_context = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "severity": severity,
                "context": context,
                "system_state": self._capture_system_state(),
                "traceback": traceback.format_exc() if severity == "ERROR" else None
            }

            # Add to history
            self.performance_context_history.append(performance_context)

            # Create structured log message
            log_message = self._format_performance_context_message(performance_context)

            # Log with appropriate level
            if severity == "ERROR":
                self.logger.error(log_message)
            elif severity == "WARNING":
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)

            # Log as JSON for machine parsing
            json_context = json.dumps(performance_context, indent=2, default=str)
            self.logger.debug(f"Performance Context JSON:\n{json_context}")

        except Exception as logging_error:
            self.logger.error(f"Failed to log performance context: {str(logging_error)}")

    def log_resource_state(
        self,
        error: Exception,
        operation_name: str = "unknown",
        include_gpu_info: bool = True
    ):
        """
        Log detailed resource state when errors occur.

        Args:
            error: The exception that occurred
            operation_name: Name of the operation that failed
            include_gpu_info: Whether to include GPU information
        """
        try:
            # Capture comprehensive resource state
            resource_state = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "cpu_info": self._get_cpu_info(),
                "memory_info": self._get_memory_info(),
                "disk_info": self._get_disk_info(),
                "process_info": self._get_process_info()
            }

            # Add GPU information if available and requested
            if include_gpu_info and torch.cuda.is_available():
                resource_state["gpu_info"] = self._get_gpu_info()

            # Add to snapshots
            self.resource_snapshots.append(resource_state)

            # Create formatted log message
            log_message = self._format_resource_state_message(resource_state)
            self.logger.error(log_message)

            # Log detailed JSON for debugging
            json_state = json.dumps(resource_state, indent=2, default=str)
            self.logger.debug(f"Resource State JSON:\n{json_state}")

        except Exception as logging_error:
            self.logger.error(f"Failed to log resource state: {str(logging_error)}")

    def generate_debugging_report(
        self,
        error_history: Optional[List[Exception]] = None,
        include_recommendations: bool = True
    ) -> str:
        """
        Generate comprehensive debugging report for error analysis.

        Args:
            error_history: List of recent errors (uses internal history if None)
            include_recommendations: Whether to include troubleshooting recommendations

        Returns:
            Formatted debugging report string
        """
        try:
            report_timestamp = datetime.now().isoformat()
            errors_to_analyze = error_history or [
                ctx["error_message"] for ctx in self.performance_context_history[-10:]
            ]

            # Build comprehensive report
            report = [
                "=" * 80,
                "SOWLv2 DEBUGGING REPORT",
                "=" * 80,
                f"Generated: {report_timestamp}",
                f"Total Errors Analyzed: {len(errors_to_analyze)}",
                f"Performance Context Records: {len(self.performance_context_history)}",
                f"Resource Snapshots: {len(self.resource_snapshots)}",
                "",
                "SYSTEM OVERVIEW",
                "-" * 40
            ]

            # Add current system state
            current_state = self._capture_system_state()
            for key, value in current_state.items():
                report.append(f"{key.replace('_', ' ').title()}: {value}")

            report.extend([
                "",
                "ERROR ANALYSIS",
                "-" * 40
            ])

            # Analyze error patterns
            error_analysis = self._analyze_error_patterns(errors_to_analyze)
            for category, details in error_analysis.items():
                report.append(f"\n{category.replace('_', ' ').title()}:")
                if isinstance(details, dict):
                    for key, value in details.items():
                        report.append(f"  • {key}: {value}")
                else:
                    report.append(f"  • {details}")

            # Add recent performance context
            if self.performance_context_history:
                report.extend([
                    "",
                    "RECENT PERFORMANCE CONTEXT",
                    "-" * 40
                ])

                for ctx in self.performance_context_history[-5:]:
                    report.extend([
                        f"\nOperation: {ctx['operation']}",
                        f"Time: {ctx['timestamp']}",
                        f"Error: {ctx['error_type']} - {ctx['error_message']}",
                        f"Severity: {ctx['severity']}"
                    ])

                    if ctx.get('context'):
                        report.append("Context:")
                        for key, value in ctx['context'].items():
                            report.append(f"  • {key}: {value}")

            # Add resource state analysis
            if self.resource_snapshots:
                report.extend([
                    "",
                    "RESOURCE STATE ANALYSIS",
                    "-" * 40
                ])

                latest_snapshot = self.resource_snapshots[-1]
                report.extend([
                    f"Latest Snapshot: {latest_snapshot['timestamp']}",
                    f"CPU Usage: {latest_snapshot['cpu_info'].get('usage_percent', 'N/A')}%",
                    f"Memory Usage: {latest_snapshot['memory_info'].get('usage_percent', 'N/A')}%",
                    f"Available Memory: {latest_snapshot['memory_info'].get('available_gb', 'N/A')}GB"
                ])

                if 'gpu_info' in latest_snapshot:
                    gpu_info = latest_snapshot['gpu_info']
                    report.extend([
                        f"GPU Memory Used: {gpu_info.get('memory_used_gb', 'N/A')}GB",
                        f"GPU Memory Total: {gpu_info.get('memory_total_gb', 'N/A')}GB",
                        f"GPU Utilization: {gpu_info.get('utilization_percent', 'N/A')}%"
                    ])

            # Add troubleshooting recommendations
            if include_recommendations:
                recommendations = self._generate_troubleshooting_recommendations(
                    errors_to_analyze, error_analysis
                )

                report.extend([
                    "",
                    "TROUBLESHOOTING RECOMMENDATIONS",
                    "-" * 40
                ])

                for i, recommendation in enumerate(recommendations, 1):
                    report.append(f"{i}. {recommendation}")

            # Add footer
            report.extend([
                "",
                "=" * 80,
                f"Report generated by SOWLv2 Enhanced Error Logger",
                f"For support, include this report with your issue description",
                "=" * 80
            ])

            return "\n".join(report)

        except Exception as e:
            error_msg = f"Failed to generate debugging report: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def log_with_severity(
        self,
        message: str,
        severity: str = "INFO",
        context: Optional[Dict[str, Any]] = None,
        operation: str = "unknown"
    ):
        """
        Log message with specified severity level and optional context.

        Args:
            message: Log message
            severity: Severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            context: Optional context information
            operation: Operation name for context
        """
        try:
            # Create structured log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "severity": severity,
                "message": message,
                "context": context or {}
            }

            # Format message with context
            formatted_message = f"[{operation}] {message}"
            if context:
                formatted_message += f" | Context: {json.dumps(context, default=str)}"

            # Log with appropriate level
            severity_upper = severity.upper()
            if severity_upper == "DEBUG":
                self.logger.debug(formatted_message)
            elif severity_upper == "INFO":
                self.logger.info(formatted_message)
            elif severity_upper == "WARNING":
                self.logger.warning(formatted_message)
            elif severity_upper == "ERROR":
                self.logger.error(formatted_message)
            elif severity_upper == "CRITICAL":
                self.logger.critical(formatted_message)
            else:
                self.logger.info(formatted_message)

        except Exception as e:
            self.logger.error(f"Failed to log with severity: {str(e)}")

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for context."""
        try:
            return {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except Exception:
            return {"error": "Failed to capture system state"}

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            return {
                "count": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=1),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "usage_percent": memory.percent,
                "cached_gb": getattr(memory, 'cached', 0) / (1024**3)
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information."""
        try:
            disk = psutil.disk_usage('/')
            return {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "usage_percent": (disk.used / disk.total) * 100
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "pid": process.pid,
                "memory_rss_gb": memory_info.rss / (1024**3),
                "memory_vms_gb": memory_info.vms / (1024**3),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}

            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                memory_total = device_props.total_memory / (1024**3)

                gpu_info[f"device_{i}"] = {
                    "name": device_props.name,
                    "memory_total_gb": memory_total,
                    "memory_allocated_gb": memory_allocated,
                    "memory_cached_gb": memory_cached,
                    "memory_free_gb": memory_total - memory_allocated,
                    "utilization_percent": (memory_allocated / memory_total) * 100,
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                }

            return gpu_info
        except Exception as e:
            return {"error": str(e)}

    def _format_performance_context_message(self, context: Dict[str, Any]) -> str:
        """Format performance context for logging."""
        try:
            return (
                f"Performance Context - Operation: {context['operation']}, "
                f"Error: {context['error_type']}, "
                f"System: CPU={context['system_state'].get('cpu_count', 'N/A')}, "
                f"Memory={context['system_state'].get('memory_available_gb', 'N/A'):.1f}GB, "
                f"GPU={'Yes' if context['system_state'].get('cuda_available') else 'No'}"
            )
        except Exception:
            return f"Performance Context - Operation: {context.get('operation', 'unknown')}"

    def _format_resource_state_message(self, state: Dict[str, Any]) -> str:
        """Format resource state for logging."""
        try:
            cpu_usage = state['cpu_info'].get('usage_percent', 'N/A')
            memory_usage = state['memory_info'].get('usage_percent', 'N/A')
            memory_available = state['memory_info'].get('available_gb', 'N/A')

            message = (
                f"Resource State - Operation: {state['operation']}, "
                f"CPU: {cpu_usage}%, Memory: {memory_usage}% "
                f"({memory_available:.1f}GB available)"
            )

            if 'gpu_info' in state and state['gpu_info']:
                gpu_info = list(state['gpu_info'].values())[0]  # First GPU
                gpu_memory = gpu_info.get('memory_allocated_gb', 'N/A')
                gpu_util = gpu_info.get('utilization_percent', 'N/A')
                message += f", GPU: {gpu_util:.1f}% ({gpu_memory:.1f}GB used)"

            return message
        except Exception:
            return f"Resource State - Operation: {state.get('operation', 'unknown')}"

    def _analyze_error_patterns(self, errors: List[str]) -> Dict[str, Any]:
        """Analyze error patterns for common issues."""
        try:
            analysis = {
                "total_errors": len(errors),
                "memory_related": 0,
                "gpu_related": 0,
                "network_related": 0,
                "file_related": 0,
                "model_related": 0,
                "common_patterns": []
            }

            for error in errors:
                error_lower = str(error).lower()

                if any(keyword in error_lower for keyword in ['memory', 'out of memory', 'oom']):
                    analysis["memory_related"] += 1

                if any(keyword in error_lower for keyword in ['cuda', 'gpu', 'device']):
                    analysis["gpu_related"] += 1

                if any(keyword in error_lower for keyword in ['connection', 'network', 'timeout']):
                    analysis["network_related"] += 1

                if any(keyword in error_lower for keyword in ['file', 'path', 'directory']):
                    analysis["file_related"] += 1

                if any(keyword in error_lower for keyword in ['model', 'checkpoint', 'weights']):
                    analysis["model_related"] += 1

            # Identify common patterns
            if analysis["memory_related"] > len(errors) * 0.3:
                analysis["common_patterns"].append("Frequent memory issues detected")

            if analysis["gpu_related"] > len(errors) * 0.2:
                analysis["common_patterns"].append("GPU-related problems detected")

            if analysis["network_related"] > 0:
                analysis["common_patterns"].append("Network connectivity issues detected")

            return analysis
        except Exception:
            return {"error": "Failed to analyze error patterns"}

    def _generate_troubleshooting_recommendations(
        self,
        errors: List[str],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate troubleshooting recommendations based on error analysis."""
        recommendations = []

        try:
            # Memory-related recommendations
            if analysis.get("memory_related", 0) > 0:
                recommendations.extend([
                    "Reduce batch size to lower memory usage",
                    "Enable streaming processing for large videos",
                    "Clear model cache and force garbage collection",
                    "Consider using mixed precision (FP16) to reduce memory usage"
                ])

            # GPU-related recommendations
            if analysis.get("gpu_related", 0) > 0:
                recommendations.extend([
                    "Check GPU memory availability with nvidia-smi",
                    "Try falling back to CPU processing",
                    "Reduce input resolution or batch size",
                    "Update GPU drivers and CUDA installation"
                ])

            # Network-related recommendations
            if analysis.get("network_related", 0) > 0:
                recommendations.extend([
                    "Check internet connection for model downloads",
                    "Use cached models if available",
                    "Configure proxy settings if behind firewall",
                    "Retry with exponential backoff for network operations"
                ])

            # Model-related recommendations
            if analysis.get("model_related", 0) > 0:
                recommendations.extend([
                    "Verify model files are not corrupted",
                    "Check model compatibility with current hardware",
                    "Try alternative model variants",
                    "Clear model cache and re-download"
                ])

            # General recommendations
            recommendations.extend([
                "Check system resources (CPU, memory, disk space)",
                "Review configuration parameters for correctness",
                "Enable debug logging for more detailed error information",
                "Update SOWLv2 to the latest version"
            ])

            return recommendations[:10]  # Limit to top 10 recommendations

        except Exception:
            return ["Enable debug logging and check system resources"]

    def clear_history(self):
        """Clear error history and snapshots."""
        self.error_history.clear()
        self.performance_context_history.clear()
        self.resource_snapshots.clear()
        self.logger.info("Error logging history cleared")

    def export_logs(self, output_file: str) -> bool:
        """
        Export all logged data to a file.

        Args:
            output_file: Path to output file

        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "error_history": self.error_history,
                "performance_context_history": self.performance_context_history,
                "resource_snapshots": self.resource_snapshots,
                "system_state": self._capture_system_state()
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info(f"Logs exported to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export logs: {str(e)}")
            return False
