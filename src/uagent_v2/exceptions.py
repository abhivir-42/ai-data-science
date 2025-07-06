"""
Exception handling module for the enhanced uAgent implementation.

This module provides structured error handling with specific exception types
to replace the silent failures and inconsistent error handling in the original code.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataAnalysisError(Exception):
    """Base exception for data analysis errors."""
    
    def __init__(self, message: str, error_code: str = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.timestamp = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "timestamp": self.timestamp
        }


class DataValidationError(DataAnalysisError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_type: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type


class FileProcessingError(DataAnalysisError):
    """Raised when file processing fails."""
    
    def __init__(self, message: str, file_path: str = None, operation: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation


class NetworkError(DataAnalysisError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, url: str = None, status_code: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code


class SecurityError(DataAnalysisError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_check: str = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)
        self.security_check = security_check


class MemoryError(DataAnalysisError):
    """Raised when memory-related issues occur."""
    
    def __init__(self, message: str, memory_usage: int = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.memory_usage = memory_usage


class ConfigurationError(DataAnalysisError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.config_key = config_key


class AgentExecutionError(DataAnalysisError):
    """Raised when agent execution fails."""
    
    def __init__(self, message: str, agent_name: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.agent_name = agent_name


def handle_analysis_error(error: Exception, context: str = None) -> str:
    """
    Convert exceptions to user-friendly error messages.
    
    Args:
        error: The exception to handle
        context: Additional context about where the error occurred
        
    Returns:
        User-friendly error message
    """
    # Log the error for debugging
    logger = logging.getLogger(__name__)
    logger.error(f"Error in {context}: {error}", exc_info=True)
    
    # Handle specific error types
    if isinstance(error, DataValidationError):
        return f"❌ **Data validation failed**: {error.message}\n\n💡 **Solution**: Check your data format and ensure it meets the requirements."
    
    elif isinstance(error, FileProcessingError):
        return f"❌ **File processing failed**: {error.message}\n\n💡 **Solution**: Verify the file exists and is accessible."
    
    elif isinstance(error, NetworkError):
        return f"❌ **Network error**: {error.message}\n\n💡 **Solution**: Check your internet connection and try again."
    
    elif isinstance(error, SecurityError):
        return f"🚨 **Security error**: {error.message}\n\n💡 **Solution**: Ensure your file meets security requirements."
    
    elif isinstance(error, MemoryError):
        return f"⚠️ **Memory error**: {error.message}\n\n💡 **Solution**: Try processing a smaller dataset or ask for chunked delivery."
    
    elif isinstance(error, ConfigurationError):
        return f"⚙️ **Configuration error**: {error.message}\n\n💡 **Solution**: Check your environment variables and configuration."
    
    elif isinstance(error, AgentExecutionError):
        return f"🤖 **Agent execution failed**: {error.message}\n\n💡 **Solution**: The analysis agent encountered an issue. Please try again."
    
    # Handle standard exceptions
    elif isinstance(error, ValueError):
        return f"❌ **Invalid value**: {str(error)}\n\n💡 **Solution**: Check your input parameters and data format."
    
    elif isinstance(error, FileNotFoundError):
        return f"❌ **File not found**: {str(error)}\n\n💡 **Solution**: Verify the file path or URL is correct."
    
    elif isinstance(error, PermissionError):
        return f"❌ **Permission denied**: {str(error)}\n\n💡 **Solution**: Check file permissions or try a different location."
    
    elif isinstance(error, TimeoutError):
        return f"⏱️ **Timeout error**: {str(error)}\n\n💡 **Solution**: The operation took too long. Try again or use a smaller dataset."
    
    # Generic error handling
    else:
        return f"❌ **Unexpected error**: {str(error)}\n\n💡 **Solution**: Please contact support if this issue persists."


def create_error_response(error: Exception, context: str = None) -> Dict[str, Any]:
    """
    Create a structured error response for API use.
    
    Args:
        error: The exception to handle
        context: Additional context about where the error occurred
        
    Returns:
        Structured error response
    """
    error_message = handle_analysis_error(error, context)
    
    return {
        "success": False,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
            "user_message": error_message,
            "context": context,
            "severity": getattr(error, 'severity', ErrorSeverity.MEDIUM).value
        }
    }


def safe_execute(func, *args, error_context: str = None, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        error_context: Context for error reporting
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or error response
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in {error_context or func.__name__}: {e}", exc_info=True)
        raise DataAnalysisError(
            f"Failed to execute {func.__name__}: {str(e)}", 
            error_code=f"EXEC_{func.__name__.upper()}"
        ) from e 