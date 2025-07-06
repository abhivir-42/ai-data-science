"""
Enhanced uAgent v2 Implementation

This package provides an improved version of the data analysis uAgent with:
- Modular architecture for better maintainability
- Enhanced security and validation
- Memory-efficient processing
- Structured error handling
- Comprehensive configuration management

Modules:
- config: Configuration management with environment variable support
- exceptions: Structured error handling with custom exception types
- utils: Memory-efficient utilities and optimization functions
- file_handlers: Secure file upload/download and content handling
"""

from .config import UAgentConfig, default_config
from .exceptions import (
    DataAnalysisError, 
    SecurityError, 
    FileProcessingError, 
    NetworkError,
    handle_analysis_error
)
from .utils import (
    MemoryEfficientCSVProcessor,
    DataDeliveryOptimizer,
    format_file_size,
    sanitize_filename
)
from .file_handlers import (
    SecureFileUploader,
    SecureFileDownloader,
    FileContentHandler
)

__version__ = "2.0.0"
__author__ = "Abhivir Singh"
__description__ = "Enhanced uAgent implementation with security and performance improvements"

__all__ = [
    "UAgentConfig",
    "default_config",
    "DataAnalysisError",
    "SecurityError", 
    "FileProcessingError",
    "NetworkError",
    "handle_analysis_error",
    "MemoryEfficientCSVProcessor",
    "DataDeliveryOptimizer",
    "format_file_size",
    "sanitize_filename",
    "SecureFileUploader",
    "SecureFileDownloader",
    "FileContentHandler"
] 