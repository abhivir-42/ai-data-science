"""
Memory-efficient utilities for the enhanced uAgent implementation.

This module provides optimized functions for handling large datasets without
loading them multiple times in memory, addressing the memory management issues
in the original code.
"""

import io
import os
import re
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple, Iterator
from pathlib import Path
import pandas as pd
from .config import UAgentConfig
from .exceptions import MemoryError, FileProcessingError, DataValidationError, SecurityError


logger = logging.getLogger(__name__)


class MemoryEfficientCSVProcessor:
    """Memory-efficient CSV processing utilities."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self._file_cache = {}
        self._size_cache = {}
    
    def get_csv_size_estimate(self, df: pd.DataFrame) -> int:
        """
        Estimate CSV size without loading full content in memory.
        
        Args:
            df: DataFrame to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Sample-based estimation for large datasets
            sample_size = min(100, len(df))
            if sample_size == 0:
                return 0
            
            # Use string buffer to estimate size
            buffer = io.StringIO()
            df.head(sample_size).to_csv(buffer, index=False)
            sample_content = buffer.getvalue()
            sample_bytes = len(sample_content.encode('utf-8'))
            
            # Estimate full size based on sample
            if sample_size < len(df):
                estimated_size = int((sample_bytes / sample_size) * len(df))
            else:
                estimated_size = sample_bytes
            
            return estimated_size
            
        except Exception as e:
            logger.warning(f"Could not estimate CSV size: {e}")
            # Fallback: rough estimate based on DataFrame shape
            return len(df) * len(df.columns) * 20  # Rough estimate
    
    def get_file_size_safe(self, file_path: str) -> int:
        """
        Safely get file size with caching.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        try:
            # Check cache first
            if file_path in self._size_cache:
                return self._size_cache[file_path]
            
            if not os.path.exists(file_path):
                raise FileProcessingError(f"File not found: {file_path}")
            
            size = os.path.getsize(file_path)
            
            # Cache the result
            self._size_cache[file_path] = size
            
            return size
            
        except Exception as e:
            raise FileProcessingError(f"Could not get file size: {e}", file_path=file_path)
    
    def read_csv_chunked(self, file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Read CSV file in chunks to manage memory usage.
        
        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
                
        except Exception as e:
            raise FileProcessingError(f"Could not read CSV in chunks: {e}", file_path=file_path)
    
    def get_dataframe_memory_usage(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get detailed memory usage information for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with memory usage details
        """
        try:
            memory_usage = df.memory_usage(deep=True)
            
            return {
                "total_bytes": memory_usage.sum(),
                "total_mb": memory_usage.sum() / (1024 * 1024),
                "index_bytes": memory_usage.iloc[0],
                "columns_bytes": memory_usage.iloc[1:].sum(),
                "avg_per_row": memory_usage.sum() / len(df) if len(df) > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate memory usage: {e}")
            return {"total_bytes": 0, "total_mb": 0, "index_bytes": 0, "columns_bytes": 0, "avg_per_row": 0}
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns to category where appropriate
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < len(df) * 0.5:  # If less than 50% unique
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum()
            reduction = ((original_memory - optimized_memory) / original_memory) * 100
            
            logger.info(f"Memory optimization: {reduction:.1f}% reduction")
            
            return df
            
        except Exception as e:
            logger.warning(f"Could not optimize DataFrame memory: {e}")
            return df  # Return original if optimization fails


class FileValidator:
    """File validation utilities with security checks."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def validate_file_security(self, file_path: str) -> bool:
        """
        Validate file for security issues.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            True if file is safe, False otherwise
        """
        try:
            # Check file extension
            if not file_path.lower().endswith(self.config.allowed_file_types):
                raise SecurityError(f"File type not allowed: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.config.get_file_size_bytes():
                raise SecurityError(f"File too large: {file_size} bytes")
            
            # Check for path traversal attempts
            if '..' in file_path or file_path.startswith('/'):
                raise SecurityError("Path traversal detected")
            
            # Basic content validation for CSV
            if file_path.lower().endswith('.csv'):
                return self._validate_csv_content(file_path)
            
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return False
    
    def _validate_csv_content(self, file_path: str) -> bool:
        """
        Validate CSV file content.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            True if valid CSV, False otherwise
        """
        try:
            # Try to read first few rows
            df = pd.read_csv(file_path, nrows=5)
            
            # Check if we have data
            if df.empty:
                raise DataValidationError("CSV file is empty")
            
            # Check for reasonable number of columns
            if len(df.columns) > 1000:
                raise DataValidationError("Too many columns in CSV")
            
            return True
            
        except pd.errors.EmptyDataError:
            raise DataValidationError("CSV file is empty or corrupted")
        except Exception as e:
            raise DataValidationError(f"Invalid CSV format: {e}")


class DataDeliveryOptimizer:
    """Optimize data delivery based on size and format."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self.processor = MemoryEfficientCSVProcessor(config)
    
    def determine_delivery_strategy(self, df: pd.DataFrame) -> str:
        """
        Determine the best delivery strategy based on data size.
        
        Args:
            df: DataFrame to deliver
            
        Returns:
            Delivery strategy ('direct', 'chunked', 'sampled', 'link')
        """
        try:
            estimated_size = self.processor.get_csv_size_estimate(df)
            
            if estimated_size < self.config.get_small_file_size_bytes():
                return 'direct'
            elif estimated_size < self.config.get_medium_file_size_bytes():
                return 'direct'  # Still manageable
            elif estimated_size < self.config.get_file_size_bytes():
                return 'chunked'
            else:
                return 'link'  # Too large, provide download link
                
        except Exception as e:
            logger.warning(f"Could not determine delivery strategy: {e}")
            return 'sampled'  # Safe fallback
    
    def create_data_preview(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """
        Create a data preview with metadata.
        
        Args:
            df: DataFrame to preview
            max_rows: Maximum rows to include
            
        Returns:
            Formatted preview string
        """
        try:
            preview_df = df.head(max_rows)
            
            # Get memory usage
            memory_info = self.processor.get_dataframe_memory_usage(df)
            
            # Create preview
            preview_lines = [
                f"📊 **Dataset Preview** ({len(df):,} rows × {len(df.columns)} columns)",
                f"💾 **Memory Usage**: {memory_info['total_mb']:.2f} MB",
                f"🔢 **Data Types**: {df.dtypes.value_counts().to_dict()}",
                "",
                "```csv",
                preview_df.to_csv(index=False),
                "```"
            ]
            
            return "\n".join(preview_lines)
            
        except Exception as e:
            logger.error(f"Could not create data preview: {e}")
            return f"❌ Could not create preview: {e}"


# Pre-compiled regex patterns for performance
REGEX_PATTERNS = {
    'url_extraction': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+\.csv(?:\?[^\s<>"{}|\\^`\[\]]*)?'),
    'file_size': re.compile(r'(\d+(?:\.\d+)?)\s*(KB|MB|GB)', re.IGNORECASE),
    'row_count': re.compile(r'(\d+)\s*rows?', re.IGNORECASE),
    'column_count': re.compile(r'(\d+)\s*columns?', re.IGNORECASE),
}


def get_file_hash(file_path: str) -> str:
    """
    Generate hash for file content (for caching).
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash of file content
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate file hash: {e}")
        return f"hash_error_{os.path.basename(file_path)}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path traversal attempts
    sanitized = filename.replace('..', '')
    sanitized = sanitized.replace('/', '_')
    sanitized = sanitized.replace('\\', '_')
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', sanitized)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure we have a valid filename
    if not sanitized or sanitized == '.':
        sanitized = 'sanitized_file'
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        max_name_length = 250 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized 