"""
Data loader tools for AI Data Science agents.

This module provides tools for loading data from various sources, listing directories,
and searching for files.
"""

import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain.tools import tool


def estimate_token_count(data: Any) -> int:
    """
    Rough estimation of token count for data.
    
    Parameters
    ----------
    data : Any
        Data to estimate tokens for
        
    Returns
    -------
    int
        Estimated token count
    """
    if isinstance(data, str):
        # Rough approximation: 1 token per 4 characters
        return len(data) // 4
    elif isinstance(data, dict):
        import json
        return len(json.dumps(data)) // 4
    else:
        return len(str(data)) // 4


def truncate_dataframe_for_llm(df: pd.DataFrame, max_tokens: int = 50000) -> Dict[str, Any]:
    """
    Truncate a DataFrame to stay within token limits while preserving structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to truncate
    max_tokens : int
        Maximum token count to target
        
    Returns
    -------
    Dict[str, Any]
        Truncated data dict with metadata
    """
    total_rows, total_cols = df.shape
    
    # Start with a sample and expand if we have room
    sample_rows = min(50, total_rows)  # Start with 50 rows max
    sample_cols = min(20, total_cols)  # Start with 20 columns max
    
    # If dataset is very wide, prioritize showing column structure over rows
    if total_cols > 50:
        sample_rows = min(10, total_rows)
        sample_cols = min(30, total_cols)
    
    # Create a sample of the data
    df_sample = df.iloc[:sample_rows, :sample_cols].copy()
    
    # Convert to dict and estimate tokens
    sample_dict = df_sample.to_dict()
    estimated_tokens = estimate_token_count(sample_dict)
    
    # If still too large, reduce further
    while estimated_tokens > max_tokens and (sample_rows > 5 or sample_cols > 5):
        if sample_rows > 5:
            sample_rows = max(5, sample_rows // 2)
        if sample_cols > 5:
            sample_cols = max(5, sample_cols // 2)
        
        df_sample = df.iloc[:sample_rows, :sample_cols].copy()
        sample_dict = df_sample.to_dict()
        estimated_tokens = estimate_token_count(sample_dict)
    
    # Create metadata about the truncation
    metadata = {
        "original_shape": [total_rows, total_cols],
        "sample_shape": [sample_rows, sample_cols],
        "truncated": sample_rows < total_rows or sample_cols < total_cols,
        "estimated_tokens": estimated_tokens,
        "missing_rows": total_rows - sample_rows,
        "missing_columns": total_cols - sample_cols
    }
    
    if metadata["truncated"]:
        # Add info about missing columns
        if sample_cols < total_cols:
            missing_cols = list(df.columns[sample_cols:])
            metadata["missing_column_names"] = missing_cols[:10]  # Show first 10
            if len(missing_cols) > 10:
                metadata["additional_missing_columns"] = len(missing_cols) - 10
    
    return {
        "data": sample_dict,
        "metadata": metadata
    }


def calculate_optimal_chunk_size(df: pd.DataFrame, max_tokens: int = 180000) -> int:
    """
    Calculate optimal chunk size for a DataFrame to stay within token limits.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    max_tokens : int
        Maximum tokens to target (with safety margin)
        
    Returns
    -------
    int
        Optimal number of rows per chunk
    """
    total_rows, total_cols = df.shape
    
    # Estimate tokens per row (rough approximation)
    # Sample a few rows to estimate average tokens per row
    sample_size = min(5, total_rows)
    sample_df = df.head(sample_size)
    sample_dict = sample_df.to_dict()
    
    estimated_tokens_per_sample = estimate_token_count(sample_dict)
    tokens_per_row = estimated_tokens_per_sample / sample_size if sample_size > 0 else 1000
    
    # Calculate how many rows we can fit in the token limit
    optimal_rows = max(1, int(max_tokens / tokens_per_row))
    
    # Don't chunk if it's already small enough
    if optimal_rows >= total_rows:
        return total_rows
    
    return optimal_rows


def load_large_dataset_in_chunks(df: pd.DataFrame, max_tokens: int = 150000, chunk_size: int = None) -> Dict[str, Any]:
    """
    Load a large dataset by splitting it into chunks that fit within token limits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Large DataFrame to process
    max_tokens : int
        Maximum tokens per chunk
    chunk_size : int, optional
        Specific chunk size to use (overrides automatic calculation)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with chunked data and metadata
    """
    total_rows, total_cols = df.shape
    
    # Use provided chunk size or calculate one
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(df, max_tokens)
    
    # If no chunking needed, return full dataset
    if chunk_size >= total_rows:
        return {
            "data": df.to_dict(),
            "chunk_info": {
                "total_chunks": 1,
                "chunk_size": total_rows,
                "total_rows": total_rows,
                "total_columns": total_cols,
                "chunked": False
            }
        }
    
    # Calculate number of chunks needed
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
    
    # For the initial load, just return the first chunk with metadata
    # The cleaning agent will handle processing all chunks
    first_chunk = df.head(chunk_size)
    
    chunk_info = {
        "total_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "chunked": True,
        "current_chunk": 1,
        "rows_in_current_chunk": len(first_chunk),
        "remaining_rows": total_rows - chunk_size
    }
    
    return {
        "data": first_chunk.to_dict(),
        "chunk_info": chunk_info,
        "full_dataframe": df  # Store the full dataset for later processing
    }


@tool
def load_file(file_path: str) -> Dict[str, Any]:
    """
    Load a data file into a pandas DataFrame with intelligent chunking for large datasets.
    
    Supports CSV, Excel, JSON, Parquet, and other common file formats.
    For large datasets, automatically chunks the data to stay within token limits
    while preserving all columns and processing the complete dataset.
    
    Parameters
    ----------
    file_path : str
        Path to the file to load
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with DataFrame data and file information
    """
    return _load_file_impl(file_path, max_tokens=150000)


def _load_file_impl(file_path: str, max_tokens: int = 150000, **kwargs) -> Dict[str, Any]:
    """
    Internal implementation of load_file with full parameter support.
    
    Parameters
    ----------
    file_path : str
        Path to the file to load
    max_tokens : int
        Maximum tokens to target when loading large datasets (default: 150000)
    **kwargs : 
        Additional keyword arguments to pass to the pandas read function
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with DataFrame data and file information
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Load the file based on extension
        if file_ext == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_ext in ['.xls', '.xlsx', '.xlsm']:
            df = pd.read_excel(file_path, **kwargs)
        elif file_ext == '.json':
            df = pd.read_json(file_path, **kwargs)
        elif file_ext in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path, **kwargs)
        elif file_ext == '.txt':
            df = pd.read_csv(file_path, sep='\t', **kwargs)
        elif file_ext == '.pdf':
            # Handle PDF files using PDF processor tools
            from src.tools.pdf_processor import smart_extract_data_from_pdf
            pdf_result = smart_extract_data_from_pdf(file_path)
            
            if "error" in pdf_result:
                return pdf_result
            
            # If we found structured data (tables), use the first table as the main DataFrame
            if pdf_result["structured_data"]:
                first_table = pdf_result["structured_data"][0]
                if first_table.get("data"):
                    df = pd.DataFrame.from_dict(first_table["data"])
                else:
                    return {"error": "PDF contains no extractable tabular data"}
            else:
                return {"error": "No structured data found in PDF"}
        else:
            return {"error": f"Unsupported file format: {file_ext}"}
        
        # Be more aggressive about chunking for very large datasets
        total_cells = df.shape[0] * df.shape[1]
        estimated_tokens = estimate_token_count(df.head(5).to_dict()) * (df.shape[0] / 5)
        
        # Use stricter thresholds for chunking
        needs_chunking = (
            estimated_tokens > max_tokens or 
            total_cells > 20000 or  # Much more aggressive cell threshold
            df.shape[0] > 200 or   # Much more aggressive row threshold
            df.shape[1] > 50       # Chunk if too many columns
        )
        
        if needs_chunking:
            # Be much more conservative with chunk sizes
            if df.shape[0] > 1000:
                # For the house price dataset: aim for ~80k tokens max
                chunk_size = max(50, int(80000 / (estimated_tokens / df.shape[0])))
                chunk_size = min(chunk_size, 200)  # Never more than 200 rows for very large datasets
            elif df.shape[0] > 500:
                # For medium-large datasets
                chunk_size = max(30, int(100000 / (estimated_tokens / df.shape[0])))
                chunk_size = min(chunk_size, 300)
            else:
                # For smaller datasets that still need chunking
                chunk_size = max(20, int(max_tokens * 0.6 / (estimated_tokens / df.shape[0])))
            
            # Make sure chunk_size is reasonable and not too large
            chunk_size = max(min(chunk_size, df.shape[0]), 10)
            
            print(f"   📦 Large dataset detected: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   🔄 Using intelligent chunking: {chunk_size} rows per chunk")
            print(f"   📊 Estimated tokens per chunk: ~{int(estimated_tokens * chunk_size / df.shape[0])}")
            
            # Use intelligent chunking that preserves all columns
            chunked_result = load_large_dataset_in_chunks(df, max_tokens, chunk_size)
            data_dict = chunked_result["data"]
            chunk_info = chunked_result["chunk_info"]
            chunk_info["chunk_size"] = chunk_size  # Override with our calculated size
            
            # Store the full dataset for later processing by cleaning agent
            full_dataframe_included = "full_dataframe" in chunked_result
        else:
            # Small enough to send in full
            data_dict = df.to_dict()
            chunk_info = {
                "total_chunks": 1,
                "chunk_size": len(df),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "chunked": False
            }
            full_dataframe_included = False
        
        # Create comprehensive file info
        file_info = {
                "path": file_path,
                "rows": len(df),
                "columns": list(df.columns),
            "format": file_ext,
            "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "chunk_info": chunk_info,
            "estimated_tokens": int(estimated_tokens)
        }
        
        # Add basic statistics for numeric columns (sample only to save tokens)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Only include stats for first few numeric columns to save tokens
            sample_numeric_cols = numeric_cols[:3]  # Reduced from 5 to 3
            file_info["numeric_summary"] = df[sample_numeric_cols].describe().round(2).to_dict()
        
        # Add missing value info (limit to most problematic columns)
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            missing_cols = missing_info[missing_info > 0].head(10)  # Limit to top 10
            file_info["missing_values"] = missing_cols.to_dict()
        
        result = {
            "data": data_dict, 
            "file_info": file_info
        }
        
        # Note: We don't include full_dataframe in the result because it's not JSON serializable
        # The cleaning agent will handle chunked processing internally
        if needs_chunking:
            result["chunk_info"] = chunk_info
        
        return result
        
    except Exception as e:
        return {"error": f"Error loading file: {str(e)}"}


@tool
def load_directory(directory_path: str, pattern: str = "*.*", recursive: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Load multiple data files from a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to load files from
    pattern : str
        Glob pattern to filter files (e.g. "*.csv", "data_*.xlsx")
    recursive : bool
        Whether to search subdirectories
    **kwargs :
        Additional keyword arguments to pass to the pandas read functions
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with loaded DataFrames and file information
    """
    directory_path = os.path.expanduser(directory_path)
    
    if not os.path.isdir(directory_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    search_pattern = os.path.join(directory_path, pattern)
    
    if recursive:
        search_pattern = os.path.join(directory_path, "**", pattern)
    
    files = glob.glob(search_pattern, recursive=recursive)
    
    results = {}
    
    for file_path in files:
        if os.path.isfile(file_path):
            # Get filename without full path as the key
            file_name = os.path.basename(file_path)
            result = load_file(file_path)
            
            if "error" not in result:
                results[file_name] = result
    
    if not results:
        return {"error": f"No matching files found in {directory_path} with pattern {pattern}"}
    
    return {
        "files": results,
        "total_files": len(results)
    }


@tool
def list_directory_contents(directory_path: str) -> Dict[str, Any]:
    """
    List all files and directories in the specified directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to list
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with directory contents information
    """
    directory_path = os.path.expanduser(directory_path)
    
    if not os.path.isdir(directory_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    try:
        contents = os.listdir(directory_path)
        files = []
        directories = []
        
        for item in contents:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                files.append(item)
            elif os.path.isdir(item_path):
                directories.append(item)
        
        return {
            "path": directory_path,
            "files": files,
            "directories": directories,
            "total_items": len(contents)
        }
        
    except Exception as e:
        return {"error": f"Error listing directory: {str(e)}"}


@tool
def list_directory_recursive(directory_path: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Recursively list all files and directories within the specified directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to list recursively
    max_depth : int
        Maximum depth to recurse (default: 3)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with recursive directory contents
    """
    directory_path = os.path.expanduser(directory_path)
    
    if not os.path.isdir(directory_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    def _list_dir_recursive(path, current_depth=0):
        if current_depth > max_depth:
            return {"name": os.path.basename(path), "type": "directory", "max_depth_reached": True}
        
        result = {"name": os.path.basename(path), "type": "directory", "children": []}
        
        try:
            contents = os.listdir(path)
            
            for item in contents:
                item_path = os.path.join(path, item)
                
                if os.path.isfile(item_path):
                    result["children"].append({
                        "name": item,
                        "type": "file",
                        "extension": os.path.splitext(item)[1]
                    })
                elif os.path.isdir(item_path):
                    result["children"].append(
                        _list_dir_recursive(item_path, current_depth + 1)
                    )
            
            return result
            
        except Exception as e:
            return {"name": os.path.basename(path), "type": "directory", "error": str(e)}
    
    return _list_dir_recursive(directory_path)


@tool
def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file to get information about
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with file information
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if not os.path.isfile(file_path):
        return {"error": f"Path is not a file: {file_path}"}
    
    try:
        file_stats = os.stat(file_path)
        
        file_info = {
            "path": file_path,
            "name": os.path.basename(file_path),
            "directory": os.path.dirname(file_path),
            "extension": os.path.splitext(file_path)[1],
            "size_bytes": file_stats.st_size,
            "size_kb": round(file_stats.st_size / 1024, 2),
            "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "created": file_stats.st_ctime,
            "modified": file_stats.st_mtime,
            "accessed": file_stats.st_atime
        }
        
        # For CSV, Excel, and other data files, add extra information
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.pq']:
            try:
                result = load_file(file_path)
                
                if "error" not in result:
                    df = pd.DataFrame(result["data"])
                    file_info["rows"] = len(df)
                    file_info["columns"] = list(df.columns)
                    file_info["data_preview"] = df.head(5).to_dict()
            except:
                # If we can't load it as a DataFrame, just continue
                pass
                
        return file_info
        
    except Exception as e:
        return {"error": f"Error getting file info: {str(e)}"}


@tool
def search_files_by_pattern(directory_path: str, pattern: str, recursive: bool = True) -> Dict[str, Any]:
    """
    Search for files matching a pattern in a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to search in
    pattern : str
        Glob pattern to match files (e.g. "*.csv", "data_*.xlsx")
    recursive : bool
        Whether to search subdirectories (default: True)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with matching files information
    """
    directory_path = os.path.expanduser(directory_path)
    
    if not os.path.isdir(directory_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    search_pattern = os.path.join(directory_path, pattern)
    
    if recursive:
        search_pattern = os.path.join(directory_path, "**", pattern)
    
    try:
        matching_files = glob.glob(search_pattern, recursive=recursive)
        
        results = []
        
        for file_path in matching_files:
            if os.path.isfile(file_path):
                file_stats = os.stat(file_path)
                
                results.append({
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "directory": os.path.dirname(file_path),
                    "extension": os.path.splitext(file_path)[1],
                    "size_bytes": file_stats.st_size,
                    "size_kb": round(file_stats.st_size / 1024, 2),
                    "modified": file_stats.st_mtime
                })
        
        return {
            "matches": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
        return {"error": f"Error searching files: {str(e)}"} 