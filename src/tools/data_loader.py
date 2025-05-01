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


def load_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Load a data file into a pandas DataFrame.
    
    Supports CSV, Excel, JSON, Parquet, and other common file formats.
    
    Parameters
    ----------
    file_path : str
        Path to the file to load
    **kwargs : 
        Additional keyword arguments to pass to the pandas read function
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with DataFrame and file information
    """
    file_path = os.path.expanduser(file_path)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
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
        else:
            return {"error": f"Unsupported file format: {file_ext}"}
        
        return {
            "data": df.to_dict(), 
            "file_info": {
                "path": file_path,
                "rows": len(df),
                "columns": list(df.columns),
                "format": file_ext
            }
        }
        
    except Exception as e:
        return {"error": f"Error loading file: {str(e)}"}


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
            result = load_file(file_path, **kwargs)
            
            if "error" not in result:
                results[file_name] = result
    
    if not results:
        return {"error": f"No matching files found in {directory_path} with pattern {pattern}"}
    
    return {
        "files": results,
        "total_files": len(results)
    }


def list_directory_contents(directory_path: str) -> Dict[str, Any]:
    """
    List the contents of a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to list
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with directory contents
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


def list_directory_recursive(directory_path: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Recursively list the contents of a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to list
    max_depth : int
        Maximum depth to recurse
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with directory contents
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


def search_files_by_pattern(directory_path: str, pattern: str, recursive: bool = True) -> Dict[str, Any]:
    """
    Search for files matching a pattern.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to search
    pattern : str
        Glob pattern to filter files (e.g. "*.csv", "data_*.xlsx")
    recursive : bool
        Whether to search subdirectories
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with matching files
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