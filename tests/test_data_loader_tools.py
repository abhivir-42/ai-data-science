"""
Test script for verifying the functionality of data loader tools
"""

import os
import sys
import pandas as pd
from pprint import pprint

# Add src directory to path
sys.path.append(os.path.join(os.getcwd()))

# Import the data loader tools
from src.tools import (
    load_file,
    load_directory,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern
)

print("=== Testing Data Loader Tools ===\n")

# Define the data directory path
data_dir = os.path.join(os.getcwd(), "data", "samples")
csv_file = os.path.join(data_dir, "sample_data.csv")
json_file = os.path.join(data_dir, "sample_data.json")

# Test 1: Verify directory structure
print("Test 1: Directory Structure")
result = list_directory_contents(data_dir)
print(f"Directory contents: {result}")
print()

# Test 2: Load CSV file
print("Test 2: Load CSV File")
result = load_file(csv_file)
if "error" in result:
    print(f"Error: {result['error']}")
else:
    df = pd.DataFrame(result["data"])
    print(f"CSV file loaded successfully with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"First 3 rows:")
    print(df.head(3))
print()

# Test 3: Load JSON file
print("Test 3: Load JSON File")
result = load_file(json_file)
if "error" in result:
    print(f"Error: {result['error']}")
else:
    df = pd.DataFrame(result["data"])
    print(f"JSON file loaded successfully with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"First 3 rows:")
    print(df.head(3))
print()

# Test 4: Get file info
print("Test 4: Get File Info")
result = get_file_info(csv_file)
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"File info for {os.path.basename(csv_file)}:")
    for key, value in result.items():
        if key != "data_preview":  # Skip data preview for brevity
            print(f"  {key}: {value}")
print()

# Test 5: Search files by pattern
print("Test 5: Search Files By Pattern")
result = search_files_by_pattern(os.path.join(os.getcwd(), "data"), "*.csv", recursive=True)
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Found {result['total_matches']} CSV files:")
    for file_info in result["matches"]:
        print(f"  {file_info['path']} ({file_info['size_kb']} KB)")
print()

# Test 6: Load directory
print("Test 6: Load Directory")
result = load_directory(data_dir, pattern="*.csv")
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Loaded {result['total_files']} files from directory")
    for filename, file_data in result["files"].items():
        df = pd.DataFrame(file_data["data"])
        print(f"  {filename}: {len(df)} rows, {len(df.columns)} columns")
print()

print("=== All tests completed ===") 