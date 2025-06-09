"""
Test script to debug house price dataset loading
"""

import os
import sys
import pandas as pd
from pprint import pprint

# Add src directory to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tools.data_loader import load_file

print("=== Testing House Price Dataset Loading ===\n")

# Test loading the house price dataset
file_path = "data/samples/house-price-prediction-train.csv"

print(f"Testing file: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")

if os.path.exists(file_path):
    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    # Load with our function
    print("\nLoading with load_file function...")
    result = load_file(file_path)
    
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Data type: {type(result.get('data'))}")
        if "chunk_info" in result:
            print(f"Chunk info: {result['chunk_info']}")
        if "file_info" in result:
            file_info = result["file_info"]
            print(f"File info keys: {list(file_info.keys())}")
            print(f"Rows: {file_info.get('rows')}")
            print(f"Columns: {len(file_info.get('columns', []))}")
            print(f"Estimated tokens: {file_info.get('estimated_tokens')}")
        
        # Test JSON serialization
        print("\nTesting JSON serialization...")
        import json
        try:
            json_str = json.dumps(result)
            print("✅ JSON serialization successful")
            print(f"JSON length: {len(json_str)} characters")
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            
            # Try to identify the problematic part
            for key, value in result.items():
                try:
                    json.dumps({key: value})
                    print(f"✅ {key}: OK")
                except Exception as ke:
                    print(f"❌ {key}: {ke}")
else:
    print("File not found!") 