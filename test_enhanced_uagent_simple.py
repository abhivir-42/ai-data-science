#!/usr/bin/env python3
"""
Simple test script for Enhanced Data Analysis uAgent Wrapper
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

# Import the uAgent function directly
from data_analysis_uagent import data_analysis_agent_func

def test_simple_text_input():
    """Test simple text input"""
    print("🧪 Testing simple text input...")
    
    query = "Analyze the iris dataset for species classification"
    
    try:
        result = data_analysis_agent_func(query)
        print("✅ Simple text input test passed")
        print("📊 Result preview:")
        print(result[:500] + "..." if len(result) > 500 else result)
        return True
    except Exception as e:
        print(f"❌ Simple text input test failed: {e}")
        return False

def test_structured_input():
    """Test structured dict input"""
    print("\n🧪 Testing structured dict input...")
    
    query = {
        "csv_url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "user_request": "Build a classification model to predict species",
        "target_variable": "species",
        "problem_type": "classification",
        "max_runtime_minutes": 10
    }
    
    try:
        result = data_analysis_agent_func(query)
        print("✅ Structured dict input test passed")
        print("📊 Result preview:")
        print(result[:500] + "..." if len(result) > 500 else result)
        return True
    except Exception as e:
        print(f"❌ Structured dict input test failed: {e}")
        return False

def test_uagent_input_format():
    """Test uAgent standard input format"""
    print("\n🧪 Testing uAgent standard input format...")
    
    query = {
        "input": "Clean and analyze the titanic dataset for survival prediction"
    }
    
    try:
        result = data_analysis_agent_func(query)
        print("✅ uAgent input format test passed")
        print("📊 Result preview:")
        print(result[:500] + "..." if len(result) > 500 else result)
        return True
    except Exception as e:
        print(f"❌ uAgent input format test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🧪 Testing error handling...")
    
    # Test missing request
    query = ""
    
    try:
        result = data_analysis_agent_func(query)
        if "Missing Request" in result:
            print("✅ Error handling test passed")
            return True
        else:
            print("❌ Error handling test failed - no proper error message")
            return False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Data Analysis uAgent Wrapper")
    print("=" * 50)
    
    tests = [
        test_simple_text_input,
        test_structured_input, 
        test_uagent_input_format,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced uAgent wrapper is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 