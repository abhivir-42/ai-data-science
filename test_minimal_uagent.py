#!/usr/bin/env python3
"""
Simple test script for the minimal Data Analysis uAgent wrapper
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

def test_simple_text():
    """Test simple text input like the LangGraph example"""
    print("🧪 Testing simple text input...")
    
    query = "Analyze the iris dataset for species classification"
    
    try:
        result = data_analysis_agent_func(query)
        print("✅ Simple text test passed")
        print("📊 Result preview:")
        print(result[:300] + "..." if len(result) > 300 else result)
        return True
    except Exception as e:
        print(f"❌ Simple text test failed: {e}")
        return False

def test_uagent_format():
    """Test uAgent standard input format"""
    print("\n🧪 Testing uAgent input format...")
    
    query = {
        "input": "Clean and analyze the titanic dataset for survival prediction"
    }
    
    try:
        result = data_analysis_agent_func(query)
        print("✅ uAgent format test passed")
        print("📊 Result preview:")
        print(result[:300] + "..." if len(result) > 300 else result)
        return True
    except Exception as e:
        print(f"❌ uAgent format test failed: {e}")
        return False

def test_with_url():
    """Test with explicit URL"""
    print("\n🧪 Testing with explicit URL...")
    
    query = "Perform classification analysis on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    try:
        result = data_analysis_agent_func(query)
        print("✅ URL test passed")
        print("📊 Result preview:")
        print(result[:300] + "..." if len(result) > 300 else result)
        return True
    except Exception as e:
        print(f"❌ URL test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🧪 Testing error handling...")
    
    query = "analyze some unknown dataset"
    
    try:
        result = data_analysis_agent_func(query)
        if "Could not detect" in result or "Analysis Error" in result:
            print("✅ Error handling test passed")
            print("📊 Error message:")
            print(result[:300] + "..." if len(result) > 300 else result)
            return True
        else:
            print("❌ Error handling test failed - no proper error message")
            return False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Minimal Data Analysis uAgent Wrapper")
    print("=" * 50)
    
    tests = [
        test_simple_text,
        test_uagent_format,
        test_with_url,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Minimal uAgent wrapper is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 