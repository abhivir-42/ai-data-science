#!/usr/bin/env python3
"""
Test script to verify performance optimization for the Data Analysis Agent.

This tests:
1. Adaptive runtime calculation based on dataset size
2. Faster execution for small datasets
3. Performance comparison before/after optimization
"""

import sys
import os
import time
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_adaptive_runtime():
    """Test the adaptive runtime calculation."""
    print("🧪 Testing Adaptive Runtime Calculation")
    print("=" * 50)
    
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    agent = DataAnalysisAgent(
        output_dir="test_output",
        enable_async=False
    )
    
    # Test different dataset sizes
    test_cases = [
        {"rows": 100, "columns": 5, "expected_range": (20, 40)},    # Small dataset - should be ~30s
        {"rows": 1000, "columns": 10, "expected_range": (50, 70)},  # Medium dataset - should be ~60s
        {"rows": 10000, "columns": 20, "expected_range": (110, 130)}, # Large dataset - should be ~120s
        {"rows": 100000, "columns": 50, "expected_range": (280, 320)} # Very large dataset - should be ~300s
    ]
    
    for i, case in enumerate(test_cases, 1):
        data_shape = {"rows": case["rows"], "columns": case["columns"]}
        runtime = agent._calculate_adaptive_runtime(data_shape)
        
        print(f"Test {i}: {case['rows']} rows × {case['columns']} cols")
        print(f"  Calculated runtime: {runtime}s")
        print(f"  Expected range: {case['expected_range'][0]}-{case['expected_range'][1]}s")
        
        if case["expected_range"][0] <= runtime <= case["expected_range"][1]:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
        print()

def test_small_dataset_performance():
    """Test performance on a small dataset (flights dataset)."""
    print("🚀 Testing Small Dataset Performance")
    print("=" * 50)
    
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    agent = DataAnalysisAgent(
        output_dir="test_output",
        enable_async=False
    )
    
    # Test message for flights dataset (144 rows)
    test_message = "Clean the dataset and do feature engineering if you think that's appropriate. I want you to do regression on passengers. Here is the flights dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    
    print(f"Input: {test_message}")
    print("Processing with optimized runtime...")
    
    start_time = time.time()
    
    try:
        result = agent.analyze_from_text(test_message)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print(f"✅ Analysis completed in {total_time:.2f} seconds")
        print(f"📊 Dataset: {result.data_shape}")
        print(f"⚡ Adaptive runtime used: {result.total_runtime_seconds:.2f}s")
        print(f"🎯 Workflow executed: Cleaning={result.workflow_intent.needs_data_cleaning}, FE={result.workflow_intent.needs_feature_engineering}, ML={result.workflow_intent.needs_ml_modeling}")
        
        # Performance expectations for small dataset
        if total_time < 60:  # Should complete in under 1 minute
            print("🎉 PERFORMANCE EXCELLENT: Completed in under 1 minute")
        elif total_time < 120:  # Should complete in under 2 minutes
            print("✅ PERFORMANCE GOOD: Completed in under 2 minutes")
        else:
            print("⚠️  PERFORMANCE SLOW: Took over 2 minutes")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_uagent_wrapper_performance():
    """Test the uAgent wrapper performance."""
    print("🔗 Testing uAgent Wrapper Performance")
    print("=" * 50)
    
    # Import the wrapper function
    from data_analysis_uagent import data_analysis_agent_func
    
    test_message = "Clean the dataset. Here is the flights dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    
    print(f"Input: {test_message}")
    print("Processing via uAgent wrapper...")
    
    start_time = time.time()
    
    try:
        result = data_analysis_agent_func(test_message)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print(f"✅ uAgent wrapper completed in {total_time:.2f} seconds")
        print(f"📝 Result length: {len(result)} characters")
        
        # Check for key performance indicators in result
        if "30 seconds" in result or "adaptive runtime" in result.lower():
            print("🎯 Adaptive runtime detected in result")
        
        if "AGENTS EXECUTED" in result:
            print("✅ Proper workflow execution detected")
        
        if total_time < 90:  # Should complete in under 1.5 minutes
            print("🎉 uAgent PERFORMANCE EXCELLENT")
        else:
            print("⚠️  uAgent performance could be improved")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in uAgent wrapper: {e}")
        return None

if __name__ == "__main__":
    print("🔧 PERFORMANCE OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print()
    
    # Test 1: Adaptive runtime calculation
    test_adaptive_runtime()
    print()
    
    # Test 2: Small dataset performance
    test_small_dataset_performance()
    print()
    
    # Test 3: uAgent wrapper performance
    test_uagent_wrapper_performance()
    print()
    
    print("=" * 60)
    print("✅ Performance optimization tests completed!") 