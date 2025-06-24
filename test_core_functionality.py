#!/usr/bin/env python3
"""
Clean test of core Data Analysis Agent functionality without uAgent registration.
"""

import sys
import os
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_core_functionality():
    """Test the core DataAnalysisAgent functionality."""
    print("🧪 Testing Core Data Analysis Agent Functionality")
    print("=" * 60)
    
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    # Initialize agent
    agent = DataAnalysisAgent(
        output_dir="test_output",
        enable_async=False
    )
    
    # Test cases that mirror the actual usage
    test_cases = [
        {
            "name": "Cleaning Only",
            "input": "Clean the dataset. Here is the flights dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
            "expected_agents": ["data_cleaning"]
        },
        {
            "name": "Full ML Pipeline", 
            "input": "Build a regression model to predict passengers using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
            "expected_agents": ["data_cleaning", "feature_engineering", "h2o_ml"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"Input: {test_case['input'][:80]}...")
        
        try:
            result = agent.analyze_from_text(test_case['input'])
            
            print(f"✅ Analysis completed successfully!")
            print(f"   Agents executed: {result.agents_executed}")
            print(f"   Expected: {test_case['expected_agents']}")
            print(f"   Runtime: {result.total_runtime_seconds:.2f} seconds")
            print(f"   Confidence: {result.confidence_level}")
            print(f"   Quality score: {result.analysis_quality_score:.2f}")
            
            # Check if correct agents were executed
            expected_set = set(test_case['expected_agents'])
            actual_set = set(result.agents_executed)
            
            if expected_set.issubset(actual_set):
                print(f"   ✅ Correct agents executed")
            else:
                print(f"   ⚠️ Agent mismatch - expected subset of {expected_set}, got {actual_set}")
            
            # Check for warnings
            if result.warnings:
                print(f"   Warnings: {len(result.warnings)}")
                for warning in result.warnings[:2]:  # Show first 2 warnings
                    print(f"     - {warning}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def test_wrapper_function():
    """Test the uAgent wrapper function without registration."""
    print("\n🧪 Testing uAgent Wrapper Function (No Registration)")
    print("=" * 60)
    
    # Import wrapper without triggering registration
    sys.path.append('.')
    
    # We'll test the core logic by importing the agent directly
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    # Create agent like the wrapper does
    agent = DataAnalysisAgent(
        output_dir="output/data_analysis_uagent/",
        intent_parser_model="gpt-4o-mini",
        enable_async=False
    )
    
    # Test the core function logic
    def test_agent_func(query):
        """Simplified version of the wrapper function."""
        if isinstance(query, dict) and 'input' in query:
            query = query['input']
        
        result = agent.analyze_from_text(query)
        
        # Simplified formatting
        return f"✅ Analysis complete! Agents: {result.agents_executed}, Quality: {result.analysis_quality_score:.2f}"
    
    # Test cases
    test_cases = [
        "Clean https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        {"input": "Build ML model https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"},
        "This has no URL and should fail gracefully"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n--- Wrapper Test {i} ---")
        print(f"Input: {query}")
        
        try:
            result = test_agent_func(query)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_core_functionality()
    test_wrapper_function()
    print("\n✅ All core functionality tests completed!") 