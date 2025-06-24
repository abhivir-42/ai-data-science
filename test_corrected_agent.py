#!/usr/bin/env python3
"""
Test script for the corrected Data Analysis Agent implementation.

This tests:
1. LLM-powered CSV URL extraction from text
2. Intent parsing to determine which agents to run
3. Proper workflow execution based on intent flags
4. uAgent wrapper functionality
"""

import sys
import os
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

def test_url_extraction():
    """Test LLM-powered URL extraction from text."""
    print("🧪 Testing URL extraction from text...")
    
    from src.parsers.intent_parser import DataAnalysisIntentParser
    
    parser = DataAnalysisIntentParser()
    
    # Test cases
    test_cases = [
        "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction",
        "I want to build a model using the data at https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Please analyze my dataset for patterns and insights",  # No URL - should fail
        "Perform feature engineering on https://example.com/mydata.csv"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {text}")
        
        try:
            result = parser.extract_dataset_url_from_text(text)
            print(f"URL: {result.extracted_csv_url}")
            print(f"Method: {result.extraction_method}")
            print(f"Confidence: {result.extraction_confidence}")
            if result.extraction_notes:
                print(f"Notes: {result.extraction_notes}")
        except Exception as e:
            print(f"Error: {e}")

def test_intent_parsing():
    """Test intent parsing to determine which agents should run."""
    print("\n🧪 Testing intent parsing...")
    
    from src.parsers.intent_parser import DataAnalysisIntentParser
    
    parser = DataAnalysisIntentParser()
    
    # Test cases with different intents
    test_cases = [
        ("Clean the data https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", "Only cleaning"),
        ("Build a machine learning model using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species prediction", "Full ML pipeline"),
        ("Perform feature engineering on https://example.com/data.csv", "Only feature engineering"),
        ("Analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction with complete preprocessing", "Full workflow")
    ]
    
    for text, expected in test_cases:
        print(f"\n--- {expected} ---")
        print(f"Input: {text}")
        
        try:
            # First extract URL
            url_result = parser.extract_dataset_url_from_text(text)
            if url_result.extraction_method != "none_found":
                # Then parse intent
                intent = parser.parse_with_data_preview(text, url_result.extracted_csv_url)
                print(f"Needs cleaning: {intent.needs_data_cleaning}")
                print(f"Needs feature engineering: {intent.needs_feature_engineering}")
                print(f"Needs ML modeling: {intent.needs_ml_modeling}")
                print(f"Intent confidence: {intent.intent_confidence}")
            else:
                print("No URL found - cannot parse intent")
        except Exception as e:
            print(f"Error: {e}")

def test_data_analysis_agent():
    """Test the complete DataAnalysisAgent workflow."""
    print("\n🧪 Testing DataAnalysisAgent workflow...")
    
    from src.agents.data_analysis_agent import DataAnalysisAgent
    
    agent = DataAnalysisAgent(
        output_dir="test_output",
        enable_async=False
    )
    
    # Test cases
    test_cases = [
        "Clean the data at https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",  # Only cleaning
        "Build a classification model using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv to predict species"  # Full pipeline
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {text}")
        
        try:
            result = agent.analyze_from_text(text)
            print(f"✅ Analysis completed!")
            print(f"Agents executed: {result.agents_executed}")
            print(f"Confidence: {result.confidence_level}")
            print(f"Quality score: {result.analysis_quality_score}")
            print(f"Warnings: {len(result.warnings)}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"  - {warning}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_uagent_wrapper():
    """Test the uAgent wrapper function."""
    print("\n🧪 Testing uAgent wrapper...")
    
    # Import the wrapper function
    sys.path.append('.')
    from data_analysis_uagent import data_analysis_agent_func
    
    # Test cases
    test_cases = [
        "Clean https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        {"input": "Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for classification"},
        "This text has no URL and should fail"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {query}")
        
        try:
            result = data_analysis_agent_func(query)
            print("✅ uAgent wrapper executed successfully!")
            print(f"Result length: {len(result)} characters")
            print("First 200 characters:")
            print(result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Corrected Data Analysis Agent Implementation")
    print("=" * 60)
    
    # Run all tests
    test_url_extraction()
    test_intent_parsing()
    test_data_analysis_agent()
    test_uagent_wrapper()
    
    print("\n✅ All tests completed!") 