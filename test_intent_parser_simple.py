#!/usr/bin/env python3
"""
Simple test script for DataAnalysisIntentParser and Parameter Mapping
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.parsers.intent_parser import DataAnalysisIntentParser
from src.mappers.parameter_mapper import AgentParameterMapper
from src.schemas.data_analysis_schemas import DataAnalysisRequest

# Load environment variables
load_dotenv()

def test_intent_parser():
    """Test the intent parser with a simple request"""
    
    print("🚀 Testing DataAnalysisIntentParser...")
    
    # Initialize the parser
    parser = DataAnalysisIntentParser(model_name="gpt-4o-mini")
    
    print("✅ Parser initialized successfully")
    
    # Test data
    csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    user_request = "Clean the dataset and do feature engineering. I want you to do regression on passengers."
    
    print(f"🔍 Testing with request: {user_request}")
    print(f"📊 Dataset: {csv_url}")
    
    try:
        # Test the intent parser
        intent = parser.parse_with_data_preview(user_request, csv_url)
        
        print("🎉 Intent parsing completed successfully!")
        print(f"🧹 Needs Data Cleaning: {intent.needs_data_cleaning}")
        print(f"🔧 Needs Feature Engineering: {intent.needs_feature_engineering}")
        print(f"🤖 Needs ML Modeling: {intent.needs_ml_modeling}")
        print(f"🎯 Suggested Target: {intent.suggested_target_variable}")
        print(f"📈 Intent Confidence: {intent.intent_confidence:.2f}")
        print(f"🎲 Complexity: {intent.complexity_level}")
        
        return intent
        
    except Exception as e:
        print(f"❌ Error during intent parsing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_parameter_mapping(intent):
    """Test parameter mapping"""
    
    print("\n🚀 Testing Parameter Mapping...")
    
    # Initialize the mapper
    mapper = AgentParameterMapper(base_output_dir="output/test_mapping/")
    
    print("✅ Mapper initialized successfully")
    
    # Create a sample request
    request = DataAnalysisRequest(
        csv_url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
        user_request="Clean the dataset and do feature engineering. I want you to do regression on passengers.",
        target_variable="passengers"
    )
    
    try:
        # Test data cleaning parameter mapping
        cleaning_params = mapper.map_data_cleaning_parameters(
            request, intent, request.csv_url
        )
        
        print(f"🧹 Data Cleaning Parameters: {len(cleaning_params)} parameters mapped")
        print(f"   📝 User Instructions Length: {len(cleaning_params['user_instructions'])} chars")
        print(f"   📁 File Name: {cleaning_params['file_name']}")
        
        # Test feature engineering parameter mapping
        fe_params = mapper.map_feature_engineering_parameters(
            request, intent, "cleaned_data.csv", "passengers"
        )
        
        print(f"🔧 Feature Engineering Parameters: {len(fe_params)} parameters mapped")
        print(f"   🎯 Target Variable: {fe_params['target_variable']}")
        print(f"   📝 User Instructions Length: {len(fe_params['user_instructions'])} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during parameter mapping: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    # Test intent parsing
    intent = test_intent_parser()
    if not intent:
        return False
    
    # Test parameter mapping
    success = test_parameter_mapping(intent)
    
    if success:
        print("\n🎉 All tests passed! The DataAnalysisAgent should work correctly.")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 