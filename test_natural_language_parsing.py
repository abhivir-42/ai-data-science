#!/usr/bin/env python3
"""
Test natural language parsing capabilities of the supervisor agent
without explicitly setting target variables.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.supervisor_agent import process_csv_request

def test_natural_language_parsing():
    """Test various natural language formats for target variable extraction"""
    
    test_cases = [
        {
            "name": "Iris Classification - 'predict species'",
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "request": "build a classification model to predict species"
        },
        {
            "name": "Tips Regression - 'predict tip'", 
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
            "request": "create a regression model to predict tip amount"
        },
        {
            "name": "Titanic Classification - 'predicting survived'",
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv", 
            "request": "clean data and build model predicting survived"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"{'='*60}")
        print(f"URL: {test_case['url']}")
        print(f"Request: {test_case['request']}")
        print(f"{'='*60}")
        
        try:
            # Test WITHOUT explicitly setting target variable
            result = process_csv_request(
                csv_url=test_case['url'],
                user_request=test_case['request']
                # Notice: NO target parameter - should be parsed from natural language
            )
            
            print(f"✅ SUCCESS - Test {i} completed")
            print(f"Result length: {len(result)} characters")
            
            # Print first 500 characters to verify it worked
            print(f"\nFirst 500 characters of result:")
            print("-" * 50)
            print(result[:500])
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ FAILED - Test {i} failed with error:")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing Natural Language Parsing for Target Variable Extraction")
    print("=" * 70)
    test_natural_language_parsing()
    print("\n" + "=" * 70)
    print("Natural language parsing tests completed!") 