"""
Example usage of the Supervisor Agent.

This script demonstrates how to use the Supervisor Agent to process
remote CSV files with different types of natural language requests.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.supervisor_agent import SupervisorAgent, process_csv_request
from langchain_openai import ChatOpenAI

def main():
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Sample CSV URLs for testing
    SAMPLE_URLS = {
        "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", 
        "tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    }
    
    print("=" * 60)
    print("SUPERVISOR AGENT EXAMPLES")
    print("=" * 60)
    
    # Example 1: Data cleaning only
    print("\n1. DATA CLEANING ONLY")
    print("-" * 30)
    
    result1 = process_csv_request(
        csv_url=SAMPLE_URLS["tips"],
        user_request="Clean this data and handle missing values",
        model=llm
    )
    print("Result saved. Preview:")
    print(result1[:500] + "..." if len(result1) > 500 else result1)
    
    # Example 2: Feature engineering
    print("\n\n2. FEATURE ENGINEERING") 
    print("-" * 30)
    
    result2 = process_csv_request(
        csv_url=SAMPLE_URLS["tips"],
        user_request="Clean the data and create features for machine learning",
        model=llm
    )
    print("Result saved. Preview:")
    print(result2[:500] + "..." if len(result2) > 500 else result2)
    
    # Example 3: Full ML pipeline
    print("\n\n3. FULL ML PIPELINE")
    print("-" * 30)
    
    result3 = process_csv_request(
        csv_url=SAMPLE_URLS["titanic"],
        user_request="Clean the data, engineer features, and build a classification model to predict Survived",
        target_variable="Survived",
        model=llm
    )
    print("Result saved. Preview:")
    print(result3[:500] + "..." if len(result3) > 500 else result3)
    
    # Example 4: Using the Supervisor Agent class directly
    print("\n\n4. USING SUPERVISOR AGENT CLASS")
    print("-" * 40)
    
    supervisor = SupervisorAgent(
        model=llm,
        output_dir="output/supervisor_examples/",
        log=True
    )
    
    result4 = supervisor.process_request(
        csv_url=SAMPLE_URLS["iris"],
        user_request="Just show me the quality of this data",
    )
    print("Result saved. Preview:")
    print(result4[:500] + "..." if len(result4) > 500 else result4)
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED!")
    print("Check the output/ directory for generated files.")
    print("=" * 60)

if __name__ == "__main__":
    main() 