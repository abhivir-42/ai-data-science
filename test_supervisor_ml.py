"""
Test the Supervisor Agent with ML pipeline.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.agents.supervisor_agent import SupervisorAgent, process_csv_request
from langchain_openai import ChatOpenAI

def test_ml_pipeline():
    """Test the supervisor agent with a full ML request."""
    
    print("🤖 Testing Supervisor Agent - ML Pipeline")
    print("=" * 60)
    
    # Initialize the language model
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("✅ Language model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize language model: {e}")
        return
    
    # Test URL - using Iris dataset which is perfect for classification
    test_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    print(f"\n📊 Testing with URL: {test_url}")
    print("📝 Request: 'Clean the data, create features, and build a classification model to predict species'")
    
    try:
        # Test with a full ML pipeline request
        result = process_csv_request(
            csv_url=test_url,
            user_request="Clean the data, create features, and build a classification model to predict species",
            target_variable="species",
            model=llm
        )
        
        print("\n✅ SUCCESS! Full ML pipeline executed successfully")
        print("\n📋 RESULT PREVIEW:")
        print("=" * 60)
        
        # Show first 2000 characters of the result
        preview = result[:2000] + "..." if len(result) > 2000 else result
        print(preview)
        
        print("\n📏 Full result length:", len(result), "characters")
        
        # Save full result to file for inspection
        with open("test_supervisor_ml_result.txt", "w") as f:
            f.write(result)
        print("💾 Full result saved to: test_supervisor_ml_result.txt")
        
    except Exception as e:
        print(f"❌ FAILED! Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_pipeline() 