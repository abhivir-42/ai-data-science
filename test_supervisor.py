"""
Simple test script for the Supervisor Agent.
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

def test_supervisor_agent():
    """Test the supervisor agent with a simple request."""
    
    print("🤖 Testing Supervisor Agent")
    print("=" * 50)
    
    # Initialize the language model
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("✅ Language model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize language model: {e}")
        return
    
    # Test URL - using a small, reliable dataset
    test_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    
    print(f"\n📊 Testing with URL: {test_url}")
    print("📝 Request: 'Just clean this data and show me the quality'")
    
    try:
        # Test with a simple cleaning request
        result = process_csv_request(
            csv_url=test_url,
            user_request="Just clean this data and show me the quality",
            model=llm
        )
        
        print("\n✅ SUCCESS! Supervisor agent executed successfully")
        print("\n📋 RESULT PREVIEW:")
        print("=" * 50)
        
        # Show first 1000 characters of the result
        preview = result[:1000] + "..." if len(result) > 1000 else result
        print(preview)
        
        print("\n📏 Full result length:", len(result), "characters")
        
        # Save full result to file for inspection
        with open("test_supervisor_result.txt", "w") as f:
            f.write(result)
        print("💾 Full result saved to: test_supervisor_result.txt")
        
    except Exception as e:
        print(f"❌ FAILED! Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_supervisor_agent() 