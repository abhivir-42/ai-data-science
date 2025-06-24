#!/usr/bin/env python3
"""
Simple test script for DataAnalysisAgent
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.agents.data_analysis_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

def main():
    """Test the DataAnalysisAgent with a simple request"""
    
    print("🚀 Testing DataAnalysisAgent...")
    
    # Initialize the agent
    agent = DataAnalysisAgent(
        output_dir="output/test_analysis/",
        intent_parser_model="gpt-4o-mini",
        enable_async=False  # Use synchronous mode
    )
    
    print("✅ Agent initialized successfully")
    
    # Test data
    csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
    user_request = "Clean the dataset and do feature engineering. I want you to do regression on passengers."
    
    print(f"🔍 Testing with request: {user_request}")
    print(f"📊 Dataset: {csv_url}")
    
    try:
        # Call the agent
        result = agent.analyze(
            csv_url=csv_url,
            user_request=user_request,
            target_variable="passengers"
        )
        
        print("🎉 Analysis completed successfully!")
        print(f"📈 Analysis Quality: {result.analysis_quality_score:.2f}")
        print(f"🎯 Confidence: {result.confidence_level}")
        print(f"🤖 Agents Executed: {', '.join(result.agents_executed)}")
        print(f"⏱️ Runtime: {result.total_runtime_seconds:.2f} seconds")
        
        # Show insights
        print("\n💡 Key Insights:")
        for insight in result.key_insights:
            print(f"  • {insight}")
        
        print("\n🎯 Recommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")
        
        print(f"\n📖 Data Story:\n{result.data_story}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 