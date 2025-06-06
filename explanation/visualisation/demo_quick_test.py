"""
QUICK TEST - Data Visualization Agent
====================================

Quick test to verify the DataVisualizationAgent works correctly.
This is a minimal test before running the full demo suite.
"""

import os
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from src.agents import DataVisualizationAgent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def quick_test():
    """Quick test of DataVisualizationAgent functionality."""
    
    print("🧪 QUICK TEST - Data Visualization Agent")
    print("=" * 50)
    
    try:
        # Initialize the model
        print("🤖 Initializing ChatOpenAI...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        print("✅ Model initialized successfully")
        
        # Create simple test data
        print("\\n📊 Creating test data...")
        np.random.seed(42)
        test_data = {
            'x': range(10),
            'y': np.random.randint(10, 100, 10),
            'category': ['A', 'B'] * 5
        }
        df = pd.DataFrame(test_data)
        print(f"✅ Test data created: {df.shape}")
        print(df.head())
        
        # Initialize agent
        print("\\n🤖 Initializing DataVisualizationAgent...")
        agent = DataVisualizationAgent(
            model=llm,
            n_samples=10,
            log=False,  # Disable logging for quick test
            human_in_the_loop=False,
            bypass_recommended_steps=True,  # Skip recommendations for speed
            bypass_explain_code=True,  # Skip explanations for speed
        )
        print("✅ Agent initialized successfully")
        
        # Test basic visualization
        print("\\n📈 Testing basic visualization...")
        agent.invoke_agent(
            data_raw=df,
            user_instructions="Create a simple bar chart showing y values by x",
            max_retries=2,
            retry_count=0
        )
        
        # Check results
        response = agent.get_response()
        plotly_graph = agent.get_plotly_graph()
        
        if plotly_graph:
            print("✅ Visualization created successfully!")
            print(f"📊 Graph type: {type(plotly_graph).__name__}")
            
            # Try to save the chart
            import plotly.io as pio
            pio.write_html(plotly_graph, "output/quick_test_chart.html")
            print("✅ Chart saved as: output/quick_test_chart.html")
            
            return True
        else:
            print("❌ Failed to create visualization")
            print(f"Error: {response.get('data_visualization_error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in quick test: {str(e)}")
        return False

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Run quick test
    success = quick_test()
    
    if success:
        print("\\n🎉 QUICK TEST PASSED!")
        print("The DataVisualizationAgent is working correctly.")
        print("You can now run the full demo suite with: python run_all_demos.py")
    else:
        print("\\n❌ QUICK TEST FAILED!")
        print("Please check the error messages above and fix any issues.")
        print("Make sure your OpenAI API key is set in the .env file.") 