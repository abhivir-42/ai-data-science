"""
DEMO 1: Basic Usage - Sales Data Visualization
==============================================

This demo showcases the basic capabilities of the DataVisualizationAgent:
- Automatic chart type selection
- Interactive Plotly visualizations
- Code generation and logging
- Multiple chart types from the same dataset

Perfect for presentations showing core functionality.
"""

import os
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from src.agents import DataVisualizationAgent
import plotly.io as pio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def create_sample_sales_data():
    """Create a realistic sales dataset for demonstration."""
    np.random.seed(42)
    
    # Generate sample data
    n_records = 200
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    data = {
        'Month': np.random.choice(months, n_records),
        'Sales': np.random.normal(50000, 15000, n_records),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], n_records),
        'Marketing_Spend': np.random.normal(5000, 1500, n_records),
        'Customer_Count': np.random.poisson(100, n_records),
        'Quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], n_records)
    }
    
    # Add some correlation
    for i in range(n_records):
        # Marketing spend affects sales
        data['Sales'][i] += data['Marketing_Spend'][i] * 0.8
        # Customer count affects sales  
        data['Sales'][i] += data['Customer_Count'][i] * 200
    
    # Ensure positive values
    data['Sales'] = np.maximum(data['Sales'], 10000)
    data['Marketing_Spend'] = np.maximum(data['Marketing_Spend'], 1000)
    
    return pd.DataFrame(data)

def demo_basic_usage():
    """Demonstrate basic usage of DataVisualizationAgent."""
    
    print("🚀 DEMO 1: Basic Usage - Sales Data Visualization")
    print("=" * 60)
    
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Create sample data
    print("\n📊 Creating sample sales data...")
    df = create_sample_sales_data()
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Initialize the agent with logging enabled
    print("\n🤖 Initializing DataVisualizationAgent...")
    agent = DataVisualizationAgent(
        model=llm,
        n_samples=30,
        log=True,
        log_path="logs/demo_1",
        file_name="sales_visualization.py",
        function_name="sales_chart",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
    )
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Automatic Chart Selection",
            "instruction": "Create a chart showing sales performance across different regions",
            "description": "Let the agent automatically choose the best chart type"
        },
        {
            "name": "Scatter Plot with Correlation", 
            "instruction": "Create a scatter plot of Marketing Spend vs Sales with a trend line to show correlation",
            "description": "Specific chart type with relationship analysis"
        },
        {
            "name": "Time Series Analysis",
            "instruction": "Show sales trends by month using a line chart with proper formatting",
            "description": "Time-based visualization"
        },
        {
            "name": "Categorical Comparison",
            "instruction": "Compare average sales by product using a bar chart with good colors",
            "description": "Category-based comparison"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Instruction: {scenario['instruction']}")
        print("-" * 40)
        
        try:
            # Invoke the agent
            agent.invoke_agent(
                data_raw=df,
                user_instructions=scenario['instruction'],
                max_retries=3,
                retry_count=0
            )
            
            # Get results
            response = agent.get_response()
            plotly_graph = agent.get_plotly_graph()
            generated_code = agent.get_data_visualization_function()
            
            if plotly_graph:
                print("✅ Visualization created successfully!")
                
                # Save the chart
                chart_filename = f"demo_1_scenario_{i}_{scenario['name'].lower().replace(' ', '_')}.html"
                pio.write_html(plotly_graph, f"output/{chart_filename}")
                print(f"📁 Chart saved as: output/{chart_filename}")
                
                # Show some info about the generated code
                print(f"📝 Generated function name: {response.get('data_visualization_function_name', 'N/A')}")
                newline = '\n'
                print(f"🔧 Code lines: {len(generated_code.split(newline)) if generated_code else 0}")
                
                results.append({
                    'scenario': scenario['name'],
                    'success': True,
                    'chart_file': chart_filename,
                    'code_length': len(generated_code.split(newline)) if generated_code else 0
                })
                
            else:
                print("❌ Failed to create visualization")
                results.append({
                    'scenario': scenario['name'], 
                    'success': False,
                    'error': response.get('data_visualization_error', 'Unknown error')
                })
                
        except Exception as e:
            print(f"❌ Error in scenario {i}: {str(e)}")
            results.append({
                'scenario': scenario['name'],
                'success': False, 
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful visualizations: {successful}/{len(results)}")
    
    print("\n📊 Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['scenario']}")
        if result['success']:
            print(f"   📁 File: {result.get('chart_file', 'N/A')}")
            print(f"   📝 Code lines: {result.get('code_length', 'N/A')}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
    
    print("\n🎉 Demo 1 completed!")
    print("🔍 Check the 'output/' directory for generated charts")
    print("📝 Check the 'logs/demo_1' directory for generated code")
    
    return results

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs/demo_1", exist_ok=True)
    
    # Run the demo
    results = demo_basic_usage() 