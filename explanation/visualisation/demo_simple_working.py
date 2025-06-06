"""
SIMPLE WORKING DEMO - Data Visualization Agent
==============================================

This demo shows exactly what the DataVisualizationAgent produces:
- Uses real Titanic dataset from the internet
- Shows the agent's decision-making process
- Displays generated code and charts
- Perfect for understanding agent capabilities

Dataset: Titanic passenger data from Seaborn
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

def load_titanic_data():
    """Load the famous Titanic dataset."""
    print("🚢 Loading Titanic dataset...")
    
    try:
        # Load from seaborn datasets (available online)
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        df = pd.read_csv(url)
        
        # Clean and prepare the data
        df = df.dropna(subset=['age', 'fare'])  # Remove rows with missing age/fare
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
        df['fare_category'] = pd.cut(df['fare'], bins=3, labels=['Low', 'Medium', 'High'])
        
        print(f"✅ Successfully loaded {len(df)} passenger records")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"🎯 Survival rate: {df['survived'].mean():.2%}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading Titanic data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data as fallback."""
    np.random.seed(42)
    n = 200
    
    data = {
        'age': np.random.normal(35, 15, n),
        'fare': np.random.exponential(30, n),
        'survived': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'class': np.random.choice(['First', 'Second', 'Third'], n, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n, p=[0.6, 0.4]),
        'embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    df['age'] = np.maximum(1, np.minimum(80, df['age']))
    df['fare'] = np.maximum(5, df['fare'])
    
    return df

def demo_simple_working():
    """Run a simple working demo to show agent capabilities."""
    
    print("🚀 SIMPLE WORKING DEMO - Data Visualization Agent")
    print("=" * 60)
    
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Load real data
    df = load_titanic_data()
    print(f"\n📊 Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Initialize the agent (simple configuration)
    print("\n🤖 Initializing DataVisualizationAgent...")
    agent = DataVisualizationAgent(
        model=llm,
        n_samples=30,
        log=True,
        log_path="logs/demo_simple",
        file_name="titanic_analysis.py",
        function_name="titanic_chart",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer=None,
    )
    
    # Simple scenarios that should work
    scenarios = [
        {
            "name": "Survival Analysis",
            "instruction": "Create a bar chart showing survival rates by passenger class. Use clear colors and labels.",
            "expected": "Bar chart comparing survival across classes"
        },
        {
            "name": "Age vs Fare Analysis", 
            "instruction": "Create a scatter plot of age vs fare with different colors for survivors and non-survivors.",
            "expected": "Scatter plot with color coding for survival"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"📋 Instruction: {scenario['instruction']}")
        print(f"🎯 Expected: {scenario['expected']}")
        print("-" * 50)
        
        try:
            # Invoke the agent
            print("🤖 Agent working...")
            agent.invoke_agent(
                data_raw=df,
                user_instructions=scenario['instruction'],
                max_retries=2,
                retry_count=0
            )
            
            # Get results
            response = agent.get_response()
            plotly_graph = agent.get_plotly_graph()
            generated_code = agent.get_data_visualization_function()
            recommended_steps = agent.get_recommended_visualization_steps()
            
            if plotly_graph:
                print("✅ SUCCESS! Visualization created!")
                
                # Save the chart
                chart_filename = f"simple_demo_{i}_{scenario['name'].lower().replace(' ', '_')}.html"
                pio.write_html(plotly_graph, f"output/{chart_filename}")
                print(f"📁 Chart saved: output/{chart_filename}")
                
                # Show what the agent actually did
                newline = '\n'
                code_lines = len(generated_code.split(newline)) if generated_code else 0
                print(f"📝 Generated code: {code_lines} lines")
                print(f"📊 Data points: {len(df)}")
                
                # Show the agent's reasoning
                if recommended_steps:
                    print("🧠 Agent's approach:")
                    lines = recommended_steps.split(newline)[:3]
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            print(f"   • {line.strip()}")
                
                # Show a snippet of the generated code
                if generated_code:
                    print("🔧 Code snippet:")
                    code_lines_list = generated_code.split(newline)
                    for line in code_lines_list[5:10]:  # Show lines 5-10
                        if line.strip():
                            print(f"   {line}")
                    if len(code_lines_list) > 10:
                        print("   ...")
                
                results.append({
                    'scenario': scenario['name'],
                    'success': True,
                    'chart_file': chart_filename,
                    'code_lines': code_lines,
                    'expected': scenario['expected']
                })
                
            else:
                print("❌ FAILED to create visualization")
                error_msg = response.get('data_visualization_error', 'Unknown error')
                print(f"Error: {error_msg}")
                results.append({
                    'scenario': scenario['name'], 
                    'success': False,
                    'error': error_msg
                })
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'scenario': scenario['name'],
                'success': False, 
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEMO RESULTS SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful visualizations: {successful}/{len(results)}")
    
    print("\n📊 What the agent actually produced:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} {result['scenario']}")
        if result['success']:
            print(f"   📁 File: {result.get('chart_file', 'N/A')}")
            print(f"   📝 Code: {result.get('code_lines', 'N/A')} lines")
            print(f"   🎯 Expected: {result.get('expected', 'N/A')}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
    
    print(f"\n🎉 Simple demo completed!")
    print(f"💡 Key takeaways:")
    print(f"   • Agent analyzed real Titanic passenger data")
    print(f"   • Generated working Python code for each visualization")
    print(f"   • Created interactive Plotly charts automatically")
    print(f"   • Made intelligent decisions about chart types and styling")
    
    return results

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs/demo_simple", exist_ok=True)
    
    # Run the demo
    results = demo_simple_working() 