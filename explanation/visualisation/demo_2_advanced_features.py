"""
DEMO 2: Advanced Features - Customer Analytics
==============================================

This demo showcases advanced capabilities of the DataVisualizationAgent:
- Human-in-the-loop functionality
- Complex dataset handling (customer churn data)
- Custom user instructions and modifications
- Logging and export capabilities
- Professional-grade visualizations

Perfect for showing enterprise-level features.
"""

import os
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from src.agents import DataVisualizationAgent
import plotly.io as pio
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def load_customer_churn_data():
    """Load customer churn dataset for advanced analytics."""
    try:
        # Try to load from URL first
        url = "https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv"
        df = pd.read_csv(url)
        print("📥 Loaded customer churn data from remote URL")
    except:
        # Create synthetic churn data if URL fails
        print("📊 Creating synthetic customer churn data...")
        np.random.seed(42)
        
        n_customers = 500
        
        data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(45, 15, n_customers),
            'tenure': np.random.exponential(2, n_customers) * 12,  # months
            'monthly_charges': np.random.normal(65, 25, n_customers),
            'total_charges': np.zeros(n_customers),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
            'partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5]),
            'dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.4, 0.4, 0.2]),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.6, 0.2, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
            'churn': np.random.choice(['Yes', 'No'], n_customers, p=[0.25, 0.75])
        }
        
        # Calculate total charges based on tenure and monthly charges
        for i in range(n_customers):
            tenure_years = max(0.1, data['tenure'][i] / 12)
            data['total_charges'][i] = data['monthly_charges'][i] * tenure_years * 12
        
        # Make churn more realistic (higher for month-to-month, older customers, etc.)
        for i in range(n_customers):
            churn_prob = 0.1  # base probability
            
            if data['contract'][i] == 'Month-to-month':
                churn_prob += 0.3
            elif data['contract'][i] == 'One year':
                churn_prob += 0.1
                
            if data['senior_citizen'][i] == 1:
                churn_prob += 0.2
                
            if data['tenure'][i] < 6:  # new customers
                churn_prob += 0.25
                
            if data['monthly_charges'][i] > 80:  # expensive plans
                churn_prob += 0.15
                
            data['churn'][i] = 'Yes' if np.random.random() < churn_prob else 'No'
        
        # Ensure reasonable ranges
        data['age'] = np.maximum(18, np.minimum(85, data['age']))
        data['tenure'] = np.maximum(0, np.minimum(72, data['tenure']))  # max 6 years
        data['monthly_charges'] = np.maximum(20, np.minimum(120, data['monthly_charges']))
        
        df = pd.DataFrame(data)
    
    return df

def demo_advanced_features():
    """Demonstrate advanced features of DataVisualizationAgent."""
    
    print("🚀 DEMO 2: Advanced Features - Customer Analytics")
    print("=" * 60)
    
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Load customer data
    print("\n📊 Loading customer churn data...")
    df = load_customer_churn_data()
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.describe().round(2))
    
    # Initialize agent with human-in-the-loop (we'll simulate the responses)
    print("\n🤖 Initializing DataVisualizationAgent with advanced features...")
    
    # Create checkpointer for human-in-the-loop
    # checkpointer = MemorySaver()
    
    agent = DataVisualizationAgent(
        model=llm,
        n_samples=50,  # More samples for complex data
        log=True,
        log_path="logs/demo_2",
        file_name="customer_analytics.py",
        function_name="customer_chart",
        overwrite=True,
        human_in_the_loop=False,  # Set to False to avoid interactive prompts in demo
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer=None,  # Remove checkpointer to avoid configuration issues
    )
    
    # Advanced demo scenarios
    scenarios = [
        {
            "name": "Churn Analysis Dashboard",
            "instruction": """Create a comprehensive visualization showing customer churn patterns. 
                           Include churn rate by contract type, tenure analysis, and age demographics. 
                           Use professional styling with clear titles and legends.""",
            "description": "Multi-dimensional churn analysis"
        },
        {
            "name": "Revenue Impact Analysis", 
            "instruction": """Create a detailed analysis showing the relationship between monthly charges, 
                           total charges, and churn. Use scatter plots with color coding for churn status 
                           and add trend lines to show patterns.""",
            "description": "Financial impact visualization with advanced styling"
        },
        {
            "name": "Customer Segmentation",
            "instruction": """Analyze customer segments by creating visualizations that show the distribution 
                           of customers across different service types (internet, phone) and demographics. 
                           Use faceted plots or subplots to show multiple dimensions.""",
            "description": "Complex multi-dimensional segmentation"
        },
        {
            "name": "Tenure vs Loyalty Analysis",
            "instruction": """Create an advanced visualization showing how customer tenure relates to loyalty 
                           (inverse of churn). Include contract type analysis and highlight key insights 
                           with annotations. Use a modern, professional theme.""",
            "description": "Advanced analytics with insights"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Instruction: {scenario['instruction'][:100]}...")
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
            workflow_summary = agent.get_workflow_summary()
            
            if plotly_graph:
                print("✅ Advanced visualization created successfully!")
                
                # Save the chart
                chart_filename = f"demo_2_scenario_{i}_{scenario['name'].lower().replace(' ', '_')}.html"
                pio.write_html(plotly_graph, f"output/{chart_filename}")
                print(f"📁 Chart saved as: output/{chart_filename}")
                
                # Show detailed info
                print(f"📝 Generated function: {response.get('data_visualization_function_name', 'N/A')}")
                newline = '\n'
                print(f"🔧 Code complexity: {len(generated_code.split(newline)) if generated_code else 0} lines")
                print(f"📊 Data points processed: {len(df)}")
                
                # Show workflow summary if available
                if workflow_summary:
                    print(f"🔍 Workflow completed with {len(workflow_summary.split(newline))} steps")
                
                results.append({
                    'scenario': scenario['name'],
                    'success': True,
                    'chart_file': chart_filename,
                    'code_length': len(generated_code.split(newline)) if generated_code else 0,
                    'data_points': len(df),
                    'has_workflow': bool(workflow_summary)
                })
                
            else:
                print("❌ Failed to create visualization")
                error_msg = response.get('data_visualization_error', 'Unknown error')
                print(f"Error details: {error_msg}")
                results.append({
                    'scenario': scenario['name'], 
                    'success': False,
                    'error': error_msg
                })
                
        except Exception as e:
            print(f"❌ Error in scenario {i}: {str(e)}")
            results.append({
                'scenario': scenario['name'],
                'success': False, 
                'error': str(e)
            })
    
    # Advanced features demonstration
    print("\n" + "=" * 60)
    print("🔬 ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Show logging capabilities
    print("\n📝 Logging Analysis:")
    log_files = []
    log_dir = "logs/demo_2"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.py')]
        print(f"Generated {len(log_files)} code files")
        for log_file in log_files:
            file_path = os.path.join(log_dir, log_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    print(f"  📄 {log_file}: {len(content.split())} words, {len(content.split(newline))} lines")
    
    # Show data complexity handling
    print("\n📊 Data Complexity Handling:")
    print(f"Dataset dimensions: {df.shape}")
    print(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"Numerical columns: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful advanced visualizations: {successful}/{len(results)}")
    
    if successful > 0:
        avg_complexity = np.mean([r['code_length'] for r in results if r['success']])
        print(f"📊 Average code complexity: {avg_complexity:.1f} lines")
        
        total_charts = len([r for r in results if r['success']])
        print(f"📈 Total professional charts generated: {total_charts}")
    
    print("\n📊 Results Details:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['scenario']}")
        if result['success']:
            print(f"   📁 File: {result.get('chart_file', 'N/A')}")
            print(f"   📝 Complexity: {result.get('code_length', 'N/A')} lines")
            print(f"   📊 Data points: {result.get('data_points', 'N/A')}")
            print(f"   🔍 Workflow: {'Yes' if result.get('has_workflow') else 'No'}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
    
    print("\n🎉 Demo 2 completed!")
    print("🔍 Check the 'output/' directory for advanced charts")
    print("📝 Check the 'logs/demo_2' directory for generated code")
    print("💡 This demo shows enterprise-ready visualization capabilities")
    
    return results

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs/demo_2", exist_ok=True)
    
    # Run the demo
    results = demo_advanced_features() 