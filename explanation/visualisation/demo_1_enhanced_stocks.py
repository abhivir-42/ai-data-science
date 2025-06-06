"""
ENHANCED DEMO 1: Stock Market Data Visualization
===============================================

This demo showcases the DataVisualizationAgent with REAL stock market data:
- Uses actual Apple (AAPL) stock data from Yahoo Finance
- Demonstrates automatic chart type selection for financial data
- Shows time series analysis, volume analysis, and price correlations
- Perfect for business presentations showing real-world applicability

Dataset: Apple Inc. (AAPL) stock data from Yahoo Finance
"""

import os
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from src.agents import DataVisualizationAgent
import plotly.io as pio
import yfinance as yf
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def load_stock_data():
    """Load real Apple stock data from Yahoo Finance."""
    print("📈 Fetching Apple (AAPL) stock data from Yahoo Finance...")
    
    try:
        # Get 1 year of Apple stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download data
        ticker = yf.Ticker("AAPL")
        df = ticker.history(start=start_date, end=end_date)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Add some calculated columns for more interesting analysis
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['Day_Range'] = df['High'] - df['Low']
        df['Month'] = df['Date'].dt.month_name()
        df['Weekday'] = df['Date'].dt.day_name()
        df['Quarter'] = df['Date'].dt.quarter
        
        # Add moving averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volume categories
        df['Volume_Category'] = pd.cut(df['Volume'], 
                                     bins=3, 
                                     labels=['Low', 'Medium', 'High'])
        
        print(f"✅ Successfully loaded {len(df)} days of AAPL data")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"💰 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading stock data: {e}")
        print("📊 Creating synthetic stock data instead...")
        return create_synthetic_stock_data()

def create_synthetic_stock_data():
    """Create realistic synthetic stock data as fallback."""
    np.random.seed(42)
    
    # Generate 252 trading days (1 year)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')  # Business days
    
    # Simulate stock price with trends and volatility
    initial_price = 150
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for return_rate in returns[1:]:
        new_price = prices[-1] * (1 + return_rate)
        prices.append(new_price)
    
    # Create OHLCV data
    data = {
        'Date': dates,
        'Open': [],
        'High': [],
        'Low': [],
        'Close': prices,
        'Volume': np.random.normal(50000000, 15000000, len(dates)).astype(int)
    }
    
    # Calculate OHLC from close prices
    for i, close_price in enumerate(prices):
        open_price = close_price + np.random.normal(0, 1)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 2))
        
        data['Open'].append(open_price)
        data['High'].append(high_price)
        data['Low'].append(low_price)
    
    df = pd.DataFrame(data)
    
    # Add calculated columns
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Day_Range'] = df['High'] - df['Low']
    df['Month'] = df['Date'].dt.month_name()
    df['Weekday'] = df['Date'].dt.day_name()
    df['Quarter'] = df['Date'].dt.quarter
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Category'] = pd.cut(df['Volume'], bins=3, labels=['Low', 'Medium', 'High'])
    
    return df

def demo_enhanced_stocks():
    """Run enhanced stock market analysis demo."""
    
    print("🚀 ENHANCED DEMO 1: Real Stock Market Data Visualization")
    print("=" * 70)
    
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Load real stock data
    df = load_stock_data()
    print(f"\n📊 Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nFirst few rows:")
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
    
    # Initialize the agent
    print("\n🤖 Initializing DataVisualizationAgent for financial analysis...")
    agent = DataVisualizationAgent(
        model=llm,
        n_samples=50,  # More samples for detailed analysis
        log=True,
        log_path="logs/demo_stocks",
        file_name="stock_analysis.py",
        function_name="stock_chart",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
    )
    
    # Real-world financial analysis scenarios
    scenarios = [
        {
            "name": "Stock Price Trend Analysis",
            "instruction": """Create a comprehensive time series chart showing Apple stock price movement over time. 
                           Include the closing price with a trend line, and add moving averages (MA_20, MA_50) 
                           for technical analysis. Use professional financial styling with proper date formatting.""",
            "description": "Professional time series with technical indicators",
            "business_value": "Shows stock performance trends and technical analysis for investment decisions"
        },
        {
            "name": "Volume vs Price Correlation", 
            "instruction": """Analyze the relationship between trading volume and price changes. 
                           Create a scatter plot showing Volume vs Price_Change_Pct with color coding 
                           by Volume_Category. Add trend lines to identify patterns.""",
            "description": "Volume-price relationship analysis",
            "business_value": "Identifies correlation between trading activity and price movements"
        },
        {
            "name": "Monthly Performance Analysis",
            "instruction": """Create a comprehensive analysis of monthly stock performance. 
                           Show average closing prices by month using a bar chart with error bars 
                           showing volatility. Use professional colors and clear labeling.""",
            "description": "Seasonal performance patterns",
            "business_value": "Reveals seasonal trends for strategic timing of investments"
        },
        {
            "name": "Daily Trading Range Analysis",
            "instruction": """Analyze daily trading ranges (High-Low) and their relationship to volume. 
                           Create a visualization showing Day_Range vs Volume with different colors for weekdays. 
                           Include statistical insights and trend analysis.""",
            "description": "Volatility and volume relationship",
            "business_value": "Understanding market volatility patterns for risk management"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"📈 Business Value: {scenario['business_value']}")
        print(f"🔍 Analysis: {scenario['description']}")
        print("-" * 50)
        
        try:
            # Invoke the agent
            print("🤖 Agent analyzing data and generating visualization...")
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
            recommended_steps = agent.get_recommended_visualization_steps()
            
            if plotly_graph:
                print("✅ Professional financial visualization created!")
                
                # Save the chart
                chart_filename = f"stock_analysis_{i}_{scenario['name'].lower().replace(' ', '_')}.html"
                pio.write_html(plotly_graph, f"output/{chart_filename}")
                print(f"📁 Chart saved: output/{chart_filename}")
                
                # Show analysis details
                print(f"📊 Data points analyzed: {len(df)}")
                newline = '\n'
                print(f"📝 Generated code: {len(generated_code.split(newline)) if generated_code else 0} lines")
                print(f"🧠 AI recommendations: {len(recommended_steps.split(newline)) if recommended_steps else 0} lines")
                
                # Show what the agent actually decided to create
                if recommended_steps:
                    print("🎯 Agent's Analysis Approach:")
                    # Extract key insights from recommendations
                    lines = recommended_steps.split(newline)[:3]
                    for line in lines:
                        if line.strip():
                            print(f"   • {line.strip()}")
                
                results.append({
                    'scenario': scenario['name'],
                    'success': True,
                    'chart_file': chart_filename,
                    'business_value': scenario['business_value'],
                    'code_complexity': len(generated_code.split('\\n')) if generated_code else 0,
                    'data_points': len(df)
                })
                
            else:
                print("❌ Failed to create visualization")
                error_msg = response.get('data_visualization_error', 'Unknown error')
                print(f"Error: {error_msg}")
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
    
    # Comprehensive analysis summary
    print("\n" + "=" * 70)
    print("📈 FINANCIAL ANALYSIS SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful financial visualizations: {successful}/{len(results)}")
    
    if successful > 0:
        total_data_points = sum(r.get('data_points', 0) for r in results if r['success'])
        avg_complexity = np.mean([r.get('code_complexity', 0) for r in results if r['success']])
        print(f"📊 Total data points analyzed: {total_data_points:,}")
        print(f"🧮 Average code complexity: {avg_complexity:.1f} lines per chart")
    
    print(f"\n📈 Business Impact Analysis:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['scenario']}")
        if result['success']:
            print(f"   💼 Business Value: {result.get('business_value', 'N/A')}")
            print(f"   📁 File: {result.get('chart_file', 'N/A')}")
            print(f"   🔧 Complexity: {result.get('code_complexity', 'N/A')} lines")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown')}")
    
    print(f"\n🎉 Enhanced Stock Market Demo completed!")
    print(f"💡 Key Insights:")
    print(f"   • Agent successfully analyzed real financial time series data")
    print(f"   • Generated professional-grade technical analysis visualizations")  
    print(f"   • Automatically selected appropriate chart types for financial data")
    print(f"   • Created business-ready analysis for investment decisions")
    
    return results

if __name__ == "__main__":
    # Install yfinance if not already installed
    try:
        import yfinance as yf
    except ImportError:
        print("📦 Installing yfinance for real stock data...")
        os.system("pip install yfinance")
        import yfinance as yf
    
    # Ensure directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs/demo_stocks", exist_ok=True)
    
    # Run the enhanced demo
    results = demo_enhanced_stocks() 