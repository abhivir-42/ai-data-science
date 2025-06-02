"""
Debug script for the data cleaning agent to identify and fix the recursion issue.

This script will:
1. Test the data cleaning agent with very simple data and instructions
2. Add debugging output to see exactly what's happening
3. Use shorter retry limits to prevent infinite loops
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


from src.agents.data_cleaning_agent import DataCleaningAgent

# Load environment variables
load_dotenv()

def test_simple_data():
    """Test with the simplest possible dataset."""
    print("🧪 Testing with simple synthetic data...")
    
    # Create extremely simple test data
    simple_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10.0, None, 30.0, 40.0, 50.0],
        'category': ['A', 'B', None, 'A', 'B']
    })
    
    print(f"📊 Test data shape: {simple_df.shape}")
    print(f"📋 Columns: {list(simple_df.columns)}")
    print(f"❌ Missing values: {simple_df.isnull().sum().sum()}")
    print("\nData preview:")
    print(simple_df)
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create cleaning agent with minimal settings
    cleaning_agent = DataCleaningAgent(
        model=llm,
        n_samples=5,  # Very small sample size
        log=True,
        log_path="./debug_logs",
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False
    )
    
    # Simple cleaning instructions
    instructions = "Fill missing numeric values with 0. Fill missing text values with 'Unknown'. That's all."
    
    print(f"\n🧹 Instructions: {instructions}")
    print("🚀 Starting cleaning process...")
    
    try:
        # Run with short retry limit 
        cleaning_agent.invoke_agent(
            data_raw=simple_df,
            user_instructions=instructions,
            max_retries=2,  # Short retry limit
            retry_count=0
        )
        
        # Get the response to debug
        response = cleaning_agent.get_response()
        print(f"\n🔍 Agent response keys: {list(response.keys()) if response else 'None'}")
        
        if response:
            print(f"📋 Has data_cleaned: {'data_cleaned' in response}")
            print(f"📋 Has data_cleaner_error: {'data_cleaner_error' in response}")
            print(f"📋 Retry count: {response.get('retry_count', 'Not set')}")
            print(f"📋 Max retries: {response.get('max_retries', 'Not set')}")
            
            if 'data_cleaner_error' in response:
                error = response['data_cleaner_error']
                print(f"🚨 Last error: {error}")
            
            if 'data_cleaned' in response:
                data_cleaned_raw = response.get('data_cleaned')
                print(f"📋 Raw cleaned data type: {type(data_cleaned_raw)}")
                print(f"📋 Raw cleaned data keys: {list(data_cleaned_raw.keys()) if isinstance(data_cleaned_raw, dict) else 'Not a dict'}")
        
        cleaned_df = cleaning_agent.get_data_cleaned()
        
        if cleaned_df is not None:
            print(f"\n✅ Success! Cleaned data shape: {cleaned_df.shape}")
            print(f"❌ Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
            print("\nCleaned data preview:")
            print(cleaned_df.head())
            
            # Get the generated function
            cleaning_function = cleaning_agent.get_data_cleaner_function()
            if cleaning_function:
                print(f"\n📝 Generated cleaning function length: {len(cleaning_function)} characters")
                print("Function preview (first 200 chars):")
                print(cleaning_function[:200] + "..." if len(cleaning_function) > 200 else cleaning_function)
            
            return True
        else:
            print("❌ Cleaning returned None")
            
            # Try to manually test the cleaning function
            cleaning_function = cleaning_agent.get_data_cleaner_function()
            if cleaning_function:
                print("\n🧪 Manually testing the cleaning function...")
                try:
                    # Create a local namespace and execute the function
                    namespace = {}
                    exec(cleaning_function, namespace)
                    if 'data_cleaner' in namespace:
                        manual_result = namespace['data_cleaner'](simple_df)
                        print(f"✅ Manual test result shape: {manual_result.shape}")
                        print(f"📊 Manual test missing values: {manual_result.isnull().sum().sum()}")
                        print("Manual result preview:")
                        print(manual_result.head())
                except Exception as e:
                    print(f"❌ Manual test failed: {str(e)}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error during cleaning: {str(e)}")
        
        # Print error details
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
        
        # Try to get partial results
        try:
            response = cleaning_agent.get_response()
            if response:
                print("\n📋 Agent response keys:", list(response.keys()))
                if 'data_cleaner_error' in response:
                    print(f"🔍 Last error: {response['data_cleaner_error']}")
                if 'retry_count' in response:
                    print(f"🔄 Retry count: {response['retry_count']}")
        except:
            print("Could not retrieve agent response")
        
        return False

def test_sample_data():
    """Test with the provided sample data."""
    sample_path = "examples/sample_data.csv"
    
    if not os.path.exists(sample_path):
        print(f"⚠️  Sample data not found at {sample_path}")
        return False
        
    print(f"\n🧪 Testing with sample data from {sample_path}...")
    
    # Load sample data
    sample_df = pd.read_csv(sample_path)
    print(f"📊 Sample data shape: {sample_df.shape}")
    print(f"📋 Columns: {list(sample_df.columns)}")
    print(f"❌ Missing values: {sample_df.isnull().sum().sum()}")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create cleaning agent
    cleaning_agent = DataCleaningAgent(
        model=llm,
        n_samples=10,  # Small sample size
        log=True,
        log_path="./debug_logs",
        human_in_the_loop=False
    )
    
    # Very simple instructions
    instructions = "Fill missing values only. Don't do anything complex."
    
    print(f"\n🧹 Instructions: {instructions}")
    print("🚀 Starting cleaning process...")
    
    try:
        cleaning_agent.invoke_agent(
            data_raw=sample_df,
            user_instructions=instructions,
            max_retries=2,
            retry_count=0
        )
        
        cleaned_df = cleaning_agent.get_data_cleaned()
        
        if cleaned_df is not None:
            print(f"\n✅ Success! Cleaned data shape: {cleaned_df.shape}")
            print(f"❌ Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
            return True
        else:
            print("❌ Cleaning returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error during cleaning: {str(e)}")
        return False

def main():
    """Run debugging tests."""
    print("🐛 Data Cleaning Agent Debug Script")
    print("=" * 50)
    
    # Test 1: Simple synthetic data
    print("\n" + "="*50)
    print("TEST 1: Simple Synthetic Data")
    print("="*50)
    
    success1 = test_simple_data()
    
    # Test 2: Sample data (only if first test passes)
    if success1:
        print("\n" + "="*50)
        print("TEST 2: Sample Data")
        print("="*50)
        
        success2 = test_sample_data()
        
        if success2:
            print("\n🎉 Both tests passed! The agent is working correctly.")
        else:
            print("\n⚠️  Simple test passed but sample data test failed.")
    else:
        print("\n❌ Simple test failed. Need to investigate further.")
    
    print("\n" + "="*50)
    print("Debug session complete.")
    print("Check ./debug_logs/ for detailed logs.")
    print("="*50)

if __name__ == "__main__":
    main() 