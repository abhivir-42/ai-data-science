"""
Data Loader Tools Agent Showcase

This example demonstrates all the capabilities of the DataLoaderToolsAgent,
including loading different file types, searching for files, and exploring
directory structures.

Features demonstrated:
1. Loading individual files (CSV, JSON, Excel)
2. Loading entire directories
3. Searching for files by pattern
4. Getting file information
5. Recursive directory exploration
6. Error handling and edge cases

Usage:
    python examples/data_loader_showcase.py
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.data_loader_tools_agent import DataLoaderToolsAgent

# Load environment variables
load_dotenv()

def create_test_files():
    """Create some test files for demonstration."""
    os.makedirs("test_data", exist_ok=True)
    
    # Create a JSON file
    json_data = {
        "users": [
            {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
            {"id": 2, "name": "Bob", "age": 25, "city": "San Francisco"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago"}
        ]
    }
    
    with open("test_data/users.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    # Create a simple CSV
    sales_data = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "product": ["Widget A", "Widget B", "Widget A"],
        "sales": [100, 150, 200],
        "region": ["North", "South", "East"]
    })
    
    sales_data.to_csv("test_data/sales.csv", index=False)
    
    print("✅ Test files created in 'test_data/' directory")

def demonstrate_basic_loading(agent):
    """Demonstrate basic file loading capabilities."""
    print("\n" + "="*60)
    print("📂 BASIC FILE LOADING DEMONSTRATION")
    print("="*60)
    
    # Test 1: Load CSV file
    print("\n📋 Test 1: Loading CSV file")
    try:
        agent.invoke_agent(
            user_instructions="Load the CSV file from examples/sample_data.csv and show me its structure"
        )
        
        df = agent.get_artifacts(as_dataframe=True)
        if df is not None:
            print(f"✅ CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"   Columns: {list(df.columns)}")
            print(f"   First few rows:\n{df.head(3)}")
        
        print(f"\n🤖 AI Response: {agent.get_ai_message()}")
        print(f"🔧 Tools used: {agent.get_tool_calls()}")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
    
    # Test 2: Load JSON file
    print("\n📋 Test 2: Loading JSON file")
    try:
        agent.invoke_agent(
            user_instructions="Load the JSON file from test_data/users.json if it exists"
        )
        
        artifacts = agent.get_artifacts()
        if artifacts:
            print(f"✅ JSON loaded successfully")
            print(f"   Data preview: {str(artifacts)[:200]}...")
        
        print(f"\n🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error loading JSON: {str(e)}")

def demonstrate_directory_operations(agent):
    """Demonstrate directory exploration and bulk loading."""
    print("\n" + "="*60)
    print("📁 DIRECTORY OPERATIONS DEMONSTRATION")
    print("="*60)
    
    # Test 1: List directory contents
    print("\n📋 Test 1: Listing directory contents")
    try:
        agent.invoke_agent(
            user_instructions="Show me all files in the examples/ directory"
        )
        
        print(f"✅ Directory listing completed")
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error listing directory: {str(e)}")
    
    # Test 2: Recursive directory exploration
    print("\n📋 Test 2: Recursive directory exploration")
    try:
        agent.invoke_agent(
            user_instructions="Recursively explore the current directory and find all CSV files"
        )
        
        print(f"✅ Recursive exploration completed")
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error in recursive exploration: {str(e)}")
    
    # Test 3: Load entire directory
    print("\n📋 Test 3: Loading all files from a directory")
    try:
        agent.invoke_agent(
            user_instructions="Load all data files from the test_data/ directory if it exists"
        )
        
        artifacts = agent.get_artifacts()
        if artifacts:
            print(f"✅ Directory loading completed")
            print(f"   Files loaded: {len(artifacts) if isinstance(artifacts, dict) else 'Multiple'}")
        
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error loading directory: {str(e)}")

def demonstrate_file_search(agent):
    """Demonstrate file search capabilities."""
    print("\n" + "="*60)
    print("🔍 FILE SEARCH DEMONSTRATION")
    print("="*60)
    
    # Test 1: Search by pattern
    print("\n📋 Test 1: Search for files by pattern")
    try:
        agent.invoke_agent(
            user_instructions="Find all files with '.csv' extension in the current directory and subdirectories"
        )
        
        print(f"✅ File search completed")
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error in file search: {str(e)}")
    
    # Test 2: Search for specific file names
    print("\n📋 Test 2: Search for specific file names")
    try:
        agent.invoke_agent(
            user_instructions="Look for any files that contain 'sample' in their name"
        )
        
        print(f"✅ Name-based search completed")
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error in name search: {str(e)}")

def demonstrate_file_info(agent):
    """Demonstrate getting detailed file information."""
    print("\n" + "="*60)
    print("ℹ️  FILE INFORMATION DEMONSTRATION")
    print("="*60)
    
    # Test 1: Get info about specific file
    print("\n📋 Test 1: Getting detailed file information")
    try:
        agent.invoke_agent(
            user_instructions="Get detailed information about the examples/sample_data.csv file including size, modification date, and format details"
        )
        
        print(f"✅ File info retrieved")
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error getting file info: {str(e)}")

def demonstrate_advanced_scenarios(agent):
    """Demonstrate advanced usage scenarios."""
    print("\n" + "="*60)
    print("🚀 ADVANCED SCENARIOS DEMONSTRATION")
    print("="*60)
    
    # Test 1: Complex multi-step operation
    print("\n📋 Test 1: Complex multi-step data loading")
    try:
        agent.invoke_agent(
            user_instructions="""
            Please do the following:
            1. Find all CSV files in the examples directory
            2. Load the largest CSV file you find
            3. Provide a summary of its structure including column types and missing values
            4. Show me the first 5 rows
            """
        )
        
        df = agent.get_artifacts(as_dataframe=True)
        if df is not None:
            print(f"✅ Complex operation completed")
            print(f"   Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error in complex operation: {str(e)}")
    
    # Test 2: Error handling
    print("\n📋 Test 2: Error handling with non-existent file")
    try:
        agent.invoke_agent(
            user_instructions="Try to load a file called 'non_existent_file.csv'"
        )
        
        print(f"✅ Error handling test completed")
        print(f"🤖 AI Response: {agent.get_ai_message()}")
        
    except Exception as e:
        print(f"❌ Error in error handling test: {str(e)}")

def cleanup_test_files():
    """Clean up test files created during demonstration."""
    import shutil
    
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
        print("🗑️  Test files cleaned up")

def main():
    """Run the complete data loader showcase."""
    print("🤖 DataLoaderToolsAgent Showcase")
    print("=" * 60)
    print("This demonstration shows all the capabilities of the DataLoaderToolsAgent")
    
    # Initialize the agent
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        agent = DataLoaderToolsAgent(model=llm)
        
        print("✅ DataLoaderToolsAgent initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        return
    
    # Create test files
    create_test_files()
    
    # Run demonstrations
    try:
        demonstrate_basic_loading(agent)
        demonstrate_directory_operations(agent)
        demonstrate_file_search(agent)
        demonstrate_file_info(agent)
        demonstrate_advanced_scenarios(agent)
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
    
    finally:
        # Clean up
        cleanup_test_files()
    
    print("\n🎉 DataLoaderToolsAgent showcase completed!")
    print("\nKey capabilities demonstrated:")
    print("• Loading individual files (CSV, JSON, Excel)")
    print("• Exploring directory structures")
    print("• Searching for files by pattern")
    print("• Getting detailed file information")
    print("• Handling complex multi-step operations")
    print("• Robust error handling")
    print("\nThe DataLoaderToolsAgent can intelligently:")
    print("• Understand natural language instructions")
    print("• Choose appropriate tools for each task")
    print("• Provide detailed responses about data structure")
    print("• Handle various file formats automatically")

if __name__ == "__main__":
    main() 