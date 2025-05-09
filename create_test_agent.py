"""
Test script for verifying the functionality of DataLoaderToolsAgent
"""

import os
import sys
import pandas as pd
from langchain_openai import ChatOpenAI

# Add path to python path
sys.path.append(os.path.join(os.getcwd()))

# Import the DataLoaderToolsAgent
from src.agents import DataLoaderToolsAgent

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print("=== Testing DataLoaderToolsAgent ===\n")

# Check if we have a valid API key to run the tests
if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key":
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    # Create the agent
    data_loader_agent = DataLoaderToolsAgent(model=llm)

    # Test 1: Basic file loading
    print("Test 1: Basic File Loading")
    data_loader_agent.invoke_agent(
        user_instructions="Load the CSV file from data/samples/sample_data.csv"
    )

    # Get the loaded data
    loaded_data = data_loader_agent.get_artifacts(as_dataframe=True)
    print(f"Loaded data shape: {loaded_data.shape}")
    print(f"Columns: {', '.join(loaded_data.columns)}")
    print(f"First 3 rows:")
    print(loaded_data.head(3))

    # Get the tool calls
    tool_calls = data_loader_agent.get_tool_calls()
    print(f"Tool calls used: {tool_calls}")
    print()

    # Test 2: Directory exploration
    print("Test 2: Directory Exploration")
    data_loader_agent.invoke_agent(
        user_instructions="List all files in the data directory and find all CSV files"
    )

    # Get the AI message
    ai_message = data_loader_agent.get_ai_message()
    print(f"Agent response: {ai_message}")

    # Get the tool calls
    tool_calls = data_loader_agent.get_tool_calls()
    print(f"Tool calls used: {tool_calls}")
    print()

    # Test 3: Loading multiple files
    print("Test 3: Loading Multiple Files")
    data_loader_agent.invoke_agent(
        user_instructions="Load all CSV files from the data/samples directory"
    )

    # Get the loaded data
    loaded_data = data_loader_agent.get_artifacts()
    print(f"Loaded data: {loaded_data}")

    # Get the tool calls
    tool_calls = data_loader_agent.get_tool_calls()
    print(f"Tool calls used: {tool_calls}")
    print()

else:
    print("⚠️ No valid OpenAI API key found in environment variables.")
    print("Running mock tests to verify class structure only...\n")
    
    # Test that the agent class is properly structured
    print("Test: DataLoaderToolsAgent Class Structure")
    
    # Import the data tools directly to test
    from src.tools import load_file, search_files_by_pattern
    
    # Test the data tools directly
    print("Testing load_file tool:")
    csv_file = os.path.join(os.getcwd(), "data", "samples", "sample_data.csv")
    result = load_file(csv_file)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        df = pd.DataFrame(result["data"])
        print(f"CSV file loaded successfully with {len(df)} rows and {len(df.columns)} columns")
    
    print("\nTesting search_files_by_pattern tool:")
    result = search_files_by_pattern(os.path.join(os.getcwd(), "data"), "*.csv", recursive=True)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Found {result['total_matches']} CSV files")
    
    print("\nVerifying DataLoaderToolsAgent methods exist:")
    # Just verify that the methods exist by creating a blank agent
    agent = DataLoaderToolsAgent(model=None)
    
    methods = [
        "invoke_agent",
        "get_artifacts",
        "get_ai_message",
        "get_tool_calls"
    ]
    
    for method in methods:
        if hasattr(agent, method):
            print(f"✅ Method exists: {method}")
        else:
            print(f"❌ Missing method: {method}")

print("\n=== All agent tests completed ===") 