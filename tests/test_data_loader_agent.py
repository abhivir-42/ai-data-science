"""
Test script for verifying the functionality of DataLoaderToolsAgent with real API keys
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Add path to python path
sys.path.append(os.path.join(os.getcwd()))

# Import the DataLoaderToolsAgent
from src.agents import DataLoaderToolsAgent

# Get the OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in .env file. Please add OPENAI_API_KEY to your .env file.")

print("=== Testing DataLoaderToolsAgent with Real API Key ===\n")

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
if isinstance(loaded_data, str):
    print(f"Error: {loaded_data}")
elif isinstance(loaded_data, pd.DataFrame):
    print(f"Loaded data shape: {loaded_data.shape}")
    print(f"Columns: {', '.join(loaded_data.columns)}")
    print(f"First 3 rows:")
    print(loaded_data.head(3))
else:
    print(f"Unexpected data type: {type(loaded_data)}")
    print(f"Raw data: {loaded_data}")

# Get the tool calls
tool_calls = data_loader_agent.get_tool_calls()
print(f"Tool calls used: {tool_calls}")

# Get the AI message to debug
ai_message = data_loader_agent.get_ai_message()
print(f"Agent response: {ai_message}")
print()

# Display internal messages for debugging
print("Internal messages:")
internal_messages = data_loader_agent.get_internal_messages()
if not isinstance(internal_messages, str):
    for i, msg in enumerate(internal_messages):
        print(f"Message {i+1} - Type: {msg.type}")
        if hasattr(msg, "additional_kwargs") and "artifact" in msg.additional_kwargs:
            print(f"  Has artifact: {msg.additional_kwargs['artifact']}")
        elif hasattr(msg, "artifact"):
            print(f"  Has artifact: {msg.artifact}")
else:
    print(internal_messages)
print()

# Test 2: Directory exploration with more explicit instructions
print("Test 2: Directory Exploration")
data_loader_agent.invoke_agent(
    user_instructions="First, list all files in the data directory. Then, search for all CSV files in the data directory including subdirectories."
)

# Get the AI message
ai_message = data_loader_agent.get_ai_message()
print(f"Agent response: {ai_message}")

# Get the tool calls
tool_calls = data_loader_agent.get_tool_calls()
print(f"Tool calls used: {tool_calls}")
print()

# Test 3: Loading a single file with very explicit instructions
print("Test 3: Loading Single File with Explicit Instructions")
data_loader_agent.invoke_agent(
    user_instructions="Use the load_file tool to load the CSV file located at data/samples/sample_data.csv"
)

# Get the loaded data
loaded_data = data_loader_agent.get_artifacts(as_dataframe=True)
if isinstance(loaded_data, str):
    print(f"Error: {loaded_data}")
elif isinstance(loaded_data, pd.DataFrame):
    print(f"Loaded data shape: {loaded_data.shape}")
    print(f"Columns: {', '.join(loaded_data.columns)}")
    print(f"First 3 rows:")
    print(loaded_data.head(3))
else:
    print(f"Unexpected data type: {type(loaded_data)}")
    print(f"Raw data: {loaded_data}")

# Get the tool calls
tool_calls = data_loader_agent.get_tool_calls()
print(f"Tool calls used: {tool_calls}")

# Get the AI message
ai_message = data_loader_agent.get_ai_message()
print(f"Agent response: {ai_message}")
print()

print("\n=== DataLoaderToolsAgent tests completed ===")
print("Check the output above to verify functionality.") 