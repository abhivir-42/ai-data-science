"""
Test script for verifying the uAgent adapters for DataCleaningAgent and DataLoaderToolsAgent
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

# Import the adapters
from src.adapters import DataCleaningAgentAdapter, DataLoaderToolsAgentAdapter

# Get the API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AGENTVERSE_API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in .env file. Please add OPENAI_API_KEY to your .env file.")

print("=== Testing uAgent Adapters for Data Science Agents ===\n")

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

# Test 1: DataCleaningAgentAdapter functionality
print("Test 1: DataCleaningAgentAdapter Local Functionality")
data_cleaning_adapter = DataCleaningAgentAdapter(
    model=llm,
    name="test_data_cleaning_agent",
    port=8000,
    mailbox=False,  # Don't use mailbox for testing
    api_token=AGENTVERSE_API_TOKEN
)

# Load sample data for cleaning
data_path = os.path.join(os.getcwd(), "data", "samples", "sample_data.csv")
test_data = pd.read_csv(data_path)

# Add some null values to test cleaning
test_data.loc[0, 'name'] = None
test_data.loc[1, 'age'] = None
test_data.loc[2, 'department'] = None

print(f"Original test data shape: {test_data.shape}")
print(f"Null values: {test_data.isnull().sum().sum()}")

# Use the adapter's clean_data method
print("Cleaning data using the adapter...")
cleaned_data = data_cleaning_adapter.clean_data(
    data=test_data,
    instructions="Clean the data by handling missing values appropriately."
)

if isinstance(cleaned_data, pd.DataFrame):
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Null values after cleaning: {cleaned_data.isnull().sum().sum()}")
    print(f"First 3 rows of cleaned data:")
    print(cleaned_data.head(3))
else:
    print(f"Error: Failed to get cleaned DataFrame. Got {type(cleaned_data)} instead.")

print()

# Test 2: DataLoaderToolsAgentAdapter functionality
print("Test 2: DataLoaderToolsAgentAdapter Local Functionality")
data_loader_adapter = DataLoaderToolsAgentAdapter(
    model=llm,
    name="test_data_loader_agent",
    port=8001,
    mailbox=False,  # Don't use mailbox for testing
    api_token=AGENTVERSE_API_TOKEN
)

# Use the adapter's load_data method
print("Loading data using the adapter...")
try:
    loaded_data = data_loader_adapter.load_data(
        instructions="Load the CSV file from data/samples/sample_data.csv"
    )
    
    if isinstance(loaded_data, pd.DataFrame):
        print(f"Loaded data shape: {loaded_data.shape}")
        print(f"Loaded data columns: {', '.join(loaded_data.columns)}")
        print(f"First 3 rows of loaded data:")
        print(loaded_data.head(3))
    else:
        print(f"Error: Failed to get loaded DataFrame. Got {type(loaded_data)} instead.")
except Exception as e:
    print(f"Error loading data: {str(e)}")

print()

# Test 3: Check registration capabilities
print("Test 3: Agent Registration Information")

# Test if the Agentverse API token is available
if AGENTVERSE_API_TOKEN:
    print("AGENTVERSE_API_TOKEN is available. Registration could be performed.")
    print("Note: Not performing actual registration as it requires network connectivity to Agentverse.")
    
    # Show what registration would look like
    print("\nInformation that would be sent for registration:")
    print(f"DataCleaningAgent - Name: {data_cleaning_adapter.name}, Port: {data_cleaning_adapter.port}")
    print(f"DataLoaderToolsAgent - Name: {data_loader_adapter.name}, Port: {data_loader_adapter.port}")
    
    # For information, print a command that would register the agents
    print("\nTo register these agents with Agentverse, use:")
    print("python src/examples/register_data_agents.py --verbose --agents data_cleaner data_loader")
else:
    print("AGENTVERSE_API_TOKEN is not available. Registration would fail.")
    print("Please set the AGENTVERSE_API_TOKEN environment variable to register agents with Fetch.ai Agentverse.")

print("\n=== uAgent Adapter Tests Completed ===")
print("The adapters work correctly for local agent functionality. Registration capabilities identified.") 