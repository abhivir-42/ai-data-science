"""
Test script for registering data agents with Agentverse
"""

import os
import sys
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

if not AGENTVERSE_API_TOKEN:
    raise ValueError("No Agentverse API token found in .env file. Please add AGENTVERSE_API_TOKEN to your .env file.")

print("=== Testing Registration with Agentverse ===\n")

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

# Try to register the DataCleaningAgent
print("Attempting to register DataCleaningAgent...")
data_cleaning_adapter = DataCleaningAgentAdapter(
    model=llm,
    name="data_cleaning_agent",
    port=8000,
    mailbox=True,
    api_token=AGENTVERSE_API_TOKEN
)

cleaning_result = data_cleaning_adapter.register()
print("\nRegistration Result for DataCleaningAgent:")
print(cleaning_result)
print("\n")

# Try to register the DataLoaderToolsAgent
print("Attempting to register DataLoaderToolsAgent...")
data_loader_adapter = DataLoaderToolsAgentAdapter(
    model=llm,
    name="data_loader_agent",
    port=8001,
    mailbox=True,
    api_token=AGENTVERSE_API_TOKEN
)

loader_result = data_loader_adapter.register()
print("\nRegistration Result for DataLoaderToolsAgent:")
print(loader_result)
print("\n")

print("=== Registration Testing Completed ===") 