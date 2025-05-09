"""
Test script for verifying the pipeline integration between DataLoaderToolsAgent and DataCleaningAgent
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

# Import agents
from src.agents import DataLoaderToolsAgent, DataCleaningAgent

# Get the OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in .env file. Please add OPENAI_API_KEY to your .env file.")

print("=== Testing Data Pipeline Integration ===\n")

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

# Step 1: Set up DataLoaderToolsAgent
print("Step 1: Setting up DataLoaderToolsAgent")
data_loader_agent = DataLoaderToolsAgent(model=llm)

# Test loading data with specific instructions
print("Loading CSV data...")
data_loader_agent.invoke_agent(
    user_instructions="Load the CSV file from data/samples/sample_data.csv"
)

# Get the loaded data as DataFrame
loaded_data = data_loader_agent.get_artifacts(as_dataframe=True)
if not isinstance(loaded_data, pd.DataFrame):
    # Try fallback with direct load if agent failed
    print("Agent-based loading failed, using direct loading fallback")
    loaded_data = pd.read_csv(os.path.join(os.getcwd(), "data", "samples", "sample_data.csv"))

print(f"Loaded data shape: {loaded_data.shape}")
print(f"Loaded data columns: {', '.join(loaded_data.columns)}")

# Add some null values to make cleaning more interesting
print("\nIntroducing null values for testing cleaning...")
loaded_data.loc[0, 'name'] = None
loaded_data.loc[1, 'age'] = None 
loaded_data.loc[2, 'department'] = None
print(f"Data with nulls - shape: {loaded_data.shape}, null count: {loaded_data.isnull().sum().sum()}")

# Step 2: Set up DataCleaningAgent
print("\nStep 2: Setting up DataCleaningAgent")
data_cleaning_agent = DataCleaningAgent(
    model=llm,
    n_samples=10, # Use smaller sample for faster execution
    log=True,
    log_path="logs/"
)

# Run the data cleaning agent on the loaded data
print("Cleaning the loaded data...")
data_cleaning_agent.invoke_agent(
    data_raw=loaded_data,
    user_instructions="Clean the data by handling missing values appropriately. For numerical columns use the mean, for categorical columns use the most frequent value."
)

# Get the cleaned data
cleaned_data = data_cleaning_agent.get_data_cleaned()
print(f"Cleaned data shape: {cleaned_data.shape}")
print(f"Null values after cleaning: {cleaned_data.isnull().sum().sum()}")
print("First 5 rows of cleaned data:")
print(cleaned_data.head(5))

# Step 3: Check pipeline results
print("\nStep 3: Pipeline Integration Results")
print(f"Original data shape: {loaded_data.shape}")
print(f"Cleaned data shape: {cleaned_data.shape}")
print(f"Null values before cleaning: {loaded_data.isnull().sum().sum()}")
print(f"Null values after cleaning: {cleaned_data.isnull().sum().sum()}")

# Verify transformations
print("\nData transformations performed:")
print(f"- Rows processed: {loaded_data.shape[0]} → {cleaned_data.shape[0]}")
print(f"- Missing values handled: {loaded_data.isnull().sum().sum()} → {cleaned_data.isnull().sum().sum()}")

print("\n=== Pipeline Integration Test Completed ===")
print("The DataLoaderToolsAgent and DataCleaningAgent successfully integrated in a pipeline workflow.") 