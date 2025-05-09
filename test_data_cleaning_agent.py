"""
Test script for verifying the functionality of DataCleaningAgent with real API keys
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

# Import the DataCleaningAgent
from src.agents import DataCleaningAgent

# Get the OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in .env file. Please add OPENAI_API_KEY to your .env file.")

print("=== Testing DataCleaningAgent with Real API Key ===\n")

# Load sample data
data_path = os.path.join(os.getcwd(), "data", "samples", "sample_data.csv")
test_data = pd.read_csv(data_path)

# Add some null values to test cleaning
test_data.loc[0, 'name'] = None
test_data.loc[1, 'age'] = None
test_data.loc[2, 'department'] = None

print(f"Original test data with introduced nulls:")
print(test_data.head(5))
print(f"Original shape: {test_data.shape}")
print(f"Null values: {test_data.isnull().sum().sum()}")
print()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

# Test 1: Basic cleaning with default parameters
print("Test 1: Basic Cleaning with Default Parameters")
data_cleaning_agent = DataCleaningAgent(
    model=llm,
    n_samples=10,  # Use smaller sample for faster execution
    log=True,
    log_path="logs/"
)

# Run the agent
data_cleaning_agent.invoke_agent(
    data_raw=test_data,
    user_instructions="Clean the data using default steps, focusing on handling missing values."
)

# Get the cleaned data
cleaned_data = data_cleaning_agent.get_data_cleaned()
if isinstance(cleaned_data, pd.DataFrame):
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Null values after cleaning: {cleaned_data.isnull().sum().sum()}")
    print(f"First 5 rows of cleaned data:")
    print(cleaned_data.head(5))
    print()
else:
    print(f"Error: Failed to get cleaned DataFrame. Got {type(cleaned_data)} instead.")
    print()

# Get the cleaning function
cleaning_function = data_cleaning_agent.get_data_cleaner_function()
print("Generated cleaning function:")
print(f"{cleaning_function[:500]}...\n(truncated)")
print()

# Get recommended steps
recommended_steps = data_cleaning_agent.get_recommended_cleaning_steps()
print("Recommended cleaning steps:")
print(f"{recommended_steps[:500]}...\n(truncated)")
print()

# Test 2: Custom cleaning instructions
print("Test 2: Custom Cleaning Instructions")
data_cleaning_agent_2 = DataCleaningAgent(
    model=llm,
    n_samples=10,
    log=True,
    log_path="logs/"
)

# Run the agent with custom instructions
data_cleaning_agent_2.invoke_agent(
    data_raw=test_data,
    user_instructions="Fill missing values in the 'name' column with 'Unknown', fill missing 'age' with the median, and remove rows with missing 'department'."
)

# Get the cleaned data
cleaned_data_2 = data_cleaning_agent_2.get_data_cleaned()
if isinstance(cleaned_data_2, pd.DataFrame):
    print(f"Cleaned data shape with custom instructions: {cleaned_data_2.shape}")
    print(f"Null values after custom cleaning: {cleaned_data_2.isnull().sum().sum()}")
    print(f"First 5 rows of custom cleaned data:")
    print(cleaned_data_2.head(5))
    print()
else:
    print(f"Error: Failed to get cleaned DataFrame. Got {type(cleaned_data_2)} instead.")
    print()

print("\n=== DataCleaningAgent tests completed ===")
print("Check the output above to verify functionality.") 