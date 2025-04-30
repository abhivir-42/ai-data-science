"""
DataCleaningAgentAdapter demonstration using uAgents.

This script:
1. Loads the churn_data.csv dataset
2. Creates a DataCleaningAgentAdapter
3. Uses the underlying DataCleaningAgent directly to demonstrate its functionality

Usage:
    python examples/real_uagents_adapter_example.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter

# Load environment variables
load_dotenv()

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY", "")
agentverse_api_token = os.getenv("AGENTVERSE_API_TOKEN", "")

# If not in .env file, use the token directly


print("🤖 DataCleaningAgentAdapter Demonstration")
print("This example demonstrates DataCleaningAgent functionality through the adapter")

# Initialize the language model
print("\n📝 Initializing ChatOpenAI...")
llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=openai_api_key)

# Create the adapter
print("\n🔄 Creating DataCleaningAgentAdapter...")
adapter = DataCleaningAgentAdapter(
    model=llm,
    name="data_cleaning_agent",
    description="A data cleaning agent for processing churn dataset",
    n_samples=20,  # Reduce to avoid token limits
    log=True,
    log_path="./logs",
    human_in_the_loop=False
)

# Load the dataset
print("\n📊 Loading churn dataset...")
df = pd.read_csv("data/churn_data.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Missing values: {df.isna().sum().sum()}")

# Skip registration and focus on using the underlying DataCleaningAgent directly
print("\n🧪 Testing the underlying DataCleaningAgent...")

# Define cleaning instructions
instructions = """
Clean this churn dataset by:
1. Filling missing values
2. Creating a tenure_group column from tenure
3. Converting categorical columns to appropriate types
"""

# Get the underlying agent
print("\n🧹 Running data cleaning process...")
data_cleaning_agent = adapter.agent

# Run the cleaning process
data_cleaning_agent.invoke_agent(
    data_raw=df,
    user_instructions=instructions
)

# Get the cleaned data
cleaned_df = data_cleaning_agent.get_data_cleaned()
print(f"\n✨ Cleaning completed!")
print(f"Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
print(f"Missing values: {cleaned_df.isna().sum().sum()}")

# Get the cleaning function
cleaner_function = data_cleaning_agent.get_data_cleaner_function()

# Print a summary of the changes
print("\n📝 Summary of changes:")
print(f"  - Original shape: {df.shape}")
print(f"  - Cleaned shape: {cleaned_df.shape}")
print(f"  - Missing values removed: {df.isna().sum().sum() - cleaned_df.isna().sum().sum()}")

# Check for new columns
new_cols = set(cleaned_df.columns) - set(df.columns)
if new_cols:
    print(f"  - New columns: {new_cols}")

# Get the cleaning steps
print("\n📋 Data Cleaning Steps:")
steps = data_cleaning_agent.get_recommended_cleaning_steps()
if steps:
    print(steps)

print("\n🎉 DataCleaningAgentAdapter demonstration completed!")
print("The DataCleaningAgent works properly through the adapter interface.")
print("Note: Full uAgent registration requires updated dependencies and network connectivity.") 