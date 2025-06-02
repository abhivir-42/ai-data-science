"""
Minimal example of using DataCleaningAgent directly.

This shows the simplest possible usage of the DataCleaningAgent.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.data_cleaning_agent import DataCleaningAgent

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=os.getenv("OPENAI_API_KEY"))

# Create the agent
agent = DataCleaningAgent(
    model=llm,
    log=True,  # Optional: logs the cleaning function to a file
    human_in_the_loop=False  # Set to True if you want to review the steps
)

# Load data
df = pd.read_csv("data/churn_data.csv")
print(f"Original data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Missing values: {df.isna().sum().sum()}")

# Define cleaning instructions (optional)
instructions = """
Clean this dataset by:
1. Filling missing values
2. Creating a tenure_group column from tenure
3. Converting categorical columns to appropriate types
"""

# Run the agent - that's it!
agent.invoke_agent(
    data_raw=df,
    user_instructions=instructions
)

# Get the cleaned data
cleaned_df = agent.get_data_cleaned()
print(f"\nCleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
print(f"Missing values: {cleaned_df.isna().sum().sum()}")

# If you want, you can also get the cleaning function and steps
cleaning_function = agent.get_data_cleaner_function()
steps = agent.get_recommended_cleaning_steps()

print("\nDone! The dataset has been cleaned.") 