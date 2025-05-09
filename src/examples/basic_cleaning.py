#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Data Cleaning Example

This example demonstrates the simplest way to use the DataCleaningAgent
for cleaning datasets. It shows the minimal code needed to:
1. Set up the agent
2. Clean a dataset
3. Get the cleaned data and generated code
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the agent
from src.agents.data_cleaning_agent import DataCleaningAgent
from langchain_openai import ChatOpenAI


def main():
    """Run the basic data cleaning example"""
    print("Basic Data Cleaning Example")
    print("--------------------------\n")
    
    # Load the OpenAI API key from environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Create a simple dataset with issues for demonstration
    data = {
        'age': [25, 30, None, 45, 200, 35, 40, None],
        'income': [50000, 60000, 75000, None, 90000, 55000, None, 65000],
        'gender': ['M', 'F', None, 'M', 'F', None, 'M', 'F'],
        'education': ['Bachelor', 'Master', 'PhD', None, 'Bachelor', 'Master', 'PhD', None]
    }
    df = pd.DataFrame(data)
    
    # Print the original dataset
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Create the data cleaning agent
    print("\nCreating data cleaning agent...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.1  # Lower temperature for more consistent results
    )
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '../../logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    agent = DataCleaningAgent(
        model=llm,
        n_samples=10,  # Small dataset, so we only need a few samples
        log=True,
        log_path=logs_dir
    )
    
    # Run the agent to clean the data
    print("\nCleaning the data...")
    agent.invoke_agent(
        data_raw=df,
        user_instructions="""
        Clean this dataset by:
        1. Handling missing values (replace with mean or other appropriate value)
        2. Removing age outliers (normal human age range)
        3. Ensuring consistent data types
        """,
        max_retries=2
    )
    
    # Get the cleaned data
    cleaned_df = agent.get_data_cleaned()
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nCleaned dataset info:")
    print(cleaned_df.info())
    print(f"\nMissing values after cleaning:\n{cleaned_df.isnull().sum()}")
    
    # Show the generated cleaning code
    print("\nGenerated cleaning code:")
    print(agent.get_data_cleaner_function())
    
    print("\nExample complete!")


if __name__ == "__main__":
    main() 