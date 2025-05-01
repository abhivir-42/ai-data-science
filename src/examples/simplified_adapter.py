"""
Simplified DataCleaningAgent adapter example

This module provides a simple example of how to use the DataCleaningAgent
without dealing with the complexity of uAgent registration.
"""

import pandas as pd
from langchain_openai import ChatOpenAI

from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter

def run_data_cleaning_example(data_path="data/churn_data.csv", openai_api_key=None):
    """
    Run a simple data cleaning example using the DataCleaningAgentAdapter
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the data to clean
    openai_api_key : str, optional
        OpenAI API key to use for the model
        
    Returns
    -------
    pd.DataFrame
        The cleaned dataset
    """
    # Initialize the language model
    print("\n📝 Initializing ChatOpenAI...")
    llm = ChatOpenAI(
        model="gpt-4o",  # Use any available model
        api_key=openai_api_key
    )
    
    # Create the adapter
    print("\n🔄 Creating DataCleaningAgentAdapter...")
    adapter = DataCleaningAgentAdapter(
        model=llm,
        n_samples=20,
        log=False  # Disable logging to avoid path issues
    )
    
    # Load the dataset
    print("\n📊 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    # Define cleaning instructions
    instructions = """
    Clean this dataset by:
    1. Filling missing values
    2. Converting categorical columns to appropriate types
    3. Creating derived columns where appropriate
    """
    
    print("\n🧹 Running data cleaning process...")
    cleaned_df = adapter.clean_data(df, instructions)
    
    print(f"\n✨ Cleaning completed!")
    print(f"Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
    print(f"Missing values: {cleaned_df.isna().sum().sum()}")
    
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
    steps = adapter.agent.get_recommended_cleaning_steps()
    if steps:
        print(steps)
    
    return cleaned_df

if __name__ == "__main__":
    # Simple usage example
    clean_df = run_data_cleaning_example() 