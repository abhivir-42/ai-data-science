"""
Example Data Pipeline using DataLoaderToolsAgent and DataCleaningAgent.

This example demonstrates how to use the DataLoaderToolsAgent and DataCleaningAgent
together to create a simple data processing pipeline.
"""

import os
import pandas as pd
from langchain_openai import ChatOpenAI

from ai_data_science.agents import DataLoaderToolsAgent, DataCleaningAgent

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")

def run_data_processing_pipeline(
    data_load_instructions: str,
    data_cleaning_instructions: str = None,
    model_name: str = "gpt-4o-mini",
    verbose: bool = True
):
    """
    Run a complete data processing pipeline using DataLoaderToolsAgent and DataCleaningAgent.
    
    Parameters
    ----------
    data_load_instructions : str
        Instructions for the DataLoaderToolsAgent on what data to load
    data_cleaning_instructions : str, optional
        Instructions for the DataCleaningAgent on how to clean the data
    model_name : str, optional
        The OpenAI model to use (defaults to "gpt-4o-mini")
    verbose : bool, optional
        Whether to print progress messages (defaults to True)
        
    Returns
    -------
    pd.DataFrame
        The cleaned data as a pandas DataFrame
    """
    if verbose:
        print(f"🚀 Initializing data processing pipeline with model: {model_name}")
    
    # Initialize the language model
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )
    
    # Step 1: Load data using DataLoaderToolsAgent
    if verbose:
        print("\n📂 Step 1: Loading data with DataLoaderToolsAgent")
        print(f"Instructions: {data_load_instructions}")
    
    data_loader = DataLoaderToolsAgent(model=llm)
    data_loader.invoke_agent(user_instructions=data_load_instructions)
    
    loaded_data = data_loader.get_artifacts(as_dataframe=True)
    
    if verbose:
        print(f"✅ Data loaded successfully with shape: {loaded_data.shape}")
        print(f"Tools used: {data_loader.get_tool_calls()}")
        
    # Step 2: Clean data using DataCleaningAgent
    if verbose:
        print("\n🧹 Step 2: Cleaning data with DataCleaningAgent")
        if data_cleaning_instructions:
            print(f"Instructions: {data_cleaning_instructions}")
        else:
            print("Using default cleaning steps")
    
    data_cleaner = DataCleaningAgent(model=llm)
    data_cleaner.invoke_agent(
        data_raw=loaded_data,
        user_instructions=data_cleaning_instructions
    )
    
    cleaned_data = data_cleaner.get_data_cleaned()
    
    if verbose:
        print(f"✅ Data cleaned successfully with shape: {cleaned_data.shape}")
        
        # Show what transformations were done
        print("\n📊 Data Transformation Summary:")
        print(f"Original data shape: {loaded_data.shape}")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        
        # Calculate and show changes
        rows_removed = loaded_data.shape[0] - cleaned_data.shape[0]
        if rows_removed > 0:
            print(f"Rows removed: {rows_removed} ({rows_removed/loaded_data.shape[0]:.1%} of original data)")
            
        cols_removed = loaded_data.shape[1] - cleaned_data.shape[1]
        if cols_removed > 0:
            print(f"Columns removed: {cols_removed}")
            removed_cols = set(loaded_data.columns) - set(cleaned_data.columns)
            print(f"Removed columns: {removed_cols}")
        
        # Show the cleaned data steps
        print("\n🧪 Data Cleaning Steps Applied:")
        cleaner_fn = data_cleaner.get_data_cleaner_function()
        print(cleaner_fn)
    
    return cleaned_data


if __name__ == "__main__":
    # Example usage
    
    # Replace with your actual data source
    load_instructions = """
    Load the CSV file located at 'data/sample_data.csv'.
    If the file doesn't exist, search the data directory for any CSV files and load the first one found.
    """
    
    cleaning_instructions = """
    Perform the standard data cleaning steps, but don't remove outliers.
    Make sure to convert date columns to datetime objects.
    """
    
    cleaned_df = run_data_processing_pipeline(
        data_load_instructions=load_instructions,
        data_cleaning_instructions=cleaning_instructions,
        verbose=True
    )
    
    print("\n🎉 Final Cleaned Data Preview:")
    print(cleaned_df.head()) 