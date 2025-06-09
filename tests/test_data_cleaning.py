"""
Comprehensive test script for the Data Cleaning Agent
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

def create_test_data():
    """Create test data with various issues that need cleaning"""
    # Create a sample dataset with missing values, duplicates, and outliers
    np.random.seed(42)
    
    data = {
        'id': range(1, 51),
        'name': [f'Person {i}' for i in range(1, 51)],
        'age': np.random.randint(18, 80, 50),
        'income': np.random.randint(20000, 150000, 50),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 50),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 50),
        'is_customer': np.random.choice([True, False], 50),
        'join_date': pd.date_range(start='2020-01-01', periods=50),
        'satisfaction_score': np.random.randint(1, 11, 50)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add missing values
    df.loc[np.random.choice(df.index, 10), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 8), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'education'] = np.nan
    df.loc[np.random.choice(df.index, 7), 'satisfaction_score'] = np.nan
    
    # Add duplicates
    duplicate_indices = np.random.choice(df.index, 5, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Add outliers
    df.loc[50, 'age'] = 120  # Extreme age
    df.loc[51, 'income'] = 1000000  # Extreme income
    df.loc[52, 'satisfaction_score'] = 20  # Extreme satisfaction score (out of 10)
    
    return df

def test_data_cleaning_agent():
    """Test the data cleaning agent with various scenarios"""
    # Load environment variables (for API keys)
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key":
        print("ERROR: OpenAI API key not found or not set!")
        print("Please set your API key in the .env file.")
        return
    
    # Create test data
    test_df = create_test_data()
    print(f"Created test data with shape: {test_df.shape}")
    print(f"Missing values: {test_df.isna().sum().sum()}")
    print(f"Duplicates: {test_df.duplicated().sum()}")
    print("Sample data:")
    print(test_df.head())
    print("\nData statistics:")
    print(test_df.describe())
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Test Case 1: Basic cleaning (default behavior)
    print("\n--- Test Case 1: Basic Cleaning ---")
    agent1 = DataCleaningAgent(
        model=llm,
        n_samples=10,
        log=True,
        log_path="./logs",
        human_in_the_loop=False
    )
    
    agent1.invoke_agent(
        data_raw=test_df,
        user_instructions="Clean the data by handling missing values, removing duplicates, and treating outliers.",
        max_retries=2
    )
    
    cleaned_df1 = agent1.get_data_cleaned()
    print(f"Original shape: {test_df.shape}")
    print(f"Cleaned shape: {cleaned_df1.shape}")
    print(f"Missing values after cleaning: {cleaned_df1.isna().sum().sum()}")
    print(f"Duplicates after cleaning: {cleaned_df1.duplicated().sum()}")
    print(f"Age range after cleaning: {cleaned_df1['age'].min()} - {cleaned_df1['age'].max()}")
    print(f"Income range after cleaning: {cleaned_df1['income'].min()} - {cleaned_df1['income'].max()}")
    
    # Test Case 2: Keep outliers
    print("\n--- Test Case 2: Keep Outliers ---")
    agent2 = DataCleaningAgent(
        model=llm,
        n_samples=10,
        log=True,
        log_path="./logs",
        human_in_the_loop=False
    )
    
    agent2.invoke_agent(
        data_raw=test_df,
        user_instructions="Clean the data by handling missing values and removing duplicates, but DO NOT remove outliers.",
        max_retries=2
    )
    
    cleaned_df2 = agent2.get_data_cleaned()
    print(f"Original shape: {test_df.shape}")
    print(f"Cleaned shape: {cleaned_df2.shape}")
    print(f"Missing values after cleaning: {cleaned_df2.isna().sum().sum()}")
    print(f"Duplicates after cleaning: {cleaned_df2.duplicated().sum()}")
    print(f"Age range after cleaning: {cleaned_df2['age'].min()} - {cleaned_df2['age'].max()}")
    print(f"Income range after cleaning: {cleaned_df2['income'].min()} - {cleaned_df2['income'].max()}")
    
    # Test Case 3: Only handle missing values
    print("\n--- Test Case 3: Only Handle Missing Values ---")
    agent3 = DataCleaningAgent(
        model=llm,
        n_samples=10,
        log=True,
        log_path="./logs",
        human_in_the_loop=False
    )
    
    agent3.invoke_agent(
        data_raw=test_df,
        user_instructions="Only handle missing values in the data. Do not remove duplicates or outliers.",
        max_retries=2
    )
    
    cleaned_df3 = agent3.get_data_cleaned()
    print(f"Original shape: {test_df.shape}")
    print(f"Cleaned shape: {cleaned_df3.shape}")
    print(f"Missing values after cleaning: {cleaned_df3.isna().sum().sum()}")
    print(f"Duplicates after cleaning: {cleaned_df3.duplicated().sum()}")

if __name__ == "__main__":
    test_data_cleaning_agent() 