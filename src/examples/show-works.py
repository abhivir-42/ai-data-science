#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example showing how the Data Cleaning Agent works with a real dataset.

This script demonstrates the complete workflow of:
1. Loading a dataset with issues
2. Creating a data cleaning agent
3. Running the agent to clean the data
4. Analyzing the results and showing the generated code

Tested with the Kaggle Housing Prices dataset.
"""

# Standard imports
import os
import sys
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add the root directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our agent
from src.agents.data_cleaning_agent import DataCleaningAgent
from langchain_openai import ChatOpenAI

# Set up colored output for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title):
    """Print a section title with formatting"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}== {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")

def print_step(step_name):
    """Print a step name with formatting"""
    print(f"{Colors.BOLD}{Colors.CYAN}>> {step_name}...{Colors.ENDC}")

def print_success(message):
    """Print a success message with formatting"""
    print(f"{Colors.BOLD}{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message with formatting"""
    print(f"{Colors.BOLD}{Colors.RED}✗ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message with formatting"""
    print(f"{Colors.BOLD}{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def main():
    """Run the main demonstration workflow"""
    print_section("DATA CLEANING AGENT DEMONSTRATION")
    
    # Step 1: Load environment variables
    print_step("Loading environment variables")
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print_error("OPENAI_API_KEY not found in environment variables")
        print("Please set the OPENAI_API_KEY environment variable in the .env file")
        return
    print_success("Environment variables loaded")
    
    # Step 2: Load the dataset
    print_step("Loading dataset")
    try:
        data_path = os.path.join(os.path.dirname(__file__), '../../data/samples/train.csv')
        df = pd.read_csv(data_path)
        print_success(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print_error(f"Failed to load dataset: {str(e)}")
        return
    
    # Step 3: Analyze the dataset
    print_step("Analyzing dataset")
    print(f"First 5 rows:")
    print(df.head())
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_stats = pd.DataFrame({
        'Missing Values': missing,
        'Percent Missing': missing_percent
    })
    missing_stats = missing_stats[missing_stats['Missing Values'] > 0].sort_values('Percent Missing', ascending=False)
    
    print("\nMissing values analysis:")
    print(missing_stats.head(10))  # Show top 10 columns with missing values
    
    # Step 4: Create the cleaning agent
    print_step("Creating data cleaning agent")
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            openai_api_key=openai_api_key,
            temperature=0.2  # Lower temperature for more deterministic responses
        )
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        cleaning_agent = DataCleaningAgent(
            model=llm,
            n_samples=50,  # Reduced sample size to avoid token limits
            log=True,      
            log_path=logs_dir,
            file_name="housing_data_cleaner.py",
            function_name="clean_housing_data"
        )
        print_success("Data cleaning agent created")
    except Exception as e:
        print_error(f"Failed to create data cleaning agent: {str(e)}")
        return
    
    # Step 5: Run the cleaning agent
    print_step("Running data cleaning agent")
    cleaning_instructions = """
    Please clean this housing dataset by:
    1. Handling missing values in a way that preserves as much data as possible
    2. Removing extreme outliers that would skew analysis
    3. Converting categorical variables appropriately
    4. Ensuring all data types are appropriate
    5. Adding helpful comments in the code explaining each cleaning step
    """
    print(f"Cleaning instructions: {cleaning_instructions}")
    
    try:
        start_time = time.time()
        result = cleaning_agent.invoke_agent(
            data_raw=df,
            user_instructions=cleaning_instructions,
            max_retries=3
        )
        end_time = time.time()
        print_success(f"Data cleaning completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print_error(f"Error during data cleaning: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 6: Examine the results
    print_step("Examining cleaned data")
    try:
        cleaned_df = cleaning_agent.get_data_cleaned()
        
        if cleaned_df is None:
            print_error("No cleaned data returned")
            # Print response keys for debugging
            if cleaning_agent.response:
                print(f"Response keys: {list(cleaning_agent.response.keys())}")
            return
            
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print("\nFirst 5 rows of cleaned data:")
        print(cleaned_df.head())
        
        # Compare before and after
        print("\nBefore vs After Cleaning:")
        comparison = pd.DataFrame({
            'Original Missing': df.isnull().sum(),
            'Cleaned Missing': cleaned_df.isnull().sum(),
            'Original Dtype': df.dtypes,
            'Cleaned Dtype': cleaned_df.dtypes
        })
        print(comparison.head(10))  # Show first 10 columns
        
        print_success("Cleaned data examination complete")
    except Exception as e:
        print_error(f"Error examining cleaned data: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # Step 7: Display the generated cleaning code
    print_step("Displaying generated cleaning code")
    try:
        cleaning_code = cleaning_agent.get_data_cleaner_function()
        if cleaning_code:
            print("\nGenerated data cleaning code:")
            print(f"\n{Colors.BOLD}{Colors.GREEN}```python{Colors.ENDC}")
            print(cleaning_code)
            print(f"{Colors.BOLD}{Colors.GREEN}```{Colors.ENDC}")
            print_success("Generated code displayed")
        else:
            print_warning("No cleaning code was generated")
    except Exception as e:
        print_error(f"Error displaying cleaning code: {str(e)}")
    
    print_section("DEMONSTRATION COMPLETE")
    print("The data cleaning agent has successfully processed the dataset!")
    print("You can use the cleaned data for further analysis and modeling.")

if __name__ == "__main__":
    main()