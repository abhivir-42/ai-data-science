#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Presentation Demo for AI Data Science Agents

This script demonstrates the capabilities of our AI Data Science Agents
in a presentation-friendly format. It focuses on:
1. Clear, visually appealing output
2. Step-by-step demonstration of agent capabilities
3. Highlighting key features for an audience

Perfect for live demonstrations during presentations.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

# Add the root directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our agents
from src.agents.data_cleaning_agent import DataCleaningAgent
from langchain_openai import ChatOpenAI

# Set up styled print functions for better readability
def print_header(text):
    """Print a header with stars around it"""
    print("\n")
    print("*" * 80)
    print(f"*{text:^78}*")
    print("*" * 80)
    print("\n")

def print_subheader(text):
    """Print a subheader with formatting"""
    print(f"\n{'='*80}\n{text}\n{'='*80}\n")

def print_step(number, description):
    """Print a step number and description"""
    print(f"\n🔷 Step {number}: {description}")

def print_code(code):
    """Print code with markdown-style formatting"""
    print("\n```python")
    print(code)
    print("```\n")

def print_success(text):
    """Print a success message"""
    print(f"\n✅ {text}\n")

def print_info(text):
    """Print an info message"""
    print(f"\nℹ️ {text}\n")

def plot_missing_values(df, title="Missing Values Analysis"):
    """Create and display a bar plot of missing values"""
    plt.figure(figsize=(12, 6))
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percent': missing_percent
    })
    missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Percent', ascending=False)
    
    if len(missing_data) == 0:
        print("No missing values found!")
        return
    
    # Plot the missing values
    ax = sns.barplot(x=missing_data.index, y='Percent', data=missing_data)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel('Percent Missing')
    plt.tight_layout()
    
    # Return the figure for display
    return plt

def compare_columns(original_df, cleaned_df, columns, title="Before vs After Cleaning"):
    """
    Compare the distributions of columns before and after cleaning
    
    Args:
        original_df: The original DataFrame
        cleaned_df: The cleaned DataFrame
        columns: List of column names to compare
        title: Title for the plot
    """
    n_cols = len(columns)
    if n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_cols, 2, figsize=(12, 4 * n_cols))
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        if col in original_df.columns and col in cleaned_df.columns:
            # Determine the plot type based on the data type
            if pd.api.types.is_numeric_dtype(original_df[col]):
                # For numeric columns, use histograms
                original_df[col].plot.hist(ax=axes[i, 0], alpha=0.7, title=f"Original: {col}")
                cleaned_df[col].plot.hist(ax=axes[i, 1], alpha=0.7, title=f"Cleaned: {col}")
            else:
                # For categorical columns, use bar plots
                if original_df[col].nunique() < 10:  # Only for columns with few unique values
                    original_df[col].value_counts().plot.bar(ax=axes[i, 0], title=f"Original: {col}")
                    cleaned_df[col].value_counts().plot.bar(ax=axes[i, 1], title=f"Cleaned: {col}")
                else:
                    axes[i, 0].text(0.5, 0.5, f"Too many categories to display\nUnique values: {original_df[col].nunique()}", 
                                  ha='center', va='center')
                    axes[i, 1].text(0.5, 0.5, f"Too many categories to display\nUnique values: {cleaned_df[col].nunique()}", 
                                  ha='center', va='center')
            
            # Add stats to the title
            axes[i, 0].set_title(f"Original: {col} (Missing: {original_df[col].isna().sum()})")
            axes[i, 1].set_title(f"Cleaned: {col} (Missing: {cleaned_df[col].isna().sum()})")
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    return plt

def run_presentation_demo():
    """Run the main presentation demonstration"""
    print_header("AI DATA SCIENCE AGENTS DEMONSTRATION")
    
    # Step 1: Introduction
    print_step(1, "Introduction to AI Data Science Agents")
    print_info("Our AI Data Science Agents automate complex data tasks using LLMs.")
    print_info("Today we'll demonstrate the Data Cleaning Agent on a housing dataset.")
    
    # Step 2: Setup
    print_step(2, "Setting up the environment")
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Load the dataset
    try:
        data_path = os.path.join(os.path.dirname(__file__), '../../data/samples/train.csv')
        df = pd.read_csv(data_path)
        print_success(f"Dataset loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Step 3: Analyze the original dataset
    print_step(3, "Analyzing the original dataset")
    print_subheader("Sample of the original data:")
    print(df.head())
    
    print_subheader("Dataset information:")
    # Capture the output of df.info() to a string
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    print(info_string)
    
    # Visualize missing values
    print_subheader("Missing Values Analysis")
    try:
        fig = plot_missing_values(df)
        if fig:
            plt.savefig('missing_values.png')
            plt.close()
            print_info("Missing values plot saved to 'missing_values.png'")
    except Exception as e:
        print(f"Could not create missing values plot: {str(e)}")
    
    # Step 4: Set up the data cleaning agent
    print_step(4, "Creating the data cleaning agent")
    
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            openai_api_key=openai_api_key,
            temperature=0.2
        )
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Initialize the cleaning agent
        cleaning_agent = DataCleaningAgent(
            model=llm,
            n_samples=50,
            log=True,
            log_path=logs_dir,
            file_name="presentation_cleaner.py",
            function_name="clean_housing_data"
        )
        print_success("Data cleaning agent initialized successfully")
    except Exception as e:
        print(f"Error setting up cleaning agent: {str(e)}")
        return
    
    # Step 5: Run the cleaning agent
    print_step(5, "Running the data cleaning agent")
    
    # Define cleaning instructions
    cleaning_instructions = """
    Clean this housing price dataset by:
    1. Handling missing values appropriately:
       - For numeric columns with < 20% missing, impute with median
       - For categorical columns, impute with mode
       - For columns with > 40% missing, consider dropping
    2. Handle outliers in numeric columns
    3. Convert categorical variables to appropriate types
    4. Make sure all data types are appropriate
    5. Add helpful comments to explain your cleaning decisions
    """
    
    print_subheader("Cleaning Instructions:")
    print(cleaning_instructions)
    
    print_info("Starting data cleaning process (this may take a few minutes)...")
    try:
        start_time = time.time()
        result = cleaning_agent.invoke_agent(
            data_raw=df,
            user_instructions=cleaning_instructions,
            max_retries=3
        )
        end_time = time.time()
        cleaning_time = end_time - start_time
        print_success(f"Data cleaning completed in {cleaning_time:.2f} seconds")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Step 6: Examine the results
    print_step(6, "Examining the cleaned data")
    
    try:
        # Get the cleaned data
        cleaned_df = cleaning_agent.get_data_cleaned()
        
        if cleaned_df is None:
            print("Error: No cleaned data returned!")
            if cleaning_agent.response:
                print(f"Response keys: {list(cleaning_agent.response.keys())}")
            return
        
        print_subheader("Sample of the cleaned data:")
        print(cleaned_df.head())
        
        # Show shape comparison
        print_subheader("Shape Comparison:")
        print(f"Original: {df.shape} → Cleaned: {cleaned_df.shape}")
        
        # Compare missing values before and after
        print_subheader("Missing Values Comparison:")
        comparison = pd.DataFrame({
            'Original Missing': df.isnull().sum(),
            'Cleaned Missing': cleaned_df.isnull().sum() if cleaned_df is not None else pd.Series(),
            'Original %': (df.isnull().sum() / len(df) * 100).round(2),
            'Cleaned %': (cleaned_df.isnull().sum() / len(cleaned_df) * 100).round(2) if cleaned_df is not None else pd.Series()
        })
        # Only show rows where there were missing values in either dataset
        comparison = comparison[(comparison['Original Missing'] > 0) | (comparison['Cleaned Missing'] > 0)]
        print(comparison)
        
        # Visualize a few interesting columns before and after
        print_subheader("Column Distribution Comparison:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Select a few interesting numeric columns to compare
        cols_to_compare = numeric_cols[:3]  # First 3 numeric columns
        try:
            fig = compare_columns(df, cleaned_df, cols_to_compare)
            if fig:
                plt.savefig('column_comparison.png')
                plt.close()
                print_info("Column comparison plot saved to 'column_comparison.png'")
        except Exception as e:
            print(f"Could not create column comparison plots: {str(e)}")
    except Exception as e:
        print(f"Error examining cleaned data: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # Step 7: Show the generated cleaning code
    print_step(7, "Displaying the generated cleaning code")
    
    try:
        cleaning_code = cleaning_agent.get_data_cleaner_function()
        if cleaning_code:
            print_subheader("Generated Data Cleaning Code:")
            print_code(cleaning_code)
            
            # Save the code to a file
            code_path = os.path.join(os.path.dirname(__file__), '../../housing_data_cleaner.py')
            with open(code_path, 'w') as f:
                f.write(cleaning_code)
            print_success(f"Cleaning code saved to {code_path}")
        else:
            print("No cleaning code was generated.")
    except Exception as e:
        print(f"Error displaying cleaning code: {str(e)}")
    
    # Step 8: Show recommended cleaning steps
    print_step(8, "Agent's recommended cleaning steps")
    
    try:
        recommended_steps = cleaning_agent.get_recommended_cleaning_steps()
        if recommended_steps:
            print_subheader("Recommended Data Cleaning Steps:")
            print(recommended_steps)
        else:
            print("No recommended steps were generated.")
    except Exception as e:
        print(f"Error displaying recommended steps: {str(e)}")
    
    # Conclusion
    print_header("DEMONSTRATION COMPLETE")
    print_info("The AI Data Science Agent has successfully analyzed and cleaned the dataset!")
    print_info("This demonstrates how our agents can automate complex data preparation tasks.")
    print_info("The cleaned data is now ready for analysis and modeling.")

if __name__ == "__main__":
    run_presentation_demo() 