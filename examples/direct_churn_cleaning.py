"""
Direct implementation of churn data cleaning using DataCleaningAgent.

This script:
1. Loads the churn_data.csv dataset
2. Directly uses the DataCleaningAgent to clean the data
3. Generates a detailed markdown report showing actual changes

Usage:
    python examples/direct_churn_cleaning.py
"""

import os
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ai_data_science.agents.data_cleaning_agent import DataCleaningAgent

# Load environment variables (for API keys)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Set to True to use mock data
USE_MOCK = not OPENAI_API_KEY

def analyze_data(df):
    """Generate a detailed analysis of the dataframe."""
    analysis = {}
    
    # Basic statistics
    analysis["rows"] = len(df)
    analysis["columns"] = len(df.columns)
    analysis["missing_values"] = df.isna().sum().sum()
    analysis["missing_percentage"] = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    # Column types
    analysis["dtypes"] = df.dtypes.astype(str).to_dict()
    
    # Missing values by column
    analysis["missing_by_column"] = df.isna().sum().to_dict()
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        analysis["numerical_stats"] = df[numerical_cols].describe().to_dict()
    
    # Categorical column analysis
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        analysis["categorical_columns"] = categorical_cols
        analysis["categorical_stats"] = {col: df[col].value_counts().to_dict() for col in categorical_cols if df[col].nunique() < 20}
    
    return analysis

def generate_report(original_df, cleaned_df, cleaning_steps, cleaning_function, analysis_original, analysis_cleaned, output_file="churn_data_cleaning_report.md"):
    """Generate a markdown report comparing the original and cleaned datasets."""
    now = datetime.datetime.now()
    
    with open(output_file, "w") as f:
        # Header
        f.write("# Churn Data Cleaning Report\n\n")
        f.write(f"**Generated on:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Original Data Summary
        f.write("## Original Dataset Summary\n\n")
        f.write(f"- **Number of records:** {analysis_original['rows']}\n")
        f.write(f"- **Number of columns:** {analysis_original['columns']}\n")
        f.write(f"- **Missing values:** {analysis_original['missing_values']} ({analysis_original['missing_percentage']:.2f}%)\n\n")
        
        f.write("### Column Datatypes\n\n")
        f.write("| Column | Type |\n")
        f.write("|--------|------|\n")
        for col, dtype in analysis_original["dtypes"].items():
            f.write(f"| {col} | {dtype} |\n")
        f.write("\n")
        
        f.write("### Missing Values by Column\n\n")
        f.write("| Column | Missing Values |\n")
        f.write("|--------|---------------|\n")
        for col, missing in analysis_original["missing_by_column"].items():
            if missing > 0:
                f.write(f"| {col} | {missing} |\n")
        f.write("\n")
        
        f.write("### First 5 rows of original data\n\n")
        f.write("```\n")
        f.write(original_df.head().to_string())
        f.write("\n```\n\n")
        
        # Cleaning Steps
        f.write("## Data Cleaning Process\n\n")
        f.write("### Recommended Cleaning Steps\n\n")
        f.write(cleaning_steps)
        f.write("\n\n")
        
        # Cleaning Function
        f.write("### Generated Cleaning Function\n\n")
        f.write("```python\n")
        f.write(cleaning_function)
        f.write("\n```\n\n")
        
        # Cleaned Data Summary
        f.write("## Cleaned Dataset Summary\n\n")
        f.write(f"- **Number of records:** {analysis_cleaned['rows']} ({analysis_cleaned['rows'] - analysis_original['rows']} record difference)\n")
        f.write(f"- **Number of columns:** {analysis_cleaned['columns']}\n")
        f.write(f"- **Missing values:** {analysis_cleaned['missing_values']} ({analysis_cleaned['missing_percentage']:.2f}%)\n\n")
        
        f.write("### First 5 rows of cleaned data\n\n")
        f.write("```\n")
        f.write(cleaned_df.head().to_string())
        f.write("\n```\n\n")
        
        # Comparison
        f.write("## Comparison: Original vs. Cleaned\n\n")
        
        # Records comparison
        record_diff = analysis_cleaned['rows'] - analysis_original['rows']
        record_percent = (record_diff / max(1, analysis_original['rows'])) * 100
        f.write(f"- **Records:** {record_diff} ({record_percent:.2f}% change)\n")
        
        # Missing values comparison
        missing_diff = analysis_cleaned['missing_values'] - analysis_original['missing_values']
        missing_percent = (missing_diff / max(1, analysis_original['missing_values'])) * 100
        f.write(f"- **Missing values:** {missing_diff} ({missing_percent:.2f}% change)\n\n")
        
        f.write("### Changes by Column\n\n")
        f.write("| Column | Original Missing | Cleaned Missing | Change |\n")
        f.write("|--------|-----------------|-----------------|--------|\n")
        
        for col in original_df.columns:
            if col in cleaned_df.columns:
                orig_missing = analysis_original["missing_by_column"].get(col, 0)
                clean_missing = analysis_cleaned["missing_by_column"].get(col, 0)
                diff = clean_missing - orig_missing
                f.write(f"| {col} | {orig_missing} | {clean_missing} | {diff} |\n")
            else:
                f.write(f"| {col} | {analysis_original['missing_by_column'].get(col, 0)} | Column removed | N/A |\n")
        
        for col in cleaned_df.columns:
            if col not in original_df.columns:
                f.write(f"| {col} | Column added | {analysis_cleaned['missing_by_column'].get(col, 0)} | N/A |\n")
                
        f.write("\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The data cleaning process successfully addressed the following issues in the churn dataset:\n\n")
        
        # Highlight major changes
        if record_diff != 0:
            f.write(f"- {abs(record_diff)} records were {'removed' if record_diff < 0 else 'added'}\n")
        
        if missing_diff != 0:
            f.write(f"- {abs(missing_diff)} missing values were {'removed' if missing_diff < 0 else 'added'}\n")
        
        # Look for columns with significant changes
        for col in original_df.columns:
            if col in cleaned_df.columns:
                orig_missing = analysis_original["missing_by_column"].get(col, 0)
                clean_missing = analysis_cleaned["missing_by_column"].get(col, 0)
                diff = clean_missing - orig_missing
                if abs(diff) > 0:  # Mention all changes
                    f.write(f"- Column '{col}': {abs(diff)} missing values were {'filled' if diff < 0 else 'introduced'}\n")
        
        # Check for removed or added columns
        removed_cols = [col for col in original_df.columns if col not in cleaned_df.columns]
        added_cols = [col for col in cleaned_df.columns if col not in original_df.columns]
        
        if removed_cols:
            f.write(f"- {len(removed_cols)} columns were removed: {', '.join(removed_cols)}\n")
        
        if added_cols:
            f.write(f"- {len(added_cols)} columns were added: {', '.join(added_cols)}\n")
            
        # Compare data types
        changed_types = []
        for col in original_df.columns:
            if col in cleaned_df.columns:
                orig_type = analysis_original["dtypes"].get(col)
                clean_type = analysis_cleaned["dtypes"].get(col)
                if orig_type != clean_type:
                    changed_types.append(f"{col} (from {orig_type} to {clean_type})")
        
        if changed_types:
            f.write(f"- Data types were changed for: {', '.join(changed_types)}\n")
    
    return output_file

def execute_data_cleaner_function(function_code, data_raw):
    """Execute the generated data cleaning function on the raw data."""
    # Define the function in the current namespace
    exec_globals = {}
    exec(function_code, exec_globals)
    
    # Get the function from the globals
    data_cleaner = exec_globals.get('data_cleaner')
    if data_cleaner is None:
        raise ValueError("Could not find 'data_cleaner' function in the generated code")
    
    # Execute the function on the raw data
    cleaned_data = data_cleaner(data_raw)
    return cleaned_data

def main():
    """Run the churn data cleaning process and generate a report."""
    print("🤖 Churn Data Cleaning using DataCleaningAgent")
    
    # Load the churn dataset
    data_path = os.path.join(os.getcwd(), "data", "churn_data.csv")
    print(f"📊 Loading dataset from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return
    
    # Analyze original data
    print("📝 Analyzing original dataset...")
    analysis_original = analyze_data(df)
    
    # Data cleaning instructions
    cleaning_instructions = """
    Clean this churn dataset by applying the following steps:
    1. Remove any duplicate customer records based on customer_id
    2. Fill missing numerical values in monthly_charges with the median (not mean) for more robustness
    3. Fill missing categorical values in paperless_billing with "Unknown"
    4. Convert total_charges to numeric, handling any non-numeric values
    5. Identify and remove outliers in monthly_charges (values outside 3 standard deviations)
    6. Create a new categorical column tenure_group that bins tenure into categories: 0-12 months, 13-24 months, 25-48 months, 49+ months
    7. Convert all categorical columns to category data type
    8. Create a binary churn indicator (1 for Yes, 0 for No)
    9. Ensure all customer IDs are valid and properly formatted
    """
    
    if USE_MOCK:
        print("⚠️ No OpenAI API key found. Creating a manual implementation instead.")
        
        # Manual data cleaning function to demonstrate actual cleaning
        cleaning_function = """
def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    
    # Make a copy of the dataframe
    df = data_raw.copy()
    
    # 1. Remove duplicate customer records
    df = df.drop_duplicates(subset=['customer_id'], keep='first')
    
    # 2. Fill missing numerical values
    # For monthly_charges, use median
    df['monthly_charges'] = df['monthly_charges'].fillna(df['monthly_charges'].median())
    
    # 3. Fill missing categorical values
    df['paperless_billing'] = df['paperless_billing'].fillna('Unknown')
    
    # 4. Convert total_charges to numeric
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())
    
    # 5. Remove outliers in monthly_charges (3 std)
    mean = df['monthly_charges'].mean()
    std = df['monthly_charges'].std()
    df = df[(df['monthly_charges'] >= mean - 3*std) & (df['monthly_charges'] <= mean + 3*std)]
    
    # 6. Create tenure_group column
    bins = [-1, 12, 24, 48, float('inf')]
    labels = ['0-12 months', '13-24 months', '25-48 months', '49+ months']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    
    # 7. Convert categorical columns to category type
    categorical_columns = ['gender', 'phone_service', 'multiple_lines', 'internet_service',
                         'online_security', 'online_backup', 'tech_support', 'streaming_tv',
                         'streaming_movies', 'contract', 'paperless_billing', 'payment_method',
                         'churn', 'tenure_group']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 8. Create binary churn indicator
    df['churn_binary'] = df['churn'].map({'Yes': 1, 'No': 0})
    
    # 9. Ensure customer IDs are valid and properly formatted
    # For this example, we'll just make sure they're all strings
    df['customer_id'] = df['customer_id'].astype(str)
    
    return df
        """
        cleaning_steps = """
1. Remove duplicate customer records based on customer_id
2. Fill missing numerical values in monthly_charges with the median
3. Fill missing categorical values in paperless_billing with "Unknown"
4. Convert total_charges to numeric, handling any non-numeric values
5. Identify and remove outliers in monthly_charges (values outside 3 standard deviations)
6. Create a new categorical column tenure_group that bins tenure into categories:
   - 0-12 months
   - 13-24 months
   - 25-48 months
   - 49+ months
7. Convert all categorical columns to category data type
8. Create a binary churn indicator (1 for Yes, 0 for No)
9. Ensure all customer IDs are valid and properly formatted
        """
        
        # Execute the cleaning function
        print("🧹 Performing data cleaning...")
        cleaned_df = execute_data_cleaner_function(cleaning_function, df)
    else:
        # Use the real DataCleaningAgent with the OpenAI API
        print("🧹 Creating and running DataCleaningAgent...")
        
        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=OPENAI_API_KEY)
        
        # Create the agent
        agent = DataCleaningAgent(
            model=llm,
            n_samples=20,  # Reduce sample size to avoid token limit issues
            log=True,
            log_path="./logs",
            human_in_the_loop=False
        )
        
        # Run the agent directly
        print("🧹 Running data cleaning process...")
        agent.invoke_agent(
            data_raw=df,
            user_instructions=cleaning_instructions
        )
        
        # Get the cleaned data and generated function
        cleaned_df = agent.get_data_cleaned()
        cleaning_function = agent.get_data_cleaner_function()
        cleaning_steps = agent.get_recommended_cleaning_steps()
        
        if cleaned_df is None:
            print("❌ Agent failed to clean the data.")
            return
    
    # Analyze cleaned data
    print("📝 Analyzing cleaned dataset...")
    analysis_cleaned = analyze_data(cleaned_df)
    
    # Check if any actual cleaning was performed
    if (analysis_original['rows'] == analysis_cleaned['rows'] and 
        analysis_original['missing_values'] == analysis_cleaned['missing_values'] and
        len(original_df.columns) == len(cleaned_df.columns)):
        print("⚠️ Warning: The data cleaning process did not appear to make any changes to the dataset.")
        
    # Generate report
    report_file = "direct_churn_data_cleaning_report.md"
    print(f"📊 Generating report: {report_file}")
    generate_report(df, cleaned_df, cleaning_steps, cleaning_function, 
                     analysis_original, analysis_cleaned, report_file)
    
    print(f"✅ Report generated: {report_file}")
    print("🎉 Process completed!")
    print(f"💾 The full report is available in the file: {report_file}")
    
    # Print a summary of the changes made
    print("\n📋 Summary of changes:")
    record_diff = analysis_cleaned['rows'] - analysis_original['rows']
    if record_diff != 0:
        print(f"  - Records: {analysis_original['rows']} → {analysis_cleaned['rows']} ({record_diff:+d})")
    
    missing_diff = analysis_cleaned['missing_values'] - analysis_original['missing_values']
    if missing_diff != 0:
        print(f"  - Missing values: {analysis_original['missing_values']} → {analysis_cleaned['missing_values']} ({missing_diff:+d})")
    
    columns_diff = len(cleaned_df.columns) - len(df.columns)
    if columns_diff != 0:
        print(f"  - Columns: {len(df.columns)} → {len(cleaned_df.columns)} ({columns_diff:+d})")
        added_cols = [col for col in cleaned_df.columns if col not in df.columns]
        if added_cols:
            print(f"    - Added columns: {', '.join(added_cols)}")

if __name__ == "__main__":
    main() 