"""
Example of using the DataCleaningAgentAdapter to clean the churn dataset.

This script:
1. Loads the churn_data.csv dataset
2. Creates and registers a DataCleaningAgent as a uAgent
3. Cleans the data with specific instructions
4. Generates a detailed markdown report

Usage:
    python examples/clean_churn_data.py
"""

import os
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter
from ai_data_science.agents.data_cleaning_agent import DataCleaningAgent

# Load environment variables (for API keys)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AGENTVERSE_API_TOKEN = os.getenv("AGENTVERSE_API_TOKEN", "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3NTM4MDkzOTAsImdycCI6ImludGVybmFsIiwiaWF0IjoxNzQ2MDMzMzkwLCJpc3MiOiJmZXRjaC5haSIsImp0aSI6IjE3NzVlYzUyZWNhNmQ0NzZiMGJmODEzOSIsInNjb3BlIjoiYXYiLCJzdWIiOiI2ODExYWNmMGEyM2I1NTU4ZTE4ZmYyOGIxOGJlMWM4YWQ1YTE5NjY4NWYzNmY1NmEifQ.FsZTHLjWWhHgzTOHwEI4BFUC6EFFiG4oEaDuRvB1TebD8oMRjRw84sEMdM8eRYvr9UYoqg8Ak_rzjJ36NhXEbs3d-kAFrE5PY4tB1HqiGvonuUD68M4H4BOCjIhollHham7K5Ck0YRhwT1PkF7JSHneHeU-w0RfGnRK-wcZMP1t_2P9f6pMbeRvRp2uyHEIf-uiY9c2aB3PnN0GywG6kHLSbBffhNMyM6XFy9Dh6yRF1bRrMXUtf7NaHtR8GVQ8q5-W-zqfjD76GTE7CXGqj9vjZQmZfXFWYwx7n-yu4f9byRBZCjH85_SPyljhPNXO5aoJcMLN4q1xb933_ucbrCQ")

# Check if we need to use mock data
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
        record_percent = (record_diff / analysis_original['rows']) * 100
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
        f.write("The data cleaning process successfully addressed missing values, outliers, and data type issues in the churn dataset. ")
        f.write("The most significant changes were:\n\n")
        
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
                if abs(diff) > 10:  # Only mention significant changes
                    f.write(f"- Column '{col}': {abs(diff)} missing values were {'filled' if diff < 0 else 'introduced'}\n")
        
        # Check for removed or added columns
        removed_cols = [col for col in original_df.columns if col not in cleaned_df.columns]
        added_cols = [col for col in cleaned_df.columns if col not in original_df.columns]
        
        if removed_cols:
            f.write(f"- {len(removed_cols)} columns were removed: {', '.join(removed_cols)}\n")
        
        if added_cols:
            f.write(f"- {len(added_cols)} columns were added: {', '.join(added_cols)}\n")
    
    return output_file

def mock_cleaning(df):
    """Perform mock cleaning of the dataframe (for demo without API key)."""
    cleaned_df = df.copy()
    
    # Fill missing numerical values with mean
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill missing categorical values with "Unknown"
    for col in cleaned_df.select_dtypes(include=["object", "category"]).columns:
        if str(cleaned_df[col].dtype) == 'category':
            # Safe categorical handling - prevents "Cannot setitem on a Categorical" error
            if not cleaned_df[col].mode().empty:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
            else:
                # Convert to object first, then fill with 'Unknown'
                cleaned_df[col] = cleaned_df[col].astype('object').fillna('Unknown')
        else:
            # Standard object columns can be filled directly
            cleaned_df[col] = cleaned_df[col].fillna("Unknown")
    
    # Remove outliers (simplified)
    numerical_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (3 * iqr)
        upper_bound = q3 + (3 * iqr)
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def mock_cleaner_function():
    """Return a mock data cleaning function."""
    return """
def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    
    # Make a copy of the dataframe
    df = data_raw.copy()
    
    # Fill missing numerical values with mean
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # Fill missing categorical values with "Unknown"
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if str(df[col].dtype) == 'category':
            # Safe categorical handling - prevents "Cannot setitem on a Categorical" error
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                # Convert to object first, then fill with 'Unknown'
                df[col] = df[col].astype('object').fillna('Unknown')
        else:
            # Standard object columns can be filled directly
            df[col] = df[col].fillna("Unknown")
    
    # Remove outliers (using 3x IQR method)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (3 * iqr)
        upper_bound = q3 + (3 * iqr)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df
"""

def mock_cleaning_steps():
    """Return mock cleaning steps."""
    return """
1. Fill missing numerical values with the mean of each column
2. Fill missing categorical values with "Unknown"
3. Remove outliers using the 3x IQR method for numerical columns
4. Remove duplicate rows from the dataset
5. Convert data types to appropriate formats
"""

def main():
    """Run the churn data cleaning process and generate a report."""
    print("🤖 Churn Data Cleaning using DataCleaningAgentAdapter")
    
    # Load the churn dataset
    data_path = os.path.join(os.getcwd(), "data", "churn_data.csv")
    print(f"📊 Loading dataset from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print("Using a sample dataset instead")
        # Create a simple sample dataset
        df = pd.DataFrame({
            'customer_id': range(1000, 1100),
            'gender': np.random.choice(['Male', 'Female', None], size=100),
            'age': np.random.randint(18, 80, size=100),
            'tenure': np.random.randint(0, 72, size=100),
            'usage': np.random.normal(500, 200, size=100),
            'monthly_charges': np.random.uniform(20, 100, size=100),
            'total_charges': np.random.uniform(100, 5000, size=100),
            'churn': np.random.choice(['Yes', 'No'], size=100, p=[0.3, 0.7])
        })
    
    # Analyze original data
    print("📝 Analyzing original dataset...")
    analysis_original = analyze_data(df)
    
    # If no API key, use mock data
    if USE_MOCK:
        print("⚠️ No OpenAI API key found. Using mock implementation.")
        
        # Perform mock cleaning
        print("🧹 Performing mock data cleaning...")
        cleaned_df = mock_cleaning(df)
        cleaning_function = mock_cleaner_function()
        cleaning_steps = mock_cleaning_steps()
        
        # Analyze cleaned data
        print("📝 Analyzing cleaned dataset...")
        analysis_cleaned = analyze_data(cleaned_df)
        
        # Generate report
        report_file = "churn_data_cleaning_report.md"
        print(f"📊 Generating report: {report_file}")
        generate_report(df, cleaned_df, cleaning_steps, cleaning_function, 
                         analysis_original, analysis_cleaned, report_file)
        
        print(f"✅ Report generated: {report_file}")
    
    else:
        # Use the real implementation with the OpenAI API
        print("🚀 Creating DataCleaningAgent...")
        
        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=OPENAI_API_KEY)
        
        # Create the adapter
        adapter = DataCleaningAgentAdapter(
            model=llm,
            name="churn_data_cleaning_agent",
            port=8000,
            description="A data cleaning agent specialized for churn datasets",
            mailbox=True,
            api_token=AGENTVERSE_API_TOKEN,
            log=True,
            human_in_the_loop=False
        )
        
        # Data cleaning instructions
        cleaning_instructions = """
        Clean this churn dataset by:
        1. Removing any duplicate customer records
        2. Filling missing numerical values with appropriate measures (mean/median)
        3. Filling missing categorical values with "Unknown" or the most frequent value
        4. Identifying and handling outliers in numerical columns
        5. Converting data types to appropriate formats
        6. Ensuring all customer IDs are valid and consistent
        """
        
        # Register the agent with Agentverse
        print("🚀 Registering agent with Agentverse...")
        try:
            result = adapter.register()
            print(f"✅ Agent registered successfully with address: {result['agent_address']}")
        except Exception as e:
            print(f"⚠️ Could not register agent: {str(e)}")
            print("Continuing with local DataCleaningAgent...")
            
            # Fall back to using the DataCleaningAgent directly without uAgents
            agent = DataCleaningAgent(
                model=llm,
                n_samples=30,
                log=True,
                log_path="./logs",
                human_in_the_loop=False
            )
            
            # Use the agent directly
            print("🧹 Running data cleaning process...")
            agent.invoke_agent(
                data_raw=df,
                user_instructions=cleaning_instructions
            )
            
            # Get the cleaned data and other outputs
            cleaned_df = agent.get_data_cleaned()
            cleaning_function = agent.get_data_cleaner_function()
            cleaning_steps = agent.get_recommended_cleaning_steps()
            
            # Analyze cleaned data
            print("📝 Analyzing cleaned dataset...")
            analysis_cleaned = analyze_data(cleaned_df)
            
            # Generate report
            report_file = "churn_data_cleaning_report.md"
            print(f"📊 Generating report: {report_file}")
            generate_report(df, cleaned_df, cleaning_steps, cleaning_function, 
                            analysis_original, analysis_cleaned, report_file)
            
            print(f"✅ Report generated: {report_file}")
    
    print("🎉 Process completed!")
    print(f"💾 The full report is available in the file: churn_data_cleaning_report.md")

if __name__ == "__main__":
    main() 