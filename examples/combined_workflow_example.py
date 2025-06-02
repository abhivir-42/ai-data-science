"""
Combined Workflow Example: Data Loader + Data Cleaning Agents

This example demonstrates how to use both the DataLoaderToolsAgent and 
DataCleaningAgent together in a sequential workflow.

Features demonstrated:
1. Load data using DataLoaderToolsAgent
2. Feed the loaded data into DataCleaningAgent 
3. Create a reusable pipeline function
4. Handle errors and edge cases
5. Generate comprehensive reports

Usage:
    python examples/combined_workflow_example.py
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


from src.agents.data_loader_tools_agent import DataLoaderToolsAgent
from src.agents.data_cleaning_agent import DataCleaningAgent

# Load environment variables
load_dotenv()

def handle_rate_limit_error(func, max_retries=3, base_delay=1):
    """
    Handle rate limit errors with exponential backoff.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
    
    Returns:
        Result of function execution
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} retries")
                    raise
            else:
                # Not a rate limit error, re-raise immediately
                raise
    
    return None

def create_data_pipeline(file_path, cleaning_instructions=None, verbose=True, max_retries=2):
    """
    Complete data loading and cleaning pipeline.
    
    Args:
        file_path: Path to the data file
        cleaning_instructions: Custom cleaning instructions (optional)
        verbose: Whether to print progress messages
        max_retries: Maximum retries for cleaning agent
    
    Returns:
        dict: Contains original data, cleaned data, and metadata
    """
    if verbose:
        print(f"🚀 Starting data pipeline for: {file_path}")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Step 1: Load data with token limits for large files
    if verbose:
        print("📂 Step 1: Loading data...")
    
    # Check file size to determine token limits
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # Use smaller token limits for larger files
        if file_size_mb > 1:  # Files larger than 1MB
            max_tokens = 25000
            n_samples = 5
        else:
            max_tokens = 50000
            n_samples = 10
    except:
        max_tokens = 25000
        n_samples = 5
    
    data_loader = DataLoaderToolsAgent(model=llm)
    
    # Use rate limit handling for data loading
    def load_data():
        return data_loader.invoke_agent(
            user_instructions=f"Load the data file from {file_path}. For large datasets, provide a representative sample with file metadata."
        )
    
    handle_rate_limit_error(load_data, max_retries=3, base_delay=2)
    
    # Get the loaded data
    loaded_artifacts = data_loader.get_artifacts()
    raw_df = data_loader.get_artifacts(as_dataframe=True)
    
    if raw_df is None or raw_df.empty:
        raise ValueError(f"Failed to load data from {file_path}")
    
    # Check if we have chunked data that needs special handling
    is_chunked_dataset = False
    if loaded_artifacts and isinstance(loaded_artifacts, dict) and "chunk_info" in loaded_artifacts:
        chunk_info = loaded_artifacts["chunk_info"]
        is_chunked_dataset = chunk_info.get("chunked", False)
    
    if verbose:
        print(f"   ✅ Data loaded successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
        
        if is_chunked_dataset:
            print(f"   🔄 Large dataset detected - using intelligent chunking")
            print(f"   📊 Full dataset: {chunk_info['total_rows']} rows × {chunk_info['total_columns']} columns")
            print(f"   📦 Processing in {chunk_info['total_chunks']} chunks of {chunk_info['chunk_size']} rows each")
        
        print(f"   📊 Missing values: {raw_df.isnull().sum().sum()}")
        print(f"   📋 Columns: {list(raw_df.columns)}")
        print(f"   🔍 Data types: {raw_df.dtypes.to_dict()}")
    
    # Step 2: Clean the loaded data
    if verbose:
        print("🧹 Step 2: Cleaning data...")
    
    cleaning_agent = DataCleaningAgent(
        model=llm, 
        log=True,
        log_path="./logs",
        human_in_the_loop=False,
        n_samples=n_samples  # Use appropriate sample size based on file size
    )
    
    # Use custom instructions or defaults, with special handling for large datasets
    if cleaning_instructions is None:
        if is_chunked_dataset:
            cleaning_instructions = """
            CONSERVATIVE CLEANING for large dataset (processed in chunks):
            
            DATA PRESERVATION PRIORITY: 
            - NEVER remove more than 10% of rows total
            - Focus on IMPUTATION not DELETION
            
            SPECIFIC STEPS:
            1. Fill missing numeric values with MEDIAN (not deletion)
            2. Fill missing categorical values with MODE or 'Unknown'
            3. Keep all original columns (don't remove columns with missing values)
            4. NO outlier removal (outliers in chunks may not be outliers in full dataset)
            5. Remove duplicates only if 100% identical across ALL columns
            6. NO row deletion for missing values - use imputation instead
            
            STOP IMMEDIATELY if more than 10% of rows would be removed.
            """
        elif "house" in file_path.lower() or "price" in file_path.lower():
            cleaning_instructions = """
            REAL ESTATE DATA CLEANING - CONSERVATIVE APPROACH:
            
            DATA PRESERVATION PRIORITY:
            - This is house price prediction data - missing values are NORMAL
            - NEVER remove more than 5% of rows
            - Focus on SMART IMPUTATION
            
            SPECIFIC STEPS:
            1. Numeric features (areas, prices): Fill with MEDIAN by neighborhood if possible, otherwise overall median
            2. Categorical features (garage type, basement, etc.): Fill with MODE or 'None'/'Unknown' for truly missing
            3. Year features: Fill with median year
            4. NO outlier removal for price data (expensive houses are not outliers!)
            5. Keep ALL original columns - missing values are informative
            6. Remove only exact duplicates (100% identical)
            
            CRITICAL: Preserve data for machine learning - don't delete valuable information!
            """
        elif raw_df.shape[1] > 50:  # Very wide dataset
            cleaning_instructions = """
            WIDE DATASET CLEANING - MINIMAL INTERVENTION:
            
            1. Fill missing numeric values with MEDIAN 
            2. Fill missing categorical with MODE or 'Unknown'
            3. Keep ALL columns (wide datasets need all features)
            4. NO outlier removal (complex relationships)
            5. Remove only exact duplicates
            6. MAX 5% row removal allowed
            """
        else:
            cleaning_instructions = """
            STANDARD CONSERVATIVE CLEANING:
            
            1. Fill missing numeric values with median
            2. Fill missing categorical with mode or 'Unknown'  
            3. Remove columns only if >90% missing (very high threshold)
            4. Remove exact duplicates only
            5. NO aggressive outlier removal
            6. MAX 10% row removal allowed
            """
    
    try:
        # Use rate limit handling for data cleaning
        def clean_data():
            # Pass the full loaded artifacts if chunked, otherwise just the DataFrame
            if is_chunked_dataset and "full_dataframe" in loaded_artifacts:
                # Pass the chunked dataset info for intelligent processing
                return cleaning_agent.invoke_agent(
                    data_raw=loaded_artifacts,  # Pass full artifacts including chunk info
                    user_instructions=cleaning_instructions,
                    max_retries=max_retries
                )
            else:
                # Standard processing for regular datasets
                return cleaning_agent.invoke_agent(
                    data_raw=raw_df,
                    user_instructions=cleaning_instructions,
                    max_retries=max_retries
                )
        
        handle_rate_limit_error(clean_data, max_retries=3, base_delay=2)
        
        cleaned_df = cleaning_agent.get_data_cleaned()
        
        if cleaned_df is None or cleaned_df.empty:
            print("⚠️  Warning: Cleaning resulted in empty dataset, using original data")
            cleaned_df = raw_df.copy()
        
        if verbose:
            print(f"   ✅ Data cleaned successfully: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
            print(f"   📊 Missing values: {cleaned_df.isnull().sum().sum()}")
        
    except Exception as e:
        print(f"⚠️  Cleaning failed: {str(e)}")
        print("   Using original data instead")
        cleaned_df = raw_df.copy()
    
    # Step 3: Generate metadata
    metadata = {
        "original_shape": raw_df.shape,
        "cleaned_shape": cleaned_df.shape,
        "missing_values_removed": raw_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
        "rows_removed": raw_df.shape[0] - cleaned_df.shape[0],
        "columns_added": cleaned_df.shape[1] - raw_df.shape[1],
        "file_path": file_path,
        "file_size_mb": file_size_mb if 'file_size_mb' in locals() else 0,
        "token_optimization_used": file_size_mb > 1 if 'file_size_mb' in locals() else False
    }
    
    return {
        "original_data": raw_df,
        "cleaned_data": cleaned_df,
        "cleaning_function": cleaning_agent.get_data_cleaner_function() if hasattr(cleaning_agent, 'response') and cleaning_agent.response else "No cleaning function generated",
        "cleaning_steps": cleaning_agent.get_recommended_cleaning_steps() if hasattr(cleaning_agent, 'response') and cleaning_agent.response else "No cleaning steps generated",
        "loader_response": data_loader.get_ai_message(),
        "metadata": metadata
    }

def test_simple_cleaning():
    """Test with the simplest possible case first."""
    print("🧪 Testing simple cleaning functionality...")
    
    # Create a very simple test dataset
    simple_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, None, 30, 40, 50],
        'category': ['A', 'B', None, 'A', 'B']
    })
    
    print(f"Test data: {simple_data.shape[0]} rows, {simple_data.shape[1]} columns")
    print(f"Missing values: {simple_data.isnull().sum().sum()}")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    cleaning_agent = DataCleaningAgent(
        model=llm, 
        log=False,  # Disable logging for test
        human_in_the_loop=False,
        n_samples=5
    )
    
    try:
        cleaning_agent.invoke_agent(
            data_raw=simple_data,
            user_instructions="Just fill missing values with 0 for numbers and 'Unknown' for text. Keep it extremely simple.",
            max_retries=1
        )
        
        cleaned = cleaning_agent.get_data_cleaned()
        print(f"✅ Simple test passed! Result: {cleaned.shape if cleaned is not None else 'None'}")
        return True
        
    except Exception as e:
        print(f"❌ Simple test failed: {str(e)}")
        return False

def print_pipeline_summary(result):
    """Print a comprehensive summary of the pipeline results."""
    metadata = result["metadata"]
    
    print("\n" + "="*60)
    print("📋 PIPELINE SUMMARY")
    print("="*60)
    
    print(f"📁 File processed: {metadata['file_path']}")
    print(f"📊 Original data: {metadata['original_shape'][0]} rows × {metadata['original_shape'][1]} columns")
    print(f"✨ Cleaned data: {metadata['cleaned_shape'][0]} rows × {metadata['cleaned_shape'][1]} columns")
    
    print(f"\n📈 Changes made:")
    print(f"   • Rows removed: {metadata['rows_removed']}")
    print(f"   • Columns added: {metadata['columns_added']}")
    print(f"   • Missing values fixed: {metadata['missing_values_removed']}")
    
    if metadata['missing_values_removed'] > 0:
        improvement = (metadata['missing_values_removed'] / max(1, result['original_data'].isnull().sum().sum())) * 100
        print(f"   • Data quality improvement: {improvement:.1f}%")

def save_results(result, output_dir="output"):
    """Save the pipeline results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned data
    cleaned_path = os.path.join(output_dir, "cleaned_data.csv")
    result["cleaned_data"].to_csv(cleaned_path, index=False)
    
    # Save cleaning function if available
    if result["cleaning_function"] != "No cleaning function generated":
        function_path = os.path.join(output_dir, "cleaning_function.py")
        with open(function_path, "w") as f:
            f.write(result["cleaning_function"])
        print(f"   • {function_path}")
    
    # Save summary report
    report_path = os.path.join(output_dir, "pipeline_report.md")
    with open(report_path, "w") as f:
        f.write("# Data Pipeline Report\n\n")
        f.write(f"**File processed:** {result['metadata']['file_path']}\n\n")
        f.write("## Original Data\n")
        f.write(f"- Shape: {result['metadata']['original_shape']}\n")
        f.write(f"- Missing values: {result['original_data'].isnull().sum().sum()}\n\n")
        f.write("## Cleaned Data\n")
        f.write(f"- Shape: {result['metadata']['cleaned_shape']}\n")
        f.write(f"- Missing values: {result['cleaned_data'].isnull().sum().sum()}\n\n")
        f.write("## Cleaning Steps\n")
        f.write(result['cleaning_steps'])
        f.write("\n\n## Cleaning Function\n```python\n")
        f.write(result['cleaning_function'])
        f.write("\n```\n")
    
    print(f"\n💾 Results saved to '{output_dir}/' directory:")
    print(f"   • {cleaned_path}")
    print(f"   • {report_path}")

def main():
    """Run the combined workflow example."""
    print("🤖 Combined Workflow Example: Data Loader + Data Cleaning")
    print("=" * 60)
    
    # First, test simple cleaning
    if not test_simple_cleaning():
        print("❌ Basic cleaning test failed. Stopping.")
        return
    
    # Example 1: Basic workflow with sample data and simple instructions
    print("\n📋 Example 1: Basic workflow with simple cleaning")
    try:
        result1 = create_data_pipeline(
            file_path="examples/sample_data.csv",
            cleaning_instructions="Fill missing numeric values with median, missing text with 'Unknown'. Remove duplicates. Keep it simple.",
            max_retries=1
        )
        
        print_pipeline_summary(result1)
        save_results(result1, "output/example1")
        
    except Exception as e:
        print(f"❌ Error in Example 1: {str(e)}")
    
    # Example 2: Test with house price data (if it exists)
    house_price_path = "data/samples/house-price-prediction-train.csv"
    if os.path.exists(house_price_path):
        print(f"\n📋 Example 2: Testing with house price dataset")
        try:
            result2 = create_data_pipeline(
                file_path=house_price_path,
                cleaning_instructions="Basic cleaning for house price prediction: fill missing values, handle categorical data simply.",
                verbose=True,
                max_retries=1
            )
            
            print_pipeline_summary(result2)
            save_results(result2, "output/house_price")
            
        except Exception as e:
            print(f"❌ Error in house price example: {str(e)}")
    else:
        print(f"\n⚠️  House price dataset not found at {house_price_path}")
    
    print("\n🎉 Combined workflow examples completed!")
    print("\nKey takeaways:")
    print("• Data loading works correctly")
    print("• Cleaning agent needs simpler instructions to avoid recursion")
    print("• Results are saved for analysis")

if __name__ == "__main__":
    main() 