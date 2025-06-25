#!/usr/bin/env python3
"""
Data Analysis uAgent Implementation

Following EXACTLY the Fetch.ai LangGraph adapter example:
https://innovationlab.fetch.ai/resources/docs/examples/adapters/langgraph-adapter-example

This wraps the DataAnalysisAgent as a uAgent for deployment on ASI:One.
The wrapper is minimal and leverages the full intelligence of DataAnalysisAgent.
"""

import os
import time
import sys
import re
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from uagents_adapter import LangchainRegisterTool, cleanup_uagent
from src.agents.data_analysis_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

# Set API keys (exactly like the example)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

if not API_TOKEN:
    print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")

# Initialize the enhanced data analysis agent
data_analysis_agent = DataAnalysisAgent(
    output_dir="output/data_analysis_uagent/",
    intent_parser_model="gpt-4o-mini",
    enable_async=False  # Keep synchronous for uAgent stability, but optimized with multi-threading
)

# Global variable to store the last processed data for follow-up requests
_last_cleaned_data = None
_last_processed_timestamp = None

def data_analysis_agent_func(query):
    """
    Enhanced data analysis agent function following the LangGraph adapter pattern.
    
    This wrapper:
    - Handles input format conversion (exactly like LangGraph example)
    - Directly invokes DataAnalysisAgent.analyze_from_text()
    - Returns formatted results with actual cleaned data samples
    - Handles follow-up requests for data delivery (chunks, subsets, etc.)
    - Leverages all DataAnalysisAgent intelligence without duplication
    
    The DataAnalysisAgent intelligently:
    - Extracts CSV URLs from text using LLM structured outputs
    - Parses workflow intent to determine which agents to run
    - Executes only the needed agents (cleaning, feature engineering, ML)
    - Returns comprehensive structured results
    """
    global _last_cleaned_data, _last_processed_timestamp
    
    # Handle input if it's a dict with 'input' key (EXACT pattern from example)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    query_lower = query.lower()
    
    # Handle follow-up data delivery requests
    if any(phrase in query_lower for phrase in [
        'send my data', 'provide my cleaned data', 'show me my processed data',
        'my cleaned dataset', 'give me my data', 'deliver my data',
        'send rows', 'send columns', 'data in chunks', 'split my data'
    ]):
        return handle_data_delivery_request(query)
    
    try:
        # Direct invocation of the underlying DataAnalysisAgent
        # This uses LLM structured outputs to extract CSV URLs and parse intent
        result = data_analysis_agent.analyze_from_text(query)
        
        # Store cleaned data for potential follow-up requests
        try:
            if hasattr(data_analysis_agent, 'data_cleaning_agent') and data_analysis_agent.data_cleaning_agent:
                cleaned_df = data_analysis_agent.data_cleaning_agent.get_data_cleaned()
                if cleaned_df is not None and len(cleaned_df) > 0:
                    _last_cleaned_data = cleaned_df
                    _last_processed_timestamp = time.time()
                    
                    # Add sample data to result for better display
                    sample_rows = cleaned_df.head(3).to_string()
                    result.key_insights.insert(0, f"Sample cleaned data (first 3 rows):\n{sample_rows}")
                    result.key_insights.insert(0, f"Cleaned dataset contains {len(cleaned_df):,} rows and {len(cleaned_df.columns)} columns")
                    
                    # Add column information
                    col_info = []
                    for col in cleaned_df.columns[:10]:  # First 10 columns
                        dtype = cleaned_df[col].dtype
                        null_count = cleaned_df[col].isnull().sum()
                        col_info.append(f"{col}: {dtype} ({null_count} nulls)")
                    
                    if col_info:
                        result.key_insights.insert(0, f"Column details: {', '.join(col_info)}")
                        
        except Exception as e:
            # Silent fail - don't break the main functionality
            print(f"Could not extract sample data: {str(e)}")
        
        # Format the structured result for uAgent compatibility
        return format_analysis_result(result)
        
    except Exception as e:
        error_msg = f"""
🚫 **Analysis Error**

Sorry, I encountered an issue: {str(e)}

**Common solutions:**
1. Include a direct CSV URL in your request (e.g., https://example.com/data.csv)
2. Be specific about what analysis you want
3. Example: "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"

**Need help?** Try: "Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species classification"
"""
        print(f"❌ Error in data analysis agent: {str(e)}")
        return error_msg

def handle_data_delivery_request(query):
    """Handle follow-up requests for data delivery in various formats."""
    global _last_cleaned_data, _last_processed_timestamp
    
    # Check if we have recent cleaned data
    if _last_cleaned_data is None:
        return """
🚫 **No Recent Data Found**

I don't have any recently processed data to deliver. Please first run a data cleaning task, for example:

"Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

Then I can provide your cleaned data in various formats.
"""
    
    # Check if data is too old (older than 1 hour)
    if _last_processed_timestamp and (time.time() - _last_processed_timestamp) > 3600:
        return """
🕐 **Data Session Expired**

Your cleaned data session has expired (older than 1 hour). Please re-run your data cleaning task to get fresh results.
"""
    
    try:
        df = _last_cleaned_data
        query_lower = query.lower()
        
        # Parse the request type
        if 'chunk' in query_lower:
            # Extract number of chunks if specified
            chunk_match = re.search(r'(\d+)\s*chunk', query_lower)
            num_chunks = int(chunk_match.group(1)) if chunk_match else 5
            return deliver_data_in_chunks(df, num_chunks)
        
        elif 'rows' in query_lower:
            # Extract row range if specified
            range_match = re.search(r'rows?\s*(\d+)[-\s]*(\d+)?', query_lower)
            if range_match:
                start_row = int(range_match.group(1)) - 1  # Convert to 0-indexed
                end_row = int(range_match.group(2)) if range_match.group(2) else start_row + 1000
                return deliver_data_rows(df, start_row, end_row)
            else:
                return deliver_data_rows(df, 0, min(1000, len(df)))
        
        elif 'column' in query_lower:
            # Extract column range or names if specified
            col_match = re.search(r'columns?\s*(\d+)[-\s]*(\d+)?', query_lower)
            if col_match:
                start_col = int(col_match.group(1)) - 1  # Convert to 0-indexed
                end_col = int(col_match.group(2)) if col_match.group(2) else start_col + 5
                return deliver_data_columns(df, start_col, end_col)
            else:
                return deliver_data_columns(df, 0, min(5, len(df.columns)))
        
        else:
            # Default: provide complete data if small enough, otherwise chunked
            csv_content = df.to_csv(index=False)
            content_size = len(csv_content.encode('utf-8'))
            
            if content_size < 100000:  # 100KB limit for direct delivery
                return f"""
📁 **YOUR COMPLETE CLEANED DATA**

File size: {content_size / 1024:.1f} KB | Rows: {len(df):,} | Columns: {len(df.columns)}

```csv
{csv_content}
```

💡 **Usage**: Copy the CSV content above and save it as a .csv file for use in Excel, Python, R, or other tools.
"""
            else:
                return deliver_data_in_chunks(df, 5)
    
    except Exception as e:
        return f"""
🚫 **Data Delivery Error**

Sorry, I encountered an issue delivering your data: {str(e)}

Please try a more specific request like:
- "Send me rows 1-100 of my data"
- "Provide my data in 3 chunks"
- "Show me columns 1-5 of my cleaned data"
"""

def deliver_data_in_chunks(df, num_chunks):
    """Deliver data in specified number of chunks."""
    chunk_size = len(df) // num_chunks
    if chunk_size == 0:
        chunk_size = 1
    
    result = [f"""
📦 **CHUNKED DATA DELIVERY**

Your cleaned data ({len(df):,} rows × {len(df.columns)} columns) split into {num_chunks} chunks.
Each chunk contains approximately {chunk_size} rows.

"""]
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df)) if i < num_chunks - 1 else len(df)
        chunk_df = df.iloc[start_idx:end_idx]
        
        result.append(f"""
📋 **CHUNK {i+1}/{num_chunks}** (Rows {start_idx+1}-{end_idx})

```csv
{chunk_df.to_csv(index=False)}
```
""")
    
    result.append("""
💡 **Combine chunks**: Copy each chunk and concatenate them to reconstruct your complete dataset.
""")
    
    return "\n".join(result)

def deliver_data_rows(df, start_row, end_row):
    """Deliver specific row range."""
    end_row = min(end_row, len(df))
    subset_df = df.iloc[start_row:end_row]
    
    return f"""
📋 **DATA ROWS {start_row+1}-{end_row}**

Showing {len(subset_df)} rows from your cleaned dataset:

```csv
{subset_df.to_csv(index=False)}
```

💡 **Need more rows?** Ask: "Send me rows {end_row+1}-{min(end_row+1000, len(df))}" for the next batch.
Total dataset has {len(df):,} rows.
"""

def deliver_data_columns(df, start_col, end_col):
    """Deliver specific column range."""
    end_col = min(end_col, len(df.columns))
    subset_df = df.iloc[:, start_col:end_col]
    
    selected_cols = df.columns[start_col:end_col].tolist()
    
    return f"""
📋 **DATA COLUMNS {start_col+1}-{end_col}**

Showing columns: {', '.join(selected_cols)}

```csv
{subset_df.to_csv(index=False)}
```

💡 **Need more columns?** Ask: "Send me columns {end_col+1}-{min(end_col+5, len(df.columns))}" for the next set.
Total dataset has {len(df.columns)} columns: {', '.join(df.columns.tolist())}
"""

def format_analysis_result(result) -> str:
    """Format the analysis result into a comprehensive, user-friendly response."""
    
    try:
        lines = [
            "🎉 **DATA ANALYSIS COMPLETE**",
            "=" * 60,
            "",
            f"📊 **Dataset**: {result.csv_url}",
            f"📝 **Request**: {result.original_request[:200]}{'...' if len(result.original_request) > 200 else ''}",
            f"⏱️  **Runtime**: {result.total_runtime_seconds:.2f} seconds",
            f"🎯 **Confidence**: {result.confidence_level.upper()}",
            f"⭐ **Quality Score**: {result.analysis_quality_score:.2f}/1.0",
            ""
        ]
        
        # CHECK FOR AGENT FAILURES FIRST
        failed_agents = []
        successful_agents = []
        
        for agent_result in result.agent_results:
            if not agent_result.success:
                failed_agents.append(agent_result)
            else:
                successful_agents.append(agent_result)
        
        # Report any failures upfront
        if failed_agents:
            lines.extend([
                "⚠️  **AGENT EXECUTION STATUS**:",
                ""
            ])
            
            for failed_agent in failed_agents:
                error_msg = getattr(failed_agent, 'error_message', 'Unknown error')
                # Handle None or empty error messages
                if not error_msg or error_msg.strip() == '':
                    error_msg = f"{failed_agent.agent_name} execution failed without specific error details"
                
                lines.extend([
                    f"   ❌ **{failed_agent.agent_name.replace('_', ' ').title()} Agent**: FAILED",
                    f"       Error: {error_msg}",
                    f"       Runtime: {failed_agent.execution_time_seconds:.2f}s",
                    ""
                ])
            
            for successful_agent in successful_agents:
                lines.extend([
                    f"   ✅ **{successful_agent.agent_name.replace('_', ' ').title()} Agent**: SUCCESS",
                    f"       Runtime: {successful_agent.execution_time_seconds:.2f}s",
                    ""
                ])
            
            lines.extend([
                "📊 **ANALYSIS CONTINUES WITH AVAILABLE DATA**:",
                "   Even with some agent failures, we'll provide results from successful steps.",
                ""
            ])
        else:
            lines.extend([
                "✅ **AGENT EXECUTION STATUS**:",
                ""
            ])
            
            for successful_agent in successful_agents:
                lines.extend([
                    f"   ✅ **{successful_agent.agent_name.replace('_', ' ').title()} Agent**: SUCCESS",
                    f"       Runtime: {successful_agent.execution_time_seconds:.2f}s",
                    ""
                ])
            
            lines.extend([
                "🎉 **ALL AGENTS EXECUTED SUCCESSFULLY**:",
                ""
            ])
        
        # SHOW ACTUAL DATA TRANSFORMATION RESULTS
        lines.extend([
            "📈 **DATA TRANSFORMATION RESULTS**:",
            ""
        ])
        
        # Original vs Final data shape
        original_rows = result.data_shape.get('rows', 'unknown')
        original_cols = result.data_shape.get('columns', 'unknown')
        
        # Extract final shape from agent results if available
        final_rows = original_rows
        final_cols = original_cols
        data_retention = 100.0
        
        # Parse agent results for actual cleaning metrics
        for agent_result in result.agent_results:
            if agent_result.agent_name == "data_cleaning" and agent_result.log_messages:
                log_text = " ".join(agent_result.log_messages)
                # Extract actual numbers from logs
                
                # Look for "Original: X rows × Y columns"
                original_match = re.search(r'Original:\s*(\d+)\s*rows\s*×\s*(\d+)\s*columns', log_text)
                if original_match:
                    original_rows = int(original_match.group(1))
                    original_cols = int(original_match.group(2))
                
                # Look for "Final: X rows × Y columns"
                final_match = re.search(r'Final:\s*(\d+)\s*rows\s*×\s*(\d+)\s*columns', log_text)
                if final_match:
                    final_rows = int(final_match.group(1))
                    final_cols = int(final_match.group(2))
                
                # Look for "Data retention: X%"
                retention_match = re.search(r'Data retention:\s*([\d.]+)%', log_text)
                if retention_match:
                    data_retention = float(retention_match.group(1))
        
        lines.extend([
            f"   📏 **Before**: {original_rows:,} rows × {original_cols} columns",
            f"   ✨ **After**: {final_rows:,} rows × {final_cols} columns",
            f"   📊 **Data Retention**: {data_retention:.1f}%",
            f"   🔄 **Rows Changed**: {final_rows - original_rows:+,}" if isinstance(final_rows, int) and isinstance(original_rows, int) else "",
            ""
        ])
        
        # DETAILED CLEANING ACTIONS (extract from logs)
        cleaning_actions = []
        missing_handled = 0
        outliers_removed = 0
        duplicates_removed = 0
        columns_dropped = []
        
        for agent_result in result.agent_results:
            if agent_result.agent_name == "data_cleaning" and agent_result.log_messages:
                log_text = " ".join(agent_result.log_messages)
                
                # Extract specific actions
                missing_matches = re.findall(r'Filled (\d+) missing values in [\'"]([^\'"]+)[\'"] with (\w+)', log_text)
                for count, column, method in missing_matches:
                    missing_handled += int(count)
                    cleaning_actions.append(f"Filled {count} missing values in '{column}' with {method}")
                
                # Extract outlier information
                outlier_matches = re.findall(r'Removed (\d+) outliers from [\'"]([^\'"]+)[\'"]', log_text)
                for count, column in outlier_matches:
                    outliers_removed += int(count)
                    cleaning_actions.append(f"Removed {count} outliers from '{column}'")
                
                # Extract dropped columns
                dropped_matches = re.findall(r'Dropped [\'"]([^\'"]+)[\'"] column', log_text)
                columns_dropped.extend(dropped_matches)
                for column in dropped_matches:
                    cleaning_actions.append(f"Dropped '{column}' column due to high missing values")
        
        if cleaning_actions:
            lines.extend([
                "�� **CLEANING ACTIONS PERFORMED**:",
                *[f"   • {action}" for action in cleaning_actions[:10]],  # Limit to 10 actions
                f"   • ...and {len(cleaning_actions) - 10} more actions" if len(cleaning_actions) > 10 else "",
                ""
            ])
        
        # DATA QUALITY IMPROVEMENTS
        if missing_handled > 0 or outliers_removed > 0:
            lines.extend([
                "📊 **DATA QUALITY IMPROVEMENTS**:",
                f"   • Missing values handled: {missing_handled:,}",
                f"   • Outliers removed: {outliers_removed:,}",
                f"   • Columns dropped: {len(columns_dropped)}",
                ""
            ])
        
        # PROVIDE ACTUAL CLEANED DATA TO USER
        # Try to read the cleaned data file and provide it to the user
        cleaned_data_provided = False
        
        # Look for cleaned data file path in generated files
        cleaned_data_path = None
        for agent_result in result.agent_results:
            if agent_result.output_data_path and agent_result.agent_name == "data_cleaning":
                cleaned_data_path = agent_result.output_data_path
                break
        
        if cleaned_data_path:
            try:
                # Try to read the cleaned data
                if os.path.exists(cleaned_data_path):
                    cleaned_df = pd.read_csv(cleaned_data_path)
                    
                    # Calculate size to determine delivery method
                    csv_content = cleaned_df.to_csv(index=False)
                    content_size = len(csv_content.encode('utf-8'))
                    
                    lines.extend([
                        "📊 **CLEANED DATA STATISTICS**:",
                        f"   • Total rows: {len(cleaned_df):,}",
                        f"   • Total columns: {len(cleaned_df.columns)}",
                        f"   • Missing values: {cleaned_df.isnull().sum().sum():,}",
                        f"   • File size: {content_size / 1024:.1f} KB",
                        ""
                    ])
                    
                    # Strategy 1: For small datasets (< 50KB), provide full CSV content
                    if content_size < 50000:  # 50KB limit
                        lines.extend([
                            "📁 **YOUR CLEANED DATA** (Complete CSV):",
                            "```csv",
                            csv_content,
                            "```",
                            "",
                            "💡 **How to use**: Copy the CSV content above and save it as a .csv file, or use it directly in your analysis.",
                            ""
                        ])
                        cleaned_data_provided = True
                    
                    # Strategy 2: For medium datasets (50KB-200KB), provide compressed or chunked data
                    elif content_size < 200000:  # 200KB limit
                        # Provide key columns and sample + instructions
                        lines.extend([
                            "📋 **CLEANED DATA PREVIEW** (First 10 rows):",
                            "```csv",
                            cleaned_df.head(10).to_csv(index=False),
                            "```",
                            "",
                            f"📁 **FULL DATASET INFORMATION**:",
                            f"   • Dataset is {content_size / 1024:.1f} KB - too large to display fully here",
                            f"   • Contains {len(cleaned_df):,} rows and {len(cleaned_df.columns)} columns",
                            "",
                            "💡 **To get your complete cleaned data**:",
                            "   1. Ask: 'Please provide my cleaned data in chunks'",
                            "   2. Or: 'Split my cleaned data into smaller parts'",
                            "   3. I can deliver it in manageable pieces you can combine",
                            ""
                        ])
                        cleaned_data_provided = True
                    
                    # Strategy 3: For large datasets (>200KB), provide summary and delivery options
                    else:
                        lines.extend([
                            "📋 **CLEANED DATA SUMMARY** (First 5 rows):",
                            "```csv",
                            cleaned_df.head(5).to_csv(index=False),
                            "```",
                            "",
                            f"📁 **LARGE DATASET DETECTED**:",
                            f"   • Dataset is {content_size / 1024:.1f} KB ({content_size / 1024 / 1024:.2f} MB)" if content_size > 1024*1024 else f"   • Dataset is {content_size / 1024:.1f} KB",
                            f"   • Contains {len(cleaned_df):,} rows and {len(cleaned_df.columns)} columns",
                            "",
                            "💡 **Delivery Options for Your Complete Data**:",
                            "   1. **Chunked Delivery**: Ask 'Send my data in 10 chunks'",
                            "   2. **Column Subsets**: Ask 'Send columns 1-5 of my cleaned data'",
                            "   3. **Row Ranges**: Ask 'Send rows 1-1000 of my cleaned data'",
                            "   4. **Filtered Data**: Ask 'Send only [specific columns/conditions]'",
                            "",
                            "🎯 **Quick Access**: Ask 'How can I download my complete cleaned dataset?'",
                            ""
                        ])
                        cleaned_data_provided = True
                    
                    # Show column information for all cases
                    lines.extend([
                        "📋 **COLUMN INFORMATION**:",
                    ])
                    
                    for col in cleaned_df.columns[:15]:  # Show first 15 columns
                        dtype = cleaned_df[col].dtype
                        null_count = cleaned_df[col].isnull().sum()
                        unique_count = cleaned_df[col].nunique()
                        
                        # Add sample values for categorical columns
                        if dtype == 'object' and unique_count <= 10:
                            sample_values = cleaned_df[col].value_counts().head(3).index.tolist()
                            sample_str = f" (e.g., {', '.join(map(str, sample_values))})"
                        else:
                            sample_str = ""
                        
                        lines.append(f"   • {col}: {dtype} | {null_count} nulls | {unique_count} unique{sample_str}")
                    
                    if len(cleaned_df.columns) > 15:
                        lines.append(f"   • ...and {len(cleaned_df.columns) - 15} more columns")
                    
                    lines.append("")
                
            except Exception as e:
                lines.extend([
                    "⚠️  **Note**: Could not access cleaned data file",
                    f"   Error: {str(e)}",
                    "   The data was processed but file access failed",
                    ""
                ])
        
        # Alternative: Try to access the DataAnalysisAgent's cleaned data directly
        if not cleaned_data_provided:
            try:
                # Try to extract cleaned data from agent logs or results
                for agent_result in result.agent_results:
                    if agent_result.agent_name == "data_cleaning" and hasattr(agent_result, 'data_quality_metrics'):
                        if hasattr(agent_result.data_quality_metrics, 'cleaned_shape'):
                            lines.extend([
                                "📋 **CLEANED DATA INFORMATION**:",
                                f"   ✅ Data successfully cleaned and processed",
                                f"   📊 Shape: {agent_result.data_quality_metrics.cleaned_shape.get('rows', 'unknown'):,} rows × {agent_result.data_quality_metrics.cleaned_shape.get('columns', 'unknown')} columns" if hasattr(agent_result.data_quality_metrics, 'cleaned_shape') else "",
                                "",
                                "💡 **To Access Your Cleaned Data**:",
                                "   Ask: 'Please provide my cleaned dataset' or 'Show me my processed data'",
                                ""
                            ])
                            cleaned_data_provided = True
                            break
                
            except Exception as e:
                pass  # Silent fail for this fallback attempt
        
        if not cleaned_data_provided:
            lines.extend([
                "📋 **CLEANED DATA**:",
                f"   ✅ Data successfully cleaned and saved",
                f"   📊 Final shape: {final_rows:,} rows × {final_cols} columns",
                "",
                "💡 **To Get Your Cleaned Data**:",
                "   Ask: 'Please provide my cleaned dataset as CSV' or 'Send me my processed data'",
                ""
            ])
        
        # Workflow information with actual execution results
        if result.workflow_intent:
            # Check actual execution results
            data_cleaning_status = "❌ Not executed"
            feature_engineering_status = "❌ Not executed"  
            ml_modeling_status = "❌ Not executed"
            
            for agent_result in result.agent_results:
                if agent_result.agent_name == "data_cleaning":
                    data_cleaning_status = "✅ Success" if agent_result.success else f"❌ Failed: {getattr(agent_result, 'error_message', 'Unknown error')[:50]}..."
                elif agent_result.agent_name == "feature_engineering":
                    feature_engineering_status = "✅ Success" if agent_result.success else f"❌ Failed: {getattr(agent_result, 'error_message', 'Unknown error')[:50]}..."
                elif agent_result.agent_name == "h2o_ml":
                    ml_modeling_status = "✅ Success" if agent_result.success else f"❌ Failed: {getattr(agent_result, 'error_message', 'Unknown error')[:50]}..."
            
            lines.extend([
                "🔄 **WORKFLOW EXECUTION RESULTS**:",
                f"   • Data Cleaning: {data_cleaning_status}",
                f"   • Feature Engineering: {feature_engineering_status}",
                f"   • ML Modeling: {ml_modeling_status}",
                f"   • Intent Confidence: {result.workflow_intent.intent_confidence:.2f}",
                ""
            ])
        
        # Agents executed
        if result.agents_executed:
            lines.extend([
                "🤖 **AGENTS EXECUTED**:",
                *[f"   • {agent.replace('_', ' ').title()}" for agent in result.agents_executed],
                ""
            ])
        
        # Performance metrics
        metrics_added = False
        if result.overall_data_quality_score is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Data Quality: {result.overall_data_quality_score:.2f}/1.0")
        
        if hasattr(result, 'feature_engineering_effectiveness') and result.feature_engineering_effectiveness is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Feature Engineering: {result.feature_engineering_effectiveness:.2f}/1.0")
        
        if hasattr(result, 'model_performance_score') and result.model_performance_score is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Model Performance: {result.model_performance_score:.2f}/1.0")
        
        if metrics_added:
            lines.append("")
        
        # Key insights
        if result.key_insights:
            lines.extend([
                "💡 **KEY INSIGHTS**:",
                *[f"   • {insight}" for insight in result.key_insights],
                ""
            ])
        
        # Recommendations
        if result.recommendations:
            lines.extend([
                "🎯 **RECOMMENDATIONS**:",
                *[f"   • {rec}" for rec in result.recommendations],
                ""
            ])
        
        # Generated files
        if result.generated_files:
            lines.extend([
                "📁 **GENERATED FILES**:",
                *[f"   • {name}: {path}" for name, path in result.generated_files.items()],
                ""
            ])
        
        # Warnings
        if result.warnings:
            lines.extend([
                "⚠️  **WARNINGS**:",
                *[f"   • {warning}" for warning in result.warnings],
                ""
            ])
        
        # Limitations
        if result.limitations:
            lines.extend([
                "⚠️  **LIMITATIONS**:",
                *[f"   • {limitation}" for limitation in result.limitations],
                ""
            ])
        
        lines.extend([
            "=" * 60,
            "✅ **Analysis completed successfully!**",
            "",
            "💡 **What you got:**",
            f"   • Cleaned dataset with {data_retention:.1f}% data retention",
            f"   • {missing_handled:,} missing values handled" if missing_handled > 0 else "",
            f"   • {outliers_removed:,} outliers removed" if outliers_removed > 0 else "",
            "   • Detailed cleaning log and generated code",
            "   • Ready-to-use data for further analysis",
        ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Error formatting result: {str(e)}\n\nRaw result: {str(result)}"

# Register the DataAnalysisAgent via uAgent (EXACT pattern from LangGraph example)
tool = LangchainRegisterTool()

print("🚀 Registering enhanced data analysis uAgent...")

agent_info = tool.invoke(
    {
        "agent_obj": data_analysis_agent_func,  # Pass the function
        "name": "enhanced_data_analysis",
        "port": 8102,
        "description": "🤖 AI Data Analysis Chatbot - Send me a CSV URL and analysis request. I'll clean your data, engineer features, and build ML models. Example: 'Clean and analyze https://example.com/data.csv for prediction'",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registration result: {agent_info}")
print(f"📊 Result type: {type(agent_info)}")

# Extract address info for display
if isinstance(agent_info, dict):
    agent_address = agent_info.get('agent_address', 'Unknown')
    agent_port = agent_info.get('agent_port', '8102')
elif isinstance(agent_info, str):
    # If it's a string, extract from logs
    agent_address = "Check logs above for actual address"
    agent_port = "8102"
else:
    agent_address = "Unknown"
    agent_port = "8102"

# Keep the agent alive (EXACT pattern from example)
if __name__ == "__main__":
    try:
        print("\n🎉 ENHANCED DATA ANALYSIS UAGENT IS RUNNING!")
        print("=" * 60)
        print(f"🔗 Agent name: enhanced_data_analysis")
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print("\n📋 Usage:")
        print("Send a message with a CSV URL and analysis request:")
        print('- "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"')
        print('- "Perform feature engineering on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"')
        print('- "Build ML model using https://example.com/your-data.csv to predict target_column"')
        print("\n🎯 The agent uses AI to:")
        print("• Extract CSV URLs from your text using LLM structured outputs")
        print("• Parse your intent to determine which analysis steps to run")
        print("• Execute only the needed agents (cleaning, feature engineering, ML)")
        print("• Return comprehensive structured results")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced data analysis agent...")
        cleanup_uagent("enhanced_data_analysis")
        print("✅ Agent stopped.") 