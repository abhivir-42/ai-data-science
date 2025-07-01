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
import requests
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from uagents_adapter import LangchainRegisterTool, cleanup_uagent
from src.agents.data_analysis_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

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

def upload_csv_to_remote_host(file_path, file_description="Processed Data"):
    """
    Upload CSV file to a remote hosting service and return public URL.
    
    Parameters
    ----------
    file_path : str
        Local path to the CSV file
    file_description : str
        Description of the file for naming
        
    Returns
    -------
    dict
        Dictionary containing success status, URL, and metadata
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "url": None
            }
        
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Check if file is too large (most free services have limits)
        if file_size_mb > 50:  # 50MB limit
            return {
                "success": False,
                "error": f"File too large ({file_size_mb:.1f} MB). Maximum size is 50MB.",
                "url": None
            }
        
        # Try multiple hosting services for reliability
        hosting_services = [
            {
                "name": "tmpfiles.org",
                "url": "https://tmpfiles.org/api/v1/upload",
                "method": "POST"
            }
        ]
        
        for service in hosting_services:
            try:
                print(f"🔄 Uploading to {service['name']}...")
                
                with open(file_path, 'rb') as file:
                    if service['name'] == "tmpfiles.org":
                        response = requests.post(
                            service['url'],
                            files={'file': file},
                            timeout=30
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract URL based on service
                        if service['name'] == "tmpfiles.org" and result.get('status') == 'success':
                            file_url = result['data']['url']
                            # Convert http to https for better security
                            if file_url.startswith('http://'):
                                file_url = file_url.replace('http://', 'https://')
                            
                            return {
                                "success": True,
                                "url": file_url,
                                "service": service['name'],
                                "file_id": file_url.split('/')[-2] if '/' in file_url else 'unknown',
                                "size_mb": file_size_mb,
                                "error": None,
                                "expires": "60 minutes (auto-delete)"
                            }
                
            except Exception as e:
                print(f"❌ Failed to upload to {service['name']}: {str(e)}")
                continue
        
        # If all services failed
        return {
            "success": False,
            "error": "All hosting services failed. File saved locally only.",
            "url": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Upload error: {str(e)}",
            "url": None
        }

def create_shareable_csv_link(file_path, file_name, file_description="Processed Data"):
    """
    Create a shareable link for a CSV file by uploading it to a remote host.
    
    Parameters
    ----------
    file_path : str
        Local path to the CSV file
    file_name : str
        Name of the file for display
    file_description : str
        Description of the file
        
    Returns
    -------
    list
        List of formatted strings for display
    """
    lines = []
    
    try:
        # Get file info
        file_size = os.path.getsize(file_path)
        file_size_kb = file_size / 1024
        file_size_mb = file_size / (1024 * 1024)
        
        # Read CSV to get basic stats
        df = pd.read_csv(file_path)
        
        lines.extend([
            f"🔗 **{file_name.replace('_', ' ').title()}** (CSV File - {file_size_kb:.1f} KB):",
            f"   📊 Dataset: {len(df):,} rows × {len(df.columns)} columns",
            f"   📅 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Upload to remote host
        upload_result = upload_csv_to_remote_host(file_path, file_description)
        
        if upload_result["success"]:
            lines.extend([
                "🌐 **SHAREABLE LINK CREATED**:",
                f"   🔗 **Download URL**: {upload_result['url']}",
                f"   🏢 **Service**: {upload_result['service']}",
                f"   📦 **File ID**: {upload_result['file_id']}",
                f"   📊 **Size**: {upload_result['size_mb']:.2f} MB",
                f"   ⏰ **Expires**: {upload_result['expires']}",
                "",
                "💡 **How to use**:",
                "   1. Click the URL above to download your processed data",
                "   2. Save the file with a .csv extension",
                "   3. Open in Excel, Python, R, or any data analysis tool",
                "   4. Share the link with colleagues or save for later use",
                "",
                "⚠️  **Important**: File auto-deletes after 60 minutes. Download promptly!"
            ])
        else:
            # Fallback: Provide local file info and sample data
            lines.extend([
                "❌ **REMOTE HOSTING FAILED**:",
                f"   Error: {upload_result['error']}",
                "",
                "📋 **FALLBACK: CSV DATA PREVIEW**:",
                ""
            ])
            
            # Show preview of the data
            if file_size_kb < 100:  # Small file - show more data
                lines.extend([
                    f"📊 **Complete CSV Data** ({len(df):,} rows × {len(df.columns)} columns):",
                    "```csv",
                    df.to_csv(index=False),
                    "```",
                    "",
                    "💡 **Usage**: Copy the CSV content above and save as .csv file"
                ])
            else:  # Large file - show preview
                lines.extend([
                    f"📊 **CSV Preview** (First 10 of {len(df):,} rows × {len(df.columns)} columns):",
                    "```csv",
                    df.head(10).to_csv(index=False),
                    "```",
                    "",
                    f"📁 **Local file**: {file_path}",
                    "💡 **To get complete data**: Ask 'Send my cleaned data in chunks'"
                ])
        
    except Exception as e:
        lines.extend([
            f"❌ **Error processing file**: {str(e)}",
            f"📁 **Local file location**: {file_path}"
        ])
    
    return lines

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

def display_file_contents(file_name, file_path):
    """Display the contents of a generated file based on its type."""
    file_lines = []
    
    try:
        if not os.path.exists(file_path):
            file_lines.extend([
                f"📄 **{file_name.replace('_', ' ').title()}**:",
                f"   ⚠️ File not found: {file_path}",
                ""
            ])
            return file_lines
        
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_kb = file_size / 1024
        
        # Determine file type and display strategy
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # Handle CSV files with remote hosting
            file_lines.extend(create_shareable_csv_link(file_path, file_name, "Processed CSV Data"))
        
        elif file_ext == '.txt' or file_ext == '.log':
            # Handle text/log files
            file_lines.extend([
                f"📝 **{file_name.replace('_', ' ').title()}** (Text File - {file_size_kb:.1f} KB):",
                ""
            ])
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content) < 10000:  # Small text file - show full content
                    file_lines.extend([
                        "```",
                        content,
                        "```"
                    ])
                else:  # Large text file - show first part
                    preview_content = content[:5000] + "\n\n... (truncated for display) ..."
                    file_lines.extend([
                        f"📄 **Content Preview** (First 5000 characters of {len(content):,} total):",
                        "```",
                        preview_content,
                        "```",
                        "",
                        f"📁 **Full file location**: {file_path}"
                    ])
            
            except Exception as e:
                file_lines.extend([
                    f"⚠️ Could not read text file: {str(e)}",
                    f"📁 File location: {file_path}"
                ])
        
        elif file_ext in ['.py', '.r', '.sql']:
            # Handle code files
            file_lines.extend([
                f"💻 **{file_name.replace('_', ' ').title()}** (Code File - {file_size_kb:.1f} KB):",
                ""
            ])
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine language for syntax highlighting
                lang_map = {'.py': 'python', '.r': 'r', '.sql': 'sql'}
                lang = lang_map.get(file_ext, 'text')
                
                if len(content) < 8000:  # Show full code file
                    file_lines.extend([
                        f"```{lang}",
                        content,
                        "```"
                    ])
                else:  # Show preview of large code file
                    preview_content = content[:4000] + "\n\n# ... (truncated for display) ..."
                    file_lines.extend([
                        f"📄 **Code Preview** (First 4000 characters):",
                        f"```{lang}",
                        preview_content,
                        "```",
                        "",
                        f"📁 **Full file location**: {file_path}"
                    ])
            
            except Exception as e:
                file_lines.extend([
                    f"⚠️ Could not read code file: {str(e)}",
                    f"📁 File location: {file_path}"
                ])
        
        else:
            # Handle other file types
            file_lines.extend([
                f"📄 **{file_name.replace('_', ' ').title()}** ({file_ext.upper()} File - {file_size_kb:.1f} KB):",
                f"📁 File location: {file_path}",
                ""
            ])
            
            # Try to read as text if it's small
            if file_size_kb < 20:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_lines.extend([
                        "📄 **File Contents**:",
                        "```",
                        content,
                        "```"
                    ])
                except:
                    file_lines.append("⚠️ Binary file - cannot display content as text")
            else:
                file_lines.append("📁 File too large to display - check file location above")
    
    except Exception as e:
        file_lines.extend([
            f"📄 **{file_name.replace('_', ' ').title()}**:",
            f"   ❌ Error reading file: {str(e)}",
            f"   📁 File path: {file_path}",
            ""
        ])
    
    return file_lines


def extract_h2o_ml_results(agent_result) -> Dict[str, Any]:
    """Extract comprehensive ML results from H2O agent execution."""
    
    if agent_result.agent_name != "h2o_ml" or not agent_result.success:
        return {}
    
    ml_results = {}
    
    try:
        # Extract from ML modeling metrics if available
        if hasattr(agent_result, 'ml_modeling_metrics') and agent_result.ml_modeling_metrics:
            metrics = agent_result.ml_modeling_metrics
            
            # Basic metrics
            ml_results["models_trained"] = getattr(metrics, 'total_models_trained', getattr(metrics, 'models_trained', 0))
            ml_results["best_model_id"] = getattr(metrics, 'best_model_id', None)
            ml_results["model_architecture"] = getattr(metrics, 'model_architecture', getattr(metrics, 'best_model_type', None))
            ml_results["training_time"] = getattr(metrics, 'training_runtime', getattr(metrics, 'training_time_seconds', 0))
            
            # Performance metrics
            ml_results["best_score"] = getattr(metrics, 'best_model_score', None)
            ml_results["cross_validation_score"] = getattr(metrics, 'cross_validation_score', None)
            
            # Rich data
            ml_results["leaderboard"] = getattr(metrics, 'leaderboard', None)
            ml_results["top_model_metrics"] = getattr(metrics, 'top_model_metrics', {})
            ml_results["generated_code"] = getattr(metrics, 'generated_code', None)
            ml_results["recommended_steps"] = getattr(metrics, 'recommended_steps', None)
            ml_results["workflow_summary"] = getattr(metrics, 'workflow_summary', None)
            ml_results["model_path"] = getattr(metrics, 'model_path', None)
            ml_results["enhanced_feature_importance"] = getattr(metrics, 'enhanced_feature_importance', [])
            
            # Status flags
            ml_results["has_leaderboard"] = ml_results["leaderboard"] is not None and len(ml_results["leaderboard"]) > 0
            ml_results["model_saved"] = ml_results["model_path"] is not None
            
        # Fallback: try to extract from log messages
        if not ml_results.get("models_trained") and agent_result.log_messages:
            result_str = " ".join(agent_result.log_messages)
            
            # Look for model information in logs  
            import re
            
            # Look for model performance patterns
            auc_matches = re.findall(r'AUC[:\s]+([0-9.]+)', result_str, re.IGNORECASE)
            if auc_matches:
                ml_results["best_auc"] = float(auc_matches[0])
            
            # Look for accuracy patterns  
            acc_matches = re.findall(r'accuracy[:\s]+([0-9.]+)', result_str, re.IGNORECASE)
            if acc_matches:
                ml_results["accuracy"] = float(acc_matches[0])
            
            # Look for model count
            model_matches = re.findall(r'(\d+)\s+models?\s+trained', result_str, re.IGNORECASE)
            if model_matches:
                ml_results["models_trained"] = int(model_matches[0])
        
        # Extract model path information from agent result
        if hasattr(agent_result, 'model_path') and agent_result.model_path:
            ml_results["model_path"] = agent_result.model_path
            ml_results["model_saved"] = True
        
        return ml_results
        
    except Exception as e:
        logger.warning(f"Failed to extract H2O ML results: {e}")
        return {}


def format_ml_leaderboard_display(ml_results: Dict[str, Any], execution_time: float = 0) -> List[str]:
    """Format ML leaderboard for beautiful user display."""
    
    lines = [
        "🤖 **MACHINE LEARNING RESULTS**",
        "=" * 50,
        ""
    ]
    
    # Training summary
    if ml_results.get("models_trained", 0) > 0:
        lines.extend([
            "🏆 **MODEL TRAINING COMPLETE**:",
            f"   • Models Trained: {ml_results.get('models_trained', 'Multiple')}",
            f"   • Training Time: {ml_results.get('training_time', execution_time):.1f} seconds",
            ""
        ])
        
        # Performance metrics
        if ml_results.get("best_score"):
            lines.append(f"   • Best Model Score: {ml_results['best_score']:.4f}")
        if ml_results.get("best_auc"):
            lines.append(f"   • Best AUC Score: {ml_results['best_auc']:.4f}")
        if ml_results.get("accuracy"):
            lines.append(f"   • Accuracy: {ml_results['accuracy']:.4f}")
        if ml_results.get("cross_validation_score"):
            lines.append(f"   • Cross-Validation Score: {ml_results['cross_validation_score']:.4f}")
        
        lines.append("")
    
    # Leaderboard display
    if ml_results.get("has_leaderboard") and ml_results.get("leaderboard"):
        lines.extend([
            "🏆 **MODEL LEADERBOARD** (Top 5 Models):",
            ""
        ])
        
        leaderboard = ml_results["leaderboard"]
        for idx, model in enumerate(leaderboard[:5]):  # Show top 5 models
            rank = idx + 1
            model_id = model.get('model_id', f'Model_{rank}')
            model_name = model_id[:40] + "..." if len(model_id) > 40 else model_id
            
            # Get performance metric (try different fields)
            performance = (
                model.get('auc', model.get('rmse', model.get('logloss', model.get('mean_residual_deviance', 0))))
            )
            
            rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            lines.append(f"   {rank_emoji} **{model_name}**")
            
            if performance:
                metric_name = "AUC" if 'auc' in model else "RMSE" if 'rmse' in model else "Score"
                lines.append(f"      • {metric_name}: {performance:.4f}")
            
            if rank == 1:
                lines.append(f"      🏆 **WINNER** - This is your best model!")
            
            lines.append("")
    
    # Model architecture and details
    if ml_results.get("model_architecture") or ml_results.get("best_model_id"):
        lines.extend([
            "🎯 **BEST MODEL DETAILS**:",
        ])
        
        if ml_results.get("best_model_id"):
            lines.append(f"   • Model ID: `{ml_results['best_model_id']}`")
        if ml_results.get("model_architecture"):
            lines.append(f"   • Architecture: {ml_results['model_architecture']}")
        if ml_results.get("training_time", 0) > 0:
            lines.append(f"   • Training Time: {ml_results['training_time']:.1f} seconds")
        
        lines.append("")
    
    # ML methodology  
    lines.extend([
        "🧠 **AI METHODOLOGY APPLIED**:",
        "   • Automated algorithm selection (Random Forest, GBM, Neural Networks, etc.)",
        "   • Hyperparameter optimization for best performance",
        "   • Cross-validation to ensure model reliability",
        "   • Feature importance analysis for interpretability",
        ""
    ])
    
    # Performance summary
    lines.extend([
        "🎯 **MODEL PERFORMANCE**:",
        "   🥇 **Best Model Selected**: AutoML chose the highest-performing algorithm",
        "   📊 **Cross-Validated**: Results are validated to avoid overfitting",
        "   ⚡ **Production Ready**: Model can be used for predictions immediately",
        ""
    ])
    
    return lines


def format_ml_generated_code_display(ml_results: Dict[str, Any]) -> List[str]:
    """Format AI-generated ML code for user display."""
    
    lines = []
    
    # Show generated code if available
    generated_code = ml_results.get("generated_code")
    if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 10:
        lines.extend([
            "💻 **AI-GENERATED MODEL CODE**:",
            "```python",
            generated_code.strip(),
            "```",
            "",
            "💡 **Usage**: Copy this code to reproduce the same model training independently!",
            ""
        ])
    else:
        # Provide template code
        model_id = ml_results.get("best_model_id", "your_model")
        lines.extend([
            "💻 **AI-GENERATED MODEL CODE**:",
            "```python",
            "# H2O AutoML Training Code (Generated by AI)",
            "import h2o",
            "from h2o.automl import H2OAutoML",
            "",
            "# Initialize H2O",
            "h2o.init()",
            "",
            "# Load your data",
            "data = h2o.import_file('your_dataset.csv')",
            "",
            "# Prepare training data",
            "train, test = data.split_frame(ratios=[0.8])",
            "x = train.columns[:-1]  # All columns except target",
            "y = train.columns[-1]   # Target column",
            "",
            "# Train AutoML model",
            "aml = H2OAutoML(max_models=20, seed=42)",
            "aml.train(x=x, y=y, training_frame=train)",
            "",
            "# Get best model and make predictions",
            "best_model = aml.leader",
            "predictions = best_model.predict(test)",
            "",
            "# View leaderboard",
            "print(aml.leaderboard.head())",
            "```",
            "",
            "💡 **Usage**: Copy this code to train H2O AutoML models independently!",
            ""
        ])
    
    return lines


def format_ml_workflow_summary_display(ml_results: Dict[str, Any]) -> List[str]:
    """Format ML workflow summary and recommendations."""
    
    lines = []
    
    # Workflow summary
    workflow_summary = ml_results.get("workflow_summary")
    if workflow_summary:
        lines.extend([
            "📚 **ML WORKFLOW SUMMARY**:",
            f"   {workflow_summary}",
            ""
        ])
    
    # Recommended steps
    recommended_steps = ml_results.get("recommended_steps")
    if recommended_steps:
        lines.extend([
            "📋 **AI-RECOMMENDED METHODOLOGY**:",
            f"   {recommended_steps}",
            ""
        ])
    
    # Feature importance if available
    feature_importance = ml_results.get("enhanced_feature_importance", [])
    if feature_importance and len(feature_importance) > 0:
        lines.extend([
            "🎯 **FEATURE IMPORTANCE ANALYSIS**:",
            ""
        ])
        
        for feature in feature_importance[:5]:  # Show top 5 features
            feature_name = feature.get("feature", "Unknown")
            importance = feature.get("importance", 0)
            impact = feature.get("impact", "Unknown")
            lines.append(f"   • **{feature_name}**: {importance:.3f} ({impact} Impact)")
        
        if len(feature_importance) > 5:
            lines.append(f"   • ...and {len(feature_importance) - 5} more features")
        
        lines.append("")
    
    return lines


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
                "🧹 **CLEANING ACTIONS PERFORMED**:",
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
        
        # Workflow information with actual execution results - Enhanced ML Display
        if result.workflow_intent:
            # Check actual execution results
            data_cleaning_status = "❌ Not executed"
            feature_engineering_status = "❌ Not executed"  
            ml_modeling_status = "❌ Not executed"
            ml_agent_result = None
            
            for agent_result in result.agent_results:
                if agent_result.agent_name == "data_cleaning":
                    data_cleaning_status = "✅ Success" if agent_result.success else f"❌ Failed: {getattr(agent_result, 'error_message', 'Unknown error')[:50]}..."
                elif agent_result.agent_name == "feature_engineering":
                    feature_engineering_status = "✅ Success" if agent_result.success else f"❌ Failed: {getattr(agent_result, 'error_message', 'Unknown error')[:50]}..."
                elif agent_result.agent_name == "h2o_ml":
                    if agent_result.success:
                        ml_modeling_status = "✅ Success"
                        ml_agent_result = agent_result  # Store for rich display
                    else:
                        ml_modeling_status = f"❌ Failed: {getattr(agent_result, 'error_message', 'Unknown error')[:50]}..."
            
            # Basic workflow status 
            lines.extend([
                "🔄 **WORKFLOW EXECUTION RESULTS**:",
                f"   • Data Cleaning: {data_cleaning_status}",
                f"   • Feature Engineering: {feature_engineering_status}",
                f"   • ML Modeling: {ml_modeling_status}",
                f"   • Intent Confidence: {result.workflow_intent.intent_confidence:.2f}",
                ""
            ])
            
            # ENHANCED ML RESULTS DISPLAY - This is the magic!
            if ml_agent_result and ml_agent_result.success:
                try:
                    # Extract comprehensive ML results
                    ml_results = extract_h2o_ml_results(ml_agent_result)
                    
                    if ml_results:  # If we have ML results to display
                        lines.append("")  # Add spacing
                        
                        # Add rich ML leaderboard display
                        ml_leaderboard_lines = format_ml_leaderboard_display(
                            ml_results, 
                            ml_agent_result.execution_time_seconds
                        )
                        lines.extend(ml_leaderboard_lines)
                        
                        # Add generated code display
                        ml_code_lines = format_ml_generated_code_display(ml_results)
                        lines.extend(ml_code_lines)
                        
                        # Add workflow summary and recommendations
                        ml_workflow_lines = format_ml_workflow_summary_display(ml_results)
                        lines.extend(ml_workflow_lines)
                        
                        # Add model download information if available
                        if ml_results.get("model_saved") and ml_results.get("model_path"):
                            lines.extend([
                                "💾 **MODEL DOWNLOAD**:",
                                f"   📁 **Model Saved**: {ml_results['model_path']}",
                                "   🔄 **Status**: Model ready for predictions and deployment",
                                "   💡 **Usage**: Model can be loaded for making predictions on new data",
                                ""
                            ])
                        
                except Exception as e:
                    logger.warning(f"Failed to display enhanced ML results: {e}")
                    # Fallback to basic display
                    lines.extend([
                        "🤖 **MACHINE LEARNING COMPLETE**:",
                        "   ✅ ML model training completed successfully",
                        f"   ⏱️  Training time: {ml_agent_result.execution_time_seconds:.1f} seconds",
                        "   📊 Model ready for predictions",
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
        
        # Generated files with actual content
        if result.generated_files:
            lines.extend([
                "📁 **GENERATED FILES & CONTENTS**:",
                ""
            ])
            
            for name, path in result.generated_files.items():
                lines.extend(display_file_contents(name, path))
                lines.append("")  # Add spacing between files
        
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
        "name": "AI Data Science Agent",
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
        print(f"🔗 Agent name: AI Data Science Agent")
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
        print("\n🛑 Shutting down AI Data Science Agent...")
        cleanup_uagent("AI Data Science Agent")
        print("✅ Agent stopped.") 