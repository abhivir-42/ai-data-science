"""
Data delivery module for the enhanced uAgent implementation.

This module contains all data delivery functions extracted from the original
enhanced_uagent.py file for better code organization and maintainability.
"""

import re
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union

from .config import UAgentConfig
from .utils import MemoryEfficientCSVProcessor, DataDeliveryOptimizer
from .file_handlers import upload_csv_to_remote_host, create_shareable_csv_link

logger = logging.getLogger(__name__)


class DataDeliveryHandler:
    """Handle data delivery requests with various strategies."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self.csv_processor = MemoryEfficientCSVProcessor(config)
        self.delivery_optimizer = DataDeliveryOptimizer(config)
    
    def deliver_data_directly(self, df: pd.DataFrame) -> str:
        """Deliver data directly as CSV content."""
        try:
            csv_content = df.to_csv(index=False)
            content_size = len(csv_content.encode('utf-8'))
            
            return f"""
📁 **YOUR COMPLETE CLEANED DATA**

File size: {content_size / 1024:.1f} KB | Rows: {len(df):,} | Columns: {len(df.columns)}

```csv
{csv_content}
```

💡 **Usage**: Copy the CSV content above and save it as a .csv file for use in Excel, Python, R, or other tools.
"""
        except Exception as e:
            logger.error(f"Direct data delivery failed: {e}")
            return f"❌ **Direct delivery failed**: {str(e)}"
    
    def deliver_data_chunked(self, df: pd.DataFrame, query: str) -> str:
        """Deliver data in chunks based on query specification."""
        try:
            # Extract number of chunks if specified
            chunk_match = re.search(r'(\d+)\s*chunk', query.lower())
            num_chunks = int(chunk_match.group(1)) if chunk_match else 5
            
            # Limit chunks to reasonable number
            num_chunks = min(num_chunks, 20)
            
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
            
        except Exception as e:
            logger.error(f"Chunked data delivery failed: {e}")
            return f"❌ **Chunked delivery failed**: {str(e)}"
    
    def deliver_data_rows(self, df: pd.DataFrame, query: str) -> str:
        """Deliver specific row range."""
        try:
            # Extract row range if specified
            range_match = re.search(r'rows?\s*(\d+)[-\s]*(\d+)?', query.lower())
            if range_match:
                start_row = int(range_match.group(1)) - 1  # Convert to 0-indexed
                end_row = int(range_match.group(2)) if range_match.group(2) else start_row + 1000
            else:
                start_row = 0
                end_row = min(1000, len(df))
            
            # Ensure valid range
            start_row = max(0, start_row)
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
        except Exception as e:
            logger.error(f"Row delivery failed: {e}")
            return f"❌ **Row delivery failed**: {str(e)}"
    
    def deliver_data_columns(self, df: pd.DataFrame, query: str) -> str:
        """Deliver specific column range."""
        try:
            # Extract column range or names if specified
            col_match = re.search(r'columns?\s*(\d+)[-\s]*(\d+)?', query.lower())
            if col_match:
                start_col = int(col_match.group(1)) - 1  # Convert to 0-indexed
                end_col = int(col_match.group(2)) if col_match.group(2) else start_col + 5
            else:
                start_col = 0
                end_col = min(5, len(df.columns))
            
            # Ensure valid range
            start_col = max(0, start_col)
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
        except Exception as e:
            logger.error(f"Column delivery failed: {e}")
            return f"❌ **Column delivery failed**: {str(e)}"
    
    def deliver_data_as_link(self, df: pd.DataFrame, temp_dir: str) -> str:
        """Deliver data as a shareable link."""
        try:
            # Save DataFrame to temporary file
            import tempfile
            import os
            
            temp_file = os.path.join(temp_dir, "cleaned_data.csv")
            df.to_csv(temp_file, index=False)
            
            # Create shareable link
            link_lines = create_shareable_csv_link(temp_file, "cleaned_data", "Cleaned Dataset")
            
            return f"""
🔗 **DATA DELIVERY VIA LINK**

Your cleaned dataset ({len(df):,} rows × {len(df.columns)} columns) is being prepared for download:

{chr(10).join(link_lines)}
"""
        except Exception as e:
            logger.error(f"Link delivery failed: {e}")
            return f"❌ **Link delivery failed**: {str(e)}"
    
    def deliver_data_preview(self, df: pd.DataFrame) -> str:
        """Deliver data preview with statistics."""
        try:
            # Get basic statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Create preview
            preview_df = df.head(10)
            
            return f"""
📊 **DATA PREVIEW**

Dataset Overview:
• Total rows: {len(df):,}
• Total columns: {len(df.columns)}
• Numeric columns: {len(numeric_cols)}
• Categorical columns: {len(categorical_cols)}
• Missing values: {df.isnull().sum().sum():,}

First 10 rows:
```csv
{preview_df.to_csv(index=False)}
```

💡 **To get complete data**:
• Small dataset: Ask "Send me my complete data"
• Large dataset: Ask "Send me my data in chunks"
• Specific parts: Ask "Send me rows 1-100" or "Send me columns 1-5"
"""
        except Exception as e:
            logger.error(f"Preview delivery failed: {e}")
            return f"❌ **Preview delivery failed**: {str(e)}"


class DataDeliveryRequestHandler:
    """Handle data delivery requests with backward compatibility."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
        self.delivery_handler = DataDeliveryHandler(config)
    
    def handle_data_delivery_request(self, query: str, cleaned_data: Optional[pd.DataFrame] = None) -> str:
        """Handle follow-up requests for data delivery (backward compatibility)."""
        
        # Check if we have data to deliver
        if cleaned_data is None:
            return self._create_no_data_response()
        
        try:
            query_lower = query.lower()
            
            # Parse the request type
            if 'chunk' in query_lower:
                return self.delivery_handler.deliver_data_chunked(cleaned_data, query)
            
            elif 'rows' in query_lower:
                return self.delivery_handler.deliver_data_rows(cleaned_data, query)
            
            elif 'column' in query_lower:
                return self.delivery_handler.deliver_data_columns(cleaned_data, query)
            
            else:
                # Default: provide complete data if small enough, otherwise chunked
                csv_content = cleaned_data.to_csv(index=False)
                content_size = len(csv_content.encode('utf-8'))
                
                if content_size < 100000:  # 100KB limit for direct delivery
                    return self.delivery_handler.deliver_data_directly(cleaned_data)
                else:
                    return self.delivery_handler.deliver_data_chunked(cleaned_data, "5 chunks")
        
        except Exception as e:
            logger.error(f"Data delivery request failed: {e}")
            return f"""
🚫 **Data Delivery Error**

Sorry, I encountered an issue delivering your data: {str(e)}

Please try a more specific request like:
- "Send me rows 1-100 of my data"
- "Provide my data in 3 chunks"
- "Show me columns 1-5 of my cleaned data"
"""
    
    def _create_no_data_response(self) -> str:
        """Create response when no data is available."""
        return """
🚫 **No Recent Data Found**

I don't have any recently processed data to deliver. Please first run a data cleaning task, for example:

"Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

Then I can provide your cleaned data in various formats.
"""


# Backward compatibility functions
def deliver_data_in_chunks(df: pd.DataFrame, num_chunks: int) -> str:
    """Deliver data in specified number of chunks (backward compatibility)."""
    from .config import UAgentConfig
    config = UAgentConfig.from_env()
    handler = DataDeliveryHandler(config)
    return handler.deliver_data_chunked(df, f"{num_chunks} chunks")


def deliver_data_rows(df: pd.DataFrame, start_row: int, end_row: int) -> str:
    """Deliver specific row range (backward compatibility)."""
    from .config import UAgentConfig
    config = UAgentConfig.from_env()
    handler = DataDeliveryHandler(config)
    return handler.deliver_data_rows(df, f"rows {start_row+1}-{end_row}")


def deliver_data_columns(df: pd.DataFrame, start_col: int, end_col: int) -> str:
    """Deliver specific column range (backward compatibility)."""
    from .config import UAgentConfig
    config = UAgentConfig.from_env()
    handler = DataDeliveryHandler(config)
    return handler.deliver_data_columns(df, f"columns {start_col+1}-{end_col}")


def handle_data_delivery_request(query: str, cleaned_data: Optional[pd.DataFrame] = None) -> str:
    """Handle follow-up requests for data delivery (backward compatibility)."""
    from .config import UAgentConfig
    config = UAgentConfig.from_env()
    handler = DataDeliveryRequestHandler(config)
    return handler.handle_data_delivery_request(query, cleaned_data) 