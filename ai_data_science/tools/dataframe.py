"""
DataFrame utilities for AI Data Science.
"""

import pandas as pd
import numpy as np
from typing import List

def get_dataframe_summary(dataframes: List[pd.DataFrame], n_sample: int = 30) -> List[str]:
    """
    Generate a summary of dataframes for use by AI agents.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of dataframes to summarize
    n_sample : int, optional
        Number of rows to include in the sample, by default 30

    Returns
    -------
    List[str]
        List of summaries for each dataframe
    """
    summaries = []
    for i, df in enumerate(dataframes):
        try:
            summary = f"DataFrame {i+1} Summary:\n"
            
            # Basic information
            summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            
            # Column information
            summary += "\nColumn Information:\n"
            for col in df.columns:
                dtype = df[col].dtype
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                summary += f"- {col} (dtype: {dtype}):\n"
                summary += f"  - Missing values: {missing} ({missing_pct:.2f}%)\n"
                summary += f"  - Unique values: {unique_count}\n"
                
                # For numeric columns, add statistical summary
                if np.issubdtype(dtype, np.number):
                    summary += f"  - Min: {df[col].min()}, Max: {df[col].max()}\n"
                    summary += f"  - Mean: {df[col].mean()}, Median: {df[col].median()}\n"
                    summary += f"  - Standard deviation: {df[col].std()}\n"
                
                # For categorical/object columns, show top values
                elif dtype == 'object' or dtype.name == 'category':
                    if unique_count <= 10:
                        value_counts = df[col].value_counts().head(5)
                        summary += f"  - Top values: {', '.join([f'{v} ({c})' for v, c in value_counts.items()])}\n"
                    else:
                        summary += f"  - Too many unique values to display\n"
            
            # Sample data
            if n_sample > 0:
                sample_size = min(n_sample, df.shape[0])
                sample = df.sample(sample_size) if sample_size < df.shape[0] else df
                summary += f"\nSample Data ({sample_size} rows):\n"
                summary += sample.to_string() + "\n"
            
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"Error generating summary for DataFrame {i+1}: {str(e)}")
    
    return summaries 