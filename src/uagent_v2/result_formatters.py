"""
Result formatting module for the enhanced uAgent implementation.

This module contains all result formatting functions extracted from the original
enhanced_uagent.py file for better code organization and maintainability.
"""

import re
import os
import time
import logging
from typing import Dict, Any, List, Optional

from .config import UAgentConfig
from .file_handlers import create_shareable_csv_link, display_file_contents

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Comprehensive result formatting for data analysis results."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def format_analysis_result_enhanced(self, result) -> str:
        """Format analysis result with enhanced display and security."""
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
                "",
                "─" * 60,
                ""
            ]
            
            # Add workflow summary
            lines.extend(self.format_workflow_summary(result))
            
            # Add data transformation results
            lines.extend(self.format_data_transformation_results(result))
            
            # Add cleaned data section with secure delivery
            lines.extend(self.format_cleaned_data_section(result))
            
            # Add ML results with comprehensive display
            lines.extend(self.format_ml_results_enhanced(result))
            
            # Add agent results
            lines.extend(self.format_agent_results_enhanced(result))
            
            # Add completion summary
            lines.extend(self.format_completion_summary(result))
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}", exc_info=True)
            return f"❌ **Error formatting result**: {str(e)}\n\nRaw result: {str(result)}"
    
    def format_workflow_summary(self, result) -> List[str]:
        """Format workflow execution summary."""
        lines = []
        
        if result.workflow_intent:
            # Extract agent status information
            data_cleaning_result = None
            feature_engineering_result = None
            ml_agent_result = None
            
            for agent_result in result.agent_results:
                if agent_result.agent_name == "data_cleaning":
                    data_cleaning_result = agent_result
                elif agent_result.agent_name == "feature_engineering":
                    feature_engineering_result = agent_result
                elif agent_result.agent_name == "h2o_ml":
                    ml_agent_result = agent_result
            
            # Check actual execution results for summary
            data_cleaning_status = "❌ Not executed"
            feature_engineering_status = "❌ Not executed"  
            ml_modeling_status = "❌ Not executed"
            
            if data_cleaning_result:
                data_cleaning_status = "✅ Success" if data_cleaning_result.success else f"❌ Failed: {getattr(data_cleaning_result, 'error_message', 'Unknown error')[:50]}..."
            if feature_engineering_result:
                feature_engineering_status = "✅ Success" if feature_engineering_result.success else f"❌ Failed: {getattr(feature_engineering_result, 'error_message', 'Unknown error')[:50]}..."
            if ml_agent_result:
                if ml_agent_result.success:
                    ml_modeling_status = "✅ Success"
                else:
                    ml_modeling_status = f"❌ Failed: {getattr(ml_agent_result, 'error_message', 'Unknown error')[:50]}..."
            
            lines.extend([
                "🔄 **WORKFLOW EXECUTION SUMMARY**:",
                f"   • Data Cleaning: {data_cleaning_status}",
                f"   • Feature Engineering: {feature_engineering_status}",
                f"   • ML Modeling: {ml_modeling_status}",
                f"   • Intent Confidence: {result.workflow_intent.intent_confidence:.2f}",
                ""
            ])
        
        return lines
    
    def format_data_transformation_results(self, result) -> List[str]:
        """Format data transformation results."""
        lines = [
            "📈 **DATA TRANSFORMATION RESULTS**:",
            "─" * 40,
            ""
        ]
        
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
        
        return lines
    
    def format_cleaned_data_section(self, result) -> List[str]:
        """Format cleaned data section with delivery options."""
        lines = []
        
        # Try to find cleaned data file path
        cleaned_data_path = None
        for agent_result in result.agent_results:
            if agent_result.output_data_path and agent_result.agent_name == "data_cleaning":
                cleaned_data_path = agent_result.output_data_path
                break
        
        if cleaned_data_path and os.path.exists(cleaned_data_path):
            try:
                import pandas as pd
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
                
                # Strategy 2: For medium datasets (50KB-200KB), provide full data with better formatting
                elif content_size < 200000:  # 200KB limit
                    lines.extend([
                        "📋 **COMPLETE CLEANED DATA** (Full Dataset):",
                        "```csv",
                        csv_content,
                        "```",
                        "",
                        f"📁 **DATASET SUMMARY**:",
                        f"   • File size: {content_size / 1024:.1f} KB",
                        f"   • Contains {len(cleaned_df):,} rows and {len(cleaned_df.columns)} columns",
                        f"   • All data shown above - ready for analysis",
                        "",
                        "💡 **Usage**: Copy the complete CSV data above and save it as a .csv file, or use it directly in your analysis.",
                        ""
                    ])
                
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
                
                # Show column information for all cases
                lines.extend([
                    "📋 **COLUMN INFORMATION**:",
                    "```",
                    f"{'Column Name':<20} | {'Data Type':<10} | {'Nulls':<6} | {'Unique':<7} | Sample Values",
                    "-" * 75,
                ])
                
                for col in cleaned_df.columns[:15]:  # Show first 15 columns
                    dtype = str(cleaned_df[col].dtype)
                    null_count = cleaned_df[col].isnull().sum()
                    unique_count = cleaned_df[col].nunique()
                    
                    # Add sample values for categorical columns
                    if dtype == 'object' and unique_count <= 10:
                        sample_values = cleaned_df[col].value_counts().head(3).index.tolist()
                        sample_str = f"{', '.join(map(str, sample_values))}"
                    elif dtype != 'object' and unique_count <= 20:
                        sample_values = sorted(cleaned_df[col].dropna().unique())[:5]
                        sample_str = f"{', '.join(map(str, sample_values))}"
                    else:
                        sample_str = "..."
                    
                    # Truncate long column names and sample values for formatting
                    col_display = col[:18] + ".." if len(col) > 20 else col
                    sample_display = sample_str[:25] + "..." if len(sample_str) > 25 else sample_str
                    
                    lines.append(f"{col_display:<20} | {dtype:<10} | {null_count:<6} | {unique_count:<7} | {sample_display}")
                
                if len(cleaned_df.columns) > 15:
                    lines.append(f"{'...':<20} | {'...':<10} | {'...':<6} | {'...':<7} | and {len(cleaned_df.columns) - 15} more columns")
                
                lines.extend([
                    "```",
                    ""
                ])
                
            except Exception as e:
                lines.extend([
                    "⚠️  **Note**: Could not access cleaned data file",
                    f"   Error: {str(e)}",
                    "   The data was processed but file access failed",
                    ""
                ])
        else:
            lines.extend([
                "📋 **CLEANED DATA**:",
                f"   ✅ Data successfully cleaned and saved",
                "",
                "💡 **To Get Your Cleaned Data**:",
                "   Ask: 'Please provide my cleaned dataset as CSV' or 'Send me my processed data'",
                ""
            ])
        
        return lines
    
    def format_ml_results_enhanced(self, result) -> List[str]:
        """Format ML results with comprehensive display."""
        lines = []
        
        # Find ML agent result
        ml_agent_result = None
        for agent_result in result.agent_results:
            if agent_result.agent_name == "h2o_ml":
                ml_agent_result = agent_result
                break
        
        if ml_agent_result and ml_agent_result.success:
            try:
                from .ml_processors import MLResultProcessor
                ml_processor = MLResultProcessor(self.config)
                
                # Extract comprehensive ML results
                ml_results = ml_processor.extract_h2o_ml_results(ml_agent_result)
                
                if ml_results:
                    lines.append("")
                    lines.extend(ml_processor.format_ml_leaderboard_display(ml_results, ml_agent_result.execution_time_seconds))
                    lines.extend(ml_processor.format_ml_generated_code_display(ml_results))
                    lines.extend(ml_processor.format_ml_workflow_summary_display(ml_results))
                    
                    # Add model download information
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
                lines.extend([
                    "🤖 **MACHINE LEARNING COMPLETE**:",
                    "   ✅ ML model training completed successfully",
                    f"   ⏱️  Training time: {ml_agent_result.execution_time_seconds:.1f} seconds",
                    "   📊 Model ready for predictions",
                    ""
                ])
        
        return lines
    
    def format_agent_results_enhanced(self, result) -> List[str]:
        """Format agent results with file download links."""
        lines = []
        
        # Data Cleaning Results Section
        data_cleaning_result = None
        feature_engineering_result = None
        
        for agent_result in result.agent_results:
            if agent_result.agent_name == "data_cleaning":
                data_cleaning_result = agent_result
            elif agent_result.agent_name == "feature_engineering":
                feature_engineering_result = agent_result
        
        # 1. DATA CLEANING RESULTS WITH DOWNLOAD LINKS
        if data_cleaning_result:
            if data_cleaning_result.success and data_cleaning_result.output_data_path:
                lines.extend([
                    "🧹 **DATA CLEANING COMPLETED**",
                    "=" * 40,
                    ""
                ])
                
                # Add the cleaned data download links
                lines.extend(create_shareable_csv_link(
                    data_cleaning_result.output_data_path,
                    "cleaned_data",
                    "Cleaned Dataset"
                ))
                lines.append("")
                
            elif not data_cleaning_result.success:
                lines.extend([
                    "🧹 **DATA CLEANING FAILED**",
                    "=" * 40,
                    f"   ❌ Error: {getattr(data_cleaning_result, 'error_message', 'Unknown error')}",
                    f"   ⏱️  Runtime: {data_cleaning_result.execution_time_seconds:.2f} seconds",
                    ""
                ])
        
        # 2. FEATURE ENGINEERING RESULTS WITH DOWNLOAD LINKS  
        if feature_engineering_result:
            if feature_engineering_result.success and feature_engineering_result.output_data_path:
                lines.extend([
                    "🔧 **FEATURE ENGINEERING COMPLETED**",
                    "=" * 40,
                    ""
                ])
                
                # Add the feature engineered data download links
                lines.extend(create_shareable_csv_link(
                    feature_engineering_result.output_data_path,
                    "feature_engineered_data", 
                    "Feature Engineered Dataset"
                ))
                lines.append("")
                
            elif not feature_engineering_result.success:
                lines.extend([
                    "🔧 **FEATURE ENGINEERING FAILED**",
                    "=" * 40,
                    f"   ❌ Error: {getattr(feature_engineering_result, 'error_message', 'Unknown error')}",
                    f"   ⏱️  Runtime: {feature_engineering_result.execution_time_seconds:.2f} seconds",
                    ""
                ])
        
        # Generated files with actual content - DEDUPLICATED
        if result.generated_files:
            # Track already displayed files to avoid duplicates
            displayed_files = set()
            
            # First, collect files that were already displayed in agent sections
            for agent_result in result.agent_results:
                if agent_result.output_data_path:
                    displayed_files.add(agent_result.output_data_path)
            
            # Only show files that haven't been displayed yet
            unique_files = {}
            for name, path in result.generated_files.items():
                if path not in displayed_files:
                    unique_files[name] = path
            
            if unique_files:
                lines.extend([
                    "📁 **ADDITIONAL GENERATED FILES**:",
                    ""
                ])
                
                for name, path in unique_files.items():
                    lines.extend(display_file_contents(name, path))
                    lines.append("")  # Add spacing between files
        
        return lines
    
    def format_completion_summary(self, result) -> List[str]:
        """Format completion summary section."""
        lines = []
        
        # Key insights
        if result.key_insights:
            lines.extend([
                "💡 **KEY INSIGHTS**:",
                "─" * 20,
                *[f"   • {insight}" for insight in result.key_insights],
                ""
            ])
        
        # Recommendations
        if result.recommendations:
            lines.extend([
                "🎯 **RECOMMENDATIONS**:",
                "─" * 25,
                *[f"   • {rec}" for rec in result.recommendations],
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
        
        # Final completion message
        lines.extend([
            "=" * 60,
            "✅ **Analysis completed successfully!**",
            "",
            "💡 **What you got:**",
            "   • Cleaned dataset with comprehensive processing",
            "   • Enhanced analysis results with download links",
            "   • Ready-to-use data for further analysis",
            "   • Comprehensive ML insights and generated code",
            ""
        ])
        
        return lines 