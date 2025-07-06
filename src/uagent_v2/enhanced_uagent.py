#!/usr/bin/env python3
"""
Enhanced Data Analysis uAgent Implementation (v2.0)

This is the improved version of the data analysis uAgent that addresses all the
critical issues identified in the original implementation:

✅ Modular architecture (split from 1547-line monolith)
✅ Enhanced security with file validation
✅ Memory-efficient processing 
✅ Structured error handling (no silent failures)
✅ Configurable settings via environment variables
✅ Type safety with comprehensive type hints
✅ Performance optimizations

Following the Fetch.ai LangGraph adapter pattern while incorporating
all the improvements from the enhancement plan.
"""

import os
import time
import sys
import logging
from typing import Dict, Any, Union, Optional
from dotenv import load_dotenv

# Import the enhanced modules
from .config import UAgentConfig
from .exceptions import handle_analysis_error, DataAnalysisError, SecurityError
from .utils import MemoryEfficientCSVProcessor, DataDeliveryOptimizer
from .file_handlers import SecureFileUploader, FileContentHandler

# Import existing system components
sys.path.append('src')
from uagents_adapter import LangchainRegisterTool, cleanup_uagent
from src.agents.data_analysis_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

# Set up logging with configuration
def setup_logging(config: UAgentConfig) -> logging.Logger:
    """Set up logging with the specified configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format
    )
    return logging.getLogger(__name__)


class EnhancedDataAnalysisUAgent:
    """
    Enhanced Data Analysis uAgent with improved architecture and security.
    
    This class wraps the DataAnalysisAgent with enhanced features:
    - Secure file handling
    - Memory-efficient processing
    - Structured error handling  
    - Configurable behavior
    - Performance optimizations
    """
    
    def __init__(self, config: Optional[UAgentConfig] = None):
        """
        Initialize the enhanced uAgent.
        
        Args:
            config: Configuration object, defaults to environment-based config
        """
        self.config = config or UAgentConfig.from_env()
        self.logger = setup_logging(self.config)
        
        # Initialize core components
        self.csv_processor = MemoryEfficientCSVProcessor(self.config)
        self.delivery_optimizer = DataDeliveryOptimizer(self.config)
        self.file_uploader = SecureFileUploader(self.config)
        self.content_handler = FileContentHandler(self.config)
        
        # Initialize the underlying data analysis agent
        self.data_analysis_agent = DataAnalysisAgent(
            output_dir=self.config.output_dir,
            intent_parser_model=self.config.intent_parser_model,
            enable_async=self.config.enable_async
        )
        
        # Ensure output_dir is set correctly for testing
        if hasattr(self.data_analysis_agent, 'output_dir'):
            self.data_analysis_agent.output_dir = self.config.output_dir
        
        # Session management
        self._last_cleaned_data = None
        self._last_processed_timestamp = None
        
        self.logger.info(f"Enhanced uAgent initialized with config: {self.config.to_dict()}")
    
    def process_query(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        Process a user query with enhanced error handling and optimization.
        
        Args:
            query: User query as string or dict with 'input' key
            
        Returns:
            Formatted response string
        """
        try:
            # Normalize input format (following LangGraph adapter pattern)
            if isinstance(query, dict) and 'input' in query:
                query_text = query['input']
            else:
                query_text = str(query)
            
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            # Check for follow-up data delivery requests
            if self._is_data_delivery_request(query_text):
                return self._handle_data_delivery_request(query_text)
            
            # Process the main analysis request
            return self._process_analysis_request(query_text)
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}", exc_info=True)
            return handle_analysis_error(e, "query_processing")
    
    def _is_data_delivery_request(self, query: str) -> bool:
        """Check if query is a follow-up data delivery request."""
        query_lower = query.lower()
        data_request_phrases = [
            'send my data', 'send my cleaned data', 'provide my cleaned data', 
            'provide my data', 'show me my processed data', 'show me my data',
            'my cleaned dataset', 'give me my data', 'deliver my data',
            'send rows', 'send columns', 'data in chunks', 'split my data'
        ]
        return any(phrase in query_lower for phrase in data_request_phrases)
    
    def _handle_data_delivery_request(self, query: str) -> str:
        """Handle follow-up requests for data delivery."""
        try:
            # Check if we have recent cleaned data
            if self._last_cleaned_data is None:
                return self._create_no_data_response()
            
            # Check if data is too old
            if self._is_session_expired():
                return self._create_expired_session_response()
            
            # Determine delivery strategy and process
            strategy = self.delivery_optimizer.determine_delivery_strategy(self._last_cleaned_data)
            
            if strategy == 'direct':
                return self._deliver_data_directly()
            elif strategy == 'chunked':
                return self._deliver_data_chunked(query)
            elif strategy == 'link':
                return self._deliver_data_as_link()
            else:
                return self._deliver_data_preview()
                
        except Exception as e:
            self.logger.error(f"Data delivery failed: {e}", exc_info=True)
            return handle_analysis_error(e, "data_delivery")
    
    def _process_analysis_request(self, query: str) -> str:
        """Process the main data analysis request."""
        try:
            # Use the underlying DataAnalysisAgent with enhanced error handling
            result = self.data_analysis_agent.analyze_from_text(query)
            
            # Store cleaned data for potential follow-up requests
            self._store_cleaned_data_if_available()
            
            # Format result with enhanced display
            return self._format_analysis_result_enhanced(result)
            
        except Exception as e:
            self.logger.error(f"Analysis request failed: {e}", exc_info=True)
            return self._create_analysis_error_response(e)
    
    def _store_cleaned_data_if_available(self):
        """Store cleaned data for follow-up requests."""
        try:
            if (hasattr(self.data_analysis_agent, 'data_cleaning_agent') and 
                self.data_analysis_agent.data_cleaning_agent):
                
                cleaned_df = self.data_analysis_agent.data_cleaning_agent.get_data_cleaned()
                if cleaned_df is not None and len(cleaned_df) > 0:
                    # Optimize memory usage
                    optimized_df = self.csv_processor.optimize_dataframe_memory(cleaned_df)
                    self._last_cleaned_data = optimized_df
                    self._last_processed_timestamp = time.time()
                    
                    memory_info = self.csv_processor.get_dataframe_memory_usage(optimized_df)
                    self.logger.info(f"Stored cleaned data: {memory_info['total_mb']:.2f} MB")
                    
        except Exception as e:
            self.logger.warning(f"Could not store cleaned data: {e}")
    
    def _format_analysis_result_enhanced(self, result) -> str:
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
            
            # Add workflow summary (moved to top for better UX)
            if result.workflow_intent:
                lines.extend(self._format_workflow_summary(result))
            
            # Add data transformation results
            lines.extend(self._format_data_transformation_results(result))
            
            # Add cleaned data information with enhanced delivery
            lines.extend(self._format_cleaned_data_section(result))
            
            # Add agent-specific results with enhanced file handling
            lines.extend(self._format_agent_results_enhanced(result))
            
            # Add insights and recommendations
            if result.key_insights:
                lines.extend([
                    "💡 **KEY INSIGHTS**:",
                    "─" * 20,
                    *[f"   • {insight}" for insight in result.key_insights],
                    ""
                ])
            
            if result.recommendations:
                lines.extend([
                    "🎯 **RECOMMENDATIONS**:",
                    "─" * 25,
                    *[f"   • {rec}" for rec in result.recommendations],
                    ""
                ])
            
            # Add completion summary
            lines.extend(self._format_completion_summary(result))
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Result formatting failed: {e}", exc_info=True)
            return f"❌ Error formatting result: {str(e)}\n\nRaw result available in logs."
    
    def _format_workflow_summary(self, result) -> list:
        """Format workflow execution summary."""
        lines = ["🔄 **WORKFLOW EXECUTION SUMMARY**:"]
        
        # Extract agent status
        agent_status = {}
        for agent_result in result.agent_results:
            status = "✅ Success" if agent_result.success else "❌ Failed"
            agent_status[agent_result.agent_name] = status
        
        lines.extend([
            f"   • Data Cleaning: {agent_status.get('data_cleaning', '❌ Not executed')}",
            f"   • Feature Engineering: {agent_status.get('feature_engineering', '❌ Not executed')}",
            f"   • ML Modeling: {agent_status.get('h2o_ml', '❌ Not executed')}",
            f"   • Intent Confidence: {result.workflow_intent.intent_confidence:.2f}",
            ""
        ])
        
        return lines
    
    def _format_data_transformation_results(self, result) -> list:
        """Format data transformation results."""
        lines = [
            "📈 **DATA TRANSFORMATION RESULTS**:",
            "─" * 40,
            ""
        ]
        
        # Extract shape information
        original_shape = result.data_shape
        lines.extend([
            f"   📏 **Original**: {original_shape.get('rows', 'unknown'):,} rows × {original_shape.get('columns', 'unknown')} columns",
            ""
        ])
        
        return lines
    
    def _format_cleaned_data_section(self, result) -> list:
        """Format cleaned data section with enhanced delivery options."""
        lines = []
        
        if self._last_cleaned_data is not None:
            strategy = self.delivery_optimizer.determine_delivery_strategy(self._last_cleaned_data)
            memory_info = self.csv_processor.get_dataframe_memory_usage(self._last_cleaned_data)
            
            lines.extend([
                "📊 **CLEANED DATA AVAILABLE**:",
                f"   • Dataset: {len(self._last_cleaned_data):,} rows × {len(self._last_cleaned_data.columns)} columns",
                f"   • Memory usage: {memory_info['total_mb']:.2f} MB",
                f"   • Delivery strategy: {strategy}",
                ""
            ])
            
            if strategy == 'direct':
                lines.extend([
                    "💡 **Data ready for immediate delivery**:",
                    "   Ask: 'Send me my cleaned data' for direct access",
                    ""
                ])
            elif strategy == 'chunked':
                lines.extend([
                    "💡 **Large dataset - chunked delivery available**:",
                    "   Ask: 'Send my data in 5 chunks' for manageable pieces",
                    ""
                ])
            else:
                lines.extend([
                    "💡 **Very large dataset - link delivery recommended**:",
                    "   Ask: 'Create download link for my data' for file access",
                    ""
                ])
        
        return lines
    
    def _format_agent_results_enhanced(self, result) -> list:
        """Format agent results with enhanced file handling."""
        lines = []
        
        # Handle generated files with enhanced security
        if result.generated_files:
            displayed_files = set()
            
            lines.extend([
                "📁 **GENERATED FILES**:",
                ""
            ])
            
            for name, path in result.generated_files.items():
                if path not in displayed_files:
                    file_display = self.content_handler.create_file_display(path, name)
                    lines.extend(file_display)
                    displayed_files.add(path)
                    lines.append("")
        
        return lines
    
    def _format_completion_summary(self, result) -> list:
        """Format completion summary."""
        return [
            "=" * 60,
            "✅ **Analysis completed successfully!**",
            "",
            "💡 **What's available:**",
            f"   • Enhanced results with {len(result.agent_results)} agent executions",
            f"   • Secure file handling with validation",
            f"   • Memory-optimized data processing",
            f"   • Flexible data delivery options",
            "",
            "🔍 **Need your data?** Ask: 'Send me my cleaned data' or 'How can I access my results?'",
            ""
        ]
    
    def _create_no_data_response(self) -> str:
        """Create response when no recent data is available."""
        return """
🚫 **No Recent Data Found**

I don't have any recently processed data to deliver. Please first run a data cleaning task, for example:

"Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

Then I can provide your cleaned data in various formats.
"""
    
    def _create_expired_session_response(self) -> str:
        """Create response when session has expired."""
        return """
🕐 **Data Session Expired**

Your cleaned data session has expired (older than 1 hour). Please re-run your data cleaning task to get fresh results.
"""
    
    def _create_analysis_error_response(self, error: Exception) -> str:
        """Create response for analysis errors."""
        return f"""
🚫 **Analysis Error**

Sorry, I encountered an issue: {str(error)}

**Common solutions:**
1. Include a direct CSV URL in your request (e.g., https://example.com/data.csv)
2. Be specific about what analysis you want
3. Example: "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"

**Need help?** Try: "Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species classification"
"""
    
    def _is_session_expired(self) -> bool:
        """Check if the current session has expired."""
        if self._last_processed_timestamp is None:
            return True
        
        session_age = time.time() - self._last_processed_timestamp
        return session_age > self.config.get_session_timeout_seconds()
    
    def _deliver_data_directly(self) -> str:
        """Deliver data directly as CSV content."""
        try:
            csv_content = self._last_cleaned_data.to_csv(index=False)
            content_size = len(csv_content.encode('utf-8'))
            
            return f"""
📁 **YOUR COMPLETE CLEANED DATA**

File size: {content_size / 1024:.1f} KB | Rows: {len(self._last_cleaned_data):,} | Columns: {len(self._last_cleaned_data.columns)}

```csv
{csv_content}
```

💡 **Usage**: Copy the CSV content above and save it as a .csv file for use in Excel, Python, R, or other tools.
"""
        except Exception as e:
            self.logger.error(f"Direct delivery failed: {e}")
            return handle_analysis_error(e, "direct_data_delivery")
    
    def _deliver_data_chunked(self, query: str) -> str:
        """Deliver data in chunks."""
        try:
            # Extract number of chunks from query if specified
            import re
            chunk_match = re.search(r'(\d+)\s*chunk', query.lower())
            num_chunks = int(chunk_match.group(1)) if chunk_match else 5
            
            chunk_size = len(self._last_cleaned_data) // num_chunks
            if chunk_size == 0:
                chunk_size = 1
            
            result_lines = [f"""
📦 **CHUNKED DATA DELIVERY**

Your cleaned data ({len(self._last_cleaned_data):,} rows × {len(self._last_cleaned_data.columns)} columns) split into {num_chunks} chunks.
Each chunk contains approximately {chunk_size} rows.

"""]
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(self._last_cleaned_data)) if i < num_chunks - 1 else len(self._last_cleaned_data)
                chunk_df = self._last_cleaned_data.iloc[start_idx:end_idx]
                
                result_lines.append(f"""
📋 **CHUNK {i+1}/{num_chunks}** (Rows {start_idx+1}-{end_idx})

```csv
{chunk_df.to_csv(index=False)}
```
""")
            
            result_lines.append("""
💡 **Combine chunks**: Copy each chunk and concatenate them to reconstruct your complete dataset.
""")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            self.logger.error(f"Chunked delivery failed: {e}")
            return handle_analysis_error(e, "chunked_data_delivery")
    
    def _deliver_data_as_link(self) -> str:
        """Deliver data as downloadable link."""
        try:
            # Create temporary file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            self._last_cleaned_data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            # Upload securely
            upload_result = self.file_uploader.upload_csv_secure(temp_file.name, "Cleaned Dataset")
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if upload_result.get("success"):
                return f"""
🌐 **DOWNLOAD LINK CREATED**

Your cleaned data has been securely uploaded and is ready for download:

🔗 **Download URL**: {upload_result['url']}
🏢 **Service**: {upload_result['service']} 
📦 **File ID**: {upload_result['file_id']}
📊 **Size**: {upload_result['size_mb']:.2f} MB
⏰ **Expires**: {upload_result['expires']}

💡 **How to use**:
1. Click the URL above to download your processed data
2. Save the file with a .csv extension  
3. Open in Excel, Python, R, or any data analysis tool
4. Share the link with colleagues or save for later use

⚠️  **Important**: File auto-deletes after expiration. Download promptly!
"""
            else:
                return f"""
❌ **Upload Failed**: {upload_result.get('error')}

Falling back to data preview:

{self.delivery_optimizer.create_data_preview(self._last_cleaned_data)}
"""
                
        except Exception as e:
            self.logger.error(f"Link delivery failed: {e}")
            return handle_analysis_error(e, "link_data_delivery")
    
    def _deliver_data_preview(self) -> str:
        """Deliver data as preview with metadata."""
        try:
            return self.delivery_optimizer.create_data_preview(self._last_cleaned_data)
        except Exception as e:
            self.logger.error(f"Preview delivery failed: {e}")
            return handle_analysis_error(e, "preview_data_delivery")


# Factory function for uAgent registration
def create_enhanced_uagent_function(config: Optional[UAgentConfig] = None):
    """
    Factory function to create the enhanced uAgent function.
    
    Args:
        config: Optional configuration, defaults to environment-based
        
    Returns:
        Callable function for uAgent registration
    """
    uagent = EnhancedDataAnalysisUAgent(config)
    
    def enhanced_data_analysis_agent_func(query: Union[str, Dict[str, Any]]) -> str:
        """Enhanced data analysis agent function for uAgent registration."""
        return uagent.process_query(query)
    
    return enhanced_data_analysis_agent_func


# Main execution
def main():
    """Main execution function for the enhanced uAgent."""
    # Validate environment
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")
    
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    if not API_TOKEN:
        print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")
    
    # Create configuration
    config = UAgentConfig.from_env()
    logger = setup_logging(config)
    
    # Create enhanced uAgent function
    enhanced_agent_func = create_enhanced_uagent_function(config)
    
    # Register with uAgent system
    tool = LangchainRegisterTool()
    
    logger.info("🚀 Registering enhanced data analysis uAgent v2...")
    
    agent_info = tool.invoke({
        "agent_obj": enhanced_agent_func,
        "name": config.name,
        "port": config.port,
        "description": config.description,
        "api_token": API_TOKEN,
        "mailbox": True
    })
    
    logger.info(f"✅ Registration result: {agent_info}")
    
    # Extract address info
    if isinstance(agent_info, dict):
        agent_address = agent_info.get('agent_address', 'Unknown')
        agent_port = agent_info.get('agent_port', str(config.port))
    else:
        agent_address = "Check logs above for actual address"
        agent_port = str(config.port)
    
    # Keep agent alive
    try:
        print("\n🎉 ENHANCED DATA ANALYSIS UAGENT V2 IS RUNNING!")
        print("=" * 60)
        print(f"🔗 Agent name: {config.name}")
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print("\n📋 Enhanced Features:")
        print("• 🔒 Enhanced security with file validation")
        print("• 💾 Memory-efficient processing")
        print("• 🎯 Structured error handling")
        print("• ⚙️ Configurable via environment variables")
        print("• 📊 Optimized data delivery strategies")
        print("\n💡 Usage examples:")
        print('- "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"')
        print('- "Send me my cleaned data in 3 chunks"')
        print('- "Create download link for my processed dataset"')
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Enhanced Data Analysis uAgent v2...")
        cleanup_uagent(config.name)
        logger.info("✅ Agent stopped.")


if __name__ == "__main__":
    main() 