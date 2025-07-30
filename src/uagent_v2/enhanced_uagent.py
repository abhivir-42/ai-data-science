#!/usr/bin/env python3
"""
Enhanced Data Analysis uAgent Implementation (v2.0)

This is the improved version of the data analysis uAgent that addresses all the
critical issues identified in the original implementation with a modular architecture:

✅ Modular architecture (split into focused components from 1547-line monolith)
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
from .result_formatters import ResultFormatter
from .ml_processors import MLResultProcessor
from .data_delivery import DataDeliveryHandler, DataDeliveryRequestHandler
from .response_builders import ResponseBuilder, ErrorResponseBuilder

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
    - Modular component architecture
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
        
        # Initialize formatting and response components
        self.result_formatter = ResultFormatter(self.config)
        self.ml_processor = MLResultProcessor(self.config)
        self.data_delivery_handler = DataDeliveryHandler(self.config)
        self.data_delivery_request_handler = DataDeliveryRequestHandler(self.config)
        self.response_builder = ResponseBuilder(self.config)
        self.error_builder = ErrorResponseBuilder(self.config)
        
        # NEW: Initialize prediction formatter
        from .prediction_formatters import PredictionResponseFormatter
        self.prediction_formatter = PredictionResponseFormatter(self.config)
        
        # Initialize the underlying data analysis agent
        self.data_analysis_agent = DataAnalysisAgent(
            output_dir=self.config.output_dir,
            intent_parser_model=self.config.intent_parser_model,
            enable_async=self.config.enable_async
        )
        
        # Session management
        self._last_cleaned_data = None
        self._last_processed_timestamp = None
        
        # NEW: ML model session management  
        self._last_trained_model = None          # MLModelingMetrics object from successful ML training
        self._last_model_timestamp = None        # When model was trained
        self._last_training_result = None        # AgentExecutionResult with ML metrics
        self._last_target_variable = None        # Target variable used for training
        
        # NEW: Initialize intent parser for prediction recognition
        from src.parsers.intent_parser import DataAnalysisIntentParser
        self.intent_parser = DataAnalysisIntentParser(self.config.intent_parser_model)
        
        self.logger.info(f"Enhanced uAgent initialized with config: {self.config.to_dict()}")
    
    def process_query(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        Process a user query following the EXACT pattern from the original working uAgent.
        
        Args:
            query: User query as string or dict with 'input' key
            
        Returns:
            Formatted response string
        """
        try:
            # Handle input if it's a dict with 'input' key (EXACT pattern from original)
            if isinstance(query, dict) and 'input' in query:
                query_text = query['input']
            else:
                query_text = str(query)
            
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            query_lower = query_text.lower()
            
            # NEW: Use LLM intent parser to determine query type
            try:
                # Try to extract CSV URL from query first
                csv_url = ""
                try:
                    extraction_result = self.intent_parser.extract_dataset_url_from_text(query_text)
                    if extraction_result.extraction_confidence > 0.5:
                        csv_url = extraction_result.extracted_csv_url
                except Exception:
                    pass  # No URL found, continue with empty string
                
                # Parse intent (with or without CSV URL)
                if csv_url:
                    intent = self.intent_parser.parse_with_data_preview(query_text, csv_url)
                else:
                    # Use basic intent parsing without data preview for queries without URLs
                    # Pass context about existing model to help with prediction detection
                    model_context = {
                        "has_trained_model": self._has_trained_model(),
                        "target_variable": self._last_target_variable,
                        "model_timestamp": self._last_model_timestamp
                    }
                    intent = self.intent_parser.parse_intent(query_text, "", model_context)
                
                # Handle ML prediction requests
                if intent.needs_prediction:
                    return self._handle_prediction_request(query_text, intent)
                
                # Handle model analysis questions
                if intent.needs_model_analysis:
                    return self._handle_model_analysis_request(query_text, intent)
                    
            except Exception as e:
                self.logger.warning(f"Intent parsing failed, falling back to keyword detection: {e}")
            
            # Handle follow-up data delivery requests (EXACT pattern from original)
            # if any(phrase in query_lower for phrase in [
            #     'send my data', 'provide my cleaned data', 'show me my processed data',
            #     'my cleaned dataset', 'give me my data', 'deliver my data',
            #     'send rows', 'send columns', 'data in chunks', 'split my data'
            # ]):
            #     return self._handle_data_delivery_request(query_text)
            
            # Process the main analysis request (NO HELP DETECTION - like original)
            return self._process_analysis_request(query_text)
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}", exc_info=True)
            return self._create_error_response(e)
    
    def _create_error_response(self, error: Exception) -> str:
        """Create error response following the original pattern."""
        return f"""
🚫 **Analysis Error**

Sorry, I encountered an issue: {str(error)}

**Common solutions:**
1. Include a direct CSV URL in your request (e.g., https://example.com/data.csv)
2. Be specific about what analysis you want
3. Example: "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"

**Need help?** Try: "Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species classification"
"""
    
    def _handle_data_delivery_request(self, query: str) -> str:
        """Handle follow-up requests for data delivery."""
        try:
            # Check if we have recent cleaned data
            if self._last_cleaned_data is None:
                return self.response_builder.create_no_data_response()
            
            # Check if data is too old
            if self._is_session_expired():
                return self.response_builder.create_expired_session_response()
            
            # Use the data delivery request handler
            return self.data_delivery_request_handler.handle_data_delivery_request(
                query, self._last_cleaned_data
            )
                
        except Exception as e:
            self.logger.error(f"Data delivery failed: {e}", exc_info=True)
            return self.error_builder.build_generic_error_response(e, "data delivery")
    
    def _handle_prediction_request(self, query: str, intent) -> str:
        """Handle prediction requests using trained model."""
        try:
            # Check if we have a trained model
            if not self._has_trained_model():
                return self.prediction_formatter.format_no_model_response()
            
            # Create prediction agent
            from src.agents.ml_prediction_agent import MLPredictionAgent
            prediction_agent = MLPredictionAgent(
                self._last_trained_model, 
                self._last_target_variable, 
                self.config
            )
            
            # Execute prediction based on intent
            if intent.prediction_type == "single_prediction":
                if not intent.extracted_prediction_data:
                    return self.prediction_formatter.format_prediction_error_response(
                        Exception("No prediction data found. Please provide input values like: Age=25, Sex=male")
                    )
                
                result = prediction_agent.predict_single(intent.extracted_prediction_data)
                return self.prediction_formatter.format_single_prediction(result)
                
            elif intent.prediction_type == "batch_prediction":
                if not intent.prediction_data_source:
                    return self.prediction_formatter.format_prediction_error_response(
                        Exception("No CSV URL found. Please provide a CSV URL for batch prediction")
                    )
                
                result = prediction_agent.predict_batch(intent.prediction_data_source)
                return self.prediction_formatter.format_batch_prediction(result)
                
            else:
                return self.prediction_formatter.format_prediction_error_response(
                    Exception("Could not understand the prediction request type")
                )
                
        except Exception as e:
            self.logger.error(f"Prediction request failed: {e}")
            return self.prediction_formatter.format_prediction_error_response(e)

    def _handle_model_analysis_request(self, query: str, intent) -> str:
        """Handle model analysis questions."""
        try:
            # Check if we have a trained model
            if not self._has_trained_model():
                return self.prediction_formatter.format_no_model_response()
            
            # Create prediction agent for analysis
            from src.agents.ml_prediction_agent import MLPredictionAgent
            prediction_agent = MLPredictionAgent(
                self._last_trained_model, 
                self._last_target_variable, 
                self.config
            )
            
            # Analyze model
            result = prediction_agent.analyze_model(query)
            return self.prediction_formatter.format_model_analysis(result)
            
        except Exception as e:
            self.logger.error(f"Model analysis failed: {e}")
            return self.prediction_formatter.format_prediction_error_response(e)
    
    def _process_analysis_request(self, query: str) -> str:
        """Process the main data analysis request following the original pattern."""
        try:
            # Direct invocation of the underlying DataAnalysisAgent (EXACT pattern from original)
            result = self.data_analysis_agent.analyze_from_text(query)
            
            # Store cleaned data for potential follow-up requests (same as original)
            self._store_cleaned_data_if_available()
            
            # NEW: Store ML model for potential predictions
            self._store_ml_model_if_available(result)
            
            # Format the structured result for uAgent compatibility (same as original)
            return self.result_formatter.format_analysis_result_enhanced(result)
            
        except Exception as e:
            self.logger.error(f"Analysis request failed: {e}", exc_info=True)
            return self._create_error_response(e)
    
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
    
    def _is_session_expired(self) -> bool:
        """Check if the current session has expired."""
        if not self._last_processed_timestamp:
            return True
        
        session_age = time.time() - self._last_processed_timestamp
        max_age = self.config.session_timeout_hours * 3600
        return session_age > max_age
    
    def cleanup_session(self):
        """Clean up expired session data."""
        if self._is_session_expired():
            self._last_cleaned_data = None
            self._last_processed_timestamp = None
            self.logger.info("Session data cleaned up due to expiration")
    
    def _has_trained_model(self) -> bool:
        """Check if we have a valid trained model in session."""
        try:
            return (self._last_trained_model is not None and 
                    not self._is_model_session_expired() and
                    hasattr(self._last_trained_model, 'best_model_id') and
                    self._last_trained_model.best_model_id is not None)
        except Exception as e:
            self.logger.warning(f"Error checking model session: {e}")
            return False
    
    def _is_model_session_expired(self) -> bool:
        """Check if the ML model session has expired."""
        try:
            if not self._last_model_timestamp:
                return True
            
            # Handle corrupted timestamp data
            if not isinstance(self._last_model_timestamp, (int, float)):
                self.logger.warning(f"Invalid timestamp type: {type(self._last_model_timestamp)}")
                return True
            
            session_age = time.time() - self._last_model_timestamp
            max_age = self.config.session_timeout_hours * 3600
            return session_age > max_age
        except Exception as e:
            self.logger.warning(f"Error checking session expiration: {e}")
            return True  # Safe default: assume expired
    
    def _store_ml_model_if_available(self, result):
        """Store trained ML model information for follow-up predictions."""
        try:
            # Find ML agent result
            ml_agent_result = None
            for agent_result in result.agent_results:
                if agent_result.agent_name == "h2o_ml" and agent_result.success:
                    ml_agent_result = agent_result
                    break
            
            if ml_agent_result and ml_agent_result.ml_modeling_metrics:
                metrics = ml_agent_result.ml_modeling_metrics
                
                # Store the existing MLModelingMetrics directly (no new schema needed!)
                self._last_trained_model = metrics
                self._last_model_timestamp = time.time()
                self._last_training_result = ml_agent_result
                self._last_target_variable = self._extract_target_variable(result)
                
                self.logger.info(f"Stored trained model session: {metrics.best_model_id}")
                self.logger.info(f"Model path: {metrics.model_path}")
                self.logger.info(f"Target variable: {self._last_target_variable}")
                
        except Exception as e:
            self.logger.warning(f"Could not store ML model session: {e}")
    
    def _extract_target_variable(self, result) -> Optional[str]:
        """Extract target variable from the analysis result."""
        # Try workflow intent first
        if result.workflow_intent.suggested_target_variable:
            return result.workflow_intent.suggested_target_variable
        
        # Try to find from ML agent execution logs
        for agent_result in result.agent_results:
            if agent_result.agent_name == "h2o_ml" and agent_result.success:
                # Target variable might be in log messages or metadata
                # This would need to be extracted from the actual agent execution
                pass
        
        return None


def create_enhanced_uagent_function(config: Optional[UAgentConfig] = None):
    """
    Factory function to create an enhanced uAgent function.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Function that can be used with LangchainRegisterTool
    """
    # Create the enhanced uAgent instance
    enhanced_uagent = EnhancedDataAnalysisUAgent(config)
    
    def enhanced_data_analysis_agent_func(query: Union[str, Dict[str, Any]]) -> str:
        """Enhanced data analysis agent function for uAgent registration."""
        return enhanced_uagent.process_query(query)
    
    return enhanced_data_analysis_agent_func


def main():
    """Main function for running the enhanced uAgent."""
    # Get API keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")
    
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    if not API_TOKEN:
        print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")
    
    # Create configuration and enhanced uAgent function
    config = UAgentConfig.from_env()
    #logger = setup_logging(config)

    # Create enhanced uAgent function
    enhanced_agent_func = create_enhanced_uagent_function(config)
    
    # Register the enhanced uAgent
    tool = LangchainRegisterTool()
    
    print("🚀 Registering Data Science Agent")
    
    agent_info = tool.invoke(
        {
            "agent_obj": enhanced_agent_func,
            "name": config.name,
            "port": config.agent_port,
            "description": config.description,
            "api_token": API_TOKEN,
            "mailbox": True
        }
    )
    
    print(f"✅ Registration result: {agent_info}")
    
    # Extract address info for display
    if isinstance(agent_info, dict):
        agent_address = agent_info.get('agent_address', 'Unknown')
        agent_port = agent_info.get('agent_port', config.agent_port)
    elif isinstance(agent_info, str):
        agent_address = "Check logs above for actual address"
        agent_port = config.agent_port
    else:
        agent_address = "Unknown"
        agent_port = config.agent_port
    
    # Keep the agent alive
    try:
        print("\n🎉 ENHANCED DATA ANALYSIS UAGENT IS RUNNING!")
        print("=" * 70)
        print(f"🔗 Agent name: AI Data Science Agent")
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print(f"📊 Configuration: {config.to_dict()}")
        print("\n📋 Usage:")
        print("Send a message with a CSV URL and analysis request:")
        print('- "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"')
        print('- "Perform feature engineering on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"')
        print('- "Build ML model using https://example.com/your-data.csv to predict target_column"')
        print("\n🚀 Enhanced Features:")
        #print("• Modular architecture for better maintainability")
        #print("• Enhanced security with file validation")
        #print("• Memory-efficient processing")
        #print("• Structured error handling")
        #print("• Configurable behavior via environment variables")
        print("• Smart data delivery strategies")
        print("• Comprehensive ML result formatting")
        print("\n🎯 The agent uses AI to:")
        print("• Extract CSV URLs from your text using LLM structured outputs")
        print("• Parse your intent to determine which analysis steps to run")
        print("• Execute only the needed agents (cleaning, feature engineering, ML)")
        print("• Return comprehensive structured results with download links")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Enhanced AI Data Science Agent v2.0...")
        cleanup_uagent("Enhanced AI Data Science Agent v2.0")
        print("✅ Agent stopped.")


if __name__ == "__main__":
    main() 