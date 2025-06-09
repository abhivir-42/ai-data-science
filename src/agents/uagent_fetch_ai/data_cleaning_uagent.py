"""
Data Cleaning uAgent Adapter

This module converts the DataCleaningAgent to a uAgent that can be deployed
on the Agentverse ecosystem using the uagents-adapter package.

Key Features:
- Converts existing LangGraph agent to uAgent format
- Supports local deployment with HTTP API endpoints
- Enables agent-to-agent communication via mailbox service
- Optional Agentverse registration for ecosystem discovery
- Handles API key validation and error management
- Provides comprehensive input/output schema validation

The adapter wraps the DataCleaningAgent and exposes it as a HTTP API endpoint
that can clean and preprocess datasets using intelligent algorithms and best
practices. It supports various cleaning operations including missing value
imputation, outlier detection, data type optimization, and more.

Technical Details:
- Uses LangchainRegisterTool from uagents-adapter for registration
- Configures proper input schemas for data cleaning requirements
- Handles OpenAI API key management and validation
- Supports both local testing and production deployment
- Implements proper error handling and logging
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from uagents_adapter import LangchainRegisterTool

# Import the agent - using relative imports for better package structure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from agents.data_cleaning_agent import DataCleaningAgent


def register_data_cleaning_uagent(
    openai_api_key: Optional[str] = None,
    port: int = 8001,
    name: str = "data_cleaning_agent",
    description: str = "AI Data Science Data Cleaning Agent - Cleans and preprocesses datasets using best practices",
    mailbox: bool = True,
    api_token: Optional[str] = None,
    return_dict: bool = True
) -> dict:
    """
    Register the DataCleaningAgent as a uAgent on the Agentverse.
    
    This function creates a uAgent wrapper around the existing DataCleaningAgent,
    enabling it to be deployed as a HTTP API service and optionally registered on the
    Agentverse ecosystem for multi-agent collaboration.
    
    The registered agent will:
    1. Accept raw datasets and cleaning instructions via HTTP API
    2. Apply intelligent data cleaning and preprocessing techniques
    3. Handle missing values, outliers, and data quality issues
    4. Return cleaned datasets with detailed processing reports
    5. Support agent-to-agent communication via mailbox service
    6. Be discoverable on Agentverse if API token is provided
    
    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key for LLM functionality. If not provided, will attempt to
        retrieve from OPENAI_API_KEY environment variable.
    port : int, optional
        Port number to run the uAgent HTTP server on. Defaults to 8001.
        Must be available and not used by other services.
    name : str, optional
        Name identifier for the uAgent. Defaults to "data_cleaning_agent".
        Should be unique within your agent ecosystem.
    description : str, optional
        Human-readable description of the uAgent's capabilities.
        Used for documentation and discovery on Agentverse.
    mailbox : bool, optional
        Whether to enable Agentverse mailbox service for agent-to-agent
        communication. Defaults to True for better interoperability.
    api_token : str, optional
        Agentverse API token for ecosystem registration. If not provided,
        will attempt to retrieve from AGENTVERSE_API_TOKEN environment variable.
        Agent will only be registered locally if not provided.
    return_dict : bool, optional
        Whether to return registration result as dictionary format.
        Defaults to True for structured response handling.
        
    Returns
    -------
    dict
        Registration result containing:
        - agent_name: The registered name of the agent
        - agent_address: Unique agent address (agent1q...)
        - agent_port: Port number the agent is running on
        - status: Registration status information
        - api_endpoints: Available HTTP endpoints
        
    Raises
    ------
    ValueError
        If OpenAI API key is not provided and not available in environment
    Exception
        If agent registration fails due to port conflicts or API issues
        
    Examples
    --------
    Basic local registration:
    ```python
    from src.agents.uagent_fetch_ai import register_data_cleaning_uagent
    
    # Register with API keys from environment
    result = register_data_cleaning_uagent(
        port=8001,
        name="my_data_cleaner"
    )
    
    print(f"Agent registered with address: {result['agent_address']}")
    ```
    
    Full Agentverse registration:
    ```python
    result = register_data_cleaning_uagent(
        openai_api_key="sk-...",
        port=8001,
        name="production_cleaner",
        api_token="your_agentverse_token",
        description="Production data cleaning service"
    )
    ```
    
    Multi-agent workflow setup:
    ```python
    # Register multiple agents for collaborative workflows
    loader_agent = register_data_loader_uagent(port=8000)
    cleaner_agent = register_data_cleaning_uagent(port=8001)
    viz_agent = register_data_visualisation_uagent(port=8002)
    ```
    """
    
    # Validate and retrieve OpenAI API key
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set. "
                "Please provide an API key or set the environment variable."
            )
    
    # Attempt to retrieve Agentverse API token if not provided
    if api_token is None:
        api_token = os.getenv("AGENTVERSE_API_TOKEN")
        # Note: This is optional - agent can run locally without Agentverse registration
    
    # Initialize OpenAI language model with optimized settings for data cleaning tasks
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective model suitable for data analysis
        api_key=openai_api_key,
        temperature=0.1,  # Low temperature for consistent data processing
        max_tokens=4000,  # Sufficient for cleaning code and reports
        timeout=30.0  # Reasonable timeout for API calls
    )
    
    # Create the DataCleaningAgent instance with production-ready configuration
    data_cleaning_agent = DataCleaningAgent(
        model=llm,
        n_samples=30,  # Number of data samples to analyze for cleaning strategies
        log=False,  # Disable verbose logging for cleaner uAgent deployment
        human_in_the_loop=False,  # Disable human review for automated workflows
        bypass_recommended_steps=False,  # Keep intelligent cleaning recommendations
        bypass_explain_code=True  # Simplify output for API consumption
    )
    
    # Extract the compiled LangGraph for uAgent registration
    # The compiled graph represents the complete agent workflow
    agent_executor = data_cleaning_agent._compiled_graph
    
    # Initialize the uagents-adapter registration tool
    register_tool = LangchainRegisterTool()
    
    # Configure comprehensive registration parameters
    register_params = {
        # Core agent configuration
        "agent_obj": agent_executor,  # The actual agent to be wrapped
        "name": name,  # Agent identifier
        "port": port,  # HTTP server port
        "description": description,  # Human-readable description
        "mailbox": mailbox,  # Enable inter-agent communication
        "return_dict": return_dict,  # Structured response format
        
        # Define input schema for API validation and documentation
        "query_params": {
            "data_raw": {
                "type": "object",
                "description": "Raw dataset to be cleaned (as dictionary format with column names as keys)",
                "required": True,
                "example": {
                    "age": [25, None, 30, 45, 200],  # Contains missing and outlier values
                    "income": [50000, 75000, None, 120000, -5000],  # Contains missing and invalid values
                    "name": ["Alice", "Bob", "Charlie", "", "David"]  # Contains empty strings
                }
            },
            "user_instructions": {
                "type": "string", 
                "description": "Custom cleaning instructions and preferences. If not provided, will use intelligent best practices.",
                "required": False,
                "example": "Remove rows with missing values in critical columns, handle outliers using IQR method, and standardize categorical variables. Preserve as much data as possible."
            },
            "cleaning_strategy": {
                "type": "string",
                "description": "Preferred cleaning approach: 'conservative', 'aggressive', or 'balanced'",
                "required": False,
                "default": "balanced",
                "enum": ["conservative", "aggressive", "balanced"]
            },
            "handle_missing": {
                "type": "string",
                "description": "Strategy for missing values: 'remove', 'impute', or 'auto'",
                "required": False,
                "default": "auto",
                "enum": ["remove", "impute", "auto"]
            },
            "handle_outliers": {
                "type": "string",
                "description": "Strategy for outliers: 'remove', 'transform', 'keep', or 'auto'",
                "required": False,
                "default": "auto",
                "enum": ["remove", "transform", "keep", "auto"]
            },
            "max_retries": {
                "type": "integer",
                "description": "Maximum retry attempts for cleaning operations (default: 3)",
                "required": False,
                "default": 3,
                "minimum": 1,
                "maximum": 10
            },
            "preserve_original": {
                "type": "boolean",
                "description": "Whether to include original data in response for comparison",
                "required": False,
                "default": True
            }
        },
        
        # Provide comprehensive example for API documentation
        "example_query": (
            "Clean this sales dataset using balanced approach. Handle missing values through "
            "intelligent imputation, detect and transform outliers using statistical methods, "
            "optimize data types for efficiency, and provide detailed cleaning report with "
            "before/after statistics."
        )
    }
    
    # Add Agentverse API token if available for ecosystem registration
    if api_token:
        register_params["api_token"] = api_token
        print("🌐 Agentverse API token detected - will register on ecosystem")
    else:
        print("🏠 No Agentverse token - registering locally only")
    
    # Attempt agent registration with comprehensive error handling
    try:
        print(f"🚀 Starting registration of Data Cleaning uAgent...")
        print(f"   Name: {name}")
        print(f"   Port: {port}")
        print(f"   Mailbox: {'Enabled' if mailbox else 'Disabled'}")
        
        result = register_tool.invoke(register_params)
        
        # Handle return type - the tool returns a string, but we need to extract agent info
        if isinstance(result, str):
            # Extract agent address and other info from the string result
            import re
            
            # Try to extract agent address from the string
            agent_address_match = re.search(r'agent1q[a-z0-9]+', result)
            agent_address = agent_address_match.group(0) if agent_address_match else None
            
            # Create a dict response for consistency
            structured_result = {
                "agent_name": name,
                "agent_address": agent_address,
                "agent_port": port,
                "status": "Active",
                "message": result,
                "registration_type": "uagents-adapter",
                "mailbox_enabled": mailbox
            }
            
            print(f"✅ Data Cleaning uAgent registered successfully!")
            print(f"   Name: {structured_result['agent_name']}")
            print(f"   Address: {structured_result['agent_address']}")
            print(f"   Port: {structured_result['agent_port']}")
            print(f"   Status: {structured_result['status']}")
            
            # Agentverse registration confirmation
            if api_token:
                print(f"   🌐 Successfully registered on Agentverse!")
                print(f"   🏷️  Badge: innovationlab")
                print(f"   🔍 Your agent is now discoverable in the ecosystem")
            else:
                print(f"   🏠 Running locally - add AGENTVERSE_API_TOKEN to enable ecosystem registration")
            
            # Usage instructions
            print(f"\n📖 Usage Instructions:")
            print(f"   • HTTP API: POST http://localhost:{port}/invoke")
            print(f"   • Health check: GET http://localhost:{port}/health")
            print(f"   • Agent address: {structured_result['agent_address']}")
            
            if return_dict:
                return structured_result
            else:
                return result
                
        elif isinstance(result, dict):
            print(f"✅ Data Cleaning uAgent registered successfully!")
            print(f"   Name: {result.get('agent_name', name)}")
            print(f"   Address: {result.get('agent_address', 'N/A')}")
            print(f"   Port: {result.get('agent_port', port)}")
            print(f"   Status: {result.get('status', 'Active')}")
            
            # Display available endpoints
            if 'endpoints' in result or 'api_endpoints' in result:
                endpoints = result.get('endpoints', result.get('api_endpoints', []))
                print(f"   Available endpoints: {endpoints}")
            
            # Agentverse registration confirmation
            if api_token:
                print(f"   🌐 Successfully registered on Agentverse!")
                print(f"   🏷️  Badge: innovationlab")
                print(f"   🔍 Your agent is now discoverable in the ecosystem")
            else:
                print(f"   🏠 Running locally - add AGENTVERSE_API_TOKEN to enable ecosystem registration")
            
            # Usage instructions
            print(f"\n📖 Usage Instructions:")
            print(f"   • HTTP API: POST http://localhost:{port}/invoke")
            print(f"   • Health check: GET http://localhost:{port}/health")
            print(f"   • Agent address: {result.get('agent_address', 'N/A')}")
            
            return result
        else:
            # Unexpected return type
            print(f"⚠️ Unexpected return type: {type(result)}")
            return {"error": f"Unexpected return type: {type(result)}", "raw_result": result}
        
    except Exception as e:
        print(f"❌ Error registering Data Cleaning uAgent: {str(e)}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check if port {port} is available")
        print(f"   • Verify OpenAI API key is valid")
        print(f"   • Ensure internet connection for Agentverse registration")
        print(f"   • Try a different port if {port} is in use")
        raise


if __name__ == "__main__":
    """
    Example usage and testing script.
    
    This script demonstrates how to register the Data Cleaning uAgent
    with various configuration options. It can be run directly for testing
    or used as a reference for integration.
    """
    
    print("🧹 Data Cleaning uAgent Registration Example")
    print("=" * 50)
    
    try:
        # Register the agent with comprehensive configuration
        result = register_data_cleaning_uagent(
            port=8001,
            name="ai_data_science_cleaner",
            description=(
                "AI Data Science Data Cleaning Agent - Cleans and preprocesses datasets using "
                "industry best practices including intelligent missing value imputation, "
                "statistical outlier handling, data type optimization, and comprehensive "
                "quality assessment with detailed reporting"
            ),
            mailbox=True  # Enable for multi-agent workflows
        )
        
        print("\n🎉 Registration completed successfully!")
        print(f"🧹 Agent details: {result}")
        
        # Display next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Test the agent: curl -X POST http://localhost:8001/invoke")
        print(f"   2. Check health: curl http://localhost:8001/health")
        print(f"   3. View documentation: http://localhost:8001/docs")
        if result.get('agent_address'):
            print(f"   4. Use agent address for inter-agent communication: {result['agent_address']}")
        
    except Exception as e:
        print(f"💥 Registration failed: {str(e)}")
        print(f"\n🔧 Setup Instructions:")
        print(f"   1. Set environment variables:")
        print(f"      export OPENAI_API_KEY='your-openai-key'")
        print(f"      export AGENTVERSE_API_TOKEN='your-agentverse-token'  # Optional")
        print(f"   2. Ensure port 8001 is available")
        print(f"   3. Install required dependencies: pip install uagents-adapter")
        print(f"   4. Try running again") 