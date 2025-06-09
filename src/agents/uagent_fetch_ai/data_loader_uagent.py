"""
Data Loader uAgent Adapter

This module converts the DataLoaderToolsAgent to a uAgent that can be deployed
on the Agentverse ecosystem using the uagents-adapter package.

Key Features:
- Converts existing LangGraph agent to uAgent format
- Supports local deployment with HTTP API endpoints
- Enables agent-to-agent communication via mailbox service
- Optional Agentverse registration for ecosystem discovery
- Handles API key validation and error management
- Provides comprehensive input/output schema validation

The adapter wraps the DataLoaderToolsAgent and exposes it as a HTTP API endpoint
that can load data from various sources (CSV, Excel, JSON, Parquet, etc.) and
provide structured responses for downstream processing by other agents.

Technical Details:
- Uses LangchainRegisterTool from uagents-adapter for registration
- Configures proper input schemas for data loading requirements
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
from agents.data_loader_tools_agent import DataLoaderToolsAgent


def register_data_loader_uagent(
    openai_api_key: Optional[str] = None,
    port: int = 8000,
    name: str = "data_loader_agent",
    description: str = "AI Data Science Data Loader Agent - Loads data from various sources and formats",
    mailbox: bool = True,
    api_token: Optional[str] = None,
    return_dict: bool = True
) -> dict:
    """
    Register the DataLoaderToolsAgent as a uAgent on the Agentverse.
    
    This function creates a uAgent wrapper around the existing DataLoaderToolsAgent,
    enabling it to be deployed as a HTTP API service and optionally registered on the
    Agentverse ecosystem for multi-agent collaboration.
    
    The registered agent will:
    1. Accept data loading instructions via HTTP API
    2. Load data from various file formats (CSV, Excel, JSON, Parquet, etc.)
    3. Provide data summaries and metadata information
    4. Support agent-to-agent communication via mailbox service
    5. Be discoverable on Agentverse if API token is provided
    
    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key for LLM functionality. If not provided, will attempt to
        retrieve from OPENAI_API_KEY environment variable.
    port : int, optional
        Port number to run the uAgent HTTP server on. Defaults to 8000.
        Must be available and not used by other services.
    name : str, optional
        Name identifier for the uAgent. Defaults to "data_loader_agent".
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
    from src.agents.uagent_fetch_ai import register_data_loader_uagent
    
    # Register with API keys from environment
    result = register_data_loader_uagent(
        port=8000,
        name="my_data_loader"
    )
    
    print(f"Agent registered with address: {result['agent_address']}")
    ```
    
    Full Agentverse registration:
    ```python
    result = register_data_loader_uagent(
        openai_api_key="sk-...",
        port=8000,
        name="production_loader",
        api_token="your_agentverse_token",
        description="Production data loading service"
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
    
    # Initialize OpenAI language model with optimized settings for data loading tasks
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective model suitable for data processing tasks
        api_key=openai_api_key,
        temperature=0.1,  # Low temperature for consistent processing
        max_tokens=4000,  # Sufficient for data analysis responses
        timeout=30.0  # Reasonable timeout for API calls
    )
    
    # Create the DataLoaderToolsAgent instance with production-ready configuration
    data_loader_agent = DataLoaderToolsAgent(
        model=llm  # Pass the configured LLM instance
    )
    
    # Extract the compiled LangGraph for uAgent registration
    # The compiled graph represents the complete agent workflow
    agent_executor = data_loader_agent._compiled_graph
    
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
            "user_instructions": {
                "type": "string",
                "description": "Detailed instructions for loading data from files or directories. Specify file paths, data sources, or loading requirements.",
                "required": True,
                "example": "Load the CSV file from data/sales_data.csv and provide a comprehensive summary including column types, missing values, and basic statistics."
            },
            "file_path": {
                "type": "string",
                "description": "Optional specific file path to load data from",
                "required": False,
                "example": "data/sample_data.csv"
            },
            "file_format": {
                "type": "string",
                "description": "Optional file format hint: 'csv', 'excel', 'json', 'parquet', 'auto'",
                "required": False,
                "default": "auto",
                "enum": ["csv", "excel", "json", "parquet", "auto"]
            },
            "encoding": {
                "type": "string",
                "description": "Optional file encoding (for text files)",
                "required": False,
                "default": "utf-8",
                "example": "utf-8"
            },
            "separator": {
                "type": "string",
                "description": "Optional delimiter for CSV files",
                "required": False,
                "default": ",",
                "example": ","
            },
            "max_rows": {
                "type": "integer",
                "description": "Optional maximum number of rows to load (for large files)",
                "required": False,
                "minimum": 1,
                "maximum": 1000000,
                "example": 10000
            }
        },
        
        # Provide comprehensive example for API documentation
        "example_query": (
            "Load the CSV file from data/customer_data.csv with semicolon separator. "
            "Provide a detailed analysis including data types, missing values, "
            "basic statistics, and data quality assessment. Limit to first 50000 rows."
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
        print(f"🚀 Starting registration of Data Loader uAgent...")
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
            
            print(f"✅ Data Loader uAgent registered successfully!")
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
            # If we got a dict (future versions might return this), use it directly
            print(f"✅ Data Loader uAgent registered successfully!")
            print(f"   Name: {result.get('agent_name', name)}")
            print(f"   Address: {result.get('agent_address', 'N/A')}")
            print(f"   Port: {result.get('agent_port', port)}")
            print(f"   Status: {result.get('status', 'Active')}")
            
            return result
        else:
            # Unexpected return type
            print(f"⚠️ Unexpected return type: {type(result)}")
            return {"error": f"Unexpected return type: {type(result)}", "raw_result": result}
        
    except Exception as e:
        print(f"❌ Error registering Data Loader uAgent: {str(e)}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check if port {port} is available")
        print(f"   • Verify OpenAI API key is valid")
        print(f"   • Ensure internet connection for Agentverse registration")
        print(f"   • Try a different port if {port} is in use")
        raise


if __name__ == "__main__":
    """
    Example usage and testing script.
    
    This script demonstrates how to register the Data Loader uAgent
    with various configuration options. It can be run directly for testing
    or used as a reference for integration.
    """
    
    print("📊 Data Loader uAgent Registration Example")
    print("=" * 50)
    
    try:
        # Register the agent with comprehensive configuration
        result = register_data_loader_uagent(
            port=8000,
            name="ai_data_science_loader",
            description=(
                "AI Data Science Data Loader Agent - Loads and processes data from various "
                "file formats including CSV, Excel, JSON, Parquet, and other structured data "
                "sources with intelligent data type detection and quality assessment"
            ),
            mailbox=True  # Enable for multi-agent workflows
        )
        
        print("\n🎉 Registration completed successfully!")
        print(f"📊 Agent details: {result}")
        
        # Display next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Test the agent: curl -X POST http://localhost:8000/invoke")
        print(f"   2. Check health: curl http://localhost:8000/health")
        print(f"   3. View documentation: http://localhost:8000/docs")
        if result.get('agent_address'):
            print(f"   4. Use agent address for inter-agent communication: {result['agent_address']}")
        
    except Exception as e:
        print(f"💥 Registration failed: {str(e)}")
        print(f"\n🔧 Setup Instructions:")
        print(f"   1. Set environment variables:")
        print(f"      export OPENAI_API_KEY='your-openai-key'")
        print(f"      export AGENTVERSE_API_TOKEN='your-agentverse-token'  # Optional")
        print(f"   2. Ensure port 8000 is available")
        print(f"   3. Install required dependencies: pip install uagents-adapter")
        print(f"   4. Try running again") 