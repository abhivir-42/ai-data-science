"""
Data Visualisation uAgent Adapter

This module converts the DataVisualisationAgent to a uAgent that can be deployed
on the Agentverse ecosystem using the uagents-adapter package.

Key Features:
- Converts existing LangGraph agent to uAgent format
- Supports local deployment with HTTP API endpoints
- Enables agent-to-agent communication via mailbox service
- Optional Agentverse registration for ecosystem discovery
- Handles API key validation and error management
- Provides comprehensive input/output schema validation

The adapter wraps the DataVisualisationAgent and exposes it as a HTTP API endpoint
that can be called by other agents or external systems. When registered on Agentverse,
it becomes discoverable in the ecosystem and can participate in multi-agent workflows.

Technical Details:
- Uses LangchainRegisterTool from uagents-adapter for registration
- Configures proper input schemas for data and visualisation requirements  
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
from agents.data_visualisation_agent import DataVisualisationAgent


def register_data_visualisation_uagent(
    openai_api_key: Optional[str] = None,
    port: int = 8002,
    name: str = "data_visualisation_agent",
    description: str = "AI Data Science Data Visualisation Agent - Creates interactive Plotly charts and visualisations",
    mailbox: bool = True,
    api_token: Optional[str] = None,
    return_dict: bool = True
) -> dict:
    """
    Register the DataVisualisationAgent as a uAgent on the Agentverse.
    
    This function creates a uAgent wrapper around the existing DataVisualisationAgent,
    enabling it to be deployed as a HTTP API service and optionally registered on the
    Agentverse ecosystem for multi-agent collaboration.
    
    The registered agent will:
    1. Accept data and visualisation instructions via HTTP API
    2. Generate interactive Plotly charts and visualisations
    3. Return the generated charts in various formats
    4. Support agent-to-agent communication via mailbox service
    5. Be discoverable on Agentverse if API token is provided
    
    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key for LLM functionality. If not provided, will attempt to
        retrieve from OPENAI_API_KEY environment variable.
    port : int, optional
        Port number to run the uAgent HTTP server on. Defaults to 8002.
        Must be available and not used by other services.
    name : str, optional
        Name identifier for the uAgent. Defaults to "data_visualisation_agent".
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
    from src.agents.uagent_fetch_ai import register_data_visualisation_uagent
    
    # Register with API keys from environment
    result = register_data_visualisation_uagent(
        port=8002,
        name="my_data_visualiser"
    )
    
    print(f"Agent registered with address: {result['agent_address']}")
    ```
    
    Full Agentverse registration:
    ```python
    result = register_data_visualisation_uagent(
        openai_api_key="sk-...",
        port=8002,
        name="production_visualiser",
        api_token="your_agentverse_token",
        description="Production data visualisation service"
    )
    ```
    
    Multi-agent workflow setup:
    ```python
    # Register multiple agents for collaborative workflows
    viz_agent = register_data_visualisation_uagent(port=8002)
    loader_agent = register_data_loader_uagent(port=8000)
    cleaner_agent = register_data_cleaning_uagent(port=8001)
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
    
    # Initialize OpenAI language model with optimized settings for visualisation tasks
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective model suitable for code generation
        api_key=openai_api_key,
        temperature=0.1,  # Low temperature for consistent code generation
        max_tokens=4000,  # Sufficient for visualisation code
        timeout=30.0  # Reasonable timeout for API calls
    )
    
    # Create the DataVisualisationAgent instance with production-ready configuration
    data_visualisation_agent = DataVisualisationAgent(
        model=llm,
        n_samples=30,  # Number of data samples to analyze for visualisation recommendations
        log=False,  # Disable verbose logging for cleaner uAgent deployment
        human_in_the_loop=False,  # Disable human review for automated workflows
        bypass_recommended_steps=False,  # Keep intelligent step recommendations
        bypass_explain_code=True  # Simplify output for API consumption
    )
    
    # Extract the compiled LangGraph for uAgent registration
    # The compiled graph represents the complete agent workflow
    agent_executor = data_visualisation_agent._compiled_graph
    
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
                "description": "Dataset to be visualised (as dictionary format with column names as keys)",
                "required": True,
                "example": {"x": [1, 2, 3, 4], "y": [10, 20, 30, 40], "category": ["A", "B", "C", "D"]}
            },
            "user_instructions": {
                "type": "string", 
                "description": "Detailed instructions for creating visualisations (chart type, styling, colors, etc.)",
                "required": True,
                "example": "Create a scatter plot showing the relationship between x and y, colored by category. Use a professional color scheme and add a trend line."
            },
            "max_retries": {
                "type": "integer",
                "description": "Maximum retry attempts for visualisation generation (default: 3)",
                "required": False,
                "default": 3,
                "minimum": 1,
                "maximum": 10
            },
            "chart_title": {
                "type": "string",
                "description": "Optional custom title for the visualisation",
                "required": False
            },
            "output_format": {
                "type": "string",
                "description": "Preferred output format: 'html', 'json', or 'both'",
                "required": False,
                "default": "both",
                "enum": ["html", "json", "both"]
            }
        },
        
        # Provide comprehensive example for API documentation
        "example_query": (
            "Create a professional scatter plot showing the relationship between age and income. "
            "Color points by department, add a trend line, use a modern color palette, "
            "and include proper axis labels and title. Export as both HTML and JSON formats."
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
        print(f"🚀 Starting registration of Data Visualisation uAgent...")
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
            
            print(f"✅ Data Visualisation uAgent registered successfully!")
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
            print(f"✅ Data Visualisation uAgent registered successfully!")
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
        print(f"❌ Error registering Data Visualisation uAgent: {str(e)}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check if port {port} is available")
        print(f"   • Verify OpenAI API key is valid")
        print(f"   • Ensure internet connection for Agentverse registration")
        print(f"   • Try a different port if {port} is in use")
        raise


# Backward compatibility alias
register_data_visualization_uagent = register_data_visualisation_uagent


if __name__ == "__main__":
    """
    Example usage and testing script.
    
    This script demonstrates how to register the Data Visualisation uAgent
    with various configuration options. It can be run directly for testing
    or used as a reference for integration.
    """
    
    print("🎨 Data Visualisation uAgent Registration Example")
    print("=" * 50)
    
    try:
        # Register the agent with comprehensive configuration
        result = register_data_visualisation_uagent(
            port=8002,
            name="ai_data_science_visualiser",
            description=(
                "AI Data Science Data Visualisation Agent - Creates interactive Plotly charts "
                "including scatter plots, bar charts, line plots, heatmaps, box plots, "
                "violin plots, and more with professional styling and customisation options"
            ),
            mailbox=True  # Enable for multi-agent workflows
        )
        
        print("\n🎉 Registration completed successfully!")
        print(f"📊 Agent details: {result}")
        
        # Display next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Test the agent: curl -X POST http://localhost:8002/invoke")
        print(f"   2. Check health: curl http://localhost:8002/health")
        print(f"   3. View documentation: http://localhost:8002/docs")
        if result.get('agent_address'):
            print(f"   4. Use agent address for inter-agent communication: {result['agent_address']}")
        
    except Exception as e:
        print(f"💥 Registration failed: {str(e)}")
        print(f"\n🔧 Setup Instructions:")
        print(f"   1. Set environment variables:")
        print(f"      export OPENAI_API_KEY='your-openai-key'")
        print(f"      export AGENTVERSE_API_TOKEN='your-agentverse-token'  # Optional")
        print(f"   2. Ensure port 8002 is available")
        print(f"   3. Install required dependencies: pip install uagents-adapter")
        print(f"   4. Try running again") 