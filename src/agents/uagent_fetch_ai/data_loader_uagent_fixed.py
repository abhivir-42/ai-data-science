"""
Data Loader uAgent - Fixed Implementation

Following the exact pattern from Fetch.ai LangGraph adapter example:
https://innovationlab.fetch.ai/resources/docs/examples/adapters/langgraph-adapter-example
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from uagents_adapter import LangchainRegisterTool, cleanup_uagent

# Import our existing agent
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from agents.data_loader_tools_agent import DataLoaderToolsAgent

# Load environment variables
load_dotenv()

# Set your API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

if not API_TOKEN:
    print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")

# Set up the LLM and agent (following the exact LangGraph pattern)
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.1,
    max_tokens=2000
)

# Create the DataLoaderToolsAgent instance
data_loader_agent = DataLoaderToolsAgent(model=model)
app = data_loader_agent._compiled_graph

# Wrap our agent into a function for uAgent (EXACTLY like the LangGraph example)
def data_loader_agent_func(query):
    """
    Wrap the DataLoaderToolsAgent following the exact LangGraph pattern.
    
    This function handles input format conversion and returns the final response.
    """
    # Handle input if it's a dict with 'input' key (EXACT pattern from example)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    # Handle if query is still a dict with user_instructions
    if isinstance(query, dict) and 'user_instructions' in query:
        query = query['user_instructions']
    
    # If query is still a dict, convert to string
    if isinstance(query, dict):
        query = str(query)
    
    # Prepare input for our agent (following our agent's expected format)
    agent_input = {
        "user_instructions": str(query)
    }
    
    try:
        # Invoke the agent (similar to LangGraph streaming pattern)
        result = app.invoke(agent_input)
        
        # Extract the final response (following LangGraph pattern)
        if isinstance(result, dict):
            # Look for the response in various possible keys
            if 'user_instructions' in result:
                final_response = result['user_instructions']
            elif 'response' in result:
                final_response = result['response']
            elif 'output' in result:
                final_response = result['output']
            else:
                final_response = str(result)
        else:
            final_response = str(result)
        
        return final_response if final_response else "No response from data loader agent"
        
    except Exception as e:
        return f"Error in data loader agent: {str(e)}"

# Register the agent via uAgent (EXACT pattern from LangGraph example)
tool = LangchainRegisterTool()
agent_info = tool.invoke(
    {
        "agent_obj": data_loader_agent_func,  # Pass the function, not the graph
        "name": "data_loader_agent",
        "port": 8100,
        "description": "A data loading agent that can load and analyze CSV, Excel, JSON and other data files",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registered Data Loader agent: {agent_info}")

# Handle both string and dict returns from registration
if isinstance(agent_info, dict):
    agent_address = agent_info.get('agent_address', 'Unknown')
    agent_port = agent_info.get('agent_port', '8100')
    agent_name = agent_info.get('agent_name', 'data_loader_agent')
else:
    # If it's a string, extract what we can
    agent_address = "Check output above for actual address"
    agent_port = "8100"
    agent_name = "data_loader_agent"

# Keep the agent alive (EXACT pattern from example)
if __name__ == "__main__":
    try:
        print("🚀 Data Loader agent is running...")
        print("📡 Send messages using the chat protocol to communicate with this agent")
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print()
        print("📋 To test with client agent:")
        print("1. Copy the agent address from the logs above")
        print("2. Update test_client_agent.py with the address")
        print("3. Run: python test_client_agent.py")
        print()
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down Data Loader agent...")
        cleanup_uagent("data_loader_agent")
        print("✅ Agent stopped.") 