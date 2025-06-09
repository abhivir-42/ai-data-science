#!/usr/bin/env python3
"""
Minimal Working uAgent Implementation

Following EXACTLY the Fetch.ai LangGraph adapter example:
https://innovationlab.fetch.ai/resources/docs/examples/adapters/langgraph-adapter-example

This is a working implementation with no extra complexity.
"""

import os
import time
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from langchain_openai import ChatOpenAI
from uagents_adapter import LangchainRegisterTool, cleanup_uagent
from agents.data_loader_tools_agent import DataLoaderToolsAgent

# Load environment variables
load_dotenv()

# Set API keys (exactly like the example)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

if not API_TOKEN:
    print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")

# Set up model and agent (following exact LangGraph pattern)
model = ChatOpenAI(temperature=0)
data_loader_agent = DataLoaderToolsAgent(model=model)
app = data_loader_agent._compiled_graph

# Wrap agent into a function for UAgent (EXACTLY like LangGraph example)
def data_loader_agent_func(query):
    """
    Wrap the agent function following the EXACT LangGraph pattern.
    
    This handles input format conversion and returns the final response.
    """
    # Handle input if it's a dict with 'input' key (EXACT pattern from example)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    # Our agent expects user_instructions format
    if isinstance(query, str):
        agent_input = {"user_instructions": query}
    else:
        agent_input = query
    
    try:
        # Invoke the agent
        result = app.invoke(agent_input)
        
        # Extract the final response (following LangGraph pattern)
        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
            if messages and hasattr(messages[-1], 'content'):
                return messages[-1].content
        
        return str(result)
        
    except Exception as e:
        return f"Error in data loader agent: {str(e)}"

# Register the agent via uAgent (EXACT pattern from LangGraph example)
tool = LangchainRegisterTool()

print("🚀 Registering minimal working uAgent...")

agent_info = tool.invoke(
    {
        "agent_obj": data_loader_agent_func,  # Pass the function
        "name": "minimal_data_loader",
        "port": 8100,
        "description": "A minimal working data loader agent",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registration result: {agent_info}")
print(f"📊 Result type: {type(agent_info)}")

# Extract address info for display
if isinstance(agent_info, dict):
    agent_address = agent_info.get('agent_address', 'Unknown')
    agent_port = agent_info.get('agent_port', '8100')
elif isinstance(agent_info, str):
    # If it's a string, extract from logs
    agent_address = "Check logs above for actual address"
    agent_port = "8100"
else:
    agent_address = "Unknown"
    agent_port = "8100"

# Keep the agent alive (EXACT pattern from example)
if __name__ == "__main__":
    try:
        print("\n🎉 MINIMAL WORKING UAGENT IS RUNNING!")
        print("=" * 50)
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print("\n📋 To test:")
        print("1. Agent is now running and registered")
        print("2. Check the logs above for the actual agent address")
        print("3. Visit the inspector URL to see the agent in Agentverse")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down agent...")
        cleanup_uagent("minimal_data_loader")
        print("✅ Agent stopped.") 