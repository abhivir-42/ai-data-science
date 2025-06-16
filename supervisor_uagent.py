#!/usr/bin/env python3
"""
Supervisor uAgent Implementation

Following EXACTLY the Fetch.ai LangGraph adapter example:
https://innovationlab.fetch.ai/resources/docs/examples/adapters/langgraph-adapter-example

This wraps the SupervisorAgent as a uAgent for deployment on ASI:One.
"""

import os
import time
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from langchain_openai import ChatOpenAI
from uagents_adapter import LangchainRegisterTool, cleanup_uagent
from src.agents.supervisor_agent import SupervisorAgent

# Load environment variables
load_dotenv()

# Set API keys (exactly like the example)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

if not API_TOKEN:
    print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")

# Set up model and supervisor agent (following exact LangGraph pattern)
model = ChatOpenAI(temperature=0)
supervisor_agent = SupervisorAgent(model=model)

# Wrap supervisor agent into a function for UAgent (EXACTLY like LangGraph example)
def supervisor_agent_func(query):
    """
    Wrap the supervisor agent function following the EXACT LangGraph pattern.
    
    This handles input format conversion and returns the final response.
    Expected input format: {"csv_url": "url", "user_request": "request", "target_variable": "optional"}
    """
    # Handle input if it's a dict with 'input' key (EXACT pattern from example)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    try:
        # Parse the query to extract required parameters
        if isinstance(query, str):
            # Try to parse as simple request - assume it's a user request
            # For demo purposes, use a default CSV URL
            csv_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            user_request = query
            target_variable = None
        elif isinstance(query, dict):
            # Extract parameters from dict
            csv_url = query.get('csv_url', 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
            user_request = query.get('user_request', query.get('query', 'Clean and analyze the data'))
            target_variable = query.get('target_variable', None)
        else:
            return "Error: Invalid input format. Expected string or dict with csv_url and user_request."
        
        # Validate required parameters
        if not csv_url or not user_request:
            return "Error: Both csv_url and user_request are required."
        
        # Process the request using the supervisor agent
        result = supervisor_agent.process_request(
            csv_url=csv_url,
            user_request=user_request,
            target_variable=target_variable
        )
        
        return result
        
    except Exception as e:
        return f"Error in supervisor agent: {str(e)}"

# Register the supervisor agent via uAgent (EXACT pattern from LangGraph example)
tool = LangchainRegisterTool()

print("🚀 Registering supervisor uAgent...")

agent_info = tool.invoke(
    {
        "agent_obj": supervisor_agent_func,  # Pass the function
        "name": "supervisor_data_science",
        "port": 8101,
        "description": "A comprehensive data science supervisor agent that orchestrates data cleaning, feature engineering, and ML modeling from remote CSV files",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registration result: {agent_info}")
print(f"📊 Result type: {type(agent_info)}")

# Extract address info for display
if isinstance(agent_info, dict):
    agent_address = agent_info.get('agent_address', 'Unknown')
    agent_port = agent_info.get('agent_port', '8101')
elif isinstance(agent_info, str):
    # If it's a string, extract from logs
    agent_address = "Check logs above for actual address"
    agent_port = "8101"
else:
    agent_address = "Unknown"
    agent_port = "8101"

# Keep the agent alive (EXACT pattern from example)
if __name__ == "__main__":
    try:
        print("\n🎉 SUPERVISOR UAGENT IS RUNNING!")
        print("=" * 50)
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print("\n📋 Usage:")
        print("Send a message with:")
        print('- Simple string: "Clean and analyze the data"')
        print('- Dict format: {"csv_url": "url", "user_request": "request", "target_variable": "optional"}')
        print("\n🔗 Default CSV (if not specified): Titanic dataset")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down supervisor agent...")
        cleanup_uagent("supervisor_data_science")
        print("✅ Agent stopped.") 