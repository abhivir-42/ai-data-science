"""
Data Cleaning uAgent - Fixed Implementation

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
from agents.data_cleaning_agent import DataCleaningAgent

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

# Create the DataCleaningAgent instance
data_cleaning_agent = DataCleaningAgent(model=model)
app = data_cleaning_agent._compiled_graph

# Wrap our agent into a function for uAgent (EXACTLY like the LangGraph example)
def data_cleaning_agent_func(query):
    """
    Wrap the DataCleaningAgent following the exact LangGraph pattern.
    
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
        
        return final_response if final_response else "No response from data cleaning agent"
        
    except Exception as e:
        return f"Error in data cleaning agent: {str(e)}"

# Register the agent via uAgent (EXACT pattern from LangGraph example)
tool = LangchainRegisterTool()
agent_info = tool.invoke(
    {
        "agent_obj": data_cleaning_agent_func,  # Pass the function, not the graph
        "name": "data_cleaning_agent",
        "port": 8101,
        "description": "A data cleaning agent that provides recommendations and code for cleaning datasets",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registered Data Cleaning agent: {agent_info}")

# Keep the agent alive (EXACT pattern from example)
if __name__ == "__main__":
    try:
        print("🚀 Data Cleaning agent is running...")
        print("📡 Send messages using the chat protocol to communicate with this agent")
        print(f"🔗 Agent address: {agent_info.get('agent_address', 'Unknown')}")
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down Data Cleaning agent...")
        cleanup_uagent("data_cleaning_agent")
        print("✅ Agent stopped.") 