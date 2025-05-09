"""
Example script for registering both DataCleaningAgent and DataLoaderToolsAgent
with Fetch.ai Agentverse.

This script demonstrates how to use the adapter pattern to register our agents with
the Fetch.ai Agentverse, allowing them to interact with other agents in the ecosystem.
"""

import os
import time
import argparse
from langchain_openai import ChatOpenAI

from src.adapters import DataCleaningAgentAdapter, DataLoaderToolsAgentAdapter

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")

# Set your Agentverse API key here or use environment variable
AGENTVERSE_API_KEY = os.environ.get("AGENTVERSE_API_KEY", "your-agentverse-api-key")

def register_agents(agents_to_register=None, verbose=True):
    """
    Register selected agents with Fetch.ai Agentverse.
    
    Parameters
    ----------
    agents_to_register : list, optional
        List of agent names to register ('data_cleaner', 'data_loader', or 'all')
    verbose : bool, optional
        Whether to print progress messages
        
    Returns
    -------
    dict
        Dictionary of registered agent information
    """
    if agents_to_register is None:
        agents_to_register = ["all"]
    
    if "all" in agents_to_register:
        agents_to_register = ["data_cleaner", "data_loader"]
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )
    
    registered_agents = {}
    
    # Register DataCleaningAgent if requested
    if "data_cleaner" in agents_to_register:
        if verbose:
            print("\n🧹 Registering DataCleaningAgent with Agentverse...")
        
        data_cleaner_adapter = DataCleaningAgentAdapter(
            model=llm,
            name="data_cleaning_agent",
            port=8000,
            api_token=AGENTVERSE_API_KEY,
            mailbox=True
        )
        
        result = data_cleaner_adapter.register()
        registered_agents["data_cleaner"] = result
        
        if verbose:
            if "error" in result:
                print(f"❌ Failed to register DataCleaningAgent: {result['error']}")
                print(f"   Solution: {result.get('solution', 'Check configuration and try again')}")
            else:
                print(f"✅ DataCleaningAgent registered successfully!")
                print(f"   Address: {result.get('address', 'unknown')}")
                print(f"   Port: 8000")
    
    # Register DataLoaderToolsAgent if requested
    if "data_loader" in agents_to_register:
        if verbose:
            print("\n📂 Registering DataLoaderToolsAgent with Agentverse...")
        
        data_loader_adapter = DataLoaderToolsAgentAdapter(
            model=llm,
            name="data_loader_agent",
            port=8001,
            api_token=AGENTVERSE_API_KEY,
            mailbox=True
        )
        
        result = data_loader_adapter.register()
        registered_agents["data_loader"] = result
        
        if verbose:
            if "error" in result:
                print(f"❌ Failed to register DataLoaderToolsAgent: {result['error']}")
                print(f"   Solution: {result.get('solution', 'Check configuration and try again')}")
            else:
                print(f"✅ DataLoaderToolsAgent registered successfully!")
                print(f"   Address: {result.get('address', 'unknown')}")
                print(f"   Port: 8001")
    
    return registered_agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register AI Data Science agents with Fetch.ai Agentverse"
    )
    
    parser.add_argument(
        "--agents", 
        nargs="+", 
        choices=["data_cleaner", "data_loader", "all"], 
        default=["all"],
        help="Agents to register (default: all)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print detailed progress information"
    )
    
    parser.add_argument(
        "--keep-alive", 
        action="store_true", 
        help="Keep the script running to maintain agent connections"
    )
    
    args = parser.parse_args()
    
    try:
        # Register agents
        registered = register_agents(
            agents_to_register=args.agents,
            verbose=args.verbose
        )
        
        # Display summary of registered agents
        print("\n🌐 Agent Registration Summary:")
        for agent_name, info in registered.items():
            if "error" in info:
                status = "❌ FAILED"
            else:
                status = "✅ SUCCESS"
            
            print(f"{agent_name}: {status}")
        
        # Keep the script running if requested
        if args.keep_alive:
            print("\n⏳ Keeping agents alive (press Ctrl+C to stop)...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down agents...")
    
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please make sure you have set your OPENAI_API_KEY and AGENTVERSE_API_KEY environment variables.") 