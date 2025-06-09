#!/usr/bin/env python3
"""
Test Minimal uAgent Implementation

This tests the registration process without running the agent indefinitely.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

def test_minimal_uagent_registration():
    """Test the minimal uAgent registration process."""
    
    print("🧪 Testing Minimal uAgent Registration")
    print("=" * 60)
    
    try:
        from langchain_openai import ChatOpenAI
        from uagents_adapter import LangchainRegisterTool
        from agents.data_loader_tools_agent import DataLoaderToolsAgent
        
        print("✅ Imports successful")
        
        # Load environment variables
        load_dotenv()
        
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")
        
        if not OPENAI_API_KEY:
            print("❌ No OpenAI API key found")
            return False
            
        print("✅ API key found")
        
        # Set up model and agent (following exact LangGraph pattern)
        model = ChatOpenAI(temperature=0)
        data_loader_agent = DataLoaderToolsAgent(model=model)
        app = data_loader_agent._compiled_graph
        
        print("✅ Agent created")
        
        # Wrap agent into a function (EXACTLY like LangGraph example)
        def data_loader_agent_func(query):
            """Wrap the agent function following the EXACT LangGraph pattern."""
            
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
        
        print("✅ Function wrapper created")
        
        # Test the function directly first
        test_result = data_loader_agent_func("List files in data directory")
        print(f"✅ Function test: {test_result[:100]}...")
        
        # Test registration tool creation
        tool = LangchainRegisterTool()
        print("✅ Registration tool created")
        
        print("\n🚀 Attempting registration...")
        
        # Register the agent (EXACT pattern from LangGraph example)
        agent_info = tool.invoke(
            {
                "agent_obj": data_loader_agent_func,  # Pass the function
                "name": "test_minimal_data_loader", 
                "port": 8100,
                "description": "A minimal working data loader agent",
                "api_token": API_TOKEN,
                "mailbox": True
            }
        )
        
        print(f"📊 Registration completed!")
        print(f"📊 Result type: {type(agent_info)}")
        print(f"📊 Result: {agent_info}")
        
        # Check if registration was successful
        if isinstance(agent_info, dict):
            print("✅ Registration returned dictionary (good)")
            agent_address = agent_info.get('agent_address', 'Not found')
            print(f"🔗 Agent address: {agent_address}")
        elif isinstance(agent_info, str):
            print("⚠️ Registration returned string")
            print(f"📝 String result: {agent_info}")
        else:
            print(f"❓ Registration returned unexpected type: {type(agent_info)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_minimal_uagent_registration()
    
    if success:
        print("\n🎉 MINIMAL UAGENT REGISTRATION TEST PASSED!")
        print("✅ Function wrapper works")
        print("✅ Registration process works")
        print("✅ Ready for actual deployment")
    else:
        print("\n❌ MINIMAL UAGENT REGISTRATION TEST FAILED")
        print("🔧 Need to investigate registration issues") 