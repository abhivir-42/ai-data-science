#!/usr/bin/env python3
"""
Test Core Agent Only - No uAgent Wrapper

This tests if our basic LangChain agent works before we try to wrap it.
"""

import sys
import os
sys.path.append('src')

def test_core_agent():
    """Test the core agent without any uAgent wrapper."""
    
    print("🧪 Testing Core LangChain Agent (No uAgent Wrapper)")
    print("=" * 60)
    
    try:
        from agents.data_loader_tools_agent import DataLoaderToolsAgent
        from langchain_openai import ChatOpenAI
        
        print("✅ Imports successful")
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No OpenAI API key found")
            return False
            
        print("✅ API key found")
        
        # Create model and agent
        model = ChatOpenAI(
            model='gpt-4o-mini', 
            api_key=api_key,
            temperature=0.1
        )
        
        print("✅ Model created")
        
        agent = DataLoaderToolsAgent(model=model)
        print("✅ Agent created")
        
        # Test the agent
        test_input = {
            'user_instructions': 'List the files in the data directory'
        }
        
        print(f"🔬 Testing with input: {test_input}")
        
        result = agent._compiled_graph.invoke(test_input)
        
        print("✅ Agent executed successfully!")
        print(f"📊 Result type: {type(result)}")
        print(f"📊 Result preview: {str(result)[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_core_agent()
    
    if success:
        print("\n🎉 CORE AGENT WORKS!")
        print("✅ Ready to proceed with uAgent wrapper")
    else:
        print("\n❌ CORE AGENT FAILED")
        print("🔧 Need to fix basic agent first") 