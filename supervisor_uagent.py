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

# Dataset URL mappings for common datasets
DATASET_URLS = {
    'iris': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
    'titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
    'wine': 'https://raw.githubusercontent.com/plotly/datasets/master/wine_data.csv',
    'boston': 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
    'diabetes': 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',
    'tips': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
    'flights': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv'
}

def detect_dataset_from_query(query_text):
    """Detect dataset name from user query and return corresponding URL."""
    query_lower = query_text.lower()
    
    for dataset_name, url in DATASET_URLS.items():
        if dataset_name in query_lower:
            return url, dataset_name
    
    # Default to titanic if no specific dataset detected
    return DATASET_URLS['titanic'], 'titanic'

def extract_target_from_query(query_text, dataset_name):
    """Extract target variable based on query and dataset."""
    query_lower = query_text.lower()
    
    # Common target variables by dataset
    common_targets = {
        'iris': 'species',
        'titanic': 'survived', 
        'wine': 'quality',
        'boston': 'medv',
        'diabetes': 'target',
        'tips': 'tip',
        'flights': 'passengers'
    }
    
    # Look for explicit target mentions
    if 'target' in query_lower and '=' in query_lower:
        # Extract target=something pattern
        import re
        match = re.search(r'target[=:]\s*["\']?(\w+)["\']?', query_lower)
        if match:
            return match.group(1)
    
    # Look for "predict X" patterns
    predict_patterns = [
        r'predict\s+["\']?(\w+)["\']?',
        r'predicting\s+["\']?(\w+)["\']?',
        r'classification\s+of\s+["\']?(\w+)["\']?',
        r'regression\s+on\s+["\']?(\w+)["\']?'
    ]
    
    import re
    for pattern in predict_patterns:
        match = re.search(pattern, query_lower)
        if match:
            return match.group(1)
    
    # Return common target for the dataset
    return common_targets.get(dataset_name, None)

# Wrap supervisor agent into a function for UAgent (EXACTLY like LangGraph example)
def supervisor_agent_func(query):
    """
    Enhanced supervisor agent function with intelligent dataset detection.
    
    This handles input format conversion and returns the final response.
    Supports:
    - Automatic dataset detection from query text
    - Common dataset URLs (iris, titanic, wine, etc.)
    - Smart target variable extraction
    - Robust error handling
    """
    # Handle input if it's a dict with 'input' key (EXACT pattern from example)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    try:
        csv_url = None
        user_request = None
        target_variable = None
        
        # Parse the query to extract required parameters
        if isinstance(query, str):
            user_request = query
            
            # Smart dataset detection from query text
            detected_url, detected_dataset = detect_dataset_from_query(query)
            csv_url = detected_url
            
            # Smart target variable extraction
            target_variable = extract_target_from_query(query, detected_dataset)
            
            print(f"🔍 Detected dataset: {detected_dataset}")
            print(f"🎯 Detected target: {target_variable}")
            
        elif isinstance(query, dict):
            # Extract parameters from dict
            user_request = query.get('user_request', query.get('query', ''))
            csv_url = query.get('csv_url')
            target_variable = query.get('target_variable')
            
            # If no CSV URL provided, try to detect from request
            if not csv_url and user_request:
                detected_url, detected_dataset = detect_dataset_from_query(user_request)
                csv_url = detected_url
                
                # If no target provided, try to extract
                if not target_variable:
                    target_variable = extract_target_from_query(user_request, detected_dataset)
                    
        else:
            return """
🚫 **Input Format Error**

I need either:
1. A simple text request like: "Analyze the iris dataset" 
2. A structured request like: {"csv_url": "your-url", "user_request": "your-task"}

**Supported datasets**: iris, titanic, wine, boston, diabetes, tips, flights
"""
        
        # Validate required parameters
        if not user_request:
            return """
🚫 **Missing Request**

Please tell me what you'd like me to do! For example:
- "Clean and analyze the iris dataset"
- "Train a classification model on the titanic data"
- "Perform feature engineering on the wine dataset"
"""
        
        if not csv_url:
            return """
🚫 **Dataset Not Found**

I couldn't detect a dataset from your request. Please specify:
- A known dataset name (iris, titanic, wine, boston, diabetes, tips, flights)
- Or provide a direct CSV URL

Example: "Analyze the iris dataset for classification"
"""
        
        # Process the request using the supervisor agent
        print(f"🚀 Processing request with:")
        print(f"   📊 Dataset URL: {csv_url}")
        print(f"   📝 Request: {user_request}")
        print(f"   🎯 Target: {target_variable}")
        
        result = supervisor_agent.process_request(
            csv_url=csv_url,
            user_request=user_request,
            target_variable=target_variable
        )
        
        return result
        
    except Exception as e:
        error_msg = f"""
🚫 **Technical Error**

Sorry, I encountered an issue: {str(e)}

**Common solutions:**
1. Check if the dataset URL is accessible
2. Ensure your request is clear (e.g., "analyze iris dataset")
3. Try specifying the target variable explicitly

**Need help?** Try: "Clean and analyze the iris dataset for species classification"
"""
        print(f"❌ Error in supervisor agent: {str(e)}")
        return error_msg

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