"""
Example of using the DataCleaningAgentAdapter to register a DataCleaningAgent as a uAgent.

This script demonstrates the concept of adapting a DataCleaningAgent to work as a uAgent.
Since we may not have access to an OpenAI API key or a complete environment,
this example uses mock data to illustrate how the integration would work.

Usage:
    python examples/uagents_adapter_example.py
"""

import pandas as pd

def create_sample_data():
    """Create a sample dataset for testing."""
    data = {
        'product_id': ['A001', 'A002', 'A003', 'A004', 'A005', None, 'A007', 'A008', 'A009', 'A010'],
        'name': ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5', 
                'Product 6', 'Product 7', 'Product 8', 'Product 9', 'Product 10'],
        'price': [10.99, 15.99, None, 8.99, 20.99, 12.99, 9.99, None, 18.99, 22.99],
        'stock': [100, 50, 75, None, 25, 60, 45, 80, None, 90],
        'category': ['Electronics', None, 'Electronics', 'Home', 'Clothing', 
                    'Home', None, 'Electronics', 'Clothing', 'Books']
    }
    return pd.DataFrame(data)

def mock_register_result():
    """Return a mock result for the agent registration."""
    return {
        "agent_name": "data_cleaning_agent",
        "agent_address": "agent1q2w3e4r5t6y7u8i9o0p1a2s3d4f5g6h7j8k9l",
        "agent_port": 8000,
        "agent_protocol": "0.3.0",
        "status": "registered",
        "message": "Agent registered successfully"
    }

def mock_data_cleaning():
    """Return a mock result for the data cleaning process."""
    # Sample cleaned data with rows containing missing values removed
    data = {
        'product_id': ['A001', 'A002', 'A004', 'A005'],
        'name': ['Product 1', 'Product 2', 'Product 4', 'Product 5'],
        'price': [10.99, 15.99, 8.99, 20.99],
        'stock': [100, 50, None, 25],
        'category': ['Electronics', 'Unknown', 'Home', 'Clothing']
    }
    return pd.DataFrame(data)

def mock_cleaner_function():
    """Return a mock data cleaning function."""
    return """
def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    
    # Make a copy of the dataframe
    df = data_raw.copy()
    
    # Fill missing categories with 'Unknown'
    df['category'] = df['category'].fillna('Unknown')
    
    # Convert price to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove rows with missing values in product_id
    df = df.dropna(subset=['product_id'])
    
    # Remove rows with missing values in price
    df = df.dropna(subset=['price'])
    
    return df
"""

def main():
    """Run the example script."""
    print("🤖 DataCleaningAgentAdapter Example (Mock)")
    print("This example demonstrates how a DataCleaningAgent can be adapted to a uAgent")
    print("In a real implementation, you would use the following code:")
    print("""
    from langchain_openai import ChatOpenAI
    from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=OPENAI_API_KEY)
    
    # Create the adapter
    adapter = DataCleaningAgentAdapter(
        model=llm,
        name="data_cleaning_agent",
        description="A data cleaning agent for processing datasets",
        mailbox=True,
        api_token="YOUR_AGENTVERSE_API_TOKEN"
    )
    
    # Register the agent with Agentverse
    result = adapter.register()
    """)
    
    print("\n🔄 Using mock data for demonstration...")
    
    # Mock registration result
    result = mock_register_result()
    
    print(f"✅ Agent registered successfully! (mock)")
    print(f"📋 Agent Information:")
    print(f"  - Name: {result['agent_name']}")
    print(f"  - Address: {result['agent_address']}")
    print(f"  - Port: {result['agent_port']}")
    print(f"  - Protocol: {result['agent_protocol']}")
    
    print("\n🔗 With a registered agent, you can interact with it through the Agentverse platform.")
    
    # Create a sample dataset
    print("\n📊 Testing with a sample dataset:")
    df = create_sample_data()
    print(df.head())
    
    # Show mock cleaned data
    print("\n✨ Cleaned data (mock result):")
    cleaned_df = mock_data_cleaning()
    print(cleaned_df.head())
    
    print("\n📝 Data cleaning function (mock):")
    print(mock_cleaner_function())
    
    print("\n🎯 To interact with the agent programmatically, you would use:")
    print("""
    from uagents import Agent, Context, Model, Protocol
    from pydantic import Field
    from typing import Dict, Any, Optional
    
    # Define message models
    class DataCleaningRequest(Model):
        query: str = Field(description="Instructions for how to clean the data")
        data_dict: Dict[str, Any] = Field(description="Dataset in dictionary format")
    
    class DataCleaningResponse(Model):
        data_cleaned: Optional[Dict[str, Any]] = Field(default=None)
        cleaner_function: Optional[str] = Field(default=None)
        workflow_summary: Optional[str] = Field(default=None)
    
    # Create a client agent
    client_agent = Agent(name="data_cleaning_client", seed="your_seed")
    protocol = Protocol("DataCleaningProtocol")
    
    # Send a request to the registered agent
    agent_address = "agent1q2w3e4r5t6y7u8i9o0p1a2s3d4f5g6h7j8k9l"  # from registration
    await client_agent.send(agent_address, DataCleaningRequest(
        query="Clean this dataset by removing duplicates and handling missing values",
        data_dict=df.to_dict()
    ))
    """)
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    main() 