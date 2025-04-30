"""
Example of how to interact with the Data Cleaning Agent through the Agentverse.

This script demonstrates how to:
1. Connect to a registered Data Cleaning Agent using its address
2. Send a request for data cleaning
3. Process the response

Usage:
    python examples/uagents_interaction_example.py <agent_address>
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from uagents import Agent, Context, Model, Protocol

# Models for the agent messages
class DataCleaningRequest(Model):
    """Request model for data cleaning with instructions and dataset."""
    query: str = Field(description="Instructions for how to clean the data")
    data_dict: Dict[str, Any] = Field(description="Dataset in dictionary format")

class DataCleaningResponse(Model):
    """Response model from data cleaning agent."""
    data_cleaned: Optional[Dict[str, Any]] = Field(default=None, description="Cleaned dataset in dictionary format")
    cleaner_function: Optional[str] = Field(default=None, description="The Python function used for cleaning")
    workflow_summary: Optional[str] = Field(default=None, description="Summary of the data cleaning workflow")
    error: Optional[str] = Field(default=None, description="Error message if data cleaning failed")

def create_sample_data():
    """Create a sample dataset for testing."""
    # Create a dataframe with missing values, outliers, and various data types
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': [f'ID_{i}' if i % 10 != 0 else None for i in range(n_samples)],
        'value_1': np.random.normal(100, 20, n_samples),
        'value_2': [np.random.randint(1, 100) if i % 5 != 0 else None for i in range(n_samples)],
        'category': [np.random.choice(['A', 'B', 'C', 'D']) if i % 7 != 0 else None for i in range(n_samples)],
        'date': [f'2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}' if i % 8 != 0 else None for i in range(n_samples)]
    }
    
    # Add some outliers
    data['value_1'][5] = 500
    data['value_1'][15] = -50
    data['value_2'][25] = 999
    
    return pd.DataFrame(data)

def main():
    """Run the example script."""
    # Get agent address from command line
    if len(sys.argv) != 2:
        print("Usage: python examples/uagents_interaction_example.py <agent_address>")
        print("Example: python examples/uagents_interaction_example.py agent1qwe8xt3...")
        sys.exit(1)
    
    agent_address = sys.argv[1]
    
    # Create a local agent to send requests
    client_agent = Agent(
        name="data_cleaning_client",
        seed="data_cleaning_client_seed",
        endpoint=["http://localhost:8001/submit"],
        port=8001,
    )
    
    # Create a protocol for interacting with the data cleaning agent
    protocol = Protocol("DataCleaningProtocol")
    
    # Create sample data
    df = create_sample_data()
    print("\n📊 Sample dataset created:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    # Instructions for data cleaning
    instructions = (
        "Please clean this dataset by: "
        "1. Removing rows where the 'id' column is missing "
        "2. Filling missing numerical values with the median "
        "3. Filling missing categorical values with 'Unknown' "
        "4. Removing outliers in 'value_1' (values outside 3 standard deviations) "
        "5. Formatting the date column properly as a datetime object"
    )
    
    # Prepare the request
    request = DataCleaningRequest(
        query=instructions,
        data_dict=df.to_dict()
    )
    
    # Handle responses from the data cleaning agent
    @protocol.on_message(model=DataCleaningResponse)
    async def handle_response(ctx: Context, sender: str, response: DataCleaningResponse):
        print("\n✅ Received response from the data cleaning agent!")
        
        if response.error:
            print(f"❌ Error: {response.error}")
            return
            
        # Print workflow summary
        if response.workflow_summary:
            print("\n📝 Workflow Summary:")
            print(response.workflow_summary)
        
        # Print the cleaner function
        if response.cleaner_function:
            print("\n🛠️ Cleaner Function:")
            print(response.cleaner_function[:300] + "..." if len(response.cleaner_function) > 300 else response.cleaner_function)
        
        # Convert the cleaned data back to a DataFrame and display it
        if response.data_cleaned:
            cleaned_df = pd.DataFrame.from_dict(response.data_cleaned)
            print("\n✨ Cleaned Data:")
            print(cleaned_df.head())
            print(f"Shape: {cleaned_df.shape}")
            print(f"Missing values: {cleaned_df.isna().sum().sum()}")
            
            # Save the cleaned data to a CSV file
            cleaned_df.to_csv("cleaned_data.csv", index=False)
            print("\n💾 Cleaned data saved to 'cleaned_data.csv'")
        
        # Stop the agent after receiving the response
        await ctx.stop()
    
    # Register the protocol
    client_agent.include(protocol)
    
    # Send the request and wait for response
    @client_agent.on_event("startup")
    async def on_startup(ctx: Context):
        print(f"🚀 Sending data cleaning request to {agent_address}...")
        await ctx.send(agent_address, request)
        print("⏳ Waiting for response...")
    
    # Run the client agent
    print("🤖 Starting client agent...")
    client_agent.run()

if __name__ == "__main__":
    main() 