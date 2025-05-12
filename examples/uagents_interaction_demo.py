#!/usr/bin/env python
"""
uAgents Interaction Demo.

This script demonstrates how multiple AI agents can interact with each other
using the uAgents ecosystem. It shows a data loading agent and a data cleaning agent
working together to load and clean a dataset.
"""

import os
import sys
import time
import pandas as pd
import asyncio
from pathlib import Path
import dotenv
import logging
import argparse
from datetime import datetime
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the adapters
from src.adapters.uagents_adapter import (
    DataCleaningAgentAdapter,
    DataLoaderToolsAgentAdapter,
    cleanup_uagent,
    load_env_keys
)

# Try to load from .env file
dotenv.load_dotenv(project_root / ".env")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AGENTVERSE_API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

def create_sample_data(num_rows=100):
    """Create a sample dataframe for testing."""
    import numpy as np
    np.random.seed(42)
    
    # Create dataframe with various data types
    df = pd.DataFrame({
        'id': range(1, num_rows + 1),
        'name': [f'Item {i}' for i in range(1, num_rows + 1)],
        'category': np.random.choice(['A', 'B', 'C', 'D'], num_rows),
        'price': np.random.uniform(10, 1000, num_rows),
        'quantity': np.random.randint(1, 100, num_rows),
        'rating': np.random.uniform(1, 5, num_rows),
        'in_stock': np.random.choice([True, False], num_rows)
    })
    
    # Add some missing values
    mask = np.random.random(num_rows) < 0.2
    df.loc[mask, 'price'] = np.nan
    
    mask = np.random.random(num_rows) < 0.15
    df.loc[mask, 'quantity'] = np.nan
    
    mask = np.random.random(num_rows) < 0.1
    df.loc[mask, 'rating'] = np.nan
    
    # Add some outliers
    outlier_idx = np.random.choice(num_rows, 5, replace=False)
    df.loc[outlier_idx, 'price'] = np.random.uniform(5000, 10000, len(outlier_idx))
    
    outlier_idx = np.random.choice(num_rows, 3, replace=False)
    df.loc[outlier_idx, 'quantity'] = np.random.randint(500, 1000, len(outlier_idx))
    
    # Save the dataframe
    csv_path = project_root / "examples" / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Sample data saved to {csv_path}")
    
    return df, csv_path

async def run_interaction_demo():
    """Run the interaction demo between loader and cleaner agents."""
    try:
        logger.info("Starting uAgents interaction demo...")
        
        # Check for required API keys
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment variables or .env file")
            return False
            
        if not AGENTVERSE_API_TOKEN:
            logger.error("AGENTVERSE_API_TOKEN not found in environment variables or .env file")
            return False
        
        # Create sample data
        _, csv_path = create_sample_data()
        
        # Import LLM
        from langchain_openai import ChatOpenAI
        
        # Initialize the model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create and register the data loader agent
        loader_adapter = DataLoaderToolsAgentAdapter(
            model=llm,
            name="data_loader_agent",
            port=8001,
            description="A data loader agent that can load data from various sources",
            mailbox=True,
            api_token=AGENTVERSE_API_TOKEN
        )
        
        # Create and register the data cleaning agent
        cleaner_adapter = DataCleaningAgentAdapter(
            model=llm,
            name="data_cleaning_agent",
            port=8000,
            description="A data cleaning agent that can process datasets",
            mailbox=True,
            api_token=AGENTVERSE_API_TOKEN,
            n_samples=20,
            log=True,
        )
        
        # Register both agents
        logger.info("Registering data loader agent...")
        loader_result = loader_adapter.register()
        
        if "error" in loader_result:
            logger.error(f"Data loader registration failed: {loader_result['error']}")
            logger.info(f"Solution: {loader_result.get('solution', 'Unknown')}")
            return False
            
        logger.info(f"Data loader registered. Address: {loader_result.get('agent_address')}")
        
        logger.info("Registering data cleaning agent...")
        cleaner_result = cleaner_adapter.register()
        
        if "error" in cleaner_result:
            logger.error(f"Data cleaner registration failed: {cleaner_result['error']}")
            logger.info(f"Solution: {cleaner_result.get('solution', 'Unknown')}")
            # Clean up the loader agent before returning
            loader_adapter.cleanup()
            return False
            
        logger.info(f"Data cleaner registered. Address: {cleaner_result.get('agent_address')}")
        
        # Get agent addresses
        loader_address = loader_result.get('agent_address')
        cleaner_address = cleaner_result.get('agent_address')
        
        if not loader_address or not cleaner_address:
            logger.error("Failed to get agent addresses")
            # Clean up before returning
            loader_adapter.cleanup()
            cleaner_adapter.cleanup()
            return False
        
        logger.info("\n\n" + "="*50)
        logger.info("COMMUNICATION SIMULATION")
        logger.info("="*50)
        
        logger.info("\nThis demo simulates how the agents would communicate in a real uAgents environment.")
        logger.info("In a real setup, you would use the uAgents SDK to send messages between agents.")
        logger.info("For this demo, we'll use the adapters' direct methods to simulate the communication.\n")
        
        # Simulate communication between agents
        logger.info("Step 1: Use the data loader agent to load the sample data")
        loaded_df = loader_adapter.load_data(
            instructions=f"Load the CSV file from {csv_path}"
        )
        
        if isinstance(loaded_df, pd.DataFrame) and not loaded_df.empty:
            logger.info(f"Data loaded successfully. Shape: {loaded_df.shape}")
            
            # Display sample
            logger.info(f"Sample of loaded data:\n{loaded_df.head()}")
            
            logger.info("\nStep 2: Send the loaded data to the data cleaning agent")
            
            # Simulate sending the data to the cleaning agent
            cleaned_df = cleaner_adapter.clean_data(
                data=loaded_df,
                instructions="Fill missing values with mean for numeric columns and mode for categorical columns. "
                           "Remove outliers that are 3x the IQR. Convert data types appropriately."
            )
            
            if cleaned_df is not None:
                logger.info(f"Data cleaned successfully. Shape before: {loaded_df.shape}, after: {cleaned_df.shape}")
                
                # Print cleaning summary
                missing_before = loaded_df.isnull().sum().sum()
                missing_after = cleaned_df.isnull().sum().sum()
                logger.info(f"Missing values before: {missing_before}, after: {missing_after}")
                
                # Display sample of cleaned data
                logger.info(f"Sample of cleaned data:\n{cleaned_df.head()}")
                
                logger.info("\n" + "-"*50)
                logger.info("Simulation completed successfully!")
            else:
                logger.error("Data cleaning failed")
        else:
            logger.error(f"Data loading failed: {loaded_df}")
        
        # Keep the agents running for a while
        logger.info("\n" + "="*50)
        logger.info("AGENTS ARE RUNNING")
        logger.info("="*50)
        logger.info("\nThe agents are now running and registered with Agentverse.")
        logger.info("In a real environment, other agents could communicate with them using the uAgents protocol.")
        logger.info("\nAgent addresses:")
        logger.info(f"Data Loader Agent: {loader_address}")
        logger.info(f"Data Cleaning Agent: {cleaner_address}")
        logger.info("\nPress Ctrl+C to stop the agents and clean up.")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping the agents...")
            
            # Clean up both agents
            logger.info("Cleaning up data loader agent...")
            loader_cleanup = loader_adapter.cleanup()
            logger.info(f"Loader cleanup result: {loader_cleanup}")
            
            logger.info("Cleaning up data cleaning agent...")
            cleaner_cleanup = cleaner_adapter.cleanup()
            logger.info(f"Cleaner cleanup result: {cleaner_cleanup}")
            
            logger.info("Agents stopped and cleaned up.")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in interaction demo: {str(e)}")
        return False

def main():
    """Main function."""
    # Print API token information
    if AGENTVERSE_API_TOKEN:
        token_preview = AGENTVERSE_API_TOKEN[:5] + "..." + AGENTVERSE_API_TOKEN[-5:]
        logger.info(f"Using Agentverse API token: {token_preview}")
    else:
        logger.warning("No Agentverse API token found")
    
    # Run the interaction demo
    asyncio.run(run_interaction_demo())

if __name__ == "__main__":
    main() 