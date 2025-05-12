#!/usr/bin/env python
"""
Test script for uAgents adapters.

This script demonstrates how to use the uAgents adapters to convert and register
AI Data Science agents as uAgents with the Fetch AI Agentverse.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import dotenv
import logging
import argparse

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

def test_data_cleaning_agent():
    """Test the DataCleaningAgentAdapter."""
    try:
        logger.info("Testing DataCleaningAgentAdapter...")
        
        # Import LLM
        from langchain_openai import ChatOpenAI
        
        # Check for API key
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment variables or .env file")
            return False
        
        # Initialize the model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create the adapter
        adapter = DataCleaningAgentAdapter(
            model=llm,
            name="data_cleaning_agent",
            port=8000,
            description="A data cleaning agent that can process datasets",
            mailbox=True,
            api_token=AGENTVERSE_API_TOKEN,
            n_samples=20,
            log=True,
        )
        
        # Create sample data
        df, _ = create_sample_data()
        
        # Test direct cleaning
        logger.info("Testing direct data cleaning...")
        cleaned_df = adapter.clean_data(
            data=df,
            instructions="Fill missing values with mean for numeric columns, "
                        "remove outliers, and convert data types appropriately."
        )
        
        if cleaned_df is not None:
            logger.info(f"Direct cleaning successful. Shape before: {df.shape}, after: {cleaned_df.shape}")
            
            # Print cleaning summary
            missing_before = df.isnull().sum().sum()
            missing_after = cleaned_df.isnull().sum().sum()
            logger.info(f"Missing values before: {missing_before}, after: {missing_after}")
        else:
            logger.error("Direct cleaning failed")
            return False
        
        # Register the agent with Agentverse
        if AGENTVERSE_API_TOKEN:
            logger.info("Registering data cleaning agent with Agentverse...")
            result = adapter.register()
            
            if "error" in result:
                logger.error(f"Registration failed: {result['error']}")
                logger.info(f"Solution: {result.get('solution', 'Unknown')}")
                return False
                
            logger.info(f"Registration successful. Agent address: {result.get('agent_address')}")
            
            # Keep the agent running for a while
            logger.info("Agent is running. Press Ctrl+C to stop.")
            try:
                keep_running = True
                while keep_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping the agent...")
                cleanup_result = adapter.cleanup()
                logger.info(f"Cleanup result: {cleanup_result}")
                keep_running = False
        else:
            logger.warning("AGENTVERSE_API_TOKEN not found, skipping registration")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error in test_data_cleaning_agent: {str(e)}")
        return False

def test_data_loader_agent():
    """Test the DataLoaderToolsAgentAdapter."""
    try:
        logger.info("Testing DataLoaderToolsAgentAdapter...")
        
        # Import LLM
        from langchain_openai import ChatOpenAI
        
        # Check for API key
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment variables or .env file")
            return False
        
        # Initialize the model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create the adapter
        adapter = DataLoaderToolsAgentAdapter(
            model=llm,
            name="data_loader_agent",
            port=8001,
            description="A data loader agent that can load data from various sources",
            mailbox=True,
            api_token=AGENTVERSE_API_TOKEN
        )
        
        # Create sample data if it doesn't exist
        csv_path = project_root / "examples" / "sample_data.csv"
        if not csv_path.exists():
            _, csv_path = create_sample_data()
        
        # Test direct loading
        logger.info("Testing direct data loading...")
        loaded_df = adapter.load_data(
            instructions=f"Load the CSV file from {csv_path}"
        )
        
        if isinstance(loaded_df, pd.DataFrame) and not loaded_df.empty:
            logger.info(f"Direct loading successful. Shape: {loaded_df.shape}")
        else:
            logger.error(f"Direct loading failed: {loaded_df}")
            return False
        
        # Register the agent with Agentverse
        if AGENTVERSE_API_TOKEN:
            logger.info("Registering data loader agent with Agentverse...")
            result = adapter.register()
            
            if "error" in result:
                logger.error(f"Registration failed: {result['error']}")
                logger.info(f"Solution: {result.get('solution', 'Unknown')}")
                return False
                
            logger.info(f"Registration successful. Agent address: {result.get('agent_address')}")
            
            # Keep the agent running for a while
            logger.info("Agent is running. Press Ctrl+C to stop.")
            try:
                keep_running = True
                while keep_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping the agent...")
                cleanup_result = adapter.cleanup()
                logger.info(f"Cleanup result: {cleanup_result}")
                keep_running = False
        else:
            logger.warning("AGENTVERSE_API_TOKEN not found, skipping registration")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error in test_data_loader_agent: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test uAgents adapters")
    parser.add_argument("--cleaner", action="store_true", help="Test the data cleaning agent")
    parser.add_argument("--loader", action="store_true", help="Test the data loader agent")
    parser.add_argument("--all", action="store_true", help="Test all agents")
    
    args = parser.parse_args()
    
    # Print API token information
    if AGENTVERSE_API_TOKEN:
        token_preview = AGENTVERSE_API_TOKEN[:5] + "..." + AGENTVERSE_API_TOKEN[-5:]
        logger.info(f"Using Agentverse API token: {token_preview}")
    else:
        logger.warning("No Agentverse API token found")
    
    if args.cleaner or args.all:
        test_data_cleaning_agent()
    
    if args.loader or args.all:
        test_data_loader_agent()
    
    if not args.cleaner and not args.loader and not args.all:
        logger.info("No test specified. Use --cleaner, --loader, or --all")
        parser.print_help()

if __name__ == "__main__":
    main() 