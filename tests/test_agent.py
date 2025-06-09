"""
Test script for the Data Cleaning Agent
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

def main():
    # Load environment variables (for API keys)
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key":
        print("ERROR: OpenAI API key not found or not set!")
        print("Please set your API key in the .env file.")
        return
    
    print("Setting up the Data Cleaning Agent...")
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create the agent
        agent = DataCleaningAgent(
            model=llm,
            n_samples=10,  # Use fewer samples for faster testing
            log=True,
            log_path="./logs",
            human_in_the_loop=False
        )
        
        print("Agent initialized successfully!")
        
        # Load sample data
        print("Loading sample data...")
        df = pd.read_csv("data/sample_data.csv")
        print(f"Data loaded with shape: {df.shape}")
        
        # Display basic info about the data
        print("\nBasic data info:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values: {df.isna().sum().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")
        
        # Run the agent
        print("\nRunning the Data Cleaning Agent...")
        agent.invoke_agent(
            data_raw=df,
            user_instructions="Fix missing values and remove duplicates but don't remove any outliers.",
            max_retries=2
        )
        
        # Get the cleaned data
        cleaned_df = agent.get_data_cleaned()
        
        print("\nCleaning completed!")
        print(f"Original data shape: {df.shape}")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Missing values after cleaning: {cleaned_df.isna().sum().sum()}")
        print(f"Duplicates after cleaning: {cleaned_df.duplicated().sum()}")
        
        # Show the cleaning steps
        print("\nRecommended cleaning steps:")
        steps = agent.get_recommended_cleaning_steps()
        print(steps[:500] + "..." if len(steps) > 500 else steps)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"ERROR: An exception occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 