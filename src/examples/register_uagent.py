"""
Register DataCleaningAgent as a uAgent with Agentverse

This script handles the process of registering the DataCleaningAgent with
the Fetch.ai Agentverse platform, including package verification and error handling.
"""

import os
import sys
import subprocess
import time
import pandas as pd
from dotenv import load_dotenv

def check_package_version(package_name):
    """Check the installed version of a package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
        return None

def update_packages():
    """Update uAgents-related packages to the latest version."""
    print("📦 Updating uAgents packages...")
    
    packages = [
        "uagents>=2.2.0",
        "uagents-adapter>=2.2.0"
    ]
    
    for package in packages:
        package_name = package.split(">=")[0]
        current_version = check_package_version(package_name)
        print(f"  - Current {package_name} version: {current_version}")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            new_version = check_package_version(package_name)
            print(f"  - Updated {package_name} version: {new_version}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update {package_name}: {str(e)}")

def register_agent():
    """Register the DataCleaningAgent as a uAgent."""
    from langchain_openai import ChatOpenAI
    from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    agentverse_api_token = os.getenv("AGENTVERSE_API_TOKEN", "")
    
    # Initialize the language model
    print("\n📝 Initializing ChatOpenAI...")
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    
    # Create the adapter
    print("\n🔄 Creating DataCleaningAgentAdapter...")
    adapter = DataCleaningAgentAdapter(
        model=llm,
        name="data_cleaning_agent",
        port=8000,
        description="A data cleaning agent for processing datasets",
        mailbox=True,
        api_token=agentverse_api_token,
        n_samples=20,
        log=False,  # Disable logging to avoid path issues
        human_in_the_loop=False
    )
    
    # Register the agent
    print("\n🌐 Registering agent with Agentverse...")
    try:
        result = adapter.register()
        
        # Check if registration was successful
        if isinstance(result, dict) and result.get('agent_address'):
            print("\n✅ Agent registered successfully!")
            print(f"📋 Agent Information:")
            print(f"  - Name: {result.get('agent_name', 'unknown')}")
            print(f"  - Address: {result.get('agent_address', 'unknown')}")
            print(f"  - Port: {result.get('agent_port', 'unknown')}")
            print(f"  - Protocol: {result.get('agent_protocol', 'unknown')}")
            return result
        elif result.get('error'):
            print(f"\n❌ Registration error: {result.get('error')}")
            print(f"💡 Solution: {result.get('solution', 'Unknown')}")
            return result
        else:
            print(f"\n⚠️ Unexpected registration result: {result}")
            return result
    except Exception as e:
        print(f"\n❌ Registration failed: {str(e)}")
        return {"error": str(e)}

def test_with_example_data(adapter):
    """Test the adapter with example data."""
    print("\n🧪 Testing the DataCleaningAgent with example data...")
    
    # Load the dataset
    try:
        print("\n📊 Loading example dataset...")
        df = pd.read_csv("data/churn_data.csv")
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ Example dataset not found. Using a simple test dataframe instead.")
        # Create a simple test dataframe
        df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': ['x', 'y', 'z', None, 'w'],
            'C': [1.1, 2.2, 3.3, 4.4, None]
        })
    
    # Define cleaning instructions
    instructions = "Clean this dataset by filling missing values and converting data types."
    
    # Clean the data
    print("\n🧹 Running data cleaning process...")
    try:
        cleaned_df = adapter.clean_data(df, instructions)
        print(f"\n✨ Cleaning completed successfully!")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Missing values remaining: {cleaned_df.isna().sum().sum()}")
        return True
    except Exception as e:
        print(f"\n❌ Data cleaning failed: {str(e)}")
        return False

def main():
    """Main function to update packages and register agent."""
    print("🔧 Starting uAgent registration process...\n")
    
    # Update packages to ensure compatibility
    update_packages()
    
    # Register the agent
    print("\n🚀 Registering the DataCleaningAgent as a uAgent...")
    result = register_agent()
    
    # Print conclusion
    print("\n🎉 Process completed!")
    if isinstance(result, dict) and result.get('agent_address'):
        print(f"The agent is registered with address: {result.get('agent_address')}")
        print("You can now interact with it through the Agentverse platform.")
    else:
        print("Registration was not successful. Check the errors above for more details.")
        print("\nTroubleshooting tips:")
        print("1. Check network connectivity")
        print("2. Verify your Agentverse API token")
        print("3. Ensure port 8000 is available")
        print("4. Try running with admin/sudo privileges if needed")

if __name__ == "__main__":
    main() 