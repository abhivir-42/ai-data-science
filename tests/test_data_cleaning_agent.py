"""
Tests for the data cleaning agent.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import warnings
from dotenv import load_dotenv

# To be able to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables (for API keys)
load_dotenv()

# Only run these tests if OpenAI API key is available
if os.getenv("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    from ai_data_science.agents import DataCleaningAgent

    class TestDataCleaningAgent(unittest.TestCase):
        """Tests for the data cleaning agent."""
        
        @classmethod
        def setUpClass(cls):
            """Set up the test class."""
            # Suppress warnings
            warnings.filterwarnings("ignore")
            
            # Create a small test dataframe with data quality issues
            cls.test_df = pd.DataFrame({
                'id': [1, 2, 3, 4, 5, 5],  # Duplicate in ID 5
                'name': ['John', 'Jane', np.nan, 'Bob', 'Alice'],  # Missing name
                'age': [25, 30, 40, np.nan, 35],  # Missing age
                'salary': [50000, 60000, 70000, 55000, np.nan]  # Missing salary
            })
            
            # Initialize the language model
            cls.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        def test_agent_initialization(self):
            """Test that the agent initializes correctly."""
            agent = DataCleaningAgent(
                model=self.llm,
                n_samples=5,
                log=False
            )
            self.assertIsNotNone(agent)
            self.assertIsNone(agent.response)
        
        def test_agent_invoke(self):
            """Test that the agent can be invoked."""
            agent = DataCleaningAgent(
                model=self.llm,
                n_samples=5,
                log=False
            )
            
            # Run the agent
            agent.invoke_agent(
                data_raw=self.test_df,
                user_instructions="Just impute missing values and remove duplicates.",
                max_retries=2
            )
            
            # Check that the response was generated
            self.assertIsNotNone(agent.response)
            
            # Check that the cleaned data is available
            cleaned_df = agent.get_data_cleaned()
            self.assertIsNotNone(cleaned_df)
            self.assertIsInstance(cleaned_df, pd.DataFrame)
            
            # Check that the data was actually cleaned
            # Should have no missing values and no duplicates
            self.assertTrue(cleaned_df.duplicated().sum() == 0)
            
            # Check that the recommendation steps and function were generated
            self.assertIsNotNone(agent.get_recommended_cleaning_steps())
            self.assertIsNotNone(agent.get_data_cleaner_function())
        
        def test_agent_with_custom_instructions(self):
            """Test that the agent follows custom instructions."""
            agent = DataCleaningAgent(
                model=self.llm,
                n_samples=5,
                log=False
            )
            
            # Run the agent with specific instructions to only remove duplicates
            agent.invoke_agent(
                data_raw=self.test_df,
                user_instructions="Only remove duplicate rows. Do not perform any other cleaning.",
                max_retries=2
            )
            
            # Get the cleaned data
            cleaned_df = agent.get_data_cleaned()
            
            # Check that duplicates were removed
            self.assertEqual(cleaned_df.duplicated().sum(), 0)
            
            # Missing values should still be present since we only asked to remove duplicates
            self.assertTrue(cleaned_df.isna().sum().sum() > 0)

if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY"):
        unittest.main()
    else:
        print("OPENAI_API_KEY not found. Skipping tests.") 