"""
H2O ML Agent Example
===================

This example demonstrates how to use the H2O ML Agent for automated machine learning
using H2O's AutoML capabilities.

Requirements:
    - H2O installed (pip install h2o)
    - MLflow installed (pip install mlflow) - optional for logging
    - OpenAI API key set in environment
"""

import os
import pandas as pd
from langchain_openai import ChatOpenAI
from src.agents.ml_agents import H2OMLAgent

# Set up paths
LOG_PATH = "logs/"
MODEL_PATH = "models/"
DATA_PATH = "data/"

def create_sample_data():
    """Create a sample dataset for demonstration"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic classification data
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),
        'debt_to_income': np.random.uniform(0, 1, n_samples),
    }
    
    # Create target variable (loan approval)
    # Higher income, credit score, and years employed = higher approval chance
    approval_prob = (
        (data['income'] / 100000) * 0.3 +
        (data['credit_score'] / 850) * 0.4 +
        (data['years_employed'] / 40) * 0.2 +
        (1 - data['debt_to_income']) * 0.1
    )
    
    # Add some noise and create binary target
    approval_prob += np.random.normal(0, 0.1, n_samples)
    data['loan_approved'] = (approval_prob > 0.5).astype(int)
    
    return pd.DataFrame(data)

def run_h2o_ml_agent_example():
    """Run the H2O ML Agent example"""
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Create sample data
    print("Creating sample loan approval dataset...")
    df = create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['loan_approved'].value_counts()}")
    
    # Create directories if they don't exist
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Initialize the H2O ML Agent
    print("\nInitializing H2O ML Agent...")
    ml_agent = H2OMLAgent(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
        n_samples=50,  # Use more samples for better data understanding
        enable_mlflow=False,  # Set to True if you want MLflow logging
        human_in_the_loop=False,  # Set to True for human review
        bypass_recommended_steps=False,  # Set to True to skip recommendations
        bypass_explain_code=False,  # Set to True to skip explanations
    )
    
    # Run the agent
    print("\nRunning H2O AutoML training...")
    user_instructions = """
    Please perform binary classification to predict 'loan_approved'.
    
    Requirements:
    - Use a maximum runtime of 60 seconds
    - Focus on maximizing AUC score
    - Exclude DeepLearning algorithms for faster training
    - Use 5-fold cross-validation
    - Balance classes if needed
    """
    
    try:
        ml_agent.invoke_agent(
            data_raw=df,
            user_instructions=user_instructions,
            target_variable="loan_approved",
            max_retries=3
        )
        
        print("\n" + "="*60)
        print("H2O ML AGENT RESULTS")
        print("="*60)
        
        # Display results
        print("\n1. LEADERBOARD:")
        leaderboard = ml_agent.get_leaderboard()
        if leaderboard is not None:
            print(leaderboard.head())
        else:
            print("No leaderboard available")
            
        print(f"\n2. BEST MODEL ID: {ml_agent.get_best_model_id()}")
        print(f"\n3. MODEL SAVED TO: {ml_agent.get_model_path()}")
        
        print("\n4. RECOMMENDED ML STEPS:")
        steps = ml_agent.get_recommended_ml_steps()
        if steps:
            print(steps)
        
        print("\n5. GENERATED H2O FUNCTION:")
        h2o_function = ml_agent.get_h2o_train_function()
        if h2o_function:
            print(f"Function saved to: {ml_agent.response.get('h2o_train_function_path')}")
        
        print("\n6. LOG SUMMARY:")
        log_summary = ml_agent.get_log_summary()
        if log_summary:
            print(log_summary)
            
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please check the logs for more details.")

def run_with_mlflow_example():
    """Example with MLflow logging enabled"""
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize the H2O ML Agent with MLflow
    print("\nRunning H2O ML Agent with MLflow logging...")
    ml_agent = H2OMLAgent(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
        enable_mlflow=True,
        mlflow_experiment_name="H2O_Loan_Approval",
        mlflow_run_name="loan_approval_automl",
    )
    
    # Run with MLflow logging
    ml_agent.invoke_agent(
        data_raw=df,
        user_instructions="Predict loan approval with MLflow logging",
        target_variable="loan_approved"
    )
    
    print("MLflow run completed. Check your MLflow UI for results.")

if __name__ == "__main__":
    print("H2O ML Agent Example")
    print("===================")
    
    # Basic example
    run_h2o_ml_agent_example()
    
    # Uncomment to run MLflow example
    # run_with_mlflow_example() 