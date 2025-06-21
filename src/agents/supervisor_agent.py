"""
Supervisor Agent for AI Data Science.

This module provides a supervisor agent that orchestrates multiple specialized agents
(data cleaning, feature engineering, and H2O ML) based on natural language user requests
and remote CSV files.
"""

import os
import re
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from src.agents.data_cleaning_agent import DataCleaningAgent
from src.agents.feature_engineering_agent import FeatureEngineeringAgent  
from src.agents.ml_agents.h2o_ml_agent import H2OMLAgent


class SupervisorAgent:
    """
    A supervisor agent that orchestrates data science workflows by intelligently 
    coordinating specialized agents based on natural language requests.
    
    The supervisor can:
    - Download and process remote CSV files
    - Parse natural language requests to determine required agents
    - Orchestrate agents in the correct sequence
    - Return comprehensive string-based reports suitable for LLM integration
    
    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used by the underlying agents.
    output_dir : str, optional
        Directory for saving output files. Defaults to "output/supervisor/".
    log : bool, optional
        Whether to enable logging for agents. Defaults to True.
    log_path : str, optional
        Directory path for storing log files. Defaults to "logs/".
    n_samples : int, optional
        Number of samples used when summarizing datasets. Defaults to 30.
    
    Methods
    -------
    process_request(csv_url: str, user_request: str, target_variable: str = None) -> str
        Main method to process a user request with a CSV file.
    """
    
    def __init__(
        self,
        model=None,
        output_dir: str = "output/supervisor/",
        log: bool = True,
        log_path: str = "logs/",
        n_samples: int = 30
    ):
        self.model = model or ChatOpenAI(model="gpt-4o-mini")
        self.output_dir = output_dir
        self.log = log
        self.log_path = log_path
        self.n_samples = n_samples
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # Initialize agents (will be created as needed)
        self.data_cleaning_agent = None
        self.feature_engineering_agent = None
        self.h2o_ml_agent = None
        
    def _download_csv(self, csv_url: str) -> pd.DataFrame:
        """Download CSV from URL and return as DataFrame."""
        try:
            # Validate URL
            parsed_url = urlparse(csv_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {csv_url}")
            
            # Download CSV
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = os.path.join(self.output_dir, f"temp_data_{timestamp}.csv")
            
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Read as DataFrame
            df = pd.read_csv(temp_file)
            
            return df, temp_file
            
        except Exception as e:
            raise Exception(f"Failed to download CSV from {csv_url}: {str(e)}")
    
    def _parse_user_intent(self, user_request: str) -> Dict[str, bool]:
        """Parse user request to determine which agents are needed."""
        request_lower = user_request.lower()
        
        # Keywords for each agent
        cleaning_keywords = [
            'clean', 'cleaning', 'preprocess', 'preprocessing', 'missing', 'null', 'nan',
            'duplicate', 'outlier', 'quality', 'validate', 'fix', 'impute', 'remove'
        ]
        
        feature_keywords = [
            'feature', 'features', 'engineer', 'engineering', 'transform', 'encode', 
            'encoding', 'categorical', 'dummy', 'one-hot', 'scale', 'normalize'
        ]
        
        ml_keywords = [
            'model', 'modeling', 'train', 'training', 'predict', 'prediction', 'ml',
            'machine learning', 'classification', 'regression', 'automl', 'h2o'
        ]
        
        # Determine needs
        needs = {
            'data_cleaning': any(keyword in request_lower for keyword in cleaning_keywords),
            'feature_engineering': any(keyword in request_lower for keyword in feature_keywords),
            'ml_modeling': any(keyword in request_lower for keyword in ml_keywords)
        }
        
        # Default logic: if ML is requested, we need cleaning and feature engineering
        if needs['ml_modeling']:
            needs['data_cleaning'] = True
            needs['feature_engineering'] = True
        
        # If feature engineering is requested, we need cleaning first
        if needs['feature_engineering']:
            needs['data_cleaning'] = True
            
        # If nothing specific is detected, default to data cleaning
        if not any(needs.values()):
            needs['data_cleaning'] = True
            
        return needs
    
    def _extract_target_variable(self, user_request: str, df: pd.DataFrame) -> Optional[str]:
        """Extract target variable from user request if specified."""
        request_lower = user_request.lower()
        
        # Look for explicit target mentions
        target_patterns = [
            r"target\s+(?:variable\s+)?['\"]?(\w+)['\"]?",
            r"predict\s+['\"]?(\w+)['\"]?",
            r"target\s*=\s*['\"]?(\w+)['\"]?",
            r"on\s+['\"]?(\w+)['\"]?",
            r"to\s+predict\s+['\"]?(\w+)['\"]?",
            r"predicting\s+['\"]?(\w+)['\"]?",
            r"classification\s+model\s+to\s+predict\s+['\"]?(\w+)['\"]?",
            r"regression\s+model\s+to\s+predict\s+['\"]?(\w+)['\"]?"
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, request_lower)
            if match:
                potential_target = match.group(1)
                # Check if this column exists in the dataframe
                if potential_target in df.columns:
                    return potential_target
                # Try case-insensitive match
                for col in df.columns:
                    if col.lower() == potential_target.lower():
                        return col
        
        return None
    
    def _initialize_agents(self):
        """Initialize agents with consistent parameters."""
        base_params = {
            'model': self.model,
            'log': self.log,
            'log_path': self.log_path,
            'n_samples': self.n_samples,
            'human_in_the_loop': False,  # Disable for automation
        }
        
        if self.data_cleaning_agent is None:
            self.data_cleaning_agent = DataCleaningAgent(**base_params)
            
        if self.feature_engineering_agent is None:
            self.feature_engineering_agent = FeatureEngineeringAgent(**base_params)
            
        if self.h2o_ml_agent is None:
            ml_params = base_params.copy()
            ml_params['model_directory'] = self.output_dir
            self.h2o_ml_agent = H2OMLAgent(**ml_params)
    
    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame and return the file path as string."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def process_request(
        self, 
        csv_url: str, 
        user_request: str, 
        target_variable: str = None
    ) -> str:
        """
        Process a user request with a CSV file URL.
        
        Parameters
        ----------
        csv_url : str
            URL of the CSV file to process
        user_request : str
            Natural language description of what to do
        target_variable : str, optional
            Target variable name (if not specified, will try to extract from request)
            
        Returns
        -------
        str
            Comprehensive report as a string including all results and file paths
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 1: Download and prepare data
            print(f"Downloading CSV from: {csv_url}")
            df, temp_csv_path = self._download_csv(csv_url)
            
            # Step 2: Parse user intent
            needs = self._parse_user_intent(user_request)
            print(f"Detected needs: {needs}")
            
            # Step 3: Extract target variable if not provided
            if target_variable is None:
                target_variable = self._extract_target_variable(user_request, df)
            
            # Step 4: Initialize agents
            self._initialize_agents()
            
            # Step 5: Execute workflow
            results = {
                'original_csv_url': csv_url,
                'user_request': user_request,
                'target_variable': target_variable,
                'needs_detected': needs,
                'timestamp': timestamp,
                'data_shape': df.shape,
                'file_paths': {},
                'reports': {}
            }
            
            current_df = df.copy()
            
            # Data Cleaning Phase
            if needs['data_cleaning']:
                print("Executing data cleaning...")
                
                self.data_cleaning_agent.invoke_agent(
                    data_raw=current_df,
                    user_instructions=user_request
                )
                
                cleaned_df = self.data_cleaning_agent.get_data_cleaned()
                if cleaned_df is not None:
                    current_df = cleaned_df
                    # Save cleaned data
                    clean_path = self._save_dataframe(
                        current_df, 
                        f"cleaned_data_{timestamp}.csv"
                    )
                    results['file_paths']['cleaned_data'] = clean_path
                
                # Get cleaning report
                cleaning_response = self.data_cleaning_agent.get_response()
                if cleaning_response:
                    results['reports']['data_cleaning'] = {
                        'workflow_summary': self.data_cleaning_agent.get_workflow_summary(),
                        'recommended_steps': self.data_cleaning_agent.get_recommended_cleaning_steps(),
                        'function_code': self.data_cleaning_agent.get_data_cleaner_function()
                    }
            
            # Feature Engineering Phase  
            if needs['feature_engineering']:
                print("Executing feature engineering...")
                
                self.feature_engineering_agent.invoke_agent(
                    data_raw=current_df,
                    user_instructions=user_request,
                    target_variable=target_variable
                )
                
                engineered_df = self.feature_engineering_agent.get_data_engineered()
                if engineered_df is not None:
                    current_df = engineered_df
                    # Save engineered data
                    feature_path = self._save_dataframe(
                        current_df,
                        f"engineered_data_{timestamp}.csv"
                    )
                    results['file_paths']['engineered_data'] = feature_path
                
                # Get feature engineering report
                feature_response = self.feature_engineering_agent.get_response()
                if feature_response:
                    results['reports']['feature_engineering'] = {
                        'workflow_summary': self.feature_engineering_agent.get_workflow_summary(),
                        'recommended_steps': self.feature_engineering_agent.get_recommended_feature_engineering_steps(),
                        'function_code': self.feature_engineering_agent.get_feature_engineer_function()
                    }
            
            # ML Modeling Phase
            if needs['ml_modeling']:
                print("Executing ML modeling...")
                
                if target_variable is None:
                    results['reports']['ml_error'] = "ML modeling requested but no target variable specified or detected."
                elif target_variable not in current_df.columns:
                    results['reports']['ml_error'] = f"Target variable '{target_variable}' not found in dataset columns: {list(current_df.columns)}"
                else:
                    self.h2o_ml_agent.invoke_agent(
                        data_raw=current_df,
                        user_instructions=user_request,
                        target_variable=target_variable
                    )
                    
                    # Get ML results
                    ml_response = self.h2o_ml_agent.get_response()
                    if ml_response:
                        results['reports']['ml_modeling'] = {
                            'workflow_summary': self.h2o_ml_agent.get_workflow_summary(),
                            'recommended_steps': self.h2o_ml_agent.get_recommended_ml_steps(),
                            'function_code': self.h2o_ml_agent.get_h2o_train_function(),
                            'leaderboard': str(self.h2o_ml_agent.get_leaderboard()),
                            'best_model_id': self.h2o_ml_agent.get_best_model_id(),
                        }
                        
                        model_path = self.h2o_ml_agent.get_model_path()
                        if model_path:
                            results['file_paths']['model'] = model_path
            
            # Step 6: Generate comprehensive string report
            report = self._generate_final_report(results)
            
            # Save the final report
            report_path = os.path.join(self.output_dir, f"supervisor_report_{timestamp}.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            error_report = f"""
SUPERVISOR AGENT ERROR REPORT
============================

Error occurred while processing request:
- CSV URL: {csv_url}
- User Request: {user_request}
- Error: {str(e)}

Please check the CSV URL and try again.
"""
            return error_report
    
    def _generate_final_report(self, results: Dict) -> str:
        """Generate a comprehensive string report from all results."""
        
        report_lines = [
            "=" * 60,
            "SUPERVISOR AGENT PROCESSING REPORT",
            "=" * 60,
            "",
            f"Timestamp: {results['timestamp']}",
            f"Original CSV URL: {results['original_csv_url']}",
            f"User Request: {results['user_request']}",
            f"Target Variable: {results['target_variable'] or 'Not specified'}",
            f"Original Data Shape: {results['data_shape']}",
            "",
            "AGENTS EXECUTED:",
            "-" * 20
        ]
        
        # Add executed agents
        for agent_type, executed in results['needs_detected'].items():
            status = "✓ EXECUTED" if executed else "✗ SKIPPED"
            report_lines.append(f"- {agent_type.replace('_', ' ').title()}: {status}")
        
        report_lines.extend(["", "FILE OUTPUTS:", "-" * 20])
        
        # Add file paths
        if results['file_paths']:
            for file_type, path in results['file_paths'].items():
                report_lines.append(f"- {file_type.replace('_', ' ').title()}: {path}")
        else:
            report_lines.append("- No files generated")
        
        # Add detailed reports for each agent
        if 'data_cleaning' in results['reports']:
            report_lines.extend([
                "",
                "DATA CLEANING RESULTS:",
                "=" * 25,
                "",
                "Workflow Summary:",
                results['reports']['data_cleaning']['workflow_summary'],
                "",
                "Recommended Steps:",
                results['reports']['data_cleaning']['recommended_steps'],
                ""
            ])
        
        if 'feature_engineering' in results['reports']:
            report_lines.extend([
                "",
                "FEATURE ENGINEERING RESULTS:",
                "=" * 30,
                "",
                "Workflow Summary:",
                results['reports']['feature_engineering']['workflow_summary'],
                "",
                "Recommended Steps:",
                results['reports']['feature_engineering']['recommended_steps'],
                ""
            ])
        
        if 'ml_modeling' in results['reports']:
            report_lines.extend([
                "",
                "MACHINE LEARNING RESULTS:",
                "=" * 27,
                "",
                "Workflow Summary:",
                results['reports']['ml_modeling']['workflow_summary'],
                "",
                "Model Leaderboard:",
                results['reports']['ml_modeling']['leaderboard'],
                "",
                f"Best Model ID: {results['reports']['ml_modeling']['best_model_id']}",
                ""
            ])
        
        if 'ml_error' in results['reports']:
            report_lines.extend([
                "",
                "ML MODELING ERROR:",
                "=" * 20,
                results['reports']['ml_error'],
                ""
            ])
        
        report_lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return "\n".join(report_lines)


# Convenience function for direct usage
def process_csv_request(
    csv_url: str, 
    user_request: str, 
    target_variable: str = None,
    model=None
) -> str:
    """
    Convenience function to process a CSV request directly.
    
    Parameters
    ----------
    csv_url : str
        URL of the CSV file to process
    user_request : str
        Natural language description of what to do
    target_variable : str, optional
        Target variable name
    model : langchain.llms.base.LLM, optional
        Language model to use
        
    Returns
    -------
    str
        Comprehensive report as a string
    """
    supervisor = SupervisorAgent(model=model)
    return supervisor.process_request(csv_url, user_request, target_variable) 