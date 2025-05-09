"""
uAgents adapter for AI Data Science agents.

This module provides adapter functionality to convert AI Data Science agents
into uAgents that can interact with other agents in the Fetch AI Agentverse.
Simplified to avoid common issues.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel

from src.agents.data_cleaning_agent import DataCleaningAgent
from src.agents.data_loader_tools_agent import DataLoaderToolsAgent


class DataCleaningAgentAdapter:
    """
    Adapter for converting DataCleaningAgent to a uAgent compatible with Fetch AI Agentverse.
    
    This adapter wraps the DataCleaningAgent and makes it compatible with the uAgents system.
    To register with Agentverse, use the register() method.
    
    Parameters
    ----------
    model : BaseLanguageModel
        The language model to use for the DataCleaningAgent
    name : str, optional
        Name of the agent (defaults to "data_cleaning_agent")
    port : int, optional
        Port to run the agent on (defaults to 8000)
    description : str, optional
        Description of the agent (defaults to a standard description)
    mailbox : bool, optional
        Whether to use the Agentverse mailbox service (defaults to True)
    api_token : str, optional
        API token for Agentverse registration
    n_samples : int, optional
        Number of samples to use for dataset summaries (defaults to 30)
    log : bool, optional
        Whether to log agent operations (defaults to False)
    log_path : str, optional
        Path to log files (defaults to None)
    human_in_the_loop : bool, optional
        Whether to use human review (defaults to False)
    
    Attributes
    ----------
    agent : DataCleaningAgent
        The wrapped DataCleaningAgent instance
    uagent_info : dict
        Information about the registered uAgent
    """
    
    def __init__(
        self,
        model: BaseLanguageModel,
        name: str = "data_cleaning_agent",
        port: int = 8000,
        description: str = None,
        mailbox: bool = True,
        api_token: Optional[str] = None,
        n_samples: int = 30,
        log: bool = False,
        log_path: Optional[str] = None,
        human_in_the_loop: bool = False,
    ):
        """Initialize the adapter with a language model and optional configuration."""
        self.model = model
        self.name = name
        self.port = port
        
        if description is None:
            self.description = (
                "A data cleaning agent that can process datasets based on "
                "user-defined instructions or default cleaning steps. "
                "It can handle missing values, outliers, duplicates, and data type conversions."
            )
        else:
            self.description = description
            
        self.mailbox = mailbox
        self.api_token = api_token
        self.n_samples = n_samples
        self.log = log
        self.log_path = log_path
        self.human_in_the_loop = human_in_the_loop
        
        # Create the DataCleaningAgent
        self.agent = DataCleaningAgent(
            model=model,
            n_samples=n_samples,
            log=log,
            log_path=log_path,
            human_in_the_loop=human_in_the_loop
        )
        
        # Initialize uAgent info
        self.uagent_info = None
    
    def register(self) -> Dict[str, Any]:
        """
        Register the DataCleaningAgent as a uAgent with the Agentverse.
        
        Returns
        -------
        Dict[str, Any]
            Information about the registered uAgent
        
        Note
        ----
        This method requires the 'uagents' and 'uagents-adapter' packages to be installed.
        For best results, use version 2.2.0 or later of both packages.
        """
        try:
            # Import here to avoid dependency issues if uagents is not installed
            from uagents_adapter.langchain import UAgentRegisterTool
            
            # Create the agent executor from the simplified adapter
            uagent_register_tool = UAgentRegisterTool()
            
            # Register the agent
            result = uagent_register_tool.invoke({
                "agent_obj": self.agent,
                "name": self.name,
                "port": self.port,
                "description": self.description,
                "mailbox": self.mailbox,
                "api_token": self.api_token,
                "return_dict": True
            })
            
            self.uagent_info = result
            return result
            
        except ImportError as e:
            return {
                "error": f"Failed to import uAgents dependencies: {str(e)}",
                "solution": "Install required packages with: pip install 'uagents-adapter[langchain]>=2.2.0'"
            }
        except Exception as e:
            return {
                "error": f"Failed to register agent: {str(e)}",
                "solution": "Ensure you have the latest uAgents version installed and check network connectivity"
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the registered uAgent.
        
        Returns
        -------
        Dict[str, Any]
            Information about the registered uAgent
        """
        if self.uagent_info is None:
            return {"status": "Agent not registered", "solution": "Call the register() method first"}
        
        return self.uagent_info
    
    def clean_data(self, data: pd.DataFrame, instructions: str = None) -> pd.DataFrame:
        """
        Clean a dataset using the DataCleaningAgent.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset to clean
        instructions : str, optional
            User instructions for cleaning the data
            
        Returns
        -------
        pd.DataFrame
            The cleaned dataset
        """
        self.agent.invoke_agent(
            data_raw=data,
            user_instructions=instructions
        )
        
        return self.agent.get_data_cleaned()


class DataLoaderToolsAgentAdapter:
    """
    Adapter for converting DataLoaderToolsAgent to a uAgent compatible with Fetch AI Agentverse.
    
    This adapter wraps the DataLoaderToolsAgent and makes it compatible with the uAgents system.
    To register with Agentverse, use the register() method.
    
    Parameters
    ----------
    model : BaseLanguageModel
        The language model to use for the DataLoaderToolsAgent
    name : str, optional
        Name of the agent (defaults to "data_loader_agent")
    port : int, optional
        Port to run the agent on (defaults to 8001)
    description : str, optional
        Description of the agent (defaults to a standard description)
    mailbox : bool, optional
        Whether to use the Agentverse mailbox service (defaults to True)
    api_token : str, optional
        API token for Agentverse registration
    
    Attributes
    ----------
    agent : DataLoaderToolsAgent
        The wrapped DataLoaderToolsAgent instance
    uagent_info : dict
        Information about the registered uAgent
    """
    
    def __init__(
        self,
        model: BaseLanguageModel,
        name: str = "data_loader_agent",
        port: int = 8001,
        description: str = None,
        mailbox: bool = True,
        api_token: Optional[str] = None,
    ):
        """Initialize the adapter with a language model and optional configuration."""
        self.model = model
        self.name = name
        self.port = port
        
        if description is None:
            self.description = (
                "A data loader agent that can interact with data loading tools and search for files. "
                "It can load data from various sources including CSV files, Excel files, JSON data, "
                "and Parquet files."
            )
        else:
            self.description = description
            
        self.mailbox = mailbox
        self.api_token = api_token
        
        # Create the DataLoaderToolsAgent with default parameters
        self.agent = DataLoaderToolsAgent(
            model=model
        )
        
        # Initialize uAgent info
        self.uagent_info = None
    
    def register(self) -> Dict[str, Any]:
        """
        Register the DataLoaderToolsAgent as a uAgent with the Agentverse.
        
        Returns
        -------
        Dict[str, Any]
            Information about the registered uAgent
        
        Note
        ----
        This method requires the 'uagents' and 'uagents-adapter' packages to be installed.
        For best results, use version 2.2.0 or later of both packages.
        """
        try:
            # Import here to avoid dependency issues if uagents is not installed
            from uagents_adapter.langchain import UAgentRegisterTool
            
            # Create the agent executor from the simplified adapter
            uagent_register_tool = UAgentRegisterTool()
            
            # Register the agent
            result = uagent_register_tool.invoke({
                "agent_obj": self.agent,
                "name": self.name,
                "port": self.port,
                "description": self.description,
                "mailbox": self.mailbox,
                "api_token": self.api_token,
                "return_dict": True
            })
            
            self.uagent_info = result
            return result
            
        except ImportError as e:
            return {
                "error": f"Failed to import uAgents dependencies: {str(e)}",
                "solution": "Install required packages with: pip install 'uagents-adapter[langchain]>=2.2.0'"
            }
        except Exception as e:
            return {
                "error": f"Failed to register agent: {str(e)}",
                "solution": "Ensure you have the latest uAgents version installed and check network connectivity"
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the registered uAgent.
        
        Returns
        -------
        Dict[str, Any]
            Information about the registered uAgent
        """
        if self.uagent_info is None:
            return {"status": "Agent not registered", "solution": "Call the register() method first"}
        
        return self.uagent_info
    
    def load_data(self, instructions: str) -> pd.DataFrame:
        """
        Load data based on the provided instructions.
        
        Parameters
        ----------
        instructions : str
            User instructions for loading data
            
        Returns
        -------
        pd.DataFrame
            The loaded dataset
        """
        self.agent.invoke_agent(
            user_instructions=instructions
        )
        
        return self.agent.get_artifacts(as_dataframe=True) 