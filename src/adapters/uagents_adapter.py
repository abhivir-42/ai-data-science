"""
uAgents adapter for AI Data Science agents.

This module provides adapter functionality to convert AI Data Science agents
into uAgents that can interact with other agents in the Fetch AI Agentverse.
"""

import os
import pandas as pd
import dotenv
from typing import Dict, Any, Optional, Callable, Union
import logging
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel

from src.agents.data_cleaning_agent import DataCleaningAgent
from src.agents.data_loader_tools_agent import DataLoaderToolsAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
def load_env_keys() -> Dict[str, Optional[str]]:
    """
    Load API keys from environment variables or .env file.
    
    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary with API keys
    """
    # Try to load from .env file in project root
    dotenv_path = Path(os.getcwd()) / "ai-data-science" / ".env"
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
    
    return {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "agentverse_api_token": os.environ.get("AGENTVERSE_API_TOKEN"),
    }


def cleanup_uagent(name: str) -> Dict[str, Any]:
    """
    Cleanup and deregister a uAgent.
    
    Parameters
    ----------
    name : str
        Name of the agent to clean up
        
    Returns
    -------
    Dict[str, Any]
        Status of the cleanup operation
    """
    try:
        from uagents_adapter import cleanup_uagent as adapter_cleanup
        result = adapter_cleanup(name)
        return {"status": "success", "message": f"Agent {name} cleaned up successfully", "details": result}
    except ImportError as e:
        return {
            "status": "error",
            "error": f"Failed to import uAgents dependencies: {str(e)}",
            "solution": "Install required packages with: pip install 'uagents-adapter>=2.2.0'"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to clean up agent: {str(e)}",
            "solution": "Ensure the agent is running and the name is correct"
        }


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
        
        # Try to get API token from init params, then from env
        if api_token is None:
            keys = load_env_keys()
            api_token = keys.get("agentverse_api_token")
            
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
        
        # Print init confirmation with partial API token for verification
        token_preview = self.api_token[:5] + "..." + self.api_token[-5:] if self.api_token else "None"
        logger.info(f"Initialized {self.name} adapter with API token: {token_preview}")
    
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
            from uagents_adapter import UAgentRegisterTool
            
            # Check if API token is available
            if not self.api_token:
                return {
                    "error": "API token not provided",
                    "solution": "Set api_token parameter or add AGENTVERSE_API_TOKEN to environment variables"
                }
            
            # Define the wrapper function to handle the agent invocation
            def data_cleaning_wrapper(query: dict) -> dict:
                """Wrapper function to invoke the DataCleaningAgent."""
                logger.info(f"Data cleaning request received: {query}")
                
                # Extract instructions and data from query
                instructions = query.get("instructions", "")
                data_dict = query.get("data", {})
                
                # Convert data to DataFrame if provided
                if data_dict:
                    data = pd.DataFrame.from_dict(data_dict)
                else:
                    return {"error": "No data provided in query"}
                
                # Invoke the agent
                self.agent.invoke_agent(
                    data_raw=data,
                    user_instructions=instructions
                )
                
                # Get results
                cleaned_data = self.agent.get_data_cleaned()
                
                # Return results
                if cleaned_data is not None:
                    response = {
                        "status": "success",
                        "data": cleaned_data.to_dict(),
                        "summary": self.agent.get_workflow_summary()
                    }
                else:
                    response = {
                        "status": "error",
                        "error": "Failed to clean data",
                        "summary": self.agent.get_workflow_summary()
                    }
                
                return response
            
            # Create the registration tool
            uagent_register_tool = UAgentRegisterTool()
            
            # Register the agent with the wrapper function
            result = uagent_register_tool.invoke({
                "agent_obj": data_cleaning_wrapper,
                "name": self.name,
                "port": self.port,
                "description": self.description,
                "mailbox": self.mailbox,
                "api_token": self.api_token,
                "return_dict": True
            })
            
            self.uagent_info = result
            logger.info(f"Agent registered successfully: {self.name}")
            return result
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return {
                "error": f"Failed to import uAgents dependencies: {str(e)}",
                "solution": "Install required packages with: pip install 'uagents-adapter>=2.2.0'"
            }
        except Exception as e:
            logger.error(f"Registration error: {e}")
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
    
    def cleanup(self) -> Dict[str, Any]:
        """
        Clean up and deregister the agent.
        
        Returns
        -------
        Dict[str, Any]
            Status of the cleanup operation
        """
        return cleanup_uagent(self.name)


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
        
        # Try to get API token from init params, then from env
        if api_token is None:
            keys = load_env_keys()
            api_token = keys.get("agentverse_api_token")
            
        self.api_token = api_token
        
        # Create the DataLoaderToolsAgent with default parameters
        self.agent = DataLoaderToolsAgent(
            model=model
        )
        
        # Initialize uAgent info
        self.uagent_info = None
        
        # Print init confirmation with partial API token for verification
        token_preview = self.api_token[:5] + "..." + self.api_token[-5:] if self.api_token else "None"
        logger.info(f"Initialized {self.name} adapter with API token: {token_preview}")
    
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
            from uagents_adapter import UAgentRegisterTool
            
            # Check if API token is available
            if not self.api_token:
                return {
                    "error": "API token not provided",
                    "solution": "Set api_token parameter or add AGENTVERSE_API_TOKEN to environment variables"
                }
            
            # Define the wrapper function to handle the agent invocation
            def data_loader_wrapper(query: dict) -> dict:
                """Wrapper function to invoke the DataLoaderToolsAgent."""
                logger.info(f"Data loading request received: {query}")
                
                # Extract instructions from query
                instructions = query.get("instructions", "")
                
                if not instructions:
                    return {"error": "No instructions provided in query"}
                
                # Invoke the agent
                self.agent.invoke_agent(
                    user_instructions=instructions
                )
                
                # Get results
                artifacts = self.agent.get_artifacts()
                
                # Return results
                if artifacts:
                    # Try to convert to DataFrame if it's a dict with 'data'
                    if isinstance(artifacts, dict) and "data" in artifacts:
                        data_df = pd.DataFrame.from_dict(artifacts["data"])
                        response = {
                            "status": "success",
                            "data": data_df.to_dict(),
                            "file_info": artifacts.get("file_info", {})
                        }
                    else:
                        response = {
                            "status": "success",
                            "data": artifacts,
                        }
                else:
                    response = {
                        "status": "error",
                        "error": "Failed to load data",
                        "tool_calls": self.agent.get_tool_calls()
                    }
                
                return response
            
            # Create the registration tool
            uagent_register_tool = UAgentRegisterTool()
            
            # Register the agent with the wrapper function
            result = uagent_register_tool.invoke({
                "agent_obj": data_loader_wrapper,
                "name": self.name,
                "port": self.port,
                "description": self.description,
                "mailbox": self.mailbox,
                "api_token": self.api_token,
                "return_dict": True
            })
            
            self.uagent_info = result
            logger.info(f"Agent registered successfully: {self.name}")
            return result
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return {
                "error": f"Failed to import uAgents dependencies: {str(e)}",
                "solution": "Install required packages with: pip install 'uagents-adapter>=2.2.0'"
            }
        except Exception as e:
            logger.error(f"Registration error: {e}")
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
    
    def cleanup(self) -> Dict[str, Any]:
        """
        Clean up and deregister the agent.
        
        Returns
        -------
        Dict[str, Any]
            Status of the cleanup operation
        """
        return cleanup_uagent(self.name) 