"""
Data Loader Tools Agent for AI Data Science.

This module provides a specialized agent for loading data from various
sources and formats based on user instructions.
"""

from typing import Any, Optional, Annotated, Sequence, List, Dict
import operator
import pandas as pd
import os
import json
from pprint import pformat

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage

from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph

from src.templates import BaseAgent
from src.utils.regex import format_agent_name
from src.tools import (
    load_file,
    load_directory,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
    extract_pdf_text,
    extract_pdf_tables,
    smart_extract_data_from_pdf,
    get_pdf_info,
)
from src.utils.messages import get_tool_call_names

# Setup
AGENT_NAME = "data_loader_tools_agent"

# Define the tools available to the agent
tools = [
    load_file,
    load_directory,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
    extract_pdf_text,
    extract_pdf_tables,
    smart_extract_data_from_pdf,
    get_pdf_info,
]

class DataLoaderToolsAgent(BaseAgent):
    """
    A Data Loader Agent that can interact with data loading tools and search for files in your file system.
    
    The agent can load data from various sources including:
    - CSV files (local or remote)
    - Excel files
    - JSON data
    - Parquet files
    - PDF documents (text and table extraction)
    
    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    create_react_agent_kwargs : dict, optional
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict, optional
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langgraph.types.Checkpointer, optional
        A checkpointer to use for saving and loading the agent's state.
        
    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled graph.
    ainvoke_agent(user_instructions: str=None, **kwargs)
        Runs the agent with the given user instructions asynchronously.
    invoke_agent(user_instructions: str=None, **kwargs)
        Runs the agent with the given user instructions.
    get_internal_messages(markdown: bool=False)
        Returns the internal messages from the agent's response.
    get_artifacts(as_dataframe: bool=False)
        Returns the data artifacts from the agent's response.
    get_ai_message(markdown: bool=False)
        Returns the AI message from the agent's response.
    get_tool_calls()
        Returns the tool calls made by the agent.
    
    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science.agents import DataLoaderToolsAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_loader_agent = DataLoaderToolsAgent(
        model=llm
    )

    data_loader_agent.invoke_agent(
        user_instructions="Load the CSV file from data/sample_data.csv"
    )

    # Get the loaded data
    data_loader_agent.get_artifacts(as_dataframe=True)
    
    # Get the tool calls that were made
    data_loader_agent.get_tool_calls()
    ```
    """
    
    def __init__(
        self, 
        model: Any,
        create_react_agent_kwargs: Optional[Dict] = None,
        invoke_react_agent_kwargs: Optional[Dict] = None,
        checkpointer: Optional[Checkpointer] = None,
    ):
        """Initialize the DataLoaderToolsAgent."""
        if create_react_agent_kwargs is None:
            create_react_agent_kwargs = {}
        
        if invoke_react_agent_kwargs is None:
            invoke_react_agent_kwargs = {}
            
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        
    def _make_compiled_graph(self):
        """
        Creates the compiled graph for the agent.
        """
        self.response = None
        return make_data_loader_tools_agent(**self._params)
    
    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        
        Parameters
        ----------
        **kwargs
            The parameters to update.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
        
    async def ainvoke_agent(
        self, 
        user_instructions: str = None, 
        **kwargs
    ):
        """
        Runs the agent with the given user instructions asynchronously.
        
        Parameters
        ----------
        user_instructions : str, optional
            The user instructions to pass to the agent.
        **kwargs
            Additional keyword arguments to pass to the agent's ainvoke method.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
            }, 
            **kwargs
        )
        self.response = response
        return None
    
    def invoke_agent(
        self, 
        user_instructions: str = None, 
        **kwargs
    ):
        """
        Runs the agent with the given user instructions.
        
        Parameters
        ----------
        user_instructions : str, optional
            The user instructions to pass to the agent.
        **kwargs
            Additional keyword arguments to pass to the agent's invoke method.
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
            },
            **kwargs
        )
        self.response = response
        return None
    
    def get_internal_messages(self, markdown: bool = False):
        """
        Returns the internal messages from the agent's response.
        
        Parameters
        ----------
        markdown : bool, optional
            Whether to return the messages as Markdown. Defaults to False.
            
        Returns
        -------
        Union[List[BaseMessage], Markdown]
            The internal messages from the agent's response.
        """
        if not self.response:
            return "No response available. Run invoke_agent() first."
            
        pretty_print = "\n\n".join([f"### {msg.type.upper()}\n\nID: {msg.id}\n\nContent:\n\n{msg.content}" for msg in self.response["internal_messages"]])       
        if markdown:
            return Markdown(pretty_print)
        else:
            return self.response["internal_messages"]
    
    def get_artifacts(self, as_dataframe: bool = False):
        """
        Returns the data artifacts from the agent's response.
        
        Parameters
        ----------
        as_dataframe : bool, optional
            Whether to return the artifacts as a pandas DataFrame. Defaults to False.
            
        Returns
        -------
        Union[Dict, pd.DataFrame]
            The data artifacts from the agent's response.
        """
        if not self.response:
            return "No response available. Run invoke_agent() first."
            
        if as_dataframe and self.response.get("data_loader_artifacts"):
            if isinstance(self.response["data_loader_artifacts"], dict) and "data" in self.response["data_loader_artifacts"]:
                # Fix the DataFrame conversion issue - use orient='dict' to preserve shape
                data_dict = self.response["data_loader_artifacts"]["data"]
                try:
                    # Try to create DataFrame properly preserving original shape
                    df = pd.DataFrame.from_dict(data_dict, orient='columns')
                    return df
                except Exception as e:
                    print(f"Warning: Error converting to DataFrame: {e}")
                    # Fallback to original method if there's an issue
                    return pd.DataFrame(data_dict)
            return pd.DataFrame(self.response["data_loader_artifacts"])
        else:
            return self.response.get("data_loader_artifacts", {})
    
    def get_ai_message(self, markdown: bool = False):
        """
        Returns the AI message from the agent's response.
        
        Parameters
        ----------
        markdown : bool, optional
            Whether to return the message as Markdown. Defaults to False.
            
        Returns
        -------
        Union[str, Markdown]
            The AI message from the agent's response.
        """
        if not self.response:
            return "No response available. Run invoke_agent() first."
            
        if markdown:
            return Markdown(self.response["messages"][0].content)
        else:
            return self.response["messages"][0].content
    
    def get_tool_calls(self):
        """
        Returns the tool calls made by the agent.
        
        Returns
        -------
        List[str]
            The tool calls made by the agent.
        """
        if not self.response:
            return "No response available. Run invoke_agent() first."
            
        return self.response.get("tool_calls", [])


def _is_valid_data_artifact(content_dict):
    """
    Check if a dictionary is a valid data artifact.
    
    Parameters
    ----------
    content_dict : dict
        Dictionary to check
        
    Returns
    -------
    bool
        True if valid data artifact
    """
    if not isinstance(content_dict, dict):
        return False
    
    # Check for standard data structure
    if "data" in content_dict:
        return True
    
    # Check for chunked data structure
    if "chunk_info" in content_dict and "full_dataframe" in content_dict:
        return True
    
    # Check for error structure
    if "error" in content_dict:
        return False
    
    return False


def make_data_loader_tools_agent(
    model: Any,
    create_react_agent_kwargs: Optional[Dict] = None,
    invoke_react_agent_kwargs: Optional[Dict] = None,
    checkpointer: Optional[Checkpointer] = None,
):
    """
    Creates a Data Loader Agent that can interact with data loading tools.
    
    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    create_react_agent_kwargs : dict, optional
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict, optional
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langgraph.types.Checkpointer, optional
        A checkpointer to use for saving and loading the agent's state.
    
    Returns
    -------
    StateGraph
        A compiled state graph that represents the data loader agent.
    """
    if create_react_agent_kwargs is None:
        create_react_agent_kwargs = {}
    
    if invoke_react_agent_kwargs is None:
        invoke_react_agent_kwargs = {}
    
    # Define GraphState for the router
    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_loader_artifacts: dict
        tool_calls: List[str]
    
    def data_loader_agent(state):
        """
        The main data loader agent node function.
        
        Parameters
        ----------
        state : GraphState
            The current state of the graph.
            
        Returns
        -------
        dict
            Updates to the graph state.
        """
        print(format_agent_name(AGENT_NAME))
        print("    ")
        
        print("    * RUN REACT TOOL-CALLING AGENT")
        
        tool_node = ToolNode(
            tools=tools
        )
        
        data_loader_agent = create_react_agent(
            model, 
            tools=tool_node, 
            state_schema=GraphState,
            checkpointer=checkpointer,
            **create_react_agent_kwargs,
        )
        
        # Create a user message from instructions
        user_message = HumanMessage(content=state["user_instructions"])
        
        response = data_loader_agent.invoke(
            {
                "messages": [user_message],
            },
            **invoke_react_agent_kwargs,
        )
        
        print("    * POST-PROCESS RESULTS")
        
        internal_messages = response['messages']

        # Ensure there is at least one AI message
        if not internal_messages:
            return {
                "internal_messages": [],
                "data_loader_artifacts": None,
                "tool_calls": [],
            }

        # Get the last AI message
        last_ai_message = None
        for msg in reversed(internal_messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if last_ai_message is None:
            last_ai_message = AIMessage(content="No AI response was generated.", role=AGENT_NAME)

        # Find and extract tool artifacts directly from tool messages
        last_tool_artifact = None
        
        # Print message types for debugging
        print(f"    * Message types: {[type(msg).__name__ for msg in internal_messages]}")
        
        # Extract data directly from ToolMessage objects
        for msg in internal_messages:
            if isinstance(msg, ToolMessage):
                if msg.name in ["load_file", "load_directory"]:
                    print(f"    * Checking tool message: {msg.name}")
                    
                    # Try to parse the content if it's a string
                    if isinstance(msg.content, str):
                        try:
                            content_dict = json.loads(msg.content)
                            if _is_valid_data_artifact(content_dict):
                                last_tool_artifact = content_dict
                                print(f"    * Found data in tool message: {msg.name} (JSON string)")
                                break
                        except Exception as e:
                            print(f"    * Failed to parse JSON from tool message: {e}")
                    
                    # If content is already a dict
                    elif isinstance(msg.content, dict) and _is_valid_data_artifact(msg.content):
                        last_tool_artifact = msg.content
                        print(f"    * Found data in tool message: {msg.name} (dict)")
                        break
                    
                    # For debugging
                    print(f"    * Tool message content type: {type(msg.content)}")
                    if isinstance(msg.content, dict):
                        print(f"    * Tool message keys: {list(msg.content.keys())}")

        # Extract tool calls from the messages
        tool_calls = get_tool_call_names(internal_messages)
        
        # If still no artifact but load tools were called, try to reconstruct
        if last_tool_artifact is None and ("load_file" in tool_calls or "load_directory" in tool_calls):
            print("    * WARNING: Data loading tool was called but no artifact found in messages")
            print("    * Attempting direct tool execution as fallback...")
            
            # Try direct tool execution as fallback
            for msg in internal_messages:
                if isinstance(msg, ToolMessage) and msg.name == "load_file":
                    # Look for file path in various places
                    file_path = None
                    
                    # Check if we can extract file path from the message
                    if hasattr(msg, 'tool_input') and isinstance(msg.tool_input, dict):
                        file_path = msg.tool_input.get("file_path")
                    elif isinstance(msg.content, str) and "file_path" in msg.content:
                        # Try to extract from content string
                        import re
                        match = re.search(r'file_path["\']?\s*:\s*["\']?([^"\']+)["\']?', msg.content)
                        if match:
                            file_path = match.group(1)
                    
                    if file_path:
                        try:
                            print(f"    * Directly loading file: {file_path}")
                            result = load_file(file_path)
                            if _is_valid_data_artifact(result):
                                last_tool_artifact = result
                                print(f"    * Successfully loaded file directly")
                                break
                        except Exception as e:
                            print(f"    * Error loading file directly: {e}")
        
        # Create a useful error message if data loading failed
        if last_tool_artifact is None and ("load_file" in tool_calls or "load_directory" in tool_calls):
            print("    * Creating empty data structure as fallback")
            # Create a minimal structure to avoid errors
            last_tool_artifact = {
                "data": {},
                "file_info": {
                    "path": "unknown",
                    "name": "unknown",
                    "error": "Data loading failed - artifact not found in messages"
                }
            }
        
        return {
            "messages": [last_ai_message], 
            "internal_messages": internal_messages,
            "data_loader_artifacts": last_tool_artifact,
            "tool_calls": tool_calls,
        }
    
    # Create the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add the data loader agent node
    workflow.add_node("data_loader_agent", data_loader_agent)
    
    # Connect the nodes
    workflow.add_edge(START, "data_loader_agent")
    workflow.add_edge("data_loader_agent", END)
    
    # Compile the workflow
    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app 