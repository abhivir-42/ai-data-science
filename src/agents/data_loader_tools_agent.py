"""
Data Loader Tools Agent for AI Data Science.

This module provides a specialized agent for loading data from various
sources and formats based on user instructions.
"""

from typing import Any, Optional, Annotated, Sequence, List, Dict
import operator
import pandas as pd
import os

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage

from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph

from ai_data_science.templates import BaseAgent
from ai_data_science.utils.regex import format_agent_name
from ai_data_science.tools import (
    load_file,
    load_directory,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
)
from ai_data_science.utils.messages import get_tool_call_names

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
]

class DataLoaderToolsAgent(BaseAgent):
    """
    A Data Loader Agent that can interact with data loading tools and search for files in your file system.
    
    The agent can load data from various sources including:
    - CSV files (local or remote)
    - Excel files
    - JSON data
    - Parquet files
    
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
                return pd.DataFrame.from_dict(self.response["data_loader_artifacts"]["data"])
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
        
        response = data_loader_agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
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
        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)

        # Get the last tool artifact safely
        last_tool_artifact = None
        for message in reversed(internal_messages):
            if hasattr(message, "additional_kwargs") and "artifact" in message.additional_kwargs:
                last_tool_artifact = message.additional_kwargs["artifact"]
                break
            elif hasattr(message, "artifact"):
                last_tool_artifact = message.artifact
                break
            elif isinstance(message, dict) and "artifact" in message:
                last_tool_artifact = message["artifact"]
                break

        # Extract tool calls from the messages
        tool_calls = get_tool_call_names(internal_messages)
        
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