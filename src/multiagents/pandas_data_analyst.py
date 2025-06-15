from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from typing import TypedDict, Annotated, Sequence, Union
import operator

import pandas as pd
import json
from IPython.display import Markdown

from src.templates import BaseAgent
from src.agents import DataWranglingAgent, DataVisualisationAgent
from src.utils.plotly import plotly_from_dict
from src.utils.regex import get_generic_summary

AGENT_NAME = "pandas_data_analyst"

class PandasDataAnalyst(BaseAgent):
    """
    PandasDataAnalyst is a multi-agent class that combines data wrangling and visualisation capabilities.

    Parameters:
    -----------
    model:
        The language model to be used for the agents.
    data_wrangling_agent: DataWranglingAgent
        The Data Wrangling Agent for transforming raw data.
    data_visualisation_agent: DataVisualisationAgent
        The Data Visualisation Agent for generating plots.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state of the multi-agent system.

    Methods:
    --------
    ainvoke_agent(user_instructions, data_raw, **kwargs)
        Asynchronously invokes the multi-agent with user instructions and raw data.
    invoke_agent(user_instructions, data_raw, **kwargs)
        Synchronously invokes the multi-agent with user instructions and raw data.
    get_data_wrangled()
        Returns the wrangled data as a Pandas DataFrame.
    get_plotly_graph()
        Returns the Plotly graph as a Plotly object.
    get_data_wrangler_function(markdown=False)
        Returns the data wrangling function as a string, optionally in Markdown.
    get_data_visualisation_function(markdown=False)
        Returns the data visualisation function as a string, optionally in Markdown.
    """

    def __init__(
        self,
        model,
        data_wrangling_agent: DataWranglingAgent,
        data_visualisation_agent: DataVisualisationAgent,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "data_wrangling_agent": data_wrangling_agent,
            "data_visualisation_agent": data_visualisation_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_pandas_data_analyst(
            model=self._params["model"],
            data_wrangling_agent=self._params["data_wrangling_agent"]._compiled_graph,
            data_visualisation_agent=self._params["data_visualisation_agent"]._compiled_graph,
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(self, user_instructions, data_raw: Union[pd.DataFrame, dict, list], max_retries: int = 3, retry_count: int = 0, **kwargs):
        """Asynchronously invokes the multi-agent."""
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": self._convert_data_input(data_raw),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        if response.get("messages"):
            response["messages"] = self._remove_consecutive_duplicates(response["messages"])
        self.response = response

    def invoke_agent(self, user_instructions, data_raw: Union[pd.DataFrame, dict, list], max_retries: int = 3, retry_count: int = 0, **kwargs):
        """Synchronously invokes the multi-agent."""
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": self._convert_data_input(data_raw),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        if response.get("messages"):
            response["messages"] = self._remove_consecutive_duplicates(response["messages"])
        self.response = response

    def get_data_wrangled(self):
        """Returns the wrangled data as a Pandas DataFrame."""
        if self.response and self.response.get("data_wrangled"):
            return pd.DataFrame(self.response.get("data_wrangled"))

    def get_plotly_graph(self):
        """Returns the Plotly graph as a Plotly object."""
        if self.response and self.response.get("plotly_graph"):
            return plotly_from_dict(self.response.get("plotly_graph"))

    def get_data_wrangler_function(self, markdown=False):
        """Returns the data wrangling function as a string."""
        if self.response and self.response.get("data_wrangler_function"):
            code = self.response.get("data_wrangler_function")
            return Markdown(f"```python\n{code}\n```") if markdown else code

    def get_data_visualisation_function(self, markdown=False):
        """Returns the data visualisation function as a string."""
        if self.response and self.response.get("data_visualisation_function"):
            code = self.response.get("data_visualisation_function")
            return Markdown(f"```python\n{code}\n```") if markdown else code

    def get_workflow_summary(self, markdown=False):
        """Returns a summary of the workflow."""
        if self.response and self.response.get("messages"):
            agents = [msg.role for msg in self.response["messages"]]
            agent_labels = [f"- **Agent {i+1}:** {role}\n" for i, role in enumerate(agents)]
            header = f"# Pandas Data Analyst Workflow Summary\n\nThis workflow contains {len(agents)} agents:\n\n" + "\n".join(agent_labels)
            reports = [get_generic_summary(json.loads(msg.content)) for msg in self.response["messages"]]
            summary = "\n\n" + header + "\n\n".join(reports)
            return Markdown(summary) if markdown else summary

    @staticmethod
    def _convert_data_input(data_raw: Union[pd.DataFrame, dict, list]) -> Union[dict, list]:
        """Converts input data to the expected format (dict or list of dicts)."""
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()
        if isinstance(data_raw, dict):
            return data_raw
        if isinstance(data_raw, list):
            return [item.to_dict() if isinstance(item, pd.DataFrame) else item for item in data_raw]
        raise ValueError("data_raw must be a DataFrame, dict, or list of DataFrames/dicts")

    @staticmethod
    def _remove_consecutive_duplicates(messages):
        """Remove consecutive duplicate messages."""
        if not messages:
            return messages
        
        result = [messages[0]]
        for msg in messages[1:]:
            if msg.content != result[-1].content:
                result.append(msg)
        return result


def make_pandas_data_analyst(
    model,
    data_wrangling_agent: CompiledStateGraph,
    data_visualisation_agent: CompiledStateGraph,
    checkpointer: Checkpointer = None
):
    """
    Creates a multi-agent system that wrangles data and optionally visualises it.

    Parameters:
    -----------
    model: The language model to be used.
    data_wrangling_agent: CompiledStateGraph
        The Data Wrangling Agent.
    data_visualisation_agent: CompiledStateGraph
        The Data Visualisation Agent.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state.

    Returns:
    --------
    CompiledStateGraph: The compiled multi-agent system.
    """
    
    llm = model
    
    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a Pandas Data Manipulation Wrangling Agent, a Charting Visualisation Agent, and a Pandas Table Agent. Your job is to tell the agents which actions to perform and determine the correct routing for the incoming user question:
        
        1. Determine what the correct format for a Users Question should be for use with a Pandas Data Wrangling Agent based on the incoming user question. Anything related to data wrangling and manipulation should be passed along. Anything related to data analysis can be handled by the Pandas Agent. Anything that uses Pandas can be passed along. Tables can be returned from this agent. Don't pass along anything about plotting or visualisation.
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        3. If a chart is requested, determine the correct format of a Users Question should be used with a Data Visualisation Agent. Anything related to plotting and visualisation should be passed along.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the Pandas Data Wrangling and Transformation agent. This will be the 'user_instructions_data_wrangling'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualisation ('chart') or a 'table' returned with the results of the Data Wrangling Agent. If unknown, not specified or 'None' is found, then select 'table'.  
        
        If a 'chart' is requested, return the 'user_instructions_data_visualisation'. If 'None' is found, return None.
        
        Return JSON with 'user_instructions_data_wrangling', 'user_instructions_data_visualisation' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {user_instructions}
        """,
        input_variables=["user_instructions"]
    )

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        user_instructions_data_wrangling: str
        user_instructions_data_visualisation: str
        routing_preprocessor_decision: str
        data_raw: Union[dict, list]
        data_wrangled: dict
        data_wrangler_function: str
        data_visualisation_function: str
        plotly_graph: dict
        plotly_error: str
        max_retries: int
        retry_count: int

    def preprocess_routing(state: PrimaryState):
        """Preprocesses the routing decision."""
        user_instructions = state["user_instructions"]
        
        chain = routing_preprocessor_prompt | llm | JsonOutputParser()
        response = chain.invoke({"user_instructions": user_instructions})
        
        return {
            "user_instructions_data_wrangling": response.get("user_instructions_data_wrangling", user_instructions),
            "user_instructions_data_visualisation": response.get("user_instructions_data_visualisation"),
            "routing_preprocessor_decision": response.get("routing_preprocessor_decision", "table")
        }

    def router_chart_or_table(state: PrimaryState):
        """Routes to either chart generation or table output."""
        return "invoke_data_visualisation_agent" if state["routing_preprocessor_decision"] == "chart" else "route_printer"

    def invoke_data_wrangling_agent(state: PrimaryState):
        """Invokes the data wrangling agent."""
        response = data_wrangling_agent.invoke({
            "user_instructions": state["user_instructions_data_wrangling"],
            "data_raw": state["data_raw"],
            "max_retries": state["max_retries"],
            "retry_count": state["retry_count"]
        })
        
        return {
            "data_wrangled": response.get("data_wrangled", {}),
            "data_wrangler_function": response.get("data_wrangler_function", "")
        }

    def invoke_data_visualisation_agent(state: PrimaryState):
        """Invokes the data visualisation agent."""
        response = data_visualisation_agent.invoke({
            "user_instructions": state["user_instructions_data_visualisation"],
            "data_raw": state["data_wrangled"],
            "max_retries": state["max_retries"],
            "retry_count": state["retry_count"]
        })
        
        return {
            "plotly_graph": response.get("plotly_graph", {}),
            "data_visualisation_function": response.get("data_visualisation_function", ""),
            "plotly_error": response.get("plotly_error", "")
        }

    def route_printer(state: PrimaryState):
        """Final routing step."""
        return {"messages": [BaseMessage(content="Analysis complete", type="ai")]}

    # Build the workflow
    workflow = StateGraph(PrimaryState)
    
    # Add nodes
    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("invoke_data_wrangling_agent", invoke_data_wrangling_agent)
    workflow.add_node("invoke_data_visualisation_agent", invoke_data_visualisation_agent)
    workflow.add_node("route_printer", route_printer)
    
    # Add edges
    workflow.add_edge(START, "preprocess_routing")
    workflow.add_edge("preprocess_routing", "invoke_data_wrangling_agent")
    workflow.add_conditional_edges("invoke_data_wrangling_agent", router_chart_or_table)
    workflow.add_edge("invoke_data_visualisation_agent", "route_printer")
    workflow.add_edge("route_printer", END)
    
    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)
    
    return app 