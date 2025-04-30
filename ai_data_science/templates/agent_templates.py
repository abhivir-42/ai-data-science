"""
Template modules for agent workflows in AI Data Science.
"""

from typing import Dict, Any, Callable, Optional, List, Type, TypeVar, Generic, Union, Literal
from abc import ABC, abstractmethod
import json
import inspect
import textwrap

from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer, Command

class BaseAgent(ABC):
    """
    Base class for AI Data Science agents.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the agent with given parameters.
        """
        self._params = kwargs
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
    
    @abstractmethod
    def _make_compiled_graph(self):
        """
        Create the compiled graph for the agent.
        """
        pass
    
    def update_params(self, **kwargs):
        """
        Update the agent's parameters and rebuild the compiled graph.
        
        Parameters
        ----------
        **kwargs
            New parameter values.
        """
        self._params.update(kwargs)
        self._compiled_graph = self._make_compiled_graph()
    
    async def ainvoke_agent(self, **kwargs):
        """
        Asynchronously invoke the agent.
        
        Parameters
        ----------
        **kwargs
            Parameters for the agent's graph.
        """
        self.response = await self._compiled_graph.ainvoke(kwargs)
        return None
    
    def invoke_agent(self, **kwargs):
        """
        Synchronously invoke the agent.
        
        Parameters
        ----------
        **kwargs
            Parameters for the agent's graph.
        """
        self.response = self._compiled_graph.invoke(kwargs)
        return None
    
    def show(self):
        """
        Display the agent's graph.
        """
        if hasattr(self._compiled_graph, "get_graph"):
            return self._compiled_graph.get_graph().draw()
        return "Graph visualization not available"
    
    def get_response(self):
        """
        Get the agent's response.
        """
        return self.response
    
    def get_state_keys(self):
        """
        Get the keys in the agent's state.
        """
        if self.response:
            return list(self.response.keys())
        return []
    
    def get_state_properties(self):
        """
        Get detailed properties of each key in the agent's state.
        """
        if not self.response:
            return {}
            
        properties = {}
        for key, value in self.response.items():
            if isinstance(value, (str, int, float, bool)):
                properties[key] = {"type": type(value).__name__, "value": str(value)[:100] + "..." if isinstance(value, str) and len(str(value)) > 100 else value}
            elif isinstance(value, dict):
                properties[key] = {"type": "dict", "length": len(value), "keys": list(value.keys())[:10]}
            elif isinstance(value, list):
                properties[key] = {"type": "list", "length": len(value)}
            else:
                properties[key] = {"type": str(type(value))}
        
        return properties


def node_func_execute_agent_code_on_data(
    state: Dict[str, Any],
    data_key: str,
    result_key: str,
    error_key: str,
    code_snippet_key: str,
    agent_function_name: str,
    pre_processing: Callable = lambda x: x,
    post_processing: Callable = lambda x: x,
    error_message_prefix: str = "An error occurred: "
) -> Dict[str, Any]:
    """
    Execute agent-generated code on data and handle any errors.
    
    Parameters
    ----------
    state : Dict[str, Any]
        The current state of the agent.
    data_key : str
        The key in the state containing the data to process.
    result_key : str
        The key to store the result in the state.
    error_key : str
        The key to store any error messages in the state.
    code_snippet_key : str
        The key containing the code to execute.
    agent_function_name : str
        The name of the function to call in the code.
    pre_processing : Callable, optional
        Function to preprocess the data, by default lambda x: x
    post_processing : Callable, optional
        Function to postprocess the result, by default lambda x: x
    error_message_prefix : str, optional
        Prefix for error messages, by default "An error occurred: "
        
    Returns
    -------
    Dict[str, Any]
        The updated state.
    """
    print(f"    * EXECUTE AGENT CODE")
    
    # Get the data and code from the state
    data = state.get(data_key)
    code_snippet = state.get(code_snippet_key)
    
    if not code_snippet or not data:
        return {
            error_key: f"{error_message_prefix}Missing code or data in state"
        }
    
    # Prepare empty namespace for execution
    namespace = {}
    
    try:
        # Execute the code in the namespace
        exec(code_snippet, namespace)
        
        # Check if the function exists in the namespace
        if agent_function_name not in namespace:
            return {
                error_key: f"{error_message_prefix}Function '{agent_function_name}' not found in the generated code"
            }
        
        # Get the function from the namespace
        function = namespace[agent_function_name]
        
        # Execute the function on the data
        data_processed = pre_processing(data)
        result = function(data_processed)
        result_processed = post_processing(result)
        
        # Store the result in the state
        return {
            result_key: result_processed,
            error_key: ""
        }
        
    except Exception as e:
        # Capture the error message and traceback
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"{error_message_prefix}{str(e)}\n\n{error_traceback}"
        
        return {
            error_key: error_message
        }

def node_func_human_review(
    state: Dict[str, Any],
    prompt_text: str,
    yes_goto: str,
    no_goto: str,
    user_instructions_key: str = "user_instructions",
    recommended_steps_key: str = "recommended_steps",
    code_snippet_key: Optional[str] = None,
) -> Command:
    """
    Enable human review of agent recommendations.
    
    Parameters
    ----------
    state : Dict[str, Any]
        The current state of the agent.
    prompt_text : str
        The text to prompt the human with.
    yes_goto : str
        Where to go if the human says "yes".
    no_goto : str
        Where to go if the human says anything other than "yes".
    user_instructions_key : str, optional
        The key containing user instructions, by default "user_instructions"
    recommended_steps_key : str, optional
        The key containing recommended steps, by default "recommended_steps"
    code_snippet_key : Optional[str], optional
        The key containing code to show, by default None
        
    Returns
    -------
    Command
        The command to execute next.
    """
    print(f"    * HUMAN IN THE LOOP")
    
    # Get the steps and instructions from the state
    steps = state.get(recommended_steps_key, "")
    
    # Format the prompt
    human_prompt = prompt_text.format(steps=steps)
    
    if code_snippet_key:
        code_snippet = state.get(code_snippet_key, "")
        human_prompt += f"\n\nCode:\n```python\n{code_snippet}\n```"
    
    # Create the human message
    message_content = human_prompt
    message = HumanMessage(content=message_content)
    
    # Return the next node based on the human's response
    def decide_next_node(state: Dict[str, Any], human_response: str) -> str:
        if human_response.strip().lower() == "yes":
            return yes_goto
        
        # If the response is not "yes", update the user instructions
        updated_instructions = state.get(user_instructions_key, "")
        if updated_instructions:
            updated_instructions += "\n\n"
        updated_instructions += human_response
        
        state[user_instructions_key] = updated_instructions
        
        return no_goto
    
    # Return the command
    return Command(
        name="human_review", 
        messages=[message],
        next_node=decide_next_node
    )

def node_func_fix_agent_code(
    state: Dict[str, Any],
    code_snippet_key: str,
    error_key: str,
    llm: Any,
    prompt_template: str,
    agent_name: str,
    log: bool = False,
    file_path: str = None,
    function_name: str = "ai_function"
) -> Dict[str, Any]:
    """
    Fix agent code based on error messages.
    
    Parameters
    ----------
    state : Dict[str, Any]
        The current state of the agent.
    code_snippet_key : str
        The key containing the code to fix.
    error_key : str
        The key containing the error message.
    llm : Any
        The language model to use for code fixing.
    prompt_template : str
        The prompt template for the language model.
    agent_name : str
        The name of the agent.
    log : bool, optional
        Whether to log the fixed code, by default False
    file_path : str, optional
        The path to log the code to, by default None
    function_name : str, optional
        The name of the function, by default "ai_function"
        
    Returns
    -------
    Dict[str, Any]
        The updated state with fixed code.
    """
    from ai_data_science.utils.regex import relocate_imports_inside_function, add_comments_to_top
    from ai_data_science.utils.logging import log_ai_function
    from ai_data_science.parsers.parsers import PythonOutputParser
    
    print(f"    * FIX AGENT CODE")
    
    # Get the code and error from the state
    code_snippet = state.get(code_snippet_key)
    error = state.get(error_key)
    
    # Create the prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["code_snippet", "error", "function_name"]
    )
    
    # Create the chain
    fix_chain = prompt | llm | PythonOutputParser()
    
    # Fix the code
    fixed_code = fix_chain.invoke({
        "code_snippet": code_snippet,
        "error": error,
        "function_name": function_name
    })
    
    # Process the fixed code
    fixed_code = relocate_imports_inside_function(fixed_code)
    fixed_code = add_comments_to_top(fixed_code, agent_name=agent_name)
    
    # Log the fixed code if requested
    if log and file_path:
        log_ai_function(
            response=fixed_code,
            file_name=file_path,
            log=log,
            log_path=None,
            overwrite=True
        )
    
    # Update the state
    return {
        code_snippet_key: fixed_code
    }

def node_func_report_agent_outputs(
    state: Dict[str, Any],
    keys_to_include: List[str],
    result_key: str = "messages",
    role: str = "agent",
    custom_title: str = None
) -> Dict[str, Any]:
    """
    Create a report of agent outputs.
    
    Parameters
    ----------
    state : Dict[str, Any]
        The current state of the agent.
    keys_to_include : List[str]
        The keys to include in the report.
    result_key : str, optional
        The key to store the result in, by default "messages"
    role : str, optional
        The role of the agent, by default "agent"
    custom_title : str, optional
        A custom title for the report, by default None
        
    Returns
    -------
    Dict[str, Any]
        The updated state with the report.
    """
    print(f"    * REPORT AGENT OUTPUTS")
    
    # Initialize the report
    report = {}
    
    # Add each requested key to the report
    for key in keys_to_include:
        if key in state:
            report[key] = state[key]
    
    # Convert the report to JSON
    report_json = json.dumps(report, indent=2)
    
    # Create the message
    title = custom_title or f"{role.capitalize()} Outputs"
    message = AIMessage(content=f"{title}:\n\n{report_json}")
    
    # Get existing messages or initialize empty list
    messages = state.get(result_key, [])
    
    # Add the new message
    updated_messages = messages + [message]
    
    # Update the state
    return {
        result_key: updated_messages
    }

def create_coding_agent_graph(
    GraphState: Type,
    node_functions: Dict[str, Callable],
    recommended_steps_node_name: str,
    create_code_node_name: str,
    execute_code_node_name: str,
    fix_code_node_name: str,
    explain_code_node_name: str,
    error_key: str,
    human_in_the_loop: bool = False,
    human_review_node_name: str = None,
    checkpointer: Checkpointer = None,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = False,
    agent_name: str = "agent"
):
    """
    Create a coding agent graph with the specified nodes.
    
    Parameters
    ----------
    GraphState : Type
        The type definition for the graph state.
    node_functions : Dict[str, Callable]
        Dictionary mapping node names to node functions.
    recommended_steps_node_name : str
        The name of the node for recommending steps.
    create_code_node_name : str
        The name of the node for creating code.
    execute_code_node_name : str
        The name of the node for executing code.
    fix_code_node_name : str
        The name of the node for fixing code.
    explain_code_node_name : str
        The name of the node for explaining code.
    error_key : str
        The key in the state containing error messages.
    human_in_the_loop : bool, optional
        Whether to use human in the loop, by default False
    human_review_node_name : str, optional
        The name of the node for human review, by default None
    checkpointer : Checkpointer, optional
        Checkpointer for the graph, by default None
    bypass_recommended_steps : bool, optional
        Whether to bypass the recommended steps node, by default False
    bypass_explain_code : bool, optional
        Whether to bypass the explain code node, by default False
    agent_name : str, optional
        The name of the agent, by default "agent"
        
    Returns
    -------
    StateGraph
        The compiled state graph.
    """
    
    # Create the workflow graph
    workflow = StateGraph(GraphState)
    
    # Configure checkpointer
    if checkpointer is None and human_in_the_loop:
        checkpointer = MemorySaver()
    
    # Add nodes to the graph
    for node_name, node_func in node_functions.items():
        workflow.add_node(node_name, node_func)
    
    # Set up the edges for the graph
    if bypass_recommended_steps:
        # Skip the recommend steps node
        workflow.set_entry_point(create_code_node_name)
    else:
        # Start with recommending steps
        workflow.set_entry_point(recommended_steps_node_name)
        
        # Connect recommended steps to either human review or create code
        if human_in_the_loop and human_review_node_name:
            workflow.add_edge(recommended_steps_node_name, human_review_node_name)
        else:
            workflow.add_edge(recommended_steps_node_name, create_code_node_name)
    
    # Connect human review to create code
    if human_in_the_loop and human_review_node_name:
        # This is handled by the Command returned by the human review node
        pass
    
    # Connect create code to execute code
    workflow.add_edge(create_code_node_name, execute_code_node_name)
    
    # Add conditional routing based on errors
    def has_error(state):
        return bool(state.get(error_key))
    
    def no_error(state):
        return not bool(state.get(error_key))
    
    def check_max_retries(state):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        return retry_count >= max_retries
    
    def increment_retry_count(state):
        return {"retry_count": state.get("retry_count", 0) + 1}
    
    # Check for errors after executing code
    workflow.add_conditional_edges(
        execute_code_node_name,
        has_error,
        {
            True: fix_code_node_name,
            False: explain_code_node_name if not bypass_explain_code else END
        }
    )
    
    # Check max retries after fixing code
    workflow.add_conditional_edges(
        fix_code_node_name,
        check_max_retries,
        {
            True: explain_code_node_name if not bypass_explain_code else END,
            False: execute_code_node_name
        }
    )
    
    # Update retry count before executing code again
    workflow.add_edge(fix_code_node_name, increment_retry_count, execute_code_node_name)
    
    # Compile the graph
    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    return app 