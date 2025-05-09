# Understanding the Supporting Templates

The data cleaning agent relies heavily on template functions and classes from the `agent_templates.py` module. These templates provide reusable components for building agent workflows using LangGraph. In this section, we'll explore the key templates used by the data cleaning agent.

## BaseAgent Class

The `BaseAgent` class is an abstract base class that provides common functionality for all agent implementations. Here's a simplified version of the class:

```python
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
        """
        self._params.update(kwargs)
        self._compiled_graph = self._make_compiled_graph()
    
    async def ainvoke_agent(self, **kwargs):
        """
        Asynchronously invoke the agent.
        """
        self.response = await self._compiled_graph.ainvoke(kwargs)
        return None
    
    def invoke_agent(self, **kwargs):
        """
        Synchronously invoke the agent.
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
        # Implementation details...
```

Key features of the `BaseAgent` class:

1. **Abstract Base Class**: The `BaseAgent` class is an abstract base class (ABC) that defines a common interface for all agent implementations.

2. **Parameter Management**: The class stores the agent's parameters in the `_params` dictionary, which can be updated using the `update_params` method.

3. **Graph Compilation**: The class delegates the creation of the compiled graph to the `_make_compiled_graph` method, which must be implemented by subclasses.

4. **Invocation Methods**: The class provides both synchronous (`invoke_agent`) and asynchronous (`ainvoke_agent`) methods for invoking the agent.

5. **Response Handling**: The class stores the agent's response in the `response` attribute and provides methods for retrieving and examining it.

The `DataCleaningAgent` class inherits from `BaseAgent` and implements the `_make_compiled_graph` method to create a compiled graph for data cleaning.

## Node Function Templates

The `agent_templates.py` module provides several template functions for creating nodes in the LangGraph workflow. These templates encapsulate common functionality and make it easier to create agents with similar workflows.

### node_func_execute_agent_code_on_data

This template function executes agent-generated code on data and handles any errors:

```python
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
```

This function:

1. Retrieves the data and code from the state.
2. Creates an empty namespace for executing the code.
3. Executes the code to define the agent function.
4. Calls the function on the preprocessed data.
5. Returns the processed result or an error message.

The function is designed to be generic and can be used for any agent that generates and executes code on data.

### node_func_human_review

This template function enables human review of agent-generated content:

```python
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
```

This function:

1. Retrieves the recommended steps from the state.
2. Formats a prompt for the human.
3. Creates a human message with the prompt.
4. Defines a function to decide the next node based on the human's response.
5. Returns a Command object with the message and next node function.

The `Command` object is a special return type in LangGraph that allows a node to:
- Send a message to the user.
- Receive a response from the user.
- Decide which node to transition to based on the response.

### node_func_fix_agent_code

This template function fixes broken agent code using the language model:

```python
def node_func_fix_agent_code(
    state: Dict[str, Any],
    code_snippet_key: str,
    error_key: str,
    llm: Any,
    prompt_template: str,
    agent_name: str = "agent",
    log: bool = False,
    file_path: str = None,
    function_name: str = "function",
) -> Dict[str, Any]:
    """
    Fix broken agent code using the language model.
    """
    print(f"    * FIX AGENT CODE")
    
    # Get the code and error from the state
    code_snippet = state.get(code_snippet_key, "")
    error = state.get(error_key, "")
    
    if not code_snippet or not error:
        return {
            error_key: "Cannot fix code: Missing code snippet or error message in state"
        }
    
    try:
        # Format the prompt
        prompt = prompt_template.format(
            code_snippet=code_snippet,
            error=error,
            function_name=function_name
        )
        
        # Invoke the language model
        response = llm.invoke(prompt)
        
        # Extract the code from the response
        parser = PythonOutputParser()
        fixed_code = parser.parse(response.content)
        
        # Process the code
        fixed_code = relocate_imports_inside_function(fixed_code)
        fixed_code = add_comments_to_top(fixed_code, agent_name=agent_name)
        
        # Log the fixed code
        if log and file_path:
            log_ai_function(
                response=fixed_code,
                file_name=os.path.basename(file_path),
                log=log,
                log_path=os.path.dirname(file_path),
                overwrite=True
            )
        
        # Return the fixed code
        return {
            code_snippet_key: fixed_code
        }
    
    except Exception as e:
        return {
            error_key: f"Error while fixing code: {str(e)}"
        }
```

This function:

1. Retrieves the broken code and error from the state.
2. Formats a prompt for the language model.
3. Invokes the language model to get a fixed version of the code.
4. Processes the response and logs it if logging is enabled.
5. Returns the fixed code.

### node_func_report_agent_outputs

This template function creates a report of the agent's outputs:

```python
def node_func_report_agent_outputs(
    state: Dict[str, Any],
    keys_to_include: List[str],
    result_key: str = "messages",
    role: str = "agent",
    custom_title: str = None
) -> Dict[str, Any]:
    """
    Create a report of agent outputs.
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
```

This function:

1. Collects the specified keys from the state.
2. Creates a JSON report with those keys.
3. Creates a message with the report.
4. Adds the message to the messages list in the state.

## Graph Construction Template

The `create_coding_agent_graph` function is a template for creating a LangGraph state machine for a coding agent:

```python
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
    
    # After fixing code, increment retry count and check if max retries reached
    workflow.add_node("increment_retry_count", increment_retry_count)
    workflow.add_edge(fix_code_node_name, "increment_retry_count")
    
    workflow.add_conditional_edges(
        "increment_retry_count",
        check_max_retries,
        {
            True: explain_code_node_name if not bypass_explain_code else END,
            False: execute_code_node_name
        }
    )
    
    # Connect explain code to end
    if not bypass_explain_code:
        workflow.add_edge(explain_code_node_name, END)
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)
```

This function:

1. Creates a `StateGraph` with the specified `GraphState` type.
2. Configures a checkpointer if human-in-the-loop is enabled.
3. Adds the specified nodes to the graph.
4. Sets up the edges between nodes, including conditional edges based on error conditions.
5. Compiles the graph and returns it.

The function creates a graph with the following structure:

- Entry point → Recommend steps (optional) → Human review (optional) → Create code → Execute code → [if error] → Fix code → Increment retry count → [if max retries not reached] → Execute code → [if no error] → Explain code (optional) → End

This template makes it easy to create different types of coding agents with similar workflows but different node functions.

## How These Templates Are Used in the Data Cleaning Agent

The data cleaning agent uses these templates as follows:

1. The `DataCleaningAgent` class inherits from `BaseAgent` and implements the `_make_compiled_graph` method to create a compiled graph for data cleaning using the `make_data_cleaning_agent` function.

2. The `make_data_cleaning_agent` function uses `node_func_execute_agent_code_on_data` to execute the generated data cleaning function on the raw data.

3. It uses `node_func_human_review` to enable human review of the recommended cleaning steps.

4. It uses `node_func_fix_agent_code` to fix the data cleaning function if an error occurs during execution.

5. It uses `node_func_report_agent_outputs` to create a report of the agent's outputs.

6. Finally, it uses `create_coding_agent_graph` to create a LangGraph state machine with the appropriate nodes and edges for the data cleaning workflow.

These templates enable the data cleaning agent to focus on the specifics of data cleaning while reusing common functionality for graph construction, code execution, human review, and error handling.
