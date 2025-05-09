# Understanding the DataCleaningAgent Class

The `DataCleaningAgent` class serves as a user-friendly wrapper around the complex LangGraph state machine that powers the data cleaning process. It provides a clean, object-oriented interface for initializing, configuring, and interacting with the data cleaning agent.

## Class Definition

```python
class DataCleaningAgent(BaseAgent):
    # Implementation details...
```

The `DataCleaningAgent` class inherits from the `BaseAgent` class, which provides common functionality for agent implementations. This inheritance allows the agent to reuse core functionality while adding data cleaning-specific features.

## Class Initialization

```python
def __init__(
    self, 
    model, 
    n_samples=30, 
    log=False, 
    log_path=None, 
    file_name="data_cleaner.py", 
    function_name="data_cleaner",
    overwrite=True, 
    human_in_the_loop=False, 
    bypass_recommended_steps=False, 
    bypass_explain_code=False,
    checkpointer: Checkpointer = None
):
    self._params = {
        "model": model,
        "n_samples": n_samples,
        "log": log,
        "log_path": log_path,
        "file_name": file_name,
        "function_name": function_name,
        "overwrite": overwrite,
        "human_in_the_loop": human_in_the_loop,
        "bypass_recommended_steps": bypass_recommended_steps,
        "bypass_explain_code": bypass_explain_code,
        "checkpointer": checkpointer
    }
    self._compiled_graph = self._make_compiled_graph()
    self.response = None
```

The initialization method takes several parameters:

- **model**: The language model (e.g., from OpenAI or Anthropic) that will power the agent.
- **n_samples**: The number of sample rows to use when summarizing the dataset for the model.
- **log**: Whether to log generated code to a file.
- **log_path**: The directory where log files should be stored.
- **file_name**: The name of the file to save the generated code to.
- **function_name**: The name of the Python function that will be generated.
- **overwrite**: Whether to overwrite existing log files.
- **human_in_the_loop**: Whether to enable human review of recommendations.
- **bypass_recommended_steps**: Whether to skip the recommendation generation step.
- **bypass_explain_code**: Whether to skip the code explanation step.
- **checkpointer**: An optional checkpointer for saving and loading the agent's state.

These parameters are stored in the `_params` dictionary for later use. The `_make_compiled_graph()` method is called to create the LangGraph state machine, and the `response` attribute is initialized to `None`.

## Making the Compiled Graph

```python
def _make_compiled_graph(self):
    """
    Create the compiled graph for the data cleaning agent. Running this method will reset the response to None.
    """
    self.response=None
    return make_data_cleaning_agent(**self._params)
```

This method creates a new LangGraph state machine using the `make_data_cleaning_agent` function (which we'll discuss in detail later) and the parameters stored in `_params`. It also resets the `response` attribute to `None`.

It's important to understand that this method only *defines* the structure of the data cleaning workflow - it doesn't actually execute any cleaning operations on data. Think of it as creating a blueprint or recipe that specifies:

1. What steps will be performed (recommending cleaning steps, creating code, executing code, etc.)
2. In what order these steps will run
3. How decisions will be made between steps

This separation between defining the workflow and executing it allows the agent to be reconfigured with different parameters without carrying over any state from previous runs. The actual execution of the data cleaning pipeline happens later when either `invoke_agent()` or `ainvoke_agent()` is called with a dataset, at which point the data will flow through this predefined workflow and the results will be stored in the `response` attribute.

## Invoking the Agent

The class provides two methods for invoking the agent:

```python
async def ainvoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
    """
    Asynchronously invokes the agent. The response is stored in the response attribute.
    """
    response = await self._compiled_graph.ainvoke({
        "user_instructions": user_instructions,
        "data_raw": data_raw.to_dict(),
        "max_retries": max_retries,
        "retry_count": retry_count,
    }, **kwargs)
    self.response = response
    return None
```

```python
def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
    """
    Invokes the agent. The response is stored in the response attribute.
    """
    response = self._compiled_graph.invoke({
        "user_instructions": user_instructions,
        "data_raw": data_raw.to_dict(),
        "max_retries": max_retries,
        "retry_count": retry_count,
    },**kwargs)
    self.response = response
    return None
```

The `ainvoke_agent` method provides an asynchronous interface, while `invoke_agent` provides a synchronous interface. Both methods take the same parameters:

- **data_raw**: The pandas DataFrame to be cleaned.
- **user_instructions**: Optional instructions for the cleaning process.
- **max_retries**: The maximum number of times to retry if errors occur.
- **retry_count**: The current retry count.
- **kwargs**: Additional keyword arguments to pass to the LangGraph invoke method.

These methods convert the DataFrame to a dictionary (bcz langgraph needs serialisable data and dict is while dataframe isn't) and pass it to the LangGraph state 
machine along with the other parameters. The response from the state machine is stored in 
the `response` attribute.

For example, if `data_raw` is a DataFrame with columns `A` and `B`, the methods convert it to a dictionary like `{"A": [...], "B": [...]}`. This dictionary, along with other parameters like `user_instructions`, `max_retries`, and `retry_count`, is then passed to the LangGraph state machine for processing. The state machine's response is stored in the `response` attribute of the `DataCleaningAgent` instance.

## Getter Methods

The `DataCleaningAgent` class provides several getter methods for retrieving information from the agent's response:

### Getting the Workflow Summary

```python
def get_workflow_summary(self, markdown=False):
    """
    Retrieves the agent's workflow summary, if logging is enabled.
    """
    if self.response and self.response.get("messages"):
        summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
        if markdown:
            return Markdown(summary)
        else:
            return summary
```

This method retrieves a summary of the workflow from the last message in the `messages` list. If the `markdown` parameter is `True`, the summary is returned as a Markdown object for display in Jupyter notebooks.

### Getting the Log Summary

```python
def get_log_summary(self, markdown=False):
    """
    Logs a summary of the agent's operations, if logging is enabled.
    """
    if self.response:
        if self.response.get('data_cleaner_function_path'):
            log_details = f"""
## Data Cleaning Agent Log Summary:

Function Path: {self.response.get('data_cleaner_function_path')}

Function Name: {self.response.get('data_cleaner_function_name')}
            """
            if markdown:
                return Markdown(log_details) 
            else:
                return log_details
```

This method returns a summary of the logging operations, including the path and name of the generated function file.

### Getting the Cleaned Data

```python
def get_data_cleaned(self):
    """
    Retrieves the cleaned data stored after running invoke_agent or clean_data methods.
    """
    if self.response:
        return pd.DataFrame(self.response.get("data_cleaned"))
```

This method retrieves the cleaned data from the agent's response as a pandas DataFrame.

### Getting the Raw Data

```python
def get_data_raw(self):
    """
    Retrieves the raw data.
    """
    if self.response:
        return pd.DataFrame(self.response.get("data_raw"))
```

This method retrieves the raw data from the agent's response as a pandas DataFrame.

### Getting the Data Cleaner Function

```python
def get_data_cleaner_function(self, markdown=False):
    """
    Retrieves the agent's pipeline function.
    """
    if self.response:
        if markdown:
            return Markdown(f"```python\n{self.response.get('data_cleaner_function')}\n```")
        else:
            return self.response.get("data_cleaner_function")
```

This method retrieves the generated data cleaning function as a string. If the `markdown` parameter is `True`, the function is returned as a Markdown code block.

### Getting the Recommended Cleaning Steps

```python
def get_recommended_cleaning_steps(self, markdown=False):
    """
    Retrieves the agent's recommended cleaning steps
    """
    if self.response:
        if markdown:
            return Markdown(self.response.get('recommended_steps'))
        else:
            return self.response.get('recommended_steps')
```

This method retrieves the recommended cleaning steps from the agent's response. If the `markdown` parameter is `True`, the steps are returned as a Markdown object.

## Summary

The `DataCleaningAgent` class provides a user-friendly interface for working with the data cleaning agent. It handles the initialization of the LangGraph state machine, provides methods for invoking the agent, and offers getter methods for retrieving information from the agent's response.

This class follows the object-oriented programming principle of encapsulation by hiding the complexity of the LangGraph state machine behind a clean interface. Users of the class don't need to understand the details of LangGraph to use the agent effectively.
