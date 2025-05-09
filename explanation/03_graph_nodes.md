# Understanding the Graph Nodes

In this section, we'll take a deeper look at each node in the data cleaning agent's LangGraph workflow. Each node represents a specific step in the data cleaning process and is implemented as a function that takes a state object as input and returns a dictionary with updates to apply to the state.

## The Graph Structure

The data cleaning agent's workflow is implemented as a directed graph, where:

- Nodes represent individual steps in the process
- Edges represent transitions between steps
- Conditional logic determines which path to take based on the current state

Here's a visualization of the complete workflow:

```
Entry Point → Recommend Cleaning Steps → [Human Review] → Create Data Cleaner Code → 
Execute Data Cleaner Code → [if error] → Fix Data Cleaner Code → Execute Data Cleaner Code → 
[if no error] → Report Agent Outputs → End
```

The nodes in square brackets (`[Human Review]` and `[if error]`) are conditional nodes that may be skipped depending on the configuration or the results of previous steps.

## Detailed Node Descriptions

Let's examine each node in detail:

### 1. Recommend Cleaning Steps

**Purpose**: This node analyzes the input data and generates a list of recommended data cleaning steps.

**Input**:
- Raw data (as a dictionary)
- User instructions (optional)

**Process**:
1. Convert the raw data dictionary to a pandas DataFrame
2. Generate a summary of the DataFrame using `get_dataframe_summary`
3. Create a PromptTemplate with instructions for the language model
4. Invoke the language model with the user instructions, any previously recommended steps, and the dataset summary
5. Format the model's response as recommended cleaning steps

**Output**:
- Recommended cleaning steps (as a formatted string)
- Dataset summary (as a string)

**Code Analysis**:

The core of this node is the prompt to the language model, which asks it to recommend data cleaning steps based on the data and user instructions:

```python
recommend_steps_prompt = PromptTemplate(
    template="""
    You are a Data Cleaning Expert. Given the following information about the data, 
    recommend a series of numbered steps to take to clean and preprocess it. 
    The steps should be tailored to the data characteristics and should be helpful 
    for a data cleaning agent that will be implemented.
    
    General Steps:
    Things that should be considered in the data cleaning steps:
    
    * Removing columns if more than 40 percent of the data is missing
    * Imputing missing values with the mean of the column if the column is numeric
    * Imputing missing values with the mode of the column if the column is categorical
    * Converting columns to the correct data type
    * Removing duplicate rows
    * Removing rows with missing values
    * Removing rows with extreme outliers (3X the interquartile range)
    
    Custom Steps:
    * Analyze the data to determine if any additional data cleaning steps are needed.
    * Recommend steps that are specific to the data provided. Include why these steps are necessary or beneficial.
    * If no additional steps are needed, simply state that no additional steps are required.
    
    IMPORTANT:
    Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
    
    User instructions:
    {user_instructions}

    Previously Recommended Steps (if any):
    {recommended_steps}

    Below are summaries of all datasets provided:
    {all_datasets_summary}

    Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.
    
    Avoid these:
    1. Do not include steps to save files.
    2. Do not include unrelated user instructions that are not related to the data cleaning.
    """,
    input_variables=["user_instructions", "recommended_steps", "all_datasets_summary"]
)
```

The prompt provides detailed guidance to the language model on what kinds of cleaning steps to recommend, how to consider the user's instructions, and how to present the recommendations.

### 2. Human Review (Optional)

**Purpose**: This node allows a human to review and modify the recommended cleaning steps.

**Input**:
- Recommended cleaning steps
- User instructions

**Process**:
1. Present the recommended steps to the human
2. If the human says "yes", continue to the next step
3. If the human says anything else, treat the response as additional instructions and go back to the recommendation step

**Output**:
- A Command object that specifies the next node in the workflow
- Updated user instructions (if the human provided modifications)

**Code Analysis**:

This node uses the `node_func_human_review` helper function from the templates module:

```python
prompt_text_human_review = "Are the following data cleaning instructions correct? (Answer 'yes' or provide modifications)\n{steps}"

def human_review(state: GraphState) -> Command[Literal["recommend_cleaning_steps", "explain_data_cleaner_code"]]:
    return node_func_human_review(
        state=state,
        prompt_text=prompt_text_human_review,
        yes_goto= 'explain_data_cleaner_code',
        no_goto="recommend_cleaning_steps",
        user_instructions_key="user_instructions",
        recommended_steps_key="recommended_steps",
        code_snippet_key="data_cleaner_function",
    )
```

The `Command` return type and `Literal` type hint are used to specify that this node can transition to either the `recommend_cleaning_steps` node or the `explain_data_cleaner_code` node, depending on the human's response.

### 3. Create Data Cleaner Code

**Purpose**: This node generates Python code to implement the recommended data cleaning steps.

**Input**:
- Recommended cleaning steps
- Dataset summary
- Function name

**Process**:
1. Create a PromptTemplate with instructions for the language model
2. Create a "data cleaning agent" by piping the prompt template to the language model and then to a PythonOutputParser
3. Invoke the data cleaning agent with the recommended steps, dataset summary, and function name
4. Process the response using helper functions to relocate imports inside the function and add comments
5. Log the function to a file if logging is enabled

**Output**:
- Generated data cleaning function (as a string)
- Function path (if logged)
- Function name
- Dataset summary

**Code Analysis**:

The prompt to the language model asks it to generate a Python function that implements the recommended cleaning steps:

```python
data_cleaning_prompt = PromptTemplate(
    template="""
    You are a Data Cleaning Agent. Your job is to create a {function_name}() function that can be run on the data provided using the following recommended steps.

    Recommended Steps:
    {recommended_steps}

    You can use Pandas, Numpy, and Scikit Learn libraries to clean the data.

    Below are summaries of all datasets provided. Use this information about the data to help determine how to clean the data:

    {all_datasets_summary}

    Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.

    Return code to provide the data cleaning function:

    def {function_name}(data_raw):
        import pandas as pd
        import numpy as np
        ...
        return data_cleaned

    Best Practices and Error Preventions:

    Always ensure that when assigning the output of fit_transform() from SimpleImputer to a Pandas DataFrame column, you call .ravel() or flatten the array, because fit_transform() returns a 2D array while a DataFrame column is 1D.
    
    """,
    input_variables=["recommended_steps", "all_datasets_summary", "function_name"]
)
```

The `PythonOutputParser` is used to extract clean Python code from the language model's response, which might include markdown formatting:

```python
data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()
```

### 4. Execute Data Cleaner Code

**Purpose**: This node executes the generated data cleaning function on the raw data.

**Input**:
- Raw data
- Generated data cleaning function
- Function name

**Process**:
1. Convert the raw data dictionary to a pandas DataFrame
2. Execute the generated function on the DataFrame
3. Convert the result back to a dictionary
4. Capture any errors that occur

**Output**:
- Cleaned data (if successful)
- Error message (if an error occurred)

**Code Analysis**:

This node uses the `node_func_execute_agent_code_on_data` helper function from the templates module:

```python
def execute_data_cleaner_code(state):
    return node_func_execute_agent_code_on_data(
        state=state,
        data_key="data_raw",
        result_key="data_cleaned",
        error_key="data_cleaner_error",
        code_snippet_key="data_cleaner_function",
        agent_function_name=state.get("data_cleaner_function_name"),
        pre_processing=lambda data: pd.DataFrame.from_dict(data),
        post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
        error_message_prefix="An error occurred during data cleaning: "
    )
```

The `pre_processing` and `post_processing` parameters are used to convert between the dictionary format used in the state and the DataFrame format expected by the generated function.

### 5. Fix Data Cleaner Code (Conditional)

**Purpose**: This node fixes the data cleaning function if an error occurred during execution.

**Input**:
- Generated data cleaning function
- Error message

**Process**:
1. Create a prompt with the broken code and error message
2. Invoke the language model with the prompt to get a fixed version of the code
3. Process the response and log it if logging is enabled

**Output**:
- Fixed data cleaning function

**Code Analysis**:

This node uses the `node_func_fix_agent_code` helper function from the templates module:

```python
def fix_data_cleaner_code(state: GraphState):
    data_cleaner_prompt = """
    You are a Data Cleaning Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is currently broken and needs to be fixed.
    
    Make sure to only return the function definition for {function_name}().
    
    Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
    
    This is the broken code (please fix): 
    {code_snippet}

    Last Known Error:
    {error}
    """

    return node_func_fix_agent_code(
        state=state,
        code_snippet_key="data_cleaner_function",
        error_key="data_cleaner_error",
        llm=llm,  
        prompt_template=data_cleaner_prompt,
        agent_name=AGENT_NAME,
        log=log,
        file_path=state.get("data_cleaner_function_path"),
        function_name=state.get("data_cleaner_function_name"),
    )
```

The prompt to the language model includes the broken code and the error message, asking it to fix the code.

### 6. Report Agent Outputs

**Purpose**: This node creates a report of the agent's outputs to be returned to the user.

**Input**:
- Recommended cleaning steps
- Generated data cleaning function
- Function path
- Function name
- Error message (if any)

**Process**:
1. Collect the specified keys from the state
2. Create a JSON report with those keys
3. Create a message with the report
4. Add the message to the messages list in the state

**Output**:
- A list of messages containing the agent's outputs

**Code Analysis**:

This node uses the `node_func_report_agent_outputs` helper function from the templates module:

```python
def report_agent_outputs(state: GraphState):
    return node_func_report_agent_outputs(
        state=state,
        keys_to_include=[
            "recommended_steps",
            "data_cleaner_function",
            "data_cleaner_function_path",
            "data_cleaner_function_name",
            "data_cleaner_error",
        ],
        result_key="messages",
        role=AGENT_NAME,
        custom_title="Data Cleaning Agent Outputs"
    )
```

This function creates a report with the specified keys from the state and adds it to the messages list.

## Graph Construction with create_coding_agent_graph

The `create_coding_agent_graph` function from the templates module is used to create the LangGraph state machine with the appropriate nodes and edges:

```python
app = create_coding_agent_graph(
    GraphState=GraphState,
    node_functions=node_functions,
    recommended_steps_node_name="recommend_cleaning_steps",
    create_code_node_name="create_data_cleaner_code",
    execute_code_node_name="execute_data_cleaner_code",
    fix_code_node_name="fix_data_cleaner_code",
    explain_code_node_name="report_agent_outputs", 
    error_key="data_cleaner_error",
    human_in_the_loop=human_in_the_loop,
    human_review_node_name="human_review",
    checkpointer=checkpointer,
    bypass_recommended_steps=bypass_recommended_steps,
    bypass_explain_code=bypass_explain_code,
    agent_name=AGENT_NAME,
)
```

This function creates a LangGraph state machine with the following structure:

1. If `bypass_recommended_steps` is `False` (default), the entry point is the `recommend_cleaning_steps` node.
2. If `human_in_the_loop` is `True`, the `recommend_cleaning_steps` node is connected to the `human_review` node, which can either continue to the `create_data_cleaner_code` node or go back to the `recommend_cleaning_steps` node.
3. Otherwise, the `recommend_cleaning_steps` node is connected directly to the `create_data_cleaner_code` node.
4. The `create_data_cleaner_code` node is connected to the `execute_data_cleaner_code` node.
5. After `execute_data_cleaner_code`, a conditional check is performed: if there's an error, the workflow continues to the `fix_data_cleaner_code` node, otherwise it continues to the `explain_code_node_name` node (which is actually the `report_agent_outputs` node in this case).
6. The `fix_data_cleaner_code` node is connected back to the `execute_data_cleaner_code` node to retry the execution.
7. The `report_agent_outputs` node is connected to the end of the workflow.

## State Management

The state is a shared dictionary that is passed from node to node as the workflow progresses. Each node can read from the state and return updates to be applied to the state. The state includes:

- **messages**: A list of messages to be returned to the user.
- **user_instructions**: The user's instructions for the data cleaning process.
- **recommended_steps**: The recommended cleaning steps generated by the agent.
- **data_raw**: The raw data as a dictionary.
- **data_cleaned**: The cleaned data as a dictionary.
- **all_datasets_summary**: A summary of all datasets provided.
- **data_cleaner_function**: The generated data cleaning function as a string.
- **data_cleaner_function_path**: The path to the log file where the function was saved.
- **data_cleaner_file_name**: The name of the log file.
- **data_cleaner_function_name**: The name of the generated function.
- **data_cleaner_error**: Any error that occurred during function execution.
- **max_retries**: The maximum number of times to retry if errors occur.
- **retry_count**: The current retry count.

## Error Handling and Retry Mechanism

The data cleaning agent includes a robust error handling and retry mechanism:

1. If an error occurs during the execution of the data cleaning function, the error is captured and stored in the `data_cleaner_error` field of the state.
2. The workflow then transitions to the `fix_data_cleaner_code` node, which uses the language model to fix the code.
3. The workflow then transitions back to the `execute_data_cleaner_code` node to retry the execution.
4. This process continues until either the execution succeeds or the maximum number of retries is reached.

The maximum number of retries is specified by the `max_retries` field in the state, which is passed to the agent's `invoke` method by the user.

## Human-in-the-Loop Capability

The data cleaning agent includes an optional human-in-the-loop capability that allows a human to review and modify the recommended cleaning steps before they are implemented:

1. After the `recommend_cleaning_steps` node generates a list of recommended steps, the workflow transitions to the `human_review` node if `human_in_the_loop` is `True`.
2. The `human_review` node presents the recommended steps to the human and asks if they are correct.
3. If the human says "yes", the workflow continues to the next step.
4. If the human says anything else, the workflow goes back to the `recommend_cleaning_steps` node with the human's response as additional user instructions.

This capability allows humans to provide feedback and guide the data cleaning process, ensuring that the agent performs the desired cleaning operations.
