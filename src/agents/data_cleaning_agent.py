"""
Data Cleaning Agent for AI Data Science.

This module provides a specialized agent for cleaning datasets based on recommended
best practices or user-defined instructions.
"""

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import os
import json
import pandas as pd
from IPython.display import Markdown
import numpy as np

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

from src.templates import(
    node_func_execute_agent_code_on_data, 
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from src.parsers.parsers import PythonOutputParser
from src.utils.regex import (
    relocate_imports_inside_function, 
    add_comments_to_top, 
    format_agent_name, 
    format_recommended_steps, 
    get_generic_summary,
)
from src.tools.dataframe import get_dataframe_summary
from src.utils.logging import log_ai_function

# Setup
AGENT_NAME = "data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


# Class
class DataCleaningAgent(BaseAgent):
    """
    Creates a data cleaning agent that can process datasets based on user-defined instructions or default cleaning steps. 
    The agent generates a Python function to clean the dataset, performs the cleaning, and logs the process, including code 
    and errors. It is designed to facilitate reproducible and customizable data cleaning workflows.

    The agent performs the following default cleaning steps unless instructed otherwise:

    - Removing columns with more than 40% missing values.
    - Imputing missing values with the mean for numeric columns.
    - Imputing missing values with the mode for categorical columns.
    - Converting columns to appropriate data types.
    - Removing duplicate rows.
    - Removing rows with missing values.
    - Removing rows with extreme outliers (values 3x the interquartile range).

    User instructions can modify, add, or remove any of these steps to tailor the cleaning process.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data cleaning function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30. Reducing this number can help 
        avoid exceeding the model's token limits.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_cleaner.py".
    function_name : str, optional
        Name of the generated data cleaning function. Defaults to "data_cleaner".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data cleaning instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended cleaning steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Cleans the provided dataset asynchronously based on user instructions.
    invoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Cleans the provided dataset synchronously based on user instructions.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_state_keys()
        Returns a list of keys from the state graph response.
    get_state_properties()
        Returns detailed properties of the state graph response.
    get_data_cleaned()
        Retrieves the cleaned dataset as a pandas DataFrame.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame.
    get_data_cleaner_function()
        Retrieves the generated Python function used for cleaning the data.
    get_recommended_cleaning_steps()
        Retrieves the agent's recommended cleaning steps.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science.agents import DataCleaningAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_cleaning_agent = DataCleaningAgent(
        model=llm, n_samples=50, log=True, log_path="logs", human_in_the_loop=True
    )

    df = pd.read_csv("data/sample_data.csv")

    data_cleaning_agent.invoke_agent(
        data_raw=df,
        user_instructions="Don't remove outliers when cleaning the data.",
        max_retries=3,
        retry_count=0
    )

    cleaned_data = data_cleaning_agent.get_data_cleaned()
    
    response = data_cleaning_agent.response
    ```
    
    Returns
    --------
    DataCleaningAgent : langchain.graphs.CompiledStateGraph 
        A data cleaning agent implemented as a compiled state graph. 
    """
    
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

    def _make_compiled_graph(self):
        """
        Create the compiled graph for the data cleaning agent. Running this method will reset the response to None.
        """
        self.response=None
        return make_data_cleaning_agent(**self._params)

    async def ainvoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
        """
        Asynchronously invokes the agent. The response is stored in the response attribute.

        Parameters:
        ----------
            data_raw (pd.DataFrame): 
                The raw dataset to be cleaned.
            user_instructions (str): 
                Instructions for data cleaning agent.
            max_retries (int): 
                Maximum retry attempts for cleaning.
            retry_count (int): 
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to ainvoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None
    
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
        """
        Invokes the agent. The response is stored in the response attribute.

        Parameters:
        ----------
            data_raw (pd.DataFrame): 
                The raw dataset to be cleaned.
            user_instructions (str): 
                Instructions for data cleaning agent.
            max_retries (int): 
                Maximum retry attempts for cleaning.
            retry_count (int): 
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to invoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        },**kwargs)
        self.response = response
        return None

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
    
    def get_data_cleaned(self):
        """
        Retrieves the cleaned data stored after running invoke_agent or clean_data methods.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame, or None if no data is available.
        
        Raises:
            ValueError: If there are issues converting the data to a DataFrame.
        """
        if not self.response:
            return None
            
        try:
            data_cleaned = self.response.get("data_cleaned")
            if data_cleaned is None:
                return None
                
            # Handle different possible formats
            if isinstance(data_cleaned, pd.DataFrame):
                return data_cleaned
            elif isinstance(data_cleaned, dict):
                # Check if it's a dict of Series or dict of lists/values
                if data_cleaned and all(isinstance(v, (pd.Series, list, tuple, np.ndarray)) for v in data_cleaned.values()):
                    df = pd.DataFrame(data_cleaned)
                    # Ensure we don't have duplicate columns from one-hot encoding or other transformations
                    if df.shape[1] > 3 * len(self.response.get("data_raw", {}).keys()):
                        # This likely means we have a format issue - try to convert from records
                        try:
                            # Try reconstructing from original columns and limiting to reasonable size
                            orig_cols = set(self.response.get("data_raw", {}).keys())
                            if len(orig_cols) > 0:
                                # Filter columns to those that make sense
                                valid_cols = [c for c in df.columns if any(oc in str(c) for oc in orig_cols) or not any(str(i) in str(c) for i in range(10))]
                                if len(valid_cols) > 0:
                                    df = df[valid_cols]
                        except:
                            pass
                    return df
                # Handle the case where it's a dict of dicts (records format)
                elif data_cleaned and all(isinstance(v, dict) for v in data_cleaned.values()):
                    return pd.DataFrame.from_records(list(data_cleaned.values()))
            
            # If we got here, the format is unexpected - try more conversion approaches
            try:
                # Try parsing as records
                return pd.DataFrame.from_records(data_cleaned)
            except:
                try:
                    # Last resort - convert to string and back to evaluate structure
                    import json
                    json_str = json.dumps(data_cleaned)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        return pd.DataFrame(parsed)
                    return pd.DataFrame(parsed)
                except:
                    raise ValueError(f"Could not convert cleaned data to DataFrame. Data format: {type(data_cleaned)}")
        except Exception as e:
            import traceback
            print(f"Error in get_data_cleaned: {str(e)}")
            print(traceback.format_exc())
            # Return a simple DataFrame with the error message for debugging
            return pd.DataFrame({"error": [str(e)], "data_type": [str(type(self.response.get("data_cleaned")))], 
                                "response_keys": [list(self.response.keys())]})
        
    def get_data_raw(self):
        """
        Retrieves the raw data.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_data_cleaner_function(self, markdown=False):
        """
        Retrieves the agent's pipeline function.
        """
        if self.response:
            if markdown:
                return Markdown(f"```python\n{self.response.get('data_cleaner_function')}\n```")
            else:
                return self.response.get("data_cleaner_function")
            
    def get_recommended_cleaning_steps(self, markdown=False):
        """
        Retrieves the agent's recommended cleaning steps
        """
        if self.response:
            if markdown:
                return Markdown(self.response.get('recommended_steps'))
            else:
                return self.response.get('recommended_steps')



# Agent

def make_data_cleaning_agent(
    model, 
    n_samples = 30, 
    log=False, 
    log_path=None, 
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    overwrite = True, 
    human_in_the_loop=False, 
    bypass_recommended_steps=False, 
    bypass_explain_code=False,
    checkpointer: Checkpointer = None
):
    """
    Creates a data cleaning agent that can be run on a dataset. The agent can be used to clean a dataset in a variety of
    ways, such as removing columns with more than 40% missing values, imputing missing
    values with the mean of the column if the column is numeric, or imputing missing
    values with the mode of the column if the column is categorical.
    The agent takes in a dataset and some user instructions, and outputs a python
    function that can be used to clean the dataset. The agent also logs the code
    generated and any errors that occur.

    The agent is instructed to to perform the following data cleaning steps:

    - Removing columns if more than 40 percent of the data is missing
    - Imputing missing values with the mean of the column if the column is numeric
    - Imputing missing values with the mode of the column if the column is categorical
    - Converting columns to the correct data type
    - Removing duplicate rows
    - Removing rows with missing values
    - Removing rows with extreme outliers (3X the interquartile range)
    - User instructions can modify, add, or remove any of the above steps

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    n_samples : int, optional
        The number of samples to use when summarizing the dataset. Defaults to 30.
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to
        "logs/".
    file_name : str, optional
        The name of the file to save the response to. Defaults to "data_cleaner.py".
    function_name : str, optional
        The name of the function that will be generated to clean the data. Defaults to "data_cleaner".
    overwrite : bool, optional
        Whether or not to overwrite the log file if it already exists. If False, a unique file name will be created. 
        Defaults to True.
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the data cleaning instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.
        
    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science.agents import DataCleaningAgent

    llm = ChatOpenAI(model = "gpt-4o-mini")

    data_cleaning_agent = DataCleaningAgent(model=llm)

    df = pd.read_csv("data/sample_data.csv")

    data_cleaning_agent.invoke_agent(
        data_raw=df,
        user_instructions="Don't remove outliers when cleaning the data.",
        max_retries=3, 
        retry_count=0
    )

    cleaned_df = data_cleaning_agent.get_data_cleaned()
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data cleaning agent as a state graph.
    """
    llm = model
    
    if human_in_the_loop:
        if checkpointer is None:
            print("Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver().")
            checkpointer = MemorySaver()
    
    # Human in th loop requires recommended steps
    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_cleaned: dict
        all_datasets_summary: str
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_file_name: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        max_retries: int
        retry_count: int

    
    def recommend_cleaning_steps(state: GraphState):
        """
        Recommend a series of data cleaning steps based on the input data. 
        These recommended steps will be appended to the user_instructions.
        """
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND CLEANING STEPS")

        # Prompt to get recommended steps from the LLM
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

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str
        }) 
        
        return {
            "recommended_steps": format_recommended_steps(recommended_steps.content.strip(), heading="# Recommended Data Cleaning Steps:"),
            "all_datasets_summary": all_datasets_summary_str
        }
    
    def create_data_cleaner_code(state: GraphState):
        
        print("    * CREATE DATA CLEANER CODE")
        
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))
            
            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)

            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)
            
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
        
        
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
                # Disable SettingWithCopyWarning temporarily for clean code
                pd.set_option('mode.chained_assignment', None)
                
                # Make a copy of the data to avoid warnings
                data = data_raw.copy()
                
                # Your cleaning code here
                
                # Re-enable warnings before returning
                pd.set_option('mode.chained_assignment', 'warn')
                
                return data

            Best Practices and Error Preventions:
            1. Always use data.loc[] for assignments rather than chained indexing to avoid SettingWithCopyWarning
            2. When replacing values, use data.loc[] syntax: data.loc[mask, 'column'] = value
            3. When instructed to replace missing values with mean/median, use explicit assignment rather than fillna
            4. Always make a copy of the dataframe before modifying it
            5. Always ensure that when assigning the output of fit_transform() from SimpleImputer to a Pandas DataFrame column, you call .ravel() or flatten the array, because fit_transform() returns a 2D array while a DataFrame column is 1D.
            6. If asked to replace missing values instead of removing rows, make sure to honor this request and don't drop rows with missing values.
            7. When removing outliers, be cautious not to remove too many rows. Consider using a generous threshold.
            """,
            input_variables=["recommended_steps", "all_datasets_summary", "function_name"]
        )

        data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()
        
        response = data_cleaning_agent.invoke({
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str,
            "function_name": function_name
        })
        
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)
        
        # For logging: store the code generated:
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )
   
        return {
            "data_cleaner_function" : response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_file_name": file_name_2,
            "data_cleaner_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str
        }
        
    # Human Review
        
    prompt_text_human_review = "Are the following data cleaning instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_cleaning_steps", "report_agent_outputs"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto='report_agent_outputs',
                no_goto="recommend_cleaning_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_cleaner_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_cleaning_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_cleaning_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_cleaner_function", 
            )
    
    def execute_data_cleaner_code(state):
        """Execute the data cleaner code on the raw data.
        
        This function handles the conversion of data between formats and captures any errors.
        """
        print("    * EXECUTE AGENT CODE")
        
        # Define a more robust post-processing function
        def robust_post_processing(result):
            """Safely convert the result to a dictionary format compatible with pandas."""
            try:
                # If result is already a DataFrame, convert to dict
                if isinstance(result, pd.DataFrame):
                    # Store original shape for debugging
                    orig_shape = result.shape
                    # Check if the result has an unreasonable number of columns
                    # (indicating potential one-hot encoding or other transformation issues)
                    data_raw = pd.DataFrame.from_dict(state.get("data_raw"))
                    if result.shape[1] > 3 * data_raw.shape[1]:
                        print(f"Warning: Cleaned data has {result.shape[1]} columns, which is much larger than the original {data_raw.shape[1]}.")
                        # Try to filter columns that make sense
                        try:
                            # Inspect column names to identify potential one-hot encoding
                            orig_cols = set(data_raw.columns)
                            valid_cols = [c for c in result.columns if any(oc in str(c) for oc in orig_cols) or not any(str(i) in str(c) for i in range(10))]
                            if len(valid_cols) > 0:
                                result = result[valid_cols]
                                print(f"Filtered to {len(valid_cols)} relevant columns")
                        except Exception as e:
                            print(f"Error filtering columns: {str(e)}")
                    
                    # Create a copy to avoid SettingWithCopyWarning
                    result_dict = result.copy().to_dict()
                    return result_dict
                    
                # If it's a dict, ensure it has the right structure
                elif isinstance(result, dict):
                    # Check if all values have the same length
                    if result and all(isinstance(v, (list, tuple, np.ndarray)) for v in result.values()):
                        # Check if all lists have the same length
                        lengths = [len(v) for v in result.values() if hasattr(v, '__len__')]
                        if lengths and all(l == lengths[0] for l in lengths):
                            return result
                    
                    # If dict of dicts (records format), convert to columnar format
                    if result and all(isinstance(v, dict) for v in result.values()):
                        columnar_dict = {}
                        for record_id, record in result.items():
                            for col, val in record.items():
                                if col not in columnar_dict:
                                    columnar_dict[col] = []
                                columnar_dict[col].append(val)
                        return columnar_dict
                    
                    # If it's just a dict but not in the right format, try a conversion approach
                    try:
                        df = pd.DataFrame.from_dict(result, orient='index')
                        return df.to_dict(orient='list')
                    except:
                        pass
                
                # If we got here, try to convert to DataFrame and then to dict
                try:
                    df = pd.DataFrame(result)
                    return df.to_dict()
                except:
                    # Last resort - convert to string and back to evaluate structure
                    import json
                    try:
                        json_str = json.dumps(result)
                        parsed = json.loads(json_str)
                        df = pd.DataFrame(parsed)
                        return df.to_dict()
                    except:
                        # If all else fails, return the original result
                        return result
            except Exception as e:
                print(f"Post-processing error: {str(e)}")
                return result
        
        # Define a pre-processing function that sets pandas options
        def preprocessing_with_options(data):
            """Convert data to DataFrame and set pandas options to avoid warnings."""
            # Disable chained assignment warnings - will be reset after execution
            prior_option = pd.get_option('mode.chained_assignment')
            pd.set_option('mode.chained_assignment', None)
            
            # Return both the dataframe and the prior option setting for restoration
            return (pd.DataFrame.from_dict(data), prior_option)
        
        # Define a post-processing wrapper that restores pandas options
        def post_processing_wrapper(result, prior_option):
            """Process the result and restore pandas options."""
            # First apply the robust post-processing
            processed_result = robust_post_processing(result)
            
            # Restore the pandas option
            pd.set_option('mode.chained_assignment', prior_option)
            
            return processed_result
        
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name"),
            pre_processing=preprocessing_with_options,
            post_processing=lambda result_and_option: post_processing_wrapper(result_and_option[0], result_and_option[1]),
            error_message_prefix="An error occurred during data cleaning: "
        )
        
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
    
    # Final reporting node
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

    node_functions = {
        "recommend_cleaning_steps": recommend_cleaning_steps,
        "human_review": human_review,
        "create_data_cleaner_code": create_data_cleaner_code,
        "execute_data_cleaner_code": execute_data_cleaner_code,
        "fix_data_cleaner_code": fix_data_cleaner_code,
        "report_agent_outputs": report_agent_outputs, 
    }

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

    return app 