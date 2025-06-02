"""
Data Cleaning Agent for AI Data Science.

This module provides a specialized agent for cleaning datasets based on recommended
best practices or user-defined instructions.
"""

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Union, Dict
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
    
    def invoke_agent(
        self,
        data_raw: Union[Dict, pd.DataFrame],
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs
    ):
        """
        Runs the data cleaning agent with given data and instructions.
        Automatically handles chunked datasets for large data processing.
        
        Parameters
        ----------
        data_raw : Union[Dict, pd.DataFrame]
            The raw data to clean. Can be a DataFrame or dict (potentially chunked)
        user_instructions : str, optional
            Custom cleaning instructions
        max_retries : int
            Maximum number of retry attempts
        retry_count : int
            Current retry count
            **kwargs
            Additional arguments passed to the compiled graph
        """
        
        # Check if we're dealing with a chunked dataset from DataLoaderToolsAgent
        if isinstance(data_raw, dict) and "full_dataframe" in data_raw and "chunk_info" in data_raw:
            # Handle chunked dataset processing
            return self._invoke_agent_chunked(
                data_raw, user_instructions, max_retries, retry_count, **kwargs
            )
        
        # Standard processing for non-chunked data
        return self._invoke_agent_standard(
            data_raw, user_instructions, max_retries, retry_count, **kwargs
        )
    
    def _invoke_agent_chunked(
        self,
        data_info: Dict,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs
    ):
        """
        Process a large dataset in chunks and combine the results.
        
        Parameters
        ----------
        data_info : Dict
            Dictionary containing chunked data and full dataframe
        user_instructions : str
            Cleaning instructions
        max_retries : int
            Maximum retries per chunk
        retry_count : int
            Current retry count
        **kwargs
            Additional arguments
        """
        full_df = data_info["full_dataframe"]
        chunk_info = data_info["chunk_info"]
        
        print(f"🔄 Processing large dataset in {chunk_info['total_chunks']} chunks...")
        print(f"   📊 Total: {chunk_info['total_rows']} rows × {chunk_info['total_columns']} columns")
        print(f"   📦 Chunk size: {chunk_info['chunk_size']} rows per chunk")
        
        cleaned_chunks = []
        cleaning_function = None
        
        # Process each chunk
        for chunk_idx in range(chunk_info['total_chunks']):
            start_row = chunk_idx * chunk_info['chunk_size']
            end_row = min(start_row + chunk_info['chunk_size'], chunk_info['total_rows'])
            
            chunk_df = full_df.iloc[start_row:end_row].copy()
            
            print(f"   🧹 Processing chunk {chunk_idx + 1}/{chunk_info['total_chunks']} (rows {start_row}-{end_row})")
            
            try:
                # Process this chunk with the standard agent
                chunk_response = self._invoke_agent_standard(
                    chunk_df, user_instructions, max_retries, retry_count, **kwargs
                )
                
                # Get cleaned data for this chunk
                if chunk_response and "data_cleaned" in chunk_response:
                    cleaned_chunk_data = chunk_response["data_cleaned"]
                    if isinstance(cleaned_chunk_data, dict):
                        cleaned_chunk_df = pd.DataFrame.from_dict(cleaned_chunk_data)
                    else:
                        cleaned_chunk_df = cleaned_chunk_data
                    
                    cleaned_chunks.append(cleaned_chunk_df)
                    
                    # Save the cleaning function from the first successful chunk
                    if cleaning_function is None and "data_cleaner_function" in chunk_response:
                        cleaning_function = chunk_response["data_cleaner_function"]
                    
                    print(f"     ✅ Chunk {chunk_idx + 1} processed: {cleaned_chunk_df.shape[0]} rows")
                else:
                    print(f"     ⚠️ Chunk {chunk_idx + 1} failed, using original data")
                    cleaned_chunks.append(chunk_df)
                    
            except Exception as e:
                print(f"     ❌ Error processing chunk {chunk_idx + 1}: {str(e)}")
                print(f"     🔄 Using original chunk data")
                cleaned_chunks.append(chunk_df)
        
        # Combine all cleaned chunks
        if cleaned_chunks:
            try:
                combined_cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
                print(f"✅ All chunks processed and combined: {combined_cleaned_df.shape[0]} rows × {combined_cleaned_df.shape[1]} columns")
                
                # Create a response similar to standard processing
                self.response = {
                    "messages": [],
                    "user_instructions": user_instructions or "Clean the data efficiently for large dataset",
                    "recommended_steps": f"Processed dataset in {chunk_info['total_chunks']} chunks for optimal performance",
                    "data_raw": full_df.to_dict() if len(full_df) < 1000 else {"info": "Large dataset - shape preserved"},
                    "data_cleaned": combined_cleaned_df.to_dict(),
                    "all_datasets_summary": f"Combined {len(cleaned_chunks)} chunks into final dataset",
                    "data_cleaner_function": cleaning_function or "# Chunked processing - function from first chunk applied to all",
                    "data_cleaner_function_path": None,
                    "data_cleaner_file_name": "data_cleaner.py",
                    "data_cleaner_function_name": "data_cleaner",
                    "data_cleaner_error": "",
                    "max_retries": max_retries,
                    "retry_count": retry_count
                }
                
                return self.response
                
            except Exception as e:
                print(f"❌ Error combining chunks: {str(e)}")
                # Fallback to processing just the first chunk
                return self._invoke_agent_standard(
                    full_df.head(chunk_info['chunk_size']), user_instructions, max_retries, retry_count, **kwargs
                )
        else:
            # No chunks processed successfully, fallback to original
            print("❌ No chunks processed successfully, using original data")
            return self._invoke_agent_standard(
                full_df.head(chunk_info['chunk_size']), user_instructions, max_retries, retry_count, **kwargs
            )
    
    def _invoke_agent_standard(
        self,
        data_raw: Union[Dict, pd.DataFrame],
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs
    ):
        """
        Standard agent processing for regular-sized datasets.
        
        Parameters
        ----------
        data_raw : Union[Dict, pd.DataFrame]
            The raw data to clean
        user_instructions : str, optional
            Custom cleaning instructions
        max_retries : int
            Maximum number of retry attempts
        retry_count : int
            Current retry count
        **kwargs
            Additional arguments passed to the compiled graph
        """
        response = self._compiled_graph.invoke(
            {
                "data_raw": data_raw,
            "user_instructions": user_instructions,
            "max_retries": max_retries,
            "retry_count": retry_count,
            },
            **kwargs
        )
        self.response = response
        return response

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
        
        Returns
        -------
        pd.DataFrame
            The cleaned data as a pandas DataFrame with original shape preserved
        """
        if self.response and "data_cleaned" in self.response:
            cleaned_data = self.response.get("data_cleaned")
            if isinstance(cleaned_data, dict):
                # Convert dictionary back to DataFrame preserving original orientation
                try:
                    return pd.DataFrame.from_dict(cleaned_data, orient='columns')
                except Exception:
                    # Fallback to default conversion if there's an issue
                    return pd.DataFrame(cleaned_data)
            elif isinstance(cleaned_data, pd.DataFrame):
                return cleaned_data
            else:
                # Try to convert whatever we have to DataFrame
                return pd.DataFrame(cleaned_data)
        return None
        
    def get_data_raw(self):
        """
        Retrieves the raw data.
        
        Returns
        -------
        pd.DataFrame
            The raw data as a pandas DataFrame with original shape preserved
        """
        if self.response and "data_raw" in self.response:
            raw_data = self.response.get("data_raw")
            if isinstance(raw_data, dict):
                # Convert dictionary back to DataFrame preserving original orientation
                try:
                    return pd.DataFrame.from_dict(raw_data, orient='columns')
                except Exception:
                    # Fallback to default conversion if there's an issue
                    return pd.DataFrame(raw_data)
            elif isinstance(raw_data, pd.DataFrame):
                return raw_data
            else:
                # Try to convert whatever we have to DataFrame
                return pd.DataFrame(raw_data)
        return None
    
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
            You are a Senior Data Scientist creating a {function_name}() function for PRODUCTION data cleaning.
            
            🚨 CRITICAL DATA PRESERVATION RULES:
            1. NEVER remove more than 20% of rows unless explicitly instructed
            2. PRIORITIZE IMPUTATION over DELETION
            3. Missing values are often INFORMATIVE - don't just delete them
            4. For real-world datasets, some missing values are EXPECTED and NORMAL
            5. Outliers in specialized domains (like real estate) may be LEGITIMATE high values
            
            📋 USER REQUIREMENTS:
            {recommended_steps}

            📊 DATASET ANALYSIS:
            {all_datasets_summary}
            
            🎯 PRODUCTION-READY IMPLEMENTATION GUIDE:
            
            ✅ REQUIRED APPROACHES:
            
            **Data Type Optimization:**
            - Convert string numbers to numeric: pd.to_numeric(df['col'], errors='coerce')
            - Handle date columns: pd.to_datetime(df['date_col'], errors='coerce')
            - Optimize memory: Use appropriate dtypes (int32 vs int64, category for strings)
            
            **Missing Value Strategy:**
            
            🔧 **For Numeric Columns:**
            - df['col'].fillna(df['col'].median()) or df['col'].fillna(df['col'].mean())
            - Use median for skewed data, mean for normal distributions
            
            🏷️ **For Categorical Columns (CRITICAL - Handle dtype carefully):**
            ```python
            # Safe categorical handling - prevents "Cannot setitem on a Categorical" error
            if col_dtype == 'category':
                # Method 1: Use existing category value (safest)
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    # If no mode exists, convert to object first, then fill
                    df[col] = df[col].astype('object').fillna('Unknown')
            elif col_dtype == 'object':
                # Standard string/object columns
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            ```
            
            📅 **For Time Series:**
            - df['col'].fillna(method='ffill') or df['col'].fillna(method='bfill')
            - pd.to_datetime(df['date_col'], errors='coerce') for date conversion
            
            🚨 **CRITICAL RULE:** Never drop rows unless >80% of values are missing
            
            **Outlier Handling (Conservative):**
            - Use IQR method only if explicitly requested: Q1 - 1.5*IQR to Q3 + 1.5*IQR
            - For specialized domains (finance, real estate), be extra conservative
            - Consider capping instead of removing: df['col'] = df['col'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
            
            **Duplicate Handling:**
            - Remove exact duplicates: df.drop_duplicates()
            - Keep first occurrence unless logic suggests otherwise
            
            ❌ CRITICAL THINGS TO AVOID:
            - dropna() without subset parameter (removes too many rows)
            - Aggressive outlier removal (Z-score > 3 is too strict for most cases)
            - Removing columns with <90% missing data
            - Creating excessive dummy variables without purpose
            - Modifying original data without explicit copy()
            
            🔧 IMPLEMENTATION TEMPLATE:

            ```python
            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                from scipy import stats
                import warnings
                
                # Suppress pandas warnings for clean output
                warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
                pd.set_option('mode.chained_assignment', None)
                
                # ALWAYS work on a copy to preserve original
                data = data_raw.copy()
                
                # Initialize tracking variables
                original_shape = data.shape
                original_rows = len(data)
                cleaning_log = []
                
                print(f"🧹 Starting data cleaning: {{original_shape[0]}} rows × {{original_shape[1]}} columns")
                
                try:
                    # STEP 1: Data Type Optimization
                    print("📊 Step 1: Optimizing data types...")
                    # Add your data type conversion logic here
                    
                    # STEP 2: Handle Missing Values (PRIORITIZE IMPUTATION)
                    print("🔧 Step 2: Handling missing values...")
                    missing_before = data.isnull().sum().sum()
                    print(f"   Missing values before cleaning: {{missing_before}}")
                    
                    for col in data.columns:
                        if data[col].isnull().sum() > 0:
                            col_dtype = str(data[col].dtype)
                            missing_count = data[col].isnull().sum()
                            print(f"   Handling {{missing_count}} missing values in '{{col}}' ({{col_dtype}})")
                            
                            # Handle numeric columns
                            if pd.api.types.is_numeric_dtype(data[col]):
                                if data[col].skew() > 2 or data[col].skew() < -2:
                                    # Use median for skewed data
                                    fill_value = data[col].median()
                                    data[col] = data[col].fillna(fill_value)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with median ({{fill_value}})")
                                else:
                                    # Use mean for normal distributions
                                    fill_value = data[col].mean()
                                    data[col] = data[col].fillna(fill_value)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with mean ({{fill_value:.2f}})")
                            
                            # Handle categorical columns (CRITICAL: Safe categorical handling)
                            elif col_dtype == 'category':
                                # For pandas categorical dtype - must handle carefully
                                if not data[col].mode().empty and not pd.isna(data[col].mode()[0]):
                                    # Use mode if it exists and is not NaN
                                    mode_value = data[col].mode()[0]
                                    data[col] = data[col].fillna(mode_value)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with mode ({{mode_value}})")
                                else:
                                    # Convert to object first, then fill with 'Unknown'
                                    data[col] = data[col].astype('object').fillna('Unknown')
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with 'Unknown' (converted from category)")
                            
                            # Handle object/string columns
                            elif col_dtype == 'object' or col_dtype.startswith('string'):
                                if not data[col].mode().empty and not pd.isna(data[col].mode()[0]):
                                    # Use mode if available
                                    mode_value = data[col].mode()[0]
                                    data[col] = data[col].fillna(mode_value)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with mode ({{mode_value}})")
                                else:
                                    # Use 'Unknown' as fallback
                                    data[col] = data[col].fillna('Unknown')
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with 'Unknown'")
                            
                            # Handle datetime columns
                            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                                # Forward fill for time series data
                                data[col] = data[col].fillna(method='ffill')
                                if data[col].isnull().sum() > 0:
                                    # If still missing, backward fill
                                    data[col] = data[col].fillna(method='bfill')
                                cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' using forward/backward fill")
                            
                            # Handle boolean columns
                            elif col_dtype == 'bool':
                                # Use mode for boolean columns
                                if not data[col].mode().empty:
                                    mode_value = data[col].mode()[0]
                                    data[col] = data[col].fillna(mode_value)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with mode ({{mode_value}})")
                                else:
                                    # Default to False if no mode
                                    data[col] = data[col].fillna(False)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with False")
                            
                            # Fallback for any other data types
                            else:
                                print(f"   ⚠️ Unknown dtype {{col_dtype}} for column '{{col}}', attempting generic fill...")
                                if not data[col].mode().empty:
                                    mode_value = data[col].mode()[0]
                                    data[col] = data[col].fillna(mode_value)
                                    cleaning_log.append(f"Filled {{missing_count}} missing values in '{{col}}' with mode ({{mode_value}})")
                    
                    missing_after = data.isnull().sum().sum()
                    print(f"   Missing values after cleaning: {{missing_after}}")
                    if missing_after < missing_before:
                        print(f"   Successfully reduced missing values by {{missing_before - missing_after}}")
                    elif missing_after > 0:
                        print(f"   ⚠️ Warning: {{missing_after}} missing values remain")
                    
                    # STEP 3: Remove Duplicates
                    print("🗑️ Step 3: Removing duplicates...")
                    duplicates_before = data.duplicated().sum()
                    if duplicates_before > 0:
                        data = data.drop_duplicates()
                        cleaning_log.append(f"Removed {{duplicates_before}} duplicate rows")
                    
                    # STEP 4: Handle Outliers (CONSERVATIVE)
                    print("📈 Step 4: Conservative outlier handling...")
                    # Add your outlier handling logic here (if requested)
                    
                    # STEP 5: Final Validation and Cleanup
                    print("✅ Step 5: Final validation...")
                    # Remove any remaining rows that are completely empty
                    empty_rows = data.isnull().all(axis=1).sum()
                    if empty_rows > 0:
                        data = data.dropna(how='all')
                        cleaning_log.append(f"Removed {{empty_rows}} completely empty rows")
                    
                except Exception as e:
                    print(f"⚠️ Error during cleaning: {{str(e)}}")
                    print("🔄 Returning original data to prevent data loss")
                    return data_raw.copy()
                
                # MANDATORY: Data preservation validation
                final_shape = data.shape
                final_rows = len(data)
                data_loss_pct = ((original_rows - final_rows) / original_rows) * 100 if original_rows > 0 else 0
                
                # Comprehensive reporting
                print(f"\\n📈 CLEANING SUMMARY:")
                print(f"   Original: {{original_shape[0]}} rows × {{original_shape[1]}} columns")
                print(f"   Final: {{final_shape[0]}} rows × {{final_shape[1]}} columns")
                print(f"   Data retention: {{100-data_loss_pct:.1f}}%")
                
                if cleaning_log:
                    print(f"\\n📝 Actions taken:")
                    for action in cleaning_log:
                        print(f"   • {{action}}")
                
                # Critical validation checks
                if data_loss_pct > 25:
                    print(f"🚨 CRITICAL WARNING: High data loss ({{data_loss_pct:.1f}}%)!")
                    print(f"   Consider using more conservative cleaning approaches")
                
                if final_rows == 0:
                    print(f"❌ FATAL ERROR: All data was removed! Returning original data")
                    return data_raw.copy()
                
                if final_rows < 10 and original_rows > 100:
                    print(f"⚠️ WARNING: Extreme data reduction ({{final_rows}} remaining from {{original_rows}})")
                
                # Reset pandas options
                pd.set_option('mode.chained_assignment', 'warn')
                warnings.resetwarnings()
                
                print(f"✅ Data cleaning completed successfully!")
                return data
            ```
            
            🎯 CRITICAL SUCCESS FACTORS:
            1. **Error Recovery**: Wrap cleaning steps in try-catch blocks
            2. **Data Validation**: Always check data retention percentage
            3. **Informative Logging**: Print what actions are being taken
            4. **Conservative Defaults**: When in doubt, preserve the data
            5. **Memory Safety**: Use .copy() and proper assignment methods
            6. **User Intent**: Follow the recommended steps carefully
            7. **Domain Awareness**: Consider the data domain when making decisions
            
            📌 FINAL REMINDERS:
            - Test each cleaning step incrementally
            - Use data.loc[] for assignments to avoid chained assignment warnings
            - Validate that your cleaning logic makes sense for the specific dataset
            - If unsure about a cleaning step, add a comment explaining your reasoning
            - Always prioritize data preservation over aggressive cleaning
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
                    # CRITICAL: Data preservation validation
                    data_raw = pd.DataFrame.from_dict(state.get("data_raw"))
                    original_rows = len(data_raw)
                    cleaned_rows = len(result)
                    
                    # Calculate data loss percentage
                    data_loss_pct = ((original_rows - cleaned_rows) / original_rows) * 100 if original_rows > 0 else 0
                    
                    print(f"Data preservation check:")
                    print(f"  Original rows: {original_rows}")
                    print(f"  Cleaned rows: {cleaned_rows}")
                    print(f"  Data loss: {data_loss_pct:.1f}%")
                    
                    # WARNING: Check for excessive data loss
                    if data_loss_pct > 20:
                        print(f"🚨 WARNING: Excessive data loss ({data_loss_pct:.1f}%)!")
                        print(f"   This may indicate overly aggressive cleaning.")
                        print(f"   Consider using imputation instead of deletion.")
                        
                        # For very aggressive cleaning, use the original data instead
                        if data_loss_pct > 50:
                            print(f"❌ CRITICAL: Data loss >50%. Using original data to prevent data destruction.")
                            result = data_raw.copy()
                    
                    # Check for unreasonable column expansion (one-hot encoding issues)
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
                    
                    # Missing values after cleaning summary
                    missing_after = result.isnull().sum().sum()
                    missing_before = data_raw.isnull().sum().sum()
                    
                    print(f"Missing values before cleaning: {missing_before}")
                    print(f"Missing values after cleaning: {missing_after}")
                    
                    # Final quality check
                    if len(result) == 0:
                        print(f"❌ CRITICAL ERROR: All data was removed during cleaning!")
                        print(f"   Reverting to original data.")
                        result = data_raw.copy()
                    elif len(result) < 10 and len(data_raw) > 100:
                        print(f"❌ SEVERE WARNING: Only {len(result)} rows remain from {len(data_raw)} original rows.")
                        print(f"   This is likely too aggressive. Consider reverting to original data.")
                    
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
        
        # Store current pandas option and set it to None to avoid warnings
        prior_option = pd.get_option('mode.chained_assignment')
        pd.set_option('mode.chained_assignment', None)
        
        try:
            result = node_func_execute_agent_code_on_data(
                state=state,
                data_key="data_raw",
                result_key="data_cleaned",
                error_key="data_cleaner_error",
                code_snippet_key="data_cleaner_function",
                agent_function_name=state.get("data_cleaner_function_name"),
                pre_processing=lambda data: pd.DataFrame.from_dict(data),  # Simple: just convert to DataFrame
                post_processing=robust_post_processing,  # Apply robust post-processing
                error_message_prefix="An error occurred during data cleaning: "
            )
        finally:
            # Always restore the pandas option
            pd.set_option('mode.chained_assignment', prior_option)
        
        return result
        
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
