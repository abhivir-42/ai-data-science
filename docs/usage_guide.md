# Data Cleaning Agent Usage Guide

The Data Cleaning Agent is designed to automate data cleaning tasks using AI capabilities. This guide explains how to use the agent, its parameters, expected inputs/outputs, and provides examples.

## Installation

Ensure you have installed the package as described in the main README:

```bash
pip install -e .
```

## Required Dependencies

To use the agent, you need:
- An OpenAI API key (or another LLM provider)
- The dependencies listed in requirements.txt

## Basic Usage

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

# Set up your language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create the agent
agent = DataCleaningAgent(model=llm)

# Load your data
df = pd.read_csv("your_dataset.csv")

# Run the agent
agent.invoke_agent(
    data_raw=df,
    user_instructions="Clean the data, but don't remove any outliers."
)

# Get the cleaned data
cleaned_df = agent.get_data_cleaned()
```

## Agent Parameters

When creating the agent, you can customize its behavior with the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | LLM | required | The language model to use |
| `n_samples` | int | 30 | Number of samples for data summarization |
| `log` | bool | False | Whether to log code generation and errors |
| `log_path` | str | None | Path for log files |
| `file_name` | str | "data_cleaner.py" | File name for generated code |
| `function_name` | str | "data_cleaner" | Function name for generated code |
| `overwrite` | bool | True | Whether to overwrite existing log files |
| `human_in_the_loop` | bool | False | Enable human review of cleaning steps |
| `bypass_recommended_steps` | bool | False | Skip default cleaning recommendations |
| `bypass_explain_code` | bool | False | Skip code explanation generation |

## Input Requirements

The `invoke_agent()` method accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_raw` | pd.DataFrame | required | The raw dataset to be cleaned |
| `user_instructions` | str | None | Custom cleaning instructions |
| `max_retries` | int | 3 | Maximum number of code fix attempts |
| `retry_count` | int | 0 | Current retry attempt |

## Output

The agent provides several methods to access results:

- `get_data_cleaned()`: Returns the cleaned DataFrame
- `get_data_raw()`: Returns the original DataFrame
- `get_data_cleaner_function()`: Returns the generated cleaning function code
- `get_recommended_cleaning_steps()`: Returns the recommended cleaning steps
- `get_workflow_summary()`: Returns a summary of the agent's workflow

## Default Cleaning Steps

If no custom instructions are provided, the agent performs these default cleaning steps:

1. Removing columns with more than 40% missing values
2. Imputing missing values with the mean for numeric columns
3. Imputing missing values with the mode for categorical columns
4. Converting columns to appropriate data types
5. Removing duplicate rows
6. Removing rows with missing values
7. Removing rows with extreme outliers (3x the interquartile range)

## Custom Cleaning Instructions

You can provide custom instructions to modify the default behavior:

```python
agent.invoke_agent(
    data_raw=df,
    user_instructions="Don't remove any outliers. Also, fill missing values in the 'age' column with the median instead of the mean."
)
```

## Human-in-the-Loop Mode

When `human_in_the_loop=True`, the agent will ask for confirmation before executing the cleaning steps:

```python
agent = DataCleaningAgent(model=llm, human_in_the_loop=True)
```

This allows you to review and modify the proposed cleaning steps before they are applied.

## Logging

Enable logging to save the generated code and track errors:

```python
agent = DataCleaningAgent(model=llm, log=True, log_path="./logs")
```

## Example with Complete Workflow

```python
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create the data cleaning agent
agent = DataCleaningAgent(
    model=llm,
    n_samples=50,
    log=True,
    log_path="./logs",
    human_in_the_loop=True
)

# Load sample data
df = pd.read_csv("data/sample_data.csv")

# Display summary of raw data
print("Raw Data Shape:", df.shape)
print("\nRaw Data Preview:")
print(df.head())
print("\nMissing Values:", df.isna().sum().sum())

# Invoke the agent with custom instructions
agent.invoke_agent(
    data_raw=df,
    user_instructions="Clean the data, but don't remove any outliers. Fill missing values in numeric columns with median instead of mean."
)

# Get the cleaned data
cleaned_df = agent.get_data_cleaned()

# Display summary of cleaned data
print("\nCleaned Data Shape:", cleaned_df.shape)
print("\nCleaned Data Preview:")
print(cleaned_df.head())
print("\nMissing Values:", cleaned_df.isna().sum().sum())

# Display the recommended cleaning steps
print("\nRecommended Cleaning Steps:")
print(agent.get_recommended_cleaning_steps())

# Display the generated cleaning function
print("\nGenerated Cleaning Function:")
print(agent.get_data_cleaner_function())
```

## Error Handling

The agent will attempt to fix errors automatically up to the number of retries specified in `max_retries`. If it cannot fix an error, it will return the error message and the original data. 