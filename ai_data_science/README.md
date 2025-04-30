# AI Data Science Library

A toolkit for data science tasks powered by AI agents.

## Overview

This library provides a collection of AI-powered agents for common data science tasks, including:

- Data cleaning and preprocessing
- Feature engineering
- Data visualization
- Model selection and evaluation

## Installation

```bash
pip install ai-data-science
```

## Quick Start

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Create the agent
agent = DataCleaningAgent(
    model=llm,
    n_samples=30,
    log=False
)

# Load your data
df = pd.read_csv("your_data.csv")

# Clean the data with natural language instructions
agent.invoke_agent(
    data_raw=df,
    user_instructions="Fill missing values, convert categorical columns to appropriate types, and remove outliers."
)

# Get the cleaned data
cleaned_df = agent.get_data_cleaned()

# Get the cleaning function for reproducibility
cleaning_function = agent.get_data_cleaner_function()
```

## Directory Structure

- `adapters/`: Adapters for integrating agents with external frameworks
- `agents/`: AI-powered agents for various data science tasks
- `examples/`: Example scripts and notebooks
- `parsers/`: Output parsers for structured agent responses
- `templates/`: Templates for agent creation
- `tools/`: Tools and utilities for agent operations
- `utils/`: Utility functions used throughout the library

## Components

### Agents

- `DataCleaningAgent`: Agent for cleaning and preprocessing datasets

### Adapters

- `DataCleaningAgentAdapter`: Adapter for registering data cleaning agents with the Fetch.ai Agentverse

## Examples

See the `examples/` directory for usage examples:

- `simplified_adapter.py`: Simple example of using the DataCleaningAgentAdapter
- `register_uagent.py`: Example of registering a DataCleaningAgent with Agentverse 