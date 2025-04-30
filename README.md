# AI Data Science

A toolkit for data science tasks powered by AI agents.

## Overview

This repository contains AI-powered agents for various data science tasks. The current implementation focuses on a data cleaning agent that can process datasets based on user-defined instructions or default cleaning steps.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-data-science.git
cd ai-data-science

# Install the package
pip install -e .
```

## Features

- **Data Cleaning Agent**: Automatically clean datasets based on common best practices or custom instructions.
  - Removes columns with excessive missing values
  - Imputes missing values with appropriate strategies
  - Converts columns to appropriate data types
  - Removes duplicate rows
  - Handles outliers
  - Customizable via user instructions

## Quick Start

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

# Initialize language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create data cleaning agent
data_cleaning_agent = DataCleaningAgent(
    model=llm, n_samples=50, log=True, log_path="logs", human_in_the_loop=True
)

# Load data
df = pd.read_csv("your_data.csv")

# Run the agent with custom instructions
data_cleaning_agent.invoke_agent(
    data_raw=df,
    user_instructions="Don't remove outliers when cleaning the data.",
    max_retries=3
)

# Get the cleaned data
cleaned_data = data_cleaning_agent.get_data_cleaned()
```

## Documentation

For detailed documentation, please refer to the [docs](./docs/) directory:

- [Implementation Plan](./docs/implementation_plan.md)
- [Usage Guide](./docs/usage_guide.md)

## Examples

Check the [examples](./examples/) directory for Jupyter notebooks demonstrating how to use the agents.

## License

MIT License

## Acknowledgements

This project was inspired by the AI Data Science Team's work on integrating AI agents into data science workflows. 