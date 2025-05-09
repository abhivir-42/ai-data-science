# Guide to Recreating the Data Cleaning Agent

This guide outlines the steps to recreate the data cleaning agent in your own projects.

## Prerequisites

- Python 3.9 or higher
- Basic understanding of pandas, LangChain, and LangGraph
- Access to an OpenAI API key or other LLM provider

## Installation

```bash
pip install langchain langchain-openai langgraph pandas numpy scikit-learn
```

## Step-by-Step Implementation

### 1. Setup Project Structure

Create the following directory structure:

```
ai-data-science/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── data_cleaning_agent.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── parsers.py
│   ├── templates/
│   │   ├── __init__.py
│   │   └── agent_templates.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── dataframe.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── regex.py
└── examples/
    └── data_cleaning_example.py
```

### 2. Implement Supporting Modules

1. **Parsers**: Create `parsers.py` with the `PythonOutputParser` class to extract code from LLM outputs.

2. **Utils**: 
   - Create `regex.py` with utility functions for text manipulation.
   - Create `logging.py` with functions for logging code to files.

3. **Tools**: Create `dataframe.py` with the `get_dataframe_summary` function for summarizing data.

4. **Templates**: Create `agent_templates.py` with:
   - `BaseAgent` abstract class
   - Node function templates (`node_func_execute_agent_code_on_data`, etc.)
   - `create_coding_agent_graph` function to create the workflow

Refer to previous sections for detailed implementations of these modules.

### 3. Implement the Data Cleaning Agent

In `data_cleaning_agent.py`, implement:

1. The `DataCleaningAgent` class that inherits from `BaseAgent`
2. The `make_data_cleaning_agent` function that creates the LangGraph workflow

### 4. Usage Example

Create an example script in `examples/data_cleaning_example.py`:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from src.agents.data_cleaning_agent import DataCleaningAgent

# Initialize language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create data cleaning agent
data_cleaning_agent = DataCleaningAgent(
    model=llm,
    n_samples=50,
    log=True,
    log_path="logs",
    human_in_the_loop=True
)

# Load data
df = pd.read_csv("data/your_dataset.csv")

# Clean data
data_cleaning_agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove outliers and handle missing values.",
    max_retries=3,
    retry_count=0
)

# Get cleaned data
cleaned_data = data_cleaning_agent.get_data_cleaned()

# View the generated cleaning function
print(data_cleaning_agent.get_data_cleaner_function())
```

## Key Implementation Details

### BaseAgent Class

Implement a minimal version focusing on:
- Storing parameters in `_params`
- Creating the compiled graph in `_make_compiled_graph`
- Implementing `invoke_agent` and `ainvoke_agent` methods
- Adding getter methods for accessing results

### Node Functions

Implement the core node functions:
1. `recommend_cleaning_steps`: Analyzes data and recommends cleaning steps
2. `create_data_cleaner_code`: Generates Python code for data cleaning
3. `execute_data_cleaner_code`: Executes the generated code on the data
4. `fix_data_cleaner_code`: Fixes errors in the generated code
5. `report_agent_outputs`: Creates a report of the agent's outputs

### Graph Construction

Use `create_coding_agent_graph` to create a workflow with:
1. Entry point: recommend_cleaning_steps (or create_data_cleaner_code if bypassing recommendations)
2. Human review (optional): Allows humans to review and modify recommendations
3. Code execution: Executes the generated code and handles errors
4. Error handling: Fixes errors and retries execution

## Extensions and Customizations

1. **Add new cleaning capabilities**: Modify the prompt templates to suggest additional cleaning steps.
2. **Integrate with databases**: Extend the agent to work with data from databases.
3. **Add visualization**: Create a complementary visualization agent to visualize the cleaned data.
4. **Improve error handling**: Add more sophisticated error analysis and fixing capabilities.

## Conclusion

By following this guide, you can recreate the data cleaning agent and customize it to your needs. For more detailed implementations, refer to the previous sections that provide in-depth explanations of each component.

For a complete reference implementation, study the code in the original repository, which includes robust error handling, comprehensive documentation, and extensive testing.
