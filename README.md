# AI Data Science Toolkit

A toolkit for automating data science tasks using AI, specifically focused on data cleaning and transformation.

## Features

- **Data Cleaning Agent**: An AI-powered agent that automatically cleans and preprocesses data based on natural language instructions.
- **Customizable Cleaning Operations**: Control what operations the agent should perform (handling missing values, removing duplicates, treating outliers, etc.).
- **Detailed Reporting**: Get detailed reports on what changes were made to your data.
- **uAgents Integration**: Deploy your Data Cleaning Agent as a uAgent in the Fetch AI Agentverse, allowing it to interact with other agents.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ai-data-science.git
   cd ai-data-science
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. For uAgents integration, install the additional package:
   ```bash
   pip install "uagents-adapter[langchain]"
   ```

5. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key
   ```

## Quick Start

Here's a simple example of how to use the Data Cleaning Agent:

```python
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from ai_data_science.agents import DataCleaningAgent

# Load environment variables (API keys)
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load your data
df = pd.read_csv("path/to/your/data.csv")

# Create the agent
agent = DataCleaningAgent(
    model=llm,
    n_samples=30,  # Number of samples to analyze
    log=True,      # Log generated code
    log_path="./logs",
    human_in_the_loop=False
)

# Run the agent with your instructions
agent.invoke_agent(
    data_raw=df,
    user_instructions="Clean the data by handling missing values, removing duplicates, and treating outliers.",
    max_retries=2
)

# Get the cleaned data
cleaned_df = agent.get_data_cleaned()

# Get the recommended cleaning steps
steps = agent.get_recommended_cleaning_steps()
print(steps)

# Get the data cleaning function
cleaning_function = agent.get_data_cleaner_function()
print(cleaning_function)
```

## Advanced Usage

### Custom Cleaning Instructions

You can give specific instructions to the agent about how to clean your data:

```python
# Only handle missing values, don't remove duplicates or outliers
agent.invoke_agent(
    data_raw=df,
    user_instructions="Only handle missing values. Do not remove duplicates or outliers.",
    max_retries=2
)

# Keep outliers but remove duplicates
agent.invoke_agent(
    data_raw=df,
    user_instructions="Clean the data and remove duplicates, but don't remove any outliers.",
    max_retries=2
)
```

### Human-in-the-Loop

You can enable human review of the recommended cleaning steps:

```python
agent = DataCleaningAgent(
    model=llm,
    human_in_the_loop=True
)

# This will prompt for human approval of the cleaning steps
agent.invoke_agent(data_raw=df)
```

## uAgents Integration

The toolkit includes an adapter for deploying your Data Cleaning Agent as a uAgent in the Fetch AI Agentverse.

### Registering a Data Cleaning Agent as a uAgent

```python
from langchain_openai import ChatOpenAI
from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the adapter
adapter = DataCleaningAgentAdapter(
    model=llm,
    name="data_cleaning_agent",
    port=8000,
    description="A data cleaning agent that processes datasets based on instructions",
    mailbox=True,
    api_token="YOUR_AGENTVERSE_API_TOKEN",  # Optional: for Agentverse registration
    log=True,
    human_in_the_loop=False
)

# Register the agent with Agentverse
result = adapter.register()

# Print agent information
print(f"Agent address: {result['agent_address']}")
print(f"Agent port: {result['agent_port']}")
```

### Interacting with a Registered Agent

You can interact with a registered Data Cleaning Agent using the uAgents framework:

```python
import pandas as pd
from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from typing import Dict, Any, Optional

# Define message models
class DataCleaningRequest(Model):
    query: str = Field(description="Instructions for how to clean the data")
    data_dict: Dict[str, Any] = Field(description="Dataset in dictionary format")

class DataCleaningResponse(Model):
    data_cleaned: Optional[Dict[str, Any]] = Field(default=None)
    cleaner_function: Optional[str] = Field(default=None)
    workflow_summary: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)

# Create a client agent
client_agent = Agent(
    name="data_cleaning_client",
    seed="data_cleaning_client_seed",
    endpoint=["http://localhost:8001/submit"],
    port=8001,
)

# Create a protocol
protocol = Protocol("DataCleaningProtocol")

# Create sample data
df = pd.read_csv("path/to/your/data.csv")

# Prepare the request
request = DataCleaningRequest(
    query="Clean this dataset by removing duplicates and handling missing values",
    data_dict=df.to_dict()
)

# Handle responses
@protocol.on_message(model=DataCleaningResponse)
async def handle_response(ctx: Context, sender: str, response: DataCleaningResponse):
    if response.data_cleaned:
        cleaned_df = pd.DataFrame.from_dict(response.data_cleaned)
        print("Cleaned data received!")
        print(cleaned_df.head())

# Register the protocol
client_agent.include(protocol)

# Send the request
@client_agent.on_event("startup")
async def on_startup(ctx: Context):
    agent_address = "agent1q..."  # Address of the registered data cleaning agent
    await ctx.send(agent_address, request)

# Run the client agent
client_agent.run()
```

For a complete example, see the `examples/uagents_adapter_example.py` and `examples/uagents_interaction_example.py` files.

## Testing

Run the included test script to verify the agent works correctly:

```bash
python test_agent.py
```

For more comprehensive testing with different scenarios:

```bash
python test_data_cleaning.py
```

## License

MIT License

## Acknowledgements

This project was inspired by the AI Data Science Team's work on integrating AI agents into data science workflows. 