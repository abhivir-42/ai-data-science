# AI Data Science Adapters

This directory contains adapters that allow AI Data Science agents to integrate with external frameworks and platforms.

## Available Adapters

### DataCleaningAgentAdapter

The `DataCleaningAgentAdapter` allows the `DataCleaningAgent` to be registered as a uAgent with the Fetch.ai Agentverse platform. This adapter wraps the LangChain/LangGraph-based data cleaning agent and makes it available as a uAgent, allowing it to communicate with other agents in the Agentverse ecosystem.

#### Features

- Automatic environment variable loading for API keys
- Support for both direct data cleaning and agent registration
- Proper error handling and logging
- Support for cleanup and deregistration

#### Usage

```python
from langchain_openai import ChatOpenAI
from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Create the adapter
adapter = DataCleaningAgentAdapter(
    model=llm,
    name="data_cleaning_agent",
    port=8000,
    description="A data cleaning agent",
    mailbox=True,
    n_samples=20,
    log=False
)

# Use the adapter to clean data directly
import pandas as pd
df = pd.read_csv("data.csv")
cleaned_df = adapter.clean_data(df, "Fill missing values and convert data types")

# Or register with Agentverse (requires uagents and uagents-adapter packages)
result = adapter.register()

# Get agent information
info = adapter.get_agent_info()
print(f"Agent address: {info.get('agent_address')}")

# Clean up when done
adapter.cleanup()
```

### DataLoaderToolsAgentAdapter

The `DataLoaderToolsAgentAdapter` allows the `DataLoaderToolsAgent` to be registered as a uAgent with the Fetch.ai Agentverse platform. This adapter provides a way to load data from various sources and formats through natural language instructions.

#### Features

- Loads data from various sources (CSV, Excel, JSON, Parquet)
- Supports file searching and directory listing
- Returns data in a structured format (pandas DataFrame)
- Integrates with the Agentverse ecosystem

#### Usage

```python
from langchain_openai import ChatOpenAI
from ai_data_science.adapters.uagents_adapter import DataLoaderToolsAgentAdapter

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Create the adapter
adapter = DataLoaderToolsAgentAdapter(
    model=llm,
    name="data_loader_agent",
    port=8001,
    description="A data loader agent",
    mailbox=True
)

# Use the adapter to load data directly
loaded_df = adapter.load_data("Load the CSV file from data/sales_data.csv")

# Or register with Agentverse
result = adapter.register()

# Get agent information
info = adapter.get_agent_info()
print(f"Agent address: {info.get('agent_address')}")

# Clean up when done
adapter.cleanup()
```

## Communication Protocol

Once registered, these agents can be communicated with using the uAgent's chat protocol:

```python
from datetime import datetime
from uuid import uuid4
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

# Send a message to an adapter-wrapped agent
message = ChatMessage(
    timestamp=datetime.utcnow(),
    msg_id=uuid4(),
    content=[TextContent(type="text", text="Load data from file.csv")]
)
await ctx.send(agent_address, message)
```

## Registration and API Keys

The adapters will automatically look for API keys in the environment variables or a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
AGENTVERSE_API_TOKEN=eyJhbGc...
```

You can also pass these keys directly when creating the adapters:

```python
adapter = DataCleaningAgentAdapter(
    model=llm,
    api_token="your-agentverse-api-token"
)
```

## Dependencies

These adapters require the following dependencies:

- `uagents-adapter >= 2.2.0`
- `python-dotenv`
- `langchain`
- `langgraph`
- `pandas`

Install the required packages with:

```bash
pip install 'uagents-adapter>=2.2.0' python-dotenv
```

## Examples

See the `examples` directory for complete usage examples:

- `test_uagents_adapters.py`: Comprehensive example showing adapter usage and registration

## Error Handling

The adapters include comprehensive error handling:

- API key validation
- Import error detection for missing dependencies
- Registration error handling
- Data processing error handling

## Cleanup

Always clean up your agents when shutting down to ensure proper deregistration:

```python
try:
    # Your agent code here
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Clean up the agent
    adapter.cleanup()
    print("Agent stopped.")
``` 