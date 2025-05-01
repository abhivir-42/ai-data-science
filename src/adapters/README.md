# AI Data Science Adapters

This directory contains adapters that allow AI Data Science agents to integrate with external frameworks and platforms.

## Available Adapters

### DataCleaningAgentAdapter

The `DataCleaningAgentAdapter` allows the `DataCleaningAgent` to be registered as a uAgent with the Fetch.ai Agentverse platform.

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
```

## Examples

See the `examples` directory for complete usage examples:

- `simplified_adapter.py`: Simple example of using the adapter without uAgent registration
- `register_uagent.py`: Complete example with package verification and registration 