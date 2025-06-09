# AI Data Science uAgent Adapters

This directory contains uAgent adapters that convert our AI Data Science LangGraph agents into uAgents that can be deployed on the [Agentverse](https://agentverse.ai) ecosystem.

## Overview

The uAgent adapters allow you to:
- ✅ Deploy your AI Data Science agents as discoverable uAgents
- 🌐 Register them on the Agentverse marketplace  
- 🤖 Enable agent-to-agent communication
- 📡 Access agents via HTTP API or mailbox service
- 🏷️ Get the "innovationlab" badge for your agents

## Available Agents

### 1. Data Loader Agent (`data_loader_uagent.py`)
- **Purpose**: Loads data from various file formats (CSV, Excel, JSON, Parquet)
- **Port**: 8000 (default)
- **Input**: File paths and loading instructions
- **Output**: Structured dataset summaries

### 2. Data Cleaning Agent (`data_cleaning_uagent.py`) 
- **Purpose**: Cleans and preprocesses datasets using best practices
- **Port**: 8001 (default)
- **Input**: Raw dataset and cleaning instructions  
- **Output**: Cleaned dataset with preservation metrics

### 3. Data Visualization Agent (`data_visualization_uagent.py`)
- **Purpose**: Creates interactive Plotly charts and visualizations
- **Port**: 8002 (default)
- **Input**: Dataset and visualization requirements
- **Output**: Interactive Plotly chart specifications

## Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install "uagents-adapter[langchain]"
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export AGENTVERSE_API_TOKEN="your-agentverse-token"  # Optional but recommended
   ```

### Option 1: Register All Agents at Once

```bash
cd src/agents/uagent_fetch_ai
python register_all_agents.py
```

### Option 2: Register Individual Agents

```python
from src.agents.uagent_fetch_ai import (
    register_data_loader_uagent,
    register_data_cleaning_uagent, 
    register_data_visualization_uagent
)

# Register the data loader agent
result = register_data_loader_uagent(
    port=8000,
    name="my_data_loader",
    api_token="your_agentverse_token"
)

print(f"Agent Address: {result['agent_address']}")
```

### Option 3: Run the Example

```bash
python example_uagent_usage.py
```

## Registration Process

When you register an agent, the following happens:

1. **Local Registration**: 
   - Agent starts running on the specified port
   - Gets a unique agent address (e.g., `agent1q...`)
   - Enables HTTP API access

2. **Agentverse Registration** (if API token provided):
   - Agent appears in your [Agentverse dashboard](https://agentverse.ai)
   - Gets discoverable in the marketplace
   - Receives "innovationlab" badge
   - Auto-generated README with input/output models
   - Mailbox service enabled for communication

## Agent Communication

### HTTP API Example
```python
import requests

# Communicate with data loader agent
response = requests.post(
    "http://localhost:8000/submit",
    json={
        "user_instructions": "Load data from data/sample.csv"
    }
)
```

### uAgent-to-uAgent Communication
```python
from uagents import Agent, Context
from uagents.communication import send

# Send message to registered agent
await send(
    destination="agent1q...",  # Agent address from registration
    message={"query": "Load sales data"},
    ctx=ctx
)
```

## Configuration Options

All registration functions support these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_key` | str | None | OpenAI API key (or from env) |
| `port` | int | 8000+ | Port to run the agent on |
| `name` | str | agent_type | Name for the uAgent |
| `description` | str | Auto | Description of capabilities |
| `mailbox` | bool | True | Use Agentverse mailbox service |
| `api_token` | str | None | Agentverse API token (or from env) |
| `return_dict` | bool | True | Return result as dictionary |

## Agentverse Integration

### Getting Your API Token
1. Visit [Agentverse.ai](https://agentverse.ai)
2. Create an account or sign in
3. Go to your profile/settings
4. Generate an API token
5. Set it as environment variable: `export AGENTVERSE_API_TOKEN="your-token"`

### What Happens on Agentverse
- Agents appear in "My Agents" section
- Auto-generated documentation with input/output models
- "innovationlab" badge indicating AI innovation
- Discoverable by other users in the ecosystem
- Usage analytics and monitoring
- Integration with ASI:One for advanced features

### Example Auto-Generated README
```markdown
# AI Data Science Loader
AI Data Science Data Loader Agent - Loads and processes data from various file formats
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

**Input Data Model**
```python
class QueryMessage(Model):
    user_instructions: str
```

**Output Data Model**  
```python
class ResponseMessage(Model):
    data_loader_artifacts: dict
    tool_calls: list
```
```

## Troubleshooting

### Common Issues

1. **"OpenAI API key not set"**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **"Port already in use"**
   - Change the port number in registration
   - Kill existing processes on that port

3. **"Import errors"**
   - Make sure you're in the correct directory
   - Check that uagents-adapter is installed

4. **"Registration failed"**
   - Check network connection
   - Verify API tokens are correct
   - Try registering without Agentverse token first

### Getting Help

- Check [Fetch.AI Documentation](https://docs.fetch.ai)
- Visit [Innovation Lab](https://innovationlab.fetch.ai)
- Join the [Fetch.AI Discord](https://discord.gg/fetchai)

## Advanced Usage

### Custom Agent Configuration
```python
# Advanced data cleaning agent setup
result = register_data_cleaning_uagent(
    port=8001,
    name="enterprise_data_cleaner",
    description="Enterprise-grade data cleaning with custom validation rules",
    api_token=api_token,
    query_params={
        "data_raw": {
            "type": "object",
            "description": "Dataset to clean",
            "required": True
        },
        "cleaning_rules": {
            "type": "object", 
            "description": "Custom cleaning configuration",
            "required": False
        }
    }
)
```

### Multi-Agent Workflows
```python
# Create a data science pipeline with multiple agents
loader_result = register_data_loader_uagent(port=8000)
cleaner_result = register_data_cleaning_uagent(port=8001) 
viz_result = register_data_visualization_uagent(port=8002)

# Chain them together in your application
data = loader_agent.load("data.csv")
clean_data = cleaner_agent.clean(data)
chart = viz_agent.visualize(clean_data, "scatter plot")
```

## License

This project is licensed under the same terms as the main AI Data Science project.

---

**Happy Agent Building! 🤖✨** 