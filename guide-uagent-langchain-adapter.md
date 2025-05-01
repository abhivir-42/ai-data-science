# Guide: Adapting LangChain Agents to Fetch.ai uAgents

This guide provides tips and best practices for adapting LangChain agents to work with the Fetch.ai uAgents framework, based on our experience implementing the DataCleaningAgent and DataLoaderToolsAgent.

## Architecture Overview

When adapting a LangChain agent to work with Fetch.ai uAgents, we follow a three-layer architecture:

1. **Core Agent Layer**: The original LangChain agent implementation
2. **Adapter Layer**: A wrapper that makes the LangChain agent compatible with uAgents
3. **Registration Layer**: Code that registers the adapted agent with the Fetch.ai Agentverse

This layered approach allows us to maintain the core functionality of our LangChain agents while making them compatible with the uAgents ecosystem.

## Step 1: Implement the LangChain Agent

Start by implementing your agent using LangChain. For example, the `DataLoaderToolsAgent` uses LangChain's ReAct agent pattern:

```python
class DataLoaderToolsAgent(BaseAgent):
    def __init__(self, model, ...):
        self._params = {
            "model": model,
            # Other parameters
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
    
    def _make_compiled_graph(self):
        self.response = None
        return make_data_loader_tools_agent(**self._params)
    
    def invoke_agent(self, user_instructions: str = None, **kwargs):
        response = self._compiled_graph.invoke(
            {"user_instructions": user_instructions},
            **kwargs
        )
        self.response = response
        return None
    
    # Other methods for accessing results
```

## Step 2: Create an Adapter Class

Next, create an adapter class that wraps your LangChain agent and makes it compatible with uAgents:

```python
class DataLoaderToolsAgentAdapter:
    def __init__(
        self,
        model: BaseLanguageModel,
        name: str = "data_loader_agent",
        port: int = 8001,
        description: str = None,
        mailbox: bool = True,
        api_token: Optional[str] = None,
    ):
        self.model = model
        self.name = name
        self.port = port
        self.description = description or "Default description"
        self.mailbox = mailbox
        self.api_token = api_token
        
        # Create the LangChain agent
        self.agent = DataLoaderToolsAgent(model=model)
        
        # Initialize uAgent info
        self.uagent_info = None
    
    # Other methods
```

## Step 3: Implement Registration Method

Implement a method to register your agent with the Fetch.ai Agentverse:

```python
def register(self) -> Dict[str, Any]:
    try:
        # Import here to avoid dependency issues if uagents is not installed
        from uagents_adapter.langchain import UAgentRegisterTool
        
        # Create the agent executor from the simplified adapter
        uagent_register_tool = UAgentRegisterTool()
        
        # Register the agent
        result = uagent_register_tool.invoke({
            "agent_obj": self.agent,
            "name": self.name,
            "port": self.port,
            "description": self.description,
            "mailbox": self.mailbox,
            "api_token": self.api_token,
            "return_dict": True
        })
        
        self.uagent_info = result
        return result
        
    except ImportError as e:
        return {
            "error": f"Failed to import uAgents dependencies: {str(e)}",
            "solution": "Install required packages with: pip install 'uagents-adapter[langchain]>=2.2.0'"
        }
    except Exception as e:
        return {
            "error": f"Failed to register agent: {str(e)}",
            "solution": "Ensure you have the latest uAgents version installed and check network connectivity"
        }
```

## Step 4: Add Helper Methods

Add helper methods to make your agent easier to use:

```python
def load_data(self, instructions: str) -> pd.DataFrame:
    """
    Load data based on the provided instructions.
    """
    self.agent.invoke_agent(
        user_instructions=instructions
    )
    
    return self.agent.get_artifacts(as_dataframe=True)
```

## Common Challenges and Solutions

### 1. Managing Dependencies

**Challenge**: The uAgents packages may introduce dependency conflicts.

**Solution**: Import uAgents dependencies only when needed, and handle ImportError gracefully:

```python
try:
    from uagents_adapter.langchain import UAgentRegisterTool
except ImportError:
    # Provide helpful error message
```

### 2. Handling Agent State

**Challenge**: LangChain agents maintain state differently than uAgents.

**Solution**: Use the adapter to translate between state models:

```python
def process_message(self, message):
    # Convert uAgent message format to LangChain format
    instructions = message.get("instructions", "")
    
    # Invoke LangChain agent
    self.agent.invoke_agent(user_instructions=instructions)
    
    # Convert LangChain response to uAgent response format
    return {
        "result": self.agent.get_artifacts(as_dataframe=True).to_dict()
    }
```

### 3. Error Handling

**Challenge**: Errors in the LangChain agent need to be properly communicated to uAgents.

**Solution**: Implement comprehensive error handling in your adapter:

```python
try:
    # Call LangChain agent
except Exception as e:
    return {
        "error": str(e),
        "status": "failed"
    }
```

## Best Practices

1. **Keep Core Functionality Separate**: Maintain a clean separation between your core LangChain agent logic and the uAgent adaptation layer.

2. **Use Consistent Naming**: Use consistent naming conventions for methods and parameters across your core agent and adapter.

3. **Handle Authentication Properly**: Store API keys securely and provide clear instructions for setting them up.

4. **Provide Fallbacks**: Allow your adapter to work in standalone mode if uAgents are not available.

5. **Document Integration Points**: Clearly document how your agent can interact with other agents in the ecosystem.

## Example: Complete Agent Registration

```python
import os
from langchain_openai import ChatOpenAI
from ai_data_science.adapters import DataLoaderToolsAgentAdapter

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create and register the agent
data_loader_adapter = DataLoaderToolsAgentAdapter(
    model=llm,
    name="data_loader_agent",
    port=8001,
    api_token=os.environ.get("AGENTVERSE_API_KEY"),
    mailbox=True
)

# Register with Agentverse
result = data_loader_adapter.register()

# Check registration status
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Success! Agent address: {result.get('address')}")
```

## Conclusion

By following this architecture and these best practices, you can successfully adapt your LangChain agents to work with the Fetch.ai uAgents framework, allowing them to participate in the wider Agentverse ecosystem.
