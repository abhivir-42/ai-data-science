Here is the guide of how to use uagents adapter from fetch ai again for your reference:
uAgents Adapter
This package provides adapters for integrating uAgents with popular AI libraries:

LangChain Adapter: Convert LangChain agents to uAgents 

Installation

# Install the base package
pip install uagents-adapter

# Install with LangChain support
pip install "uagents-adapter[langchain]"

# Install with CrewAI support
pip install "uagents-adapter[crewai]"

# Install with all extras
pip install "uagents-adapter[langchain,crewai]"

LangChain Adapter

The LangChain adapter allows you to convert any LangChain agent into a uAgent that can interact with other agents in the Agentverse ecosystem.

from langchain_core.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from uagents_adapter.langchain import UAgentRegisterTool

# Create your LangChain agent
llm = ChatOpenAI(model_name="gpt-4")
tools = [...]  # Your tools here
agent = create_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create uAgent register tool
register_tool = UAgentRegisterTool()

# Register the agent as a uAgent
result = register_tool.invoke({
    "agent_obj": agent_executor,
    "name": "my_langchain_agent",
    "port": 8000,
    "description": "My LangChain agent as a uAgent",
    "mailbox": True,  # Use Agentverse mailbox service
    "api_token": "YOUR_AGENTVERSE_API_TOKEN",  # Optional: for Agentverse registration
    "return_dict": True  # Return a dictionary instead of a string
})

print(f"Created uAgent '{result['agent_name']}' with address {result['agent_address']} on port {result['agent_port']}")

Agentverse Integration

Mailbox Service

By default, agents are created with mailbox=True, which enables the agent to use the Agentverse mailbox service. This allows agents to communicate with other agents without requiring a publicly accessible endpoint.

When mailbox is enabled:

Agents can be reached by their agent address (e.g., agent1q...)
No port forwarding or public IP is required
Messages are securely handled through the Agentverse infrastructure
Agentverse Registration

You can optionally register your agent with the Agentverse API, which makes it discoverable and usable by other users in the Agentverse ecosystem:

Obtain an API token from Agentverse.ai
Include the token when registering your agent:
result = register_tool.invoke({
    # ... other parameters
    "api_token": "YOUR_AGENTVERSE_API_TOKEN"
})
When an agent is registered with Agentverse:

It connects to the mailbox service automatically
It appears in the Agentverse directory
A README with input/output models is automatically generated
The agent gets an "innovationlab" badge
Other users can discover and interact with it
You can monitor its usage and performance through the Agentverse dashboard
Example of auto-generated README for LangChain agents:

# Agent Name
Agent Description
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

**Input Data Model**
class QueryMessage(Model): query: str


**Output Data Model**
class ResponseMessage(Model): response: str

Example of auto-generated README for CrewAI agents with parameters:

# Agent Name
Agent Description
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

**Input Data Model**
class ParameterMessage(Model): topic: str max_results: int


**Output Data Model**
class ResponseMessage(Model): response: str

We’ve just come up with uAgents‑Adapter 0.1.0 — a lightweight bridge that lets you drop your existing LangChain agents or CrewAI crews straight into the Agentverse as uAgents.
Why it rocks
One‑liner wrap‑up: Instantly convert any LangChain AgentExecutor or CrewAI Crew into a uAgent.
Zero networking hassle: The adapter auto‑connects your agent to the Agentverse mailbox, so it can talk to the world without port‑forwarding.
Instant discoverability: Pass an API token and your agent registers itself in Agentverse, complete with an auto‑generated README (inputs, outputs, examples).
Protocol‑ready: Fully compatible with the latest uAgents chat protocol (v 0.3.0), so your agent plugs right into chat.agentverse and ASI:One.
Get started
pip install uagents-adapter
# or with extras:
pip install "uagents-adapter[langchain,crewai]"
Then wrap your agent:
from uagents_adapter.langchain import UAgentRegisterTool  # or .crewai

result = UAgentRegisterTool().invoke(
    agent_obj=my_langchain_or_crewai_object,
    name="my_awesome_agent",
    mailbox=True,
    api_token="AGENTVERSE_API_TOKEN",
    description="Description to be added on readme"
)
That’s it — you’re live on Agentverse!