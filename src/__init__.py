"""
AI Data Science Library

A toolkit for data science tasks powered by AI agents.
"""

__version__ = "0.1.0"

# Import main modules
from src import agents, tools, utils, parsers, templates
from src import ds_agents, multiagents

try:
    from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter
except ImportError:
    # uAgents adapter is optional
    pass

__all__ = [
    "agents",
    "tools", 
    "utils",
    "parsers",
    "templates",
    "ds_agents",
    "multiagents"
] 