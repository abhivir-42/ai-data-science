"""
Adapters for AI Data Science agents.

This package provides adapters to convert AI Data Science agents to 
work with other frameworks, such as Fetch.ai's uAgents system.
"""

from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter, DataLoaderToolsAgentAdapter

__all__ = ["DataCleaningAgentAdapter", "DataLoaderToolsAgentAdapter"] 