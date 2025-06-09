"""
uAgent Adapters for Fetch.AI

This module contains uAgent adapters for converting LangGraph agents to uAgents
that can be deployed on the Agentverse ecosystem.
"""

from .data_loader_uagent import register_data_loader_uagent
from .data_cleaning_uagent import register_data_cleaning_uagent
from .data_visualisation_uagent import register_data_visualisation_uagent

__all__ = [
    "register_data_loader_uagent",
    "register_data_cleaning_uagent", 
    "register_data_visualisation_uagent"
] 