"""
AI Data Science Library

A toolkit for data science tasks powered by AI agents.
"""

__version__ = "0.1.0"

try:
    from ai_data_science.adapters.uagents_adapter import DataCleaningAgentAdapter
except ImportError:
    # uAgents adapter is optional
    pass 