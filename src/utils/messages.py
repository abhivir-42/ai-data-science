"""
Message utilities for AI Data Science agents.

This module provides utilities for working with messages between agents.
"""

from typing import List, Any, Sequence


def get_tool_call_names(messages: Sequence[Any]) -> List[str]:
    """
    Extract tool call names from a sequence of messages.

    This function parses the messages to identify tool calls and returns a list
    of the tool names that were called.

    Parameters
    ----------
    messages : Sequence[Any]
        A sequence of messages to parse for tool calls.

    Returns
    -------
    List[str]
        A list of tool names that were called.
    """
    tool_calls = []
    
    for message in messages:
        # Handle different message formats
        
        # LangChain BaseMessage with tool_calls in additional_kwargs
        if hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
            for tool_call in message.additional_kwargs["tool_calls"]:
                if "name" in tool_call:
                    tool_calls.append(tool_call["name"])
                elif "function" in tool_call and "name" in tool_call["function"]:
                    tool_calls.append(tool_call["function"]["name"])
        
        # Tool message with tool property
        elif hasattr(message, "tool") and message.tool:
            tool_calls.append(message.tool)
        
        # Dict messages with tool_calls
        elif isinstance(message, dict):
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if "name" in tool_call:
                        tool_calls.append(tool_call["name"])
                    elif "function" in tool_call and "name" in tool_call["function"]:
                        tool_calls.append(tool_call["function"]["name"])
            elif "tool" in message:
                tool_calls.append(message["tool"])
                
    return tool_calls 