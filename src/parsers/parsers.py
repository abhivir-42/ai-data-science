"""
Output parsers for AI Data Science.
"""

import re
from langchain_core.output_parsers import BaseOutputParser

class PythonOutputParser(BaseOutputParser):
    """
    Parser for extracting Python code from LLM output.
    
    This parser extracts code blocks marked with ```python and ``` from the LLM output.
    If no code blocks are found, it tries to extract any code blocks or just returns the raw text.
    """
    
    def parse(self, text: str) -> str:
        """
        Parse the output text to extract Python code.
        
        Parameters
        ----------
        text : str
            The output text from an LLM.
            
        Returns
        -------
        str
            The extracted Python code.
        """
        # Try to find Python code blocks
        python_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if python_match:
            return python_match.group(1).strip()
            
        # Try to find any code blocks
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        # Return the text as is if no code blocks are found
        return text.strip()


class SQLOutputParser(BaseOutputParser):
    """
    Parser for extracting SQL code from LLM output.
    
    This parser extracts SQL code blocks marked with ```sql and ``` from the LLM output.
    """
    
    def parse(self, text: str) -> str:
        """
        Parse the output text to extract SQL code.
        
        Parameters
        ----------
        text : str
            The output text from an LLM.
            
        Returns
        -------
        str
            The extracted SQL code.
        """
        # Try to find SQL code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
            
        # Try to find SQLQuery pattern
        sqlquery_match = re.search(r"SQLQuery:\s*(.*)", text)
        if sqlquery_match:
            return sqlquery_match.group(1).strip()
            
        # Try to find any code blocks
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        # Return the text as is if no code blocks are found
        return text.strip() 