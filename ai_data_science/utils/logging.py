"""
Logging utilities for AI Data Science.
"""

import os
import datetime
import uuid

def log_ai_function(response: str, file_name: str = None, log: bool = False, log_path: str = None, overwrite: bool = True) -> tuple:
    """
    Log an AI-generated function to a file.
    
    Parameters
    ----------
    response : str
        The function code to log.
    file_name : str, optional
        The name of the file to save the code to.
    log : bool, optional
        Whether to log the code. If False, returns empty strings.
    log_path : str, optional
        The path to the directory to save the log file in.
    overwrite : bool, optional
        Whether to overwrite an existing file with the same name.
        
    Returns
    -------
    tuple
        A tuple containing the file path and file name.
    """
    if not log:
        return "", ""
        
    if not log_path:
        log_path = "logs"
        
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    if not file_name:
        file_name = f"ai_function_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    elif not file_name.endswith('.py'):
        file_name += '.py'
        
    file_path = os.path.join(log_path, file_name)
    
    # If not overwriting and file exists, create a unique name
    if not overwrite and os.path.exists(file_path):
        base_name, ext = os.path.splitext(file_name)
        unique_id = str(uuid.uuid4())[:8]
        file_name = f"{base_name}_{unique_id}{ext}"
        file_path = os.path.join(log_path, file_name)
    
    with open(file_path, 'w') as f:
        f.write(response)
        
    return file_path, file_name 