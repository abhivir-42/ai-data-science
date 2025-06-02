# Complete Guide to Data Loading and Cleaning Agents

## Table of Contents
1. [Introduction](#introduction)
2. [Individual Agent Usage](#individual-agent-usage)
3. [Combined Agent Workflow](#combined-agent-workflow)
4. [Fetch AI uAgents Integration](#fetch-ai-uagents-integration)
5. [Example Files Overview](#example-files-overview)
6. [Advanced Use Cases](#advanced-use-cases)
7. [Troubleshooting](#troubleshooting)

## Introduction

This guide covers the comprehensive usage of two powerful AI agents in the ai-data-science framework:

- **DataLoaderToolsAgent**: Loads data from various sources and file formats
- **DataCleaningAgent**: Cleans and preprocesses datasets using AI-generated functions

These agents can work independently or in combination to create powerful data processing pipelines. They can also be deployed as Fetch AI uAgents for distributed, autonomous operation.

## Individual Agent Usage

### DataLoaderToolsAgent

The DataLoaderToolsAgent is designed to load data from various sources including local files, directories, and remote locations.

#### Basic Usage

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ai_data_science.agents.data_loader_tools_agent import DataLoaderToolsAgent

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Create the agent
data_loader = DataLoaderToolsAgent(model=llm)

# Load data with natural language instructions
data_loader.invoke_agent(
    user_instructions="Load the CSV file from data/sample_data.csv and show me its structure"
)

# Get the loaded data as a DataFrame
df = data_loader.get_artifacts(as_dataframe=True)
print(f"Loaded data shape: {df.shape}")

# View the AI's response
print(data_loader.get_ai_message())

# See what tools were used
print("Tools used:", data_loader.get_tool_calls())
```

#### Available Tools

The DataLoaderToolsAgent has access to several file system tools:

- `load_file`: Load individual files (CSV, Excel, JSON, Parquet)
- `load_directory`: Load all compatible files from a directory
- `list_directory_contents`: List files in a directory
- `list_directory_recursive`: Recursively list files in subdirectories
- `get_file_info`: Get detailed information about a file
- `search_files_by_pattern`: Search for files matching a pattern

#### Advanced Usage

```python
# Load multiple files from a directory
data_loader.invoke_agent(
    user_instructions="""
    Load all CSV files from the data/ directory and combine them into a single dataset.
    Show me a summary of each file before combining.
    """
)

# Search for specific file patterns
data_loader.invoke_agent(
    user_instructions="Find all Excel files in the current directory that contain 'sales' in their name"
)
```

### DataCleaningAgent

The DataCleaningAgent automatically generates Python functions to clean datasets based on AI analysis and user instructions.

#### Basic Usage

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from ai_data_science.agents.data_cleaning_agent import DataCleaningAgent

# Initialize the agent
cleaning_agent = DataCleaningAgent(
    model=llm,
    log=True,  # Log the cleaning function to a file
    human_in_the_loop=False  # Set to True for interactive review
)

# Load your data
df = pd.read_csv("data/sample_data.csv")

# Clean the data with custom instructions
cleaning_agent.invoke_agent(
    data_raw=df,
    user_instructions="""
    Clean this dataset by:
    1. Filling missing numeric values with the median
    2. Filling missing categorical values with 'Unknown'
    3. Removing outliers beyond 2 standard deviations
    4. Converting price column to proper numeric format
    """
)

# Get results
cleaned_df = cleaning_agent.get_data_cleaned()
cleaning_function = cleaning_agent.get_data_cleaner_function()
steps = cleaning_agent.get_recommended_cleaning_steps()

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_df.shape}")
```

#### Default Cleaning Steps

When no custom instructions are provided, the agent performs these default steps:

1. Remove columns with more than 40% missing values
2. Impute missing numeric values with the mean
3. Impute missing categorical values with the mode
4. Convert columns to appropriate data types
5. Remove duplicate rows
6. Remove rows with missing values
7. Remove extreme outliers (3x interquartile range)

#### Human-in-the-Loop Mode

```python
# Enable human review of cleaning steps
cleaning_agent = DataCleaningAgent(
    model=llm,
    human_in_the_loop=True,
    checkpointer=MemorySaver()  # Required for human-in-the-loop
)

# The agent will pause for your review before executing
cleaning_agent.invoke_agent(data_raw=df, user_instructions="Clean this data for analysis")
```

## Combined Agent Workflow

The most powerful approach is to combine both agents in a sequential workflow where the data loader feeds directly into the data cleaner.

### Method 1: Sequential Execution

```python
from langchain_openai import ChatOpenAI
from ai_data_science.agents.data_loader_tools_agent import DataLoaderToolsAgent
from ai_data_science.agents.data_cleaning_agent import DataCleaningAgent

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Load data
data_loader = DataLoaderToolsAgent(model=llm)
data_loader.invoke_agent(
    user_instructions="Load the CSV file from data/sample_data.csv"
)

# Get the loaded data
raw_df = data_loader.get_artifacts(as_dataframe=True)

# Step 2: Clean the loaded data
cleaning_agent = DataCleaningAgent(model=llm, log=True)
cleaning_agent.invoke_agent(
    data_raw=raw_df,
    user_instructions="Clean this e-commerce dataset for machine learning analysis"
)

# Get final results
cleaned_df = cleaning_agent.get_data_cleaned()
print(f"Pipeline complete: {raw_df.shape} -> {cleaned_df.shape}")
```

### Method 2: Custom Pipeline Function

```python
def create_data_pipeline(file_path, cleaning_instructions=None):
    """
    Complete data loading and cleaning pipeline.
    
    Args:
        file_path: Path to the data file
        cleaning_instructions: Custom cleaning instructions (optional)
    
    Returns:
        dict: Contains original data, cleaned data, and metadata
    """
    # Initialize agents
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    data_loader = DataLoaderToolsAgent(model=llm)
    cleaning_agent = DataCleaningAgent(model=llm, log=True)
    
    # Load data
    data_loader.invoke_agent(f"Load the file from {file_path}")
    raw_df = data_loader.get_artifacts(as_dataframe=True)
    
    if raw_df is None or raw_df.empty:
        raise ValueError(f"Failed to load data from {file_path}")
    
    # Clean data
    cleaning_agent.invoke_agent(
        data_raw=raw_df,
        user_instructions=cleaning_instructions or "Apply best practices for data cleaning"
    )
    
    cleaned_df = cleaning_agent.get_data_cleaned()
    
    return {
        "original_data": raw_df,
        "cleaned_data": cleaned_df,
        "cleaning_function": cleaning_agent.get_data_cleaner_function(),
        "cleaning_steps": cleaning_agent.get_recommended_cleaning_steps(),
        "loader_response": data_loader.get_ai_message(),
        "metadata": {
            "original_shape": raw_df.shape,
            "cleaned_shape": cleaned_df.shape,
            "missing_values_removed": raw_df.isna().sum().sum() - cleaned_df.isna().sum().sum()
        }
    }

# Usage
pipeline_result = create_data_pipeline(
    "data/sample_data.csv",
    "Prepare this data for machine learning by handling missing values and outliers"
)
```

## Fetch AI uAgents Integration

### Overview

Both agents can be converted to Fetch AI uAgents for autonomous, distributed operation. This allows them to:

- Run independently on the Agentverse platform
- Communicate with other agents
- Be accessible via natural language chat interfaces
- Operate in decentralized multi-agent systems

### Architecture for uAgents Deployment

When deployed as uAgents, the system architecture looks like this:

```
User (Chat Interface)
     |
     v
Supervisor Agent (Optional)
     |
     +-- Data Loader uAgent
     |        |
     |        v
     +-- Data Cleaning uAgent
              |
              v
         Cleaned Data Result
```

### Do You Need a Supervisor Agent?

**Yes, a supervisor agent is recommended** for the following reasons:

1. **Orchestration**: Manages the workflow between loader and cleaner
2. **Error Handling**: Handles failures and retries
3. **State Management**: Tracks progress across agent interactions
4. **User Interface**: Provides a single entry point for users
5. **Data Flow**: Manages data transfer between agents

### uAgent Implementation

Here's how the system works when deployed as uAgents:

#### 1. Data Loader uAgent

```python
from uagents import Agent, Context, Model
from pydantic import Field
from ai_data_science.adapters.uagents_adapter import DataLoaderAgentAdapter

# Message models
class DataLoadRequest(Model):
    file_path: str = Field(description="Path to the file to load")
    instructions: str = Field(description="Loading instructions")

class DataLoadResponse(Model):
    data_dict: dict = Field(description="Loaded data in dictionary format")
    metadata: dict = Field(description="File metadata")
    success: bool = Field(description="Whether loading was successful")
    error: str = Field(default="", description="Error message if any")

# Create the uAgent
data_loader_agent = Agent(
    name="data_loader_agent",
    seed="data_loader_seed",
    endpoint=["http://localhost:8001/submit"]
)

# Initialize the AI agent
adapter = DataLoaderAgentAdapter(model=llm)

@data_loader_agent.on_message(model=DataLoadRequest)
async def handle_load_request(ctx: Context, sender: str, request: DataLoadRequest):
    try:
        # Use the adapter to load data
        result = adapter.load_data(request.file_path, request.instructions)
        
        response = DataLoadResponse(
            data_dict=result.get("data", {}),
            metadata=result.get("metadata", {}),
            success=True
        )
    except Exception as e:
        response = DataLoadResponse(
            data_dict={},
            metadata={},
            success=False,
            error=str(e)
        )
    
    await ctx.send(sender, response)
```

#### 2. Data Cleaning uAgent

```python
class DataCleanRequest(Model):
    data_dict: dict = Field(description="Data to clean in dictionary format")
    instructions: str = Field(description="Cleaning instructions")

class DataCleanResponse(Model):
    cleaned_data: dict = Field(description="Cleaned data")
    cleaning_function: str = Field(description="Generated cleaning function")
    success: bool = Field(description="Whether cleaning was successful")
    error: str = Field(default="", description="Error message if any")

data_cleaning_agent = Agent(
    name="data_cleaning_agent",
    seed="data_cleaning_seed",
    endpoint=["http://localhost:8002/submit"]
)

@data_cleaning_agent.on_message(model=DataCleanRequest)
async def handle_clean_request(ctx: Context, sender: str, request: DataCleanRequest):
    # Implementation similar to loader agent
    pass
```

#### 3. Supervisor Agent

```python
class ProcessDataRequest(Model):
    file_path: str = Field(description="Path to data file")
    cleaning_instructions: str = Field(description="How to clean the data")

class ProcessDataResponse(Model):
    original_data: dict = Field(description="Original loaded data")
    cleaned_data: dict = Field(description="Final cleaned data")
    summary: str = Field(description="Process summary")

supervisor_agent = Agent(
    name="data_supervisor",
    seed="supervisor_seed",
    endpoint=["http://localhost:8000/submit"]
)

@supervisor_agent.on_message(model=ProcessDataRequest)
async def handle_process_request(ctx: Context, sender: str, request: ProcessDataRequest):
    # Step 1: Request data loading
    load_request = DataLoadRequest(
        file_path=request.file_path,
        instructions=f"Load data from {request.file_path}"
    )
    
    load_response = await ctx.send(LOADER_AGENT_ADDRESS, load_request)
    
    if not load_response.success:
        await ctx.send(sender, ProcessDataResponse(
            original_data={},
            cleaned_data={},
            summary=f"Failed to load data: {load_response.error}"
        ))
        return
    
    # Step 2: Request data cleaning
    clean_request = DataCleanRequest(
        data_dict=load_response.data_dict,
        instructions=request.cleaning_instructions
    )
    
    clean_response = await ctx.send(CLEANER_AGENT_ADDRESS, clean_request)
    
    # Step 3: Send final response
    final_response = ProcessDataResponse(
        original_data=load_response.data_dict,
        cleaned_data=clean_response.cleaned_data,
        summary=f"Processed {request.file_path}: {len(load_response.data_dict)} -> {len(clean_response.cleaned_data)} records"
    )
    
    await ctx.send(sender, final_response)
```

### Chat Interface Integration

With Fetch AI's ASI1 LLM and chat mode, users can interact naturally:

```
User: "Please load the sales data from /data/sales_2024.csv and clean it for analysis"

Supervisor Agent: 
1. Parses the request
2. Sends file path to Data Loader Agent
3. Receives loaded data
4. Sends data + cleaning instructions to Cleaning Agent
5. Returns cleaned data and summary to user

User: "The data looks good, but can you also remove outliers?"

Supervisor Agent:
1. Recognizes this is a refinement request
2. Sends the already-loaded data to Cleaning Agent with new instructions
3. Returns updated results
```

### Deployment Steps

1. **Convert Agents to uAgents**: Use the adapter classes
2. **Deploy to Agentverse**: Register each agent
3. **Configure Communication**: Set up message routing
4. **Test Workflow**: Verify agent interactions
5. **Connect to Chat**: Integrate with ASI1 LLM interface

## Example Files Overview

### Current Example Files Analysis

| File | Purpose | Usefulness | Run Command |
|------|---------|------------|-------------|
| `minimal_cleaning_example.py` | Basic DataCleaningAgent usage | ⭐⭐⭐⭐⭐ Essential | `python examples/minimal_cleaning_example.py` |
| `uagents_adapter_example.py` | Mock uAgent adaptation demo | ⭐⭐⭐ Good for understanding | `python examples/uagents_adapter_example.py` |
| `real_uagents_adapter_example.py` | Real adapter implementation | ⭐⭐⭐⭐ Very useful | `python examples/real_uagents_adapter_example.py` |
| `uagents_interaction_example.py` | Agent communication demo | ⭐⭐⭐⭐ Very useful | `python examples/uagents_interaction_example.py <agent_address>` |
| `clean_churn_data.py` | Comprehensive cleaning example | ⭐⭐⭐⭐ Very useful | `python examples/clean_churn_data.py` |
| `direct_churn_cleaning.py` | Similar to above | ⭐⭐ Redundant | `python examples/direct_churn_cleaning.py` |
| `sample_data.csv` | Test dataset | ⭐⭐⭐⭐⭐ Essential | N/A (data file) |
| `data_cleaning_example.ipynb` | Jupyter notebook | ⭐ Redundant/large | Open in Jupyter |
| `test_uagents_adapters.py` | Unit tests | ⭐⭐⭐ Useful for developers | `python -m pytest examples/test_uagents_adapters.py` |
| `uagents_interaction_demo.py` | Extended demo | ⭐⭐ Similar to others | `python examples/uagents_interaction_demo.py` |

### Recommended New Examples

I'll create several new, focused examples and remove redundant ones:

1. **`combined_workflow_example.py`** - Shows both agents working together
2. **`data_loader_showcase.py`** - Demonstrates all data loading capabilities
3. **`supervisor_agent_example.py`** - Shows supervisor pattern
4. **`production_pipeline_example.py`** - Real-world usage scenario

## Advanced Use Cases

### 1. Multi-File Processing

```python
def process_multiple_files(file_patterns, output_dir):
    """Process multiple files through the pipeline."""
    results = {}
    
    for pattern in file_patterns:
        # Use data loader to find and load files
        data_loader.invoke_agent(f"Find and load all files matching {pattern}")
        files_data = data_loader.get_artifacts()
        
        for file_path, data in files_data.items():
            # Clean each file
            cleaning_agent.invoke_agent(
                data_raw=pd.DataFrame(data),
                user_instructions="Standardize for machine learning"
            )
            
            # Save results
            cleaned_data = cleaning_agent.get_data_cleaned()
            output_path = os.path.join(output_dir, f"cleaned_{os.path.basename(file_path)}")
            cleaned_data.to_csv(output_path, index=False)
            
            results[file_path] = output_path
    
    return results
```

### 2. Dynamic Cleaning Based on Data Type

```python
def intelligent_cleaning(df):
    """Apply different cleaning strategies based on data characteristics."""
    
    # Analyze data characteristics
    data_types = df.dtypes
    missing_percent = df.isnull().sum() / len(df)
    
    # Build dynamic instructions
    instructions = ["Clean this dataset with the following specific requirements:"]
    
    if (missing_percent > 0.3).any():
        instructions.append("- High missing data detected: use advanced imputation techniques")
    
    if df.select_dtypes(include=['object']).shape[1] > 5:
        instructions.append("- Many categorical columns: consider encoding strategies")
    
    if df.select_dtypes(include=[np.number]).shape[1] > 10:
        instructions.append("- Many numeric columns: check for multicollinearity")
    
    cleaning_agent.invoke_agent(
        data_raw=df,
        user_instructions="\n".join(instructions)
    )
    
    return cleaning_agent.get_data_cleaned()
```

### 3. Monitoring and Logging

```python
class DataPipelineMonitor:
    """Monitor and log data pipeline execution."""
    
    def __init__(self, log_file="pipeline.log"):
        self.log_file = log_file
        self.metrics = []
    
    def run_monitored_pipeline(self, file_path, cleaning_instructions):
        start_time = time.time()
        
        try:
            # Run pipeline with monitoring
            result = create_data_pipeline(file_path, cleaning_instructions)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            data_quality_score = self.calculate_quality_score(
                result["original_data"], 
                result["cleaned_data"]
            )
            
            # Log results
            self.log_execution(file_path, execution_time, data_quality_score, True)
            
            return result
            
        except Exception as e:
            self.log_execution(file_path, time.time() - start_time, 0, False, str(e))
            raise
    
    def calculate_quality_score(self, original, cleaned):
        """Calculate a data quality improvement score."""
        orig_missing = original.isnull().sum().sum()
        clean_missing = cleaned.isnull().sum().sum()
        
        missing_improvement = max(0, (orig_missing - clean_missing) / max(1, orig_missing))
        return missing_improvement * 100
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Data Loading Issues
- **Problem**: File not found or unsupported format
- **Solution**: Use `list_directory_contents` tool to verify file paths and formats

#### 2. Cleaning Function Errors
- **Problem**: Generated cleaning function fails to execute
- **Solution**: Enable retry mechanism in DataCleaningAgent (max_retries parameter)

#### 3. Memory Issues with Large Datasets
- **Problem**: Out of memory with large files
- **Solution**: Reduce `n_samples` parameter or process data in chunks

#### 4. uAgent Communication Failures
- **Problem**: Agents can't communicate
- **Solution**: Verify agent addresses and network connectivity

#### 5. API Key Issues
- **Problem**: Missing or invalid API keys
- **Solution**: Check .env file and API key validity

### Performance Optimization

1. **Reduce Token Usage**: Lower `n_samples` for large datasets
2. **Parallel Processing**: Run multiple agents concurrently
3. **Caching**: Cache cleaning functions for similar datasets
4. **Batch Processing**: Group similar files together

### Best Practices

1. **Always Validate Results**: Check cleaned data before using
2. **Version Control**: Save cleaning functions for reproducibility
3. **Monitor Performance**: Track execution times and quality metrics
4. **Error Handling**: Implement robust error handling and retries
5. **Documentation**: Log all cleaning steps and parameters used

This comprehensive guide should give you everything needed to effectively use both agents individually and together, whether in standalone mode or as part of a distributed uAgent system on Fetch AI's platform. 