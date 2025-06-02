"""
Supervisor Agent Example for Data Processing Pipeline

This example demonstrates how a supervisor agent would coordinate between
DataLoaderToolsAgent and DataCleaningAgent in a uAgent environment.

This is a conceptual example showing:
1. How a supervisor agent would orchestrate the workflow
2. Message passing between agents
3. Error handling and state management
4. How it would work with Fetch AI's ASI1 LLM chat interface

Note: This is a simulation/mockup of how the uAgent system would work.
For actual deployment, you would use the real uAgents framework.

Usage:
    python examples/supervisor_agent_example.py
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd

# Mock message models (in real implementation, these would be uAgents models)
@dataclass
class DataLoadRequest:
    """Request to load data from a file."""
    file_path: str
    instructions: str
    request_id: str
    timestamp: str

@dataclass
class DataLoadResponse:
    """Response from data loader agent."""
    data_dict: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    error: str = ""
    request_id: str = ""
    processing_time: float = 0.0

@dataclass
class DataCleanRequest:
    """Request to clean data."""
    data_dict: Dict[str, Any]
    instructions: str
    request_id: str
    timestamp: str

@dataclass
class DataCleanResponse:
    """Response from data cleaning agent."""
    cleaned_data: Dict[str, Any]
    cleaning_function: str
    cleaning_steps: str
    success: bool
    error: str = ""
    request_id: str = ""
    processing_time: float = 0.0

@dataclass
class ProcessDataRequest:
    """User request to process data through the full pipeline."""
    file_path: str
    cleaning_instructions: str
    user_id: str
    session_id: str

@dataclass
class ProcessDataResponse:
    """Final response to user."""
    original_data_summary: Dict[str, Any]
    cleaned_data_summary: Dict[str, Any]
    processing_summary: str
    cleaning_function: str
    success: bool
    error: str = ""
    total_processing_time: float = 0.0

class ProcessingState(Enum):
    """States of the data processing pipeline."""
    IDLE = "idle"
    LOADING = "loading"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    ERROR = "error"

class MockDataLoaderAgent:
    """Mock data loader agent that simulates the actual uAgent."""
    
    def __init__(self):
        self.name = "data_loader_agent"
        self.address = "agent1q2w3e4r5t6y7u8i9o0p1a2s3d4f5g6h7j8k9l0"
    
    async def handle_request(self, request: DataLoadRequest) -> DataLoadResponse:
        """Simulate processing a data load request."""
        start_time = time.time()
        
        try:
            print(f"🔄 [DataLoader] Processing request: {request.file_path}")
            
            # Simulate loading delay
            await asyncio.sleep(1)
            
            # Mock loading logic
            if not request.file_path.endswith('.csv'):
                raise ValueError(f"Unsupported file format for {request.file_path}")
            
            # Create mock data based on file path
            if "sample_data" in request.file_path:
                mock_data = {
                    'id': [1, 2, 3, 4, 5],
                    'name': ['Item 1', 'Item 2', None, 'Item 4', 'Item 5'],
                    'price': [10.99, None, 25.50, 8.99, None],
                    'category': ['A', 'B', 'A', None, 'C']
                }
            else:
                mock_data = {
                    'col1': [1, 2, 3],
                    'col2': ['a', 'b', 'c']
                }
            
            metadata = {
                "file_size": len(str(mock_data)),
                "columns": list(mock_data.keys()),
                "rows": len(mock_data[list(mock_data.keys())[0]]),
                "file_type": "csv"
            }
            
            processing_time = time.time() - start_time
            
            return DataLoadResponse(
                data_dict=mock_data,
                metadata=metadata,
                success=True,
                request_id=request.request_id,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return DataLoadResponse(
                data_dict={},
                metadata={},
                success=False,
                error=str(e),
                request_id=request.request_id,
                processing_time=processing_time
            )

class MockDataCleaningAgent:
    """Mock data cleaning agent that simulates the actual uAgent."""
    
    def __init__(self):
        self.name = "data_cleaning_agent"
        self.address = "agent1q3w4e5r6t7y8u9i0o1p2a3s4d5f6g7h8j9k0l1"
    
    async def handle_request(self, request: DataCleanRequest) -> DataCleanResponse:
        """Simulate processing a data cleaning request."""
        start_time = time.time()
        
        try:
            print(f"🧹 [DataCleaner] Processing cleaning request")
            
            # Simulate cleaning delay
            await asyncio.sleep(2)
            
            # Mock cleaning logic
            original_df = pd.DataFrame(request.data_dict)
            
            # Simple cleaning operations
            cleaned_df = original_df.copy()
            
            # Fill missing values
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            
            # Remove rows with missing names (if name column exists)
            if 'name' in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df['name'] != 'Unknown']
            
            # Mock cleaning function
            cleaning_function = f"""
def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    
    # Make a copy of the data
    data = data_raw.copy()
    
    # Fill missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('Unknown')
        else:
            data[col] = data[col].fillna(data[col].mean())
    
    # Remove rows with missing names
    if 'name' in data.columns:
        data = data[data['name'] != 'Unknown']
    
    return data
"""
            
            cleaning_steps = """
# Recommended Data Cleaning Steps:

1. Fill missing categorical values with 'Unknown'
2. Fill missing numerical values with column mean
3. Remove rows where essential fields (like name) are missing
4. Ensure data consistency and format
"""
            
            processing_time = time.time() - start_time
            
            return DataCleanResponse(
                cleaned_data=cleaned_df.to_dict(),
                cleaning_function=cleaning_function.strip(),
                cleaning_steps=cleaning_steps.strip(),
                success=True,
                request_id=request.request_id,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return DataCleanResponse(
                cleaned_data={},
                cleaning_function="",
                cleaning_steps="",
                success=False,
                error=str(e),
                request_id=request.request_id,
                processing_time=processing_time
            )

class SupervisorAgent:
    """
    Supervisor agent that orchestrates the data processing pipeline.
    
    This agent:
    1. Receives user requests for data processing
    2. Coordinates between data loader and cleaning agents
    3. Manages state and error handling
    4. Provides comprehensive responses to users
    """
    
    def __init__(self):
        self.name = "data_supervisor"
        self.address = "agent1q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6j7k8l9"
        self.state = ProcessingState.IDLE
        self.active_sessions: Dict[str, Dict] = {}
        
        # Initialize sub-agents
        self.data_loader = MockDataLoaderAgent()
        self.data_cleaner = MockDataCleaningAgent()
    
    async def process_data_request(self, request: ProcessDataRequest) -> ProcessDataResponse:
        """
        Main entry point for data processing requests.
        
        This method orchestrates the entire pipeline:
        1. Data loading
        2. Data cleaning
        3. Response generation
        """
        session_id = request.session_id
        start_time = time.time()
        
        # Initialize session tracking
        self.active_sessions[session_id] = {
            "state": ProcessingState.LOADING,
            "start_time": start_time,
            "request": request
        }
        
        try:
            print(f"🎯 [Supervisor] Starting data processing pipeline for session {session_id}")
            
            # Step 1: Load data
            load_response = await self._load_data(request)
            
            if not load_response.success:
                self.active_sessions[session_id]["state"] = ProcessingState.ERROR
                return ProcessDataResponse(
                    original_data_summary={},
                    cleaned_data_summary={},
                    processing_summary=f"Failed to load data: {load_response.error}",
                    cleaning_function="",
                    success=False,
                    error=load_response.error,
                    total_processing_time=time.time() - start_time
                )
            
            # Step 2: Clean data
            self.active_sessions[session_id]["state"] = ProcessingState.CLEANING
            clean_response = await self._clean_data(load_response.data_dict, request)
            
            if not clean_response.success:
                self.active_sessions[session_id]["state"] = ProcessingState.ERROR
                return ProcessDataResponse(
                    original_data_summary=self._summarize_data(load_response.data_dict),
                    cleaned_data_summary={},
                    processing_summary=f"Data loaded successfully but cleaning failed: {clean_response.error}",
                    cleaning_function="",
                    success=False,
                    error=clean_response.error,
                    total_processing_time=time.time() - start_time
                )
            
            # Step 3: Generate final response
            self.active_sessions[session_id]["state"] = ProcessingState.COMPLETED
            
            total_time = time.time() - start_time
            
            original_summary = self._summarize_data(load_response.data_dict)
            cleaned_summary = self._summarize_data(clean_response.cleaned_data)
            
            processing_summary = self._generate_processing_summary(
                original_summary, 
                cleaned_summary, 
                load_response.processing_time,
                clean_response.processing_time,
                total_time
            )
            
            print(f"✅ [Supervisor] Pipeline completed successfully in {total_time:.2f}s")
            
            return ProcessDataResponse(
                original_data_summary=original_summary,
                cleaned_data_summary=cleaned_summary,
                processing_summary=processing_summary,
                cleaning_function=clean_response.cleaning_function,
                success=True,
                total_processing_time=total_time
            )
            
        except Exception as e:
            self.active_sessions[session_id]["state"] = ProcessingState.ERROR
            return ProcessDataResponse(
                original_data_summary={},
                cleaned_data_summary={},
                processing_summary=f"Unexpected error in pipeline: {str(e)}",
                cleaning_function="",
                success=False,
                error=str(e),
                total_processing_time=time.time() - start_time
            )
        
        finally:
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def _load_data(self, request: ProcessDataRequest) -> DataLoadResponse:
        """Coordinate with the data loader agent."""
        load_request = DataLoadRequest(
            file_path=request.file_path,
            instructions=f"Load data from {request.file_path}",
            request_id=f"load_{request.session_id}",
            timestamp=datetime.now().isoformat()
        )
        
        return await self.data_loader.handle_request(load_request)
    
    async def _clean_data(self, data_dict: Dict, request: ProcessDataRequest) -> DataCleanResponse:
        """Coordinate with the data cleaning agent."""
        clean_request = DataCleanRequest(
            data_dict=data_dict,
            instructions=request.cleaning_instructions,
            request_id=f"clean_{request.session_id}",
            timestamp=datetime.now().isoformat()
        )
        
        return await self.data_cleaner.handle_request(clean_request)
    
    def _summarize_data(self, data_dict: Dict) -> Dict[str, Any]:
        """Generate a summary of dataset characteristics."""
        if not data_dict:
            return {}
        
        df = pd.DataFrame(data_dict)
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    
    def _generate_processing_summary(self, original_summary, cleaned_summary, 
                                   load_time, clean_time, total_time):
        """Generate a human-readable processing summary."""
        summary_parts = [
            "🎯 Data Processing Pipeline Summary",
            f"📊 Original data: {original_summary.get('rows', 0)} rows × {original_summary.get('columns', 0)} columns",
            f"✨ Cleaned data: {cleaned_summary.get('rows', 0)} rows × {cleaned_summary.get('columns', 0)} columns",
            "",
            "⏱️ Processing Times:",
            f"  • Data loading: {load_time:.2f}s",
            f"  • Data cleaning: {clean_time:.2f}s", 
            f"  • Total time: {total_time:.2f}s",
            "",
            "🔧 Changes Made:",
            f"  • Rows removed: {original_summary.get('rows', 0) - cleaned_summary.get('rows', 0)}",
            f"  • Missing values handled: {original_summary.get('missing_values', 0) - cleaned_summary.get('missing_values', 0)}",
            "",
            "✅ Pipeline completed successfully!"
        ]
        
        return "\n".join(summary_parts)

# Chat Interface Simulation
class ChatInterface:
    """
    Simulates how users would interact with the supervisor agent
    through Fetch AI's ASI1 LLM chat interface.
    """
    
    def __init__(self, supervisor: SupervisorAgent):
        self.supervisor = supervisor
        self.session_counter = 0
    
    async def process_user_message(self, user_message: str, user_id: str = "user123") -> str:
        """
        Process a natural language message from the user.
        
        In a real implementation, this would be handled by ASI1 LLM
        which would parse the message and generate appropriate requests.
        """
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        
        # Simple message parsing (in reality, ASI1 would do this)
        if "load" in user_message.lower() and "clean" in user_message.lower():
            # Extract file path and instructions from the message
            file_path = self._extract_file_path(user_message)
            cleaning_instructions = self._extract_cleaning_instructions(user_message)
            
            # Create request
            request = ProcessDataRequest(
                file_path=file_path,
                cleaning_instructions=cleaning_instructions,
                user_id=user_id,
                session_id=session_id
            )
            
            # Process through supervisor
            response = await self.supervisor.process_data_request(request)
            
            # Format response for user
            if response.success:
                return f"""✅ **Data Processing Completed!**

{response.processing_summary}

**Cleaning Function Generated:**
```python
{response.cleaning_function[:200]}...
```

Your data has been successfully loaded and cleaned! 
- Original: {response.original_data_summary.get('rows', 0)} rows
- Cleaned: {response.cleaned_data_summary.get('rows', 0)} rows
- Processing time: {response.total_processing_time:.2f} seconds

Would you like me to perform any additional analysis or modifications?"""
            else:
                return f"❌ **Error Processing Data**\n\n{response.error}\n\nPlease check your file path and try again."
        
        else:
            return "I can help you load and clean data files! Try saying something like: 'Please load the file data/sample.csv and clean it for analysis'"
    
    def _extract_file_path(self, message: str) -> str:
        """Extract file path from user message."""
        # Simple extraction - in reality, ASI1 would do sophisticated NLP
        words = message.split()
        for word in words:
            if '.' in word and ('csv' in word or 'json' in word or 'xlsx' in word):
                return word
        return "examples/sample_data.csv"  # Default
    
    def _extract_cleaning_instructions(self, message: str) -> str:
        """Extract cleaning instructions from user message."""
        # Simple extraction - in reality, ASI1 would understand intent
        if "machine learning" in message.lower():
            return "Prepare this data for machine learning by handling missing values, removing outliers, and ensuring data consistency"
        elif "analysis" in message.lower():
            return "Clean this data for analysis by filling missing values and standardizing formats"
        else:
            return "Apply best practices for data cleaning"

async def demonstrate_supervisor_workflow():
    """Demonstrate the complete supervisor agent workflow."""
    print("🤖 Supervisor Agent Demonstration")
    print("=" * 60)
    
    # Initialize supervisor and chat interface
    supervisor = SupervisorAgent()
    chat = ChatInterface(supervisor)
    
    # Simulate user interactions
    user_messages = [
        "Please load the file examples/sample_data.csv and clean it for analysis",
        "Load data/products.csv and prepare it for machine learning",
        "Can you process the file missing_file.csv?"  # Error case
    ]
    
    for i, message in enumerate(user_messages, 1):
        print(f"\n📋 User Message {i}: {message}")
        print("-" * 50)
        
        response = await chat.process_user_message(message)
        print(response)
    
    print("\n🎉 Supervisor agent demonstration completed!")

def main():
    """Run the supervisor agent example."""
    print("""
🎯 Supervisor Agent for Data Processing Pipeline

This example shows how a supervisor agent would coordinate between
DataLoaderToolsAgent and DataCleaningAgent in a Fetch AI uAgent environment.

Key concepts demonstrated:
• Agent orchestration and workflow management
• Message passing between specialized agents
• Error handling and state management
• Integration with chat interfaces (ASI1 LLM)
• Scalable multi-agent architecture

""")
    
    # Run the demonstration
    asyncio.run(demonstrate_supervisor_workflow())
    
    print("""
📚 Key Takeaways:

1. **Supervisor Pattern**: A coordinator agent manages the workflow between specialized agents
2. **Message-Based Communication**: Agents communicate through structured messages
3. **State Management**: The supervisor tracks session state and handles errors
4. **Chat Integration**: Natural language requests are parsed and routed appropriately
5. **Scalability**: This pattern can be extended to include more specialized agents

🚀 For production deployment:
• Convert mock agents to real uAgents using the framework
• Deploy to Agentverse platform
• Integrate with Fetch AI's ASI1 LLM for natural language processing
• Add authentication, logging, and monitoring
• Implement persistent state management
""")

if __name__ == "__main__":
    main() 