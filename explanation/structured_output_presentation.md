# Data Analysis Agent: Structured Outputs & Orchestration Architecture

## Executive Summary

The Data Analysis Agent (`data_analysis_agent.py`) implements a sophisticated multi-agent orchestration system using **structured outputs with Pydantic schemas**, **intelligent parameter mapping**, and **LangGraph-inspired workflow management**. The system takes unstructured user requests, parses them into structured workflows, maps parameters intelligently across agents, and returns comprehensive structured results. It's essentially a "smart supervisor" that coordinates data cleaning, feature engineering, and ML modeling agents while maintaining full type safety and validation throughout the pipeline.

---

## Presentation Guide

### 1. **Schema-Driven Architecture Overview** (Show: `src/schemas/data_analysis_schemas.py`)

**What to show:** Lines 29-87 (`DataAnalysisRequest` class)
**What to say:** 
"This is our input schema - notice how we use Pydantic to define exactly what parameters the system accepts, with validation rules, defaults, and documentation. This isn't just documentation - it's executable validation that ensures data quality from the start."

**Key lines to highlight:**
- Line 34: `csv_url: str = Field(...)` - Structured input with validation
- Line 48: `problem_type: Optional[ProblemType] = Field(default=ProblemType.AUTO)` - Enum-based type safety
- Line 58: `missing_threshold: Optional[float] = Field(default=0.4, ge=0.0, le=1.0)` - Built-in validation rules

### 2. **LLM-Powered Intent Parsing** (Show: `src/parsers/intent_parser.py`)

**What to show:** Lines 82-86 (`_create_prompt_template` method)
**What to say:**
"Here's where we use LLM structured outputs - we're not just getting text back, we're getting validated Pydantic objects. The LLM analyzes user requests and returns structured workflow requirements that our system can act on programmatically."

**Key lines to highlight:**
- Line 54: `self.output_parser = PydanticOutputParser(pydantic_object=WorkflowIntent)` - Structured output parser
- Line 61: `self.chain = self.prompt_template | self.llm | self.output_parser` - LangChain pipeline with structured parsing

### 3. **Intent-to-Workflow Mapping** (Show: `src/schemas/data_analysis_schemas.py`)

**What to show:** Lines 125-183 (`WorkflowIntent` class)
**What to say:**
"This is our intelligent workflow intent schema - it captures not just what the user wants, but the AI's analysis of their request, including confidence scores and suggested parameters. Notice how we separate the 'what' from the 'how'."

**Key lines to highlight:**
- Line 130: `needs_data_cleaning: bool` - Boolean workflow flags
- Line 152: `suggested_target_variable: Optional[str]` - AI-suggested parameters
- Line 176: `intent_confidence: float = Field(ge=0.0, le=1.0)` - Confidence scoring

### 4. **Intelligent Parameter Mapping** (Show: `src/mappers/parameter_mapper.py`)

**What to show:** Lines 42-85 (`map_data_cleaning_parameters` method)
**What to say:**
"This is where we translate user intentions into agent-specific parameters. Each agent has different parameter requirements, so we intelligently map from our standardized request schema to what each agent actually needs."

**Key lines to highlight:**
- Line 79: `"user_instructions": self._create_cleaning_instructions(request, intent, csv_url)` - Dynamic instruction generation
- Line 161: `params = self.parameter_mapper.map_data_cleaning_parameters(request, intent, data_path)` - Parameter mapping in action

### 5. **Orchestration Engine** (Show: `src/agents/data_analysis_agent.py`)

**What to show:** Lines 248-285 (`_execute_workflow_sync` method)
**What to say:**
"This is our orchestration engine - it's like a smart conductor that decides which agents to run based on the parsed intent, manages data flow between agents, and handles error cases gracefully. Notice how each agent's output becomes the next agent's input."

**Key lines to highlight:**
- Line 256: `if intent.needs_data_cleaning:` - Intent-driven execution
- Line 260: `current_data_path = result.output_data_path` - Data pipeline management
- Line 267: `intent.suggested_target_variable or request.target_variable` - Parameter cascading

### 6. **Structured Results Generation** (Show: `src/agents/data_analysis_agent.py`)

**What to show:** Lines 576-638 (`_generate_result` method)
**What to say:**
"Finally, we generate comprehensive structured results - not just 'it worked' or 'it failed', but detailed metrics, insights, file paths, confidence levels, and actionable recommendations. This makes the system not just automated, but truly intelligent."

**Key lines to highlight:**
- Line 616: `return DataAnalysisResult(...)` - Comprehensive structured output
- Line 622: `workflow_intent=intent` - Full workflow traceability
- Line 625: `agent_results=agent_results` - Detailed execution results

### 7. **End-to-End Example** (Show: `src/agents/data_analysis_agent.py`)

**What to show:** Lines 812-883 (`analyze_from_text` method)
**What to say:**
"Here's how it all comes together - a single method that takes natural language input, extracts dataset URLs, parses intent, creates structured requests, executes the workflow, and returns comprehensive results. It's like having a data scientist that never gets tired and always documents everything perfectly."

**Key lines to highlight:**
- Line 834: `url_extraction = self.intent_parser.extract_dataset_url_from_text(text_input)` - URL extraction
- Line 847: `intent = self.intent_parser.parse_with_data_preview(text_input, csv_url)` - Intent parsing with data preview
- Line 860: `request = DataAnalysisRequest(...)` - Structured request creation
- Line 875: `agent_results = self._execute_workflow_sync(request, intent)` - Workflow execution

---

## Key Benefits Demonstrated

1. **Type Safety**: Pydantic schemas ensure data validation at every step
2. **Intelligent Orchestration**: LLM-powered intent parsing drives workflow decisions
3. **Parameter Translation**: Smart mapping between user preferences and agent requirements
4. **Comprehensive Tracking**: Full execution traceability and structured results
5. **Error Resilience**: Graceful handling of failures with detailed error reporting
6. **Extensibility**: Easy to add new agents or modify workflows

This architecture transforms unstructured user requests into reliable, traceable, and comprehensive data analysis workflows while maintaining full visibility into the process. 