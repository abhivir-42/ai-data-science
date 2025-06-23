# Enhanced Data Analysis Agent with Structured Outputs - Implementation Plan

## Overview
Create a new `data_analysis_agent` that improves upon the current supervisor agent by implementing structured outputs and schema definitions using LangChain's structured output capabilities. This agent will orchestrate `data_cleaning_agent`, `feature_engineering_agent`, and `h2o_ml_agent` with better parameter handling and user intent parsing.

## ⚠️ CRITICAL ANALYSIS: Current Issues Identified

### 1. **Current Supervisor Agent Limitations** 
After analyzing `data_cleaning_agent.py`, `feature_engineering_agent.py`, and `h2o_ml_agent.py`:

**❌ Parameter Mapping Issues:**
- Current agent uses basic string parsing, ignoring complex parameter structures
- **Data Cleaning Agent** has 11+ parameters: `model`, `n_samples`, `log`, `log_path`, `file_name`, `function_name`, `overwrite`, `human_in_the_loop`, `bypass_recommended_steps`, `bypass_explain_code`, `checkpointer`
- **Feature Engineering Agent** has similar + `target_variable` support
- **H2O ML Agent** has 15+ parameters including MLflow integration: `enable_mlflow`, `mlflow_tracking_uri`, `mlflow_experiment_name`, `mlflow_run_name`, `model_directory`

**❌ Target Variable Detection:**
- Current regex-based parsing is unreliable and fragile
- No confidence scoring or validation
- Cannot handle ambiguous cases or multiple potential targets

**❌ Output Limitations:**
- Basic string concatenation instead of structured data
- Missing critical information like execution times, error handling, and metadata
- No integration with actual agent response structures

**❌ Workflow Orchestration:**
- Simple if/else logic instead of intelligent workflow planning
- No error recovery or partial failure handling
- No consideration of data dependencies between agents

### 2. **Required Improvements**

**✅ Enhanced Schema Design:**
- Map to actual agent parameter structures
- Provide intelligent defaults based on analysis
- Support all agent-specific features (MLflow, logging, human-in-the-loop)

**✅ Advanced Intent Parsing:**
- Use LLM-based structured output parsing
- Confidence scoring for target variable detection
- Intelligent parameter mapping based on user preferences

**✅ Comprehensive Output Structure:**
- Capture all agent outputs in structured format
- Include execution metadata, timing, and quality metrics
- Support visualization and reporting capabilities

**✅ Robust Workflow Management:**
- Error handling and recovery mechanisms
- Partial result preservation
- Performance optimization based on user preferences

## Key Improvements
1. **Structured Input Schema**: Define comprehensive input schemas for user requests
2. **Enhanced Parameter Extraction**: Use LLM-based parsing instead of regex
3. **Comprehensive Output Schema**: Structured outputs that capture all agent capabilities
4. **Better Workflow Orchestration**: Smart sequencing based on actual user needs
5. **Default Value Handling**: Intelligent defaults when user doesn't specify parameters

## Step-by-Step Implementation Plan

### Step 1: Schema Definition and Validation
**Files to create/modify:**
- `src/schemas/data_analysis_schemas.py` (new)

**Tasks:**
- Define Pydantic schemas for user input validation
- Create comprehensive parameter schemas for each agent
- Define structured output schemas for the final report
- Include default values and validation rules

**Schema Components:**
- `UserRequest`: Main input schema with CSV URL and natural language request
- `DataCleaningParams`: Parameters for data cleaning agent
- `FeatureEngineeringParams`: Parameters for feature engineering agent  
- `MLParams`: Parameters for H2O ML agent
- `AnalysisReport`: Comprehensive structured output schema

### Step 2: Enhanced User Intent Parser
**Files to create/modify:**
- `src/parsers/intent_parser.py` (new)

**Tasks:**
- Create LLM-based intent parser using structured outputs
- Replace regex-based parsing with AI-powered parameter extraction
- Implement smart target variable detection
- Add dataset type and problem type detection

### Step 3: Create Enhanced Data Analysis Agent
**Files to create/modify:**
- `src/agents/data_analysis_agent.py` (new)

**Tasks:**
- Implement new supervisor agent class with structured outputs
- Integrate schema validation and parameter extraction
- Create workflow orchestration logic based on parsed intent
- Implement comprehensive error handling and recovery

### Step 4: Agent Parameter Mapping
**Files to create/modify:**
- `src/utils/parameter_mapper.py` (new)

**Tasks:**
- Create utility functions to map parsed parameters to agent-specific parameters
- Handle default values and parameter validation
- Implement parameter transformation and normalization

### Step 5: Structured Report Generation
**Files to create/modify:**
- `src/templates/report_templates.py` (new)

**Tasks:**
- Create comprehensive report templates
- Implement structured output formatting
- Add markdown and JSON export capabilities
- Include visualizations and summaries

### Step 6: Enhanced Testing and Validation
**Files to create/modify:**
- `tests/test_data_analysis_agent.py` (new)
- `tests/test_intent_parser.py` (new)

**Tasks:**
- Create comprehensive test suite
- Add validation tests for schemas
- Test with multiple dataset types and request formats
- Validate structured outputs

### Step 7: Integration and Documentation
**Files to create/modify:**
- `examples/structured_data_analysis_example.py` (new)
- Update existing documentation

**Tasks:**
- Create working examples
- Update API documentation  
- Add usage guides and best practices

## Detailed Schema Design

Based on analysis of existing agents, here are comprehensive schemas that map to actual agent capabilities:

### Enhanced Input Schema
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Union, Any
from enum import Enum

class ProblemType(str, Enum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

class ImputationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"

class EncodingStrategy(str, Enum):
    ONEHOT = "onehot"
    LABEL = "label"
    TARGET = "target"
    ORDINAL = "ordinal"

class UserRequest(BaseModel):
    """Main user input schema with comprehensive validation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    csv_url: str = Field(
        description="URL to CSV file or path to local file",
        min_length=1
    )
    user_request: str = Field(
        description="Natural language analysis request",
        min_length=10,
        max_length=2000
    )
    target_variable: Optional[str] = Field(
        default=None,
        description="Target variable for ML (auto-detected if not provided)"
    )
    problem_type: ProblemType = Field(
        default=ProblemType.AUTO,
        description="Type of ML problem"
    )
    max_runtime: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Max runtime in seconds for entire analysis"
    )

class DataCleaningParams(BaseModel):
    """Enhanced data cleaning parameters based on actual agent capabilities"""
    model_config = ConfigDict(validate_assignment=True)
    
    # Core parameters that match DataCleaningAgent
    n_samples: int = Field(default=30, ge=10, le=1000, description="Samples for data summary")
    log: bool = Field(default=True, description="Enable logging")
    file_name: str = Field(default="data_cleaner.py", description="Function file name")
    function_name: str = Field(default="data_cleaner", description="Function name")
    overwrite: bool = Field(default=True, description="Overwrite existing files")
    human_in_the_loop: bool = Field(default=False, description="Enable human review")
    bypass_recommended_steps: bool = Field(default=False, description="Skip recommendation step")
    bypass_explain_code: bool = Field(default=False, description="Skip code explanation")
    
    # Cleaning-specific parameters
    remove_missing_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Threshold for removing columns with missing values"
    )
    imputation_strategy: ImputationStrategy = Field(
        default=ImputationStrategy.MEAN,
        description="Strategy for imputing missing values"
    )
    remove_outliers: bool = Field(default=True, description="Remove outliers using IQR method")
    remove_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    convert_data_types: bool = Field(default=True, description="Optimize data types")

class FeatureEngineeringParams(BaseModel):
    """Enhanced feature engineering parameters"""
    model_config = ConfigDict(validate_assignment=True)
    
    # Core parameters that match FeatureEngineeringAgent
    n_samples: int = Field(default=30, ge=10, le=1000)
    log: bool = Field(default=True)
    file_name: str = Field(default="feature_engineer.py")
    function_name: str = Field(default="feature_engineer")
    overwrite: bool = Field(default=True)
    human_in_the_loop: bool = Field(default=False)
    bypass_recommended_steps: bool = Field(default=False)
    bypass_explain_code: bool = Field(default=False)
    
    # Feature engineering specific
    encoding_strategy: EncodingStrategy = Field(default=EncodingStrategy.ONEHOT)
    scale_features: bool = Field(default=False, description="Scale numerical features")
    create_polynomial_features: bool = Field(default=False)
    feature_selection: bool = Field(default=False)
    high_cardinality_threshold: float = Field(
        default=0.05, ge=0.01, le=0.5,
        description="Threshold for high cardinality encoding"
    )
    remove_constant_features: bool = Field(default=True)
    create_datetime_features: bool = Field(default=True)

class MLParams(BaseModel):
    """H2O ML parameters based on actual agent"""
    model_config = ConfigDict(validate_assignment=True)
    
    # Core parameters matching H2OMLAgent
    n_samples: int = Field(default=30, ge=10, le=1000)
    log: bool = Field(default=True)
    file_name: str = Field(default="h2o_automl.py")
    function_name: str = Field(default="h2o_automl")
    overwrite: bool = Field(default=True)
    human_in_the_loop: bool = Field(default=False)
    bypass_recommended_steps: bool = Field(default=False)
    bypass_explain_code: bool = Field(default=False)
    
    # H2O-specific parameters
    max_models: int = Field(default=20, ge=1, le=100, description="Maximum models to train")
    max_runtime_secs: int = Field(default=300, ge=30, le=3600, description="AutoML runtime")
    nfolds: int = Field(default=5, ge=2, le=20, description="Cross-validation folds")
    seed: int = Field(default=42, ge=1, description="Random seed")
    stopping_metric: str = Field(default="AUTO", description="Stopping metric")
    stopping_tolerance: float = Field(default=0.001, gt=0.0, description="Stopping tolerance")
    stopping_rounds: int = Field(default=3, ge=1, description="Stopping rounds")
    sort_metric: str = Field(default="AUTO", description="Leaderboard sort metric")
    balance_classes: bool = Field(default=False, description="Balance class distribution")
    exclude_algos: List[str] = Field(default=["DeepLearning"], description="Algorithms to exclude")
    
    # MLflow integration
    enable_mlflow: bool = Field(default=False, description="Enable MLflow tracking")
    mlflow_tracking_uri: Optional[str] = Field(default=None)
    mlflow_experiment_name: str = Field(default="H2O AutoML")
    mlflow_run_name: Optional[str] = Field(default=None)

class AnalysisConfig(BaseModel):
    """Complete analysis configuration"""
    user_request: UserRequest
    data_cleaning: DataCleaningParams = Field(default_factory=DataCleaningParams)
    feature_engineering: FeatureEngineeringParams = Field(default_factory=FeatureEngineeringParams)
    ml_params: MLParams = Field(default_factory=MLParams)
```

### Comprehensive Output Schema
```python
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

class DataInfo(BaseModel):
    """Dataset information and statistics"""
    shape: tuple[int, int] = Field(description="Dataset dimensions (rows, columns)")
    columns: List[str] = Field(description="Column names")
    dtypes: Dict[str, str] = Field(description="Data types by column")
    missing_values: Dict[str, int] = Field(description="Missing values by column")
    memory_usage: str = Field(description="Memory usage")
    sample_data: Dict[str, Any] = Field(description="Sample rows")

class CleaningResults(BaseModel):
    """Data cleaning results and statistics"""
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    data_loss_percentage: float = Field(ge=0.0, le=100.0)
    columns_removed: List[str] = Field(default_factory=list)
    rows_removed: int = Field(ge=0)
    missing_values_before: int
    missing_values_after: int
    duplicates_removed: int = Field(ge=0)
    outliers_removed: int = Field(ge=0)
    data_types_converted: Dict[str, str] = Field(default_factory=dict)
    cleaning_function_path: Optional[str] = None
    execution_time: float = Field(ge=0.0, description="Cleaning time in seconds")

class FeatureResults(BaseModel):
    """Feature engineering results"""
    original_features: int
    engineered_features: int
    features_added: List[str] = Field(default_factory=list)
    features_removed: List[str] = Field(default_factory=list)
    encoding_applied: Dict[str, str] = Field(default_factory=dict)
    scaling_applied: bool = False
    polynomial_features_created: bool = False
    datetime_features_created: List[str] = Field(default_factory=list)
    feature_function_path: Optional[str] = None
    execution_time: float = Field(ge=0.0)

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    metric_name: str
    metric_value: float
    
class MLResults(BaseModel):
    """Machine learning results from H2O"""
    leaderboard: List[Dict[str, Any]] = Field(description="H2O AutoML leaderboard")
    best_model_id: str = Field(description="ID of best performing model")
    best_model_metrics: Dict[str, float] = Field(description="Best model metrics")
    model_path: Optional[str] = Field(description="Saved model path")
    training_time: float = Field(ge=0.0, description="Training time in seconds")
    cross_validation_metrics: Dict[str, float] = Field(default_factory=dict)
    feature_importance: List[Dict[str, Union[str, float]]] = Field(default_factory=list)
    model_explanation: str = Field(description="Model interpretation")
    ml_function_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None

class WorkflowStep(BaseModel):
    """Individual workflow step result"""
    step_name: str
    status: str = Field(description="success/failed/skipped")
    execution_time: float = Field(ge=0.0)
    error_message: Optional[str] = None
    output_summary: str

class AnalysisReport(BaseModel):
    """Comprehensive structured analysis report"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    # Metadata
    analysis_id: str = Field(description="Unique analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    total_execution_time: float = Field(ge=0.0, description="Total analysis time in seconds")
    
    # Input summary
    request_summary: str = Field(description="Summary of user request")
    dataset_url: str = Field(description="Source dataset URL")
    target_variable: Optional[str] = None
    problem_type: str = Field(description="Detected or specified problem type")
    
    # Results
    data_info: DataInfo
    cleaning_results: CleaningResults
    feature_engineering_results: FeatureResults
    ml_results: MLResults
    
    # Workflow information
    workflow_steps: List[WorkflowStep] = Field(description="Detailed workflow execution")
    
    # Analysis outputs
    executive_summary: str = Field(description="High-level analysis summary")
    key_insights: List[str] = Field(description="Important findings from the analysis")
    recommendations: List[str] = Field(description="Actionable recommendations")
    limitations: List[str] = Field(description="Analysis limitations and caveats")
    
    # File outputs
    generated_files: Dict[str, str] = Field(
        description="Generated files (type -> path mapping)",
        default_factory=dict
    )
    visualizations: List[Dict[str, str]] = Field(
        description="Generated visualizations with descriptions",
        default_factory=list
    )
    
    # Quality metrics
    data_quality_score: float = Field(ge=0.0, le=100.0, description="Overall data quality score")
    model_confidence: float = Field(ge=0.0, le=100.0, description="Model reliability score")
    analysis_reliability: str = Field(description="HIGH/MEDIUM/LOW reliability assessment")
```

## Success Criteria
1. **Robust Input Handling**: Handle various input formats and provide helpful error messages
2. **Accurate Parameter Extraction**: 95%+ accuracy in extracting user intent and parameters
3. **Comprehensive Outputs**: Structured reports that showcase all agent capabilities
4. **Better Performance**: Faster processing through optimized workflows
5. **Maintainable Code**: Well-documented, modular, and testable implementation

## Implementation Timeline
- **Step 1-2**: Schema and parser development (Foundation)
- **Step 3-4**: Core agent and parameter mapping (Core functionality)  
- **Step 5-6**: Report generation and testing (Quality assurance)
- **Step 7**: Integration and documentation (Completion)

## Technical Considerations
- Use Pydantic v2 for schema validation
- Leverage LangChain's `with_structured_output()` method
- Implement proper error handling and recovery
- Ensure backward compatibility where possible
- Add comprehensive logging and debugging capabilities

This implementation will create a significantly more robust and capable data analysis agent that leverages the full potential of the underlying specialized agents while providing a much better user experience through structured inputs and outputs. 