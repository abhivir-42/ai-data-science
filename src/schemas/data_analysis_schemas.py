"""
Comprehensive Pydantic schemas for structured data analysis workflows.

This module defines all the data models used for input validation, workflow orchestration,
and structured output generation in the enhanced data analysis agent.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid


class ProblemType(str, Enum):
    """Enumeration of supported ML problem types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"


class ModelType(str, Enum):
    """Enumeration of supported H2O model types"""
    GBM = "GBM"
    RANDOM_FOREST = "RandomForest"
    GLM = "GLM"
    DEEP_LEARNING = "DeepLearning"
    XG_BOOST = "XGBoost"
    AUTO_ML = "AutoML"


class DataAnalysisRequest(BaseModel):
    """Complete structured input for data analysis workflows"""
    
    # Core Requirements
    csv_url: str = Field(
        description="URL to CSV file for analysis",
        example="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    user_request: str = Field(
        description="Natural language analysis request",
        min_length=10,
        example="Clean this dataset and build a machine learning model to predict survival"
    )
    
    # Advanced Configuration
    target_variable: Optional[str] = Field(
        default=None,
        description="Target variable for ML modeling (if known)",
        example="Survived"
    )
    problem_type: Optional[ProblemType] = Field(
        default=ProblemType.AUTO,
        description="Type of ML problem to solve"
    )
    max_runtime_seconds: Optional[int] = Field(
        default=300,
        ge=30,  # Reduced from 60 to 30 for performance optimization
        le=1800,
        description="Maximum runtime per agent (30-1800 seconds, optimized for dataset size)"
    )
    
    # Data Cleaning Preferences
    missing_threshold: Optional[float] = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Threshold for removing columns with missing values (0.0-1.0)"
    )
    outlier_detection: Optional[bool] = Field(
        default=True,
        description="Enable outlier detection and removal"
    )
    duplicate_removal: Optional[bool] = Field(
        default=True,
        description="Enable duplicate row removal"
    )
    
    # Feature Engineering Preferences  
    feature_selection: Optional[bool] = Field(
        default=True,
        description="Enable automatic feature selection"
    )
    datetime_features: Optional[bool] = Field(
        default=True,
        description="Generate datetime-based features if datetime columns exist"
    )
    categorical_encoding: Optional[bool] = Field(
        default=True,
        description="Enable automatic categorical variable encoding"
    )
    
    # ML Modeling Preferences
    enable_mlflow: Optional[bool] = Field(
        default=True,
        description="Enable MLflow experiment tracking"
    )
    model_types: Optional[List[ModelType]] = Field(
        default=[ModelType.GBM, ModelType.RANDOM_FOREST, ModelType.GLM],
        description="H2O model types to try"
    )
    cross_validation_folds: Optional[int] = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of cross-validation folds (3-10)"
    )
    
    @field_validator('csv_url')
    @classmethod
    def validate_csv_url(cls, v):
        """Validate that the URL looks reasonable"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('CSV URL must start with http:// or https://')
        if not v.lower().endswith('.csv'):
            raise ValueError('URL must point to a CSV file (.csv extension)')
        return v
    
    @field_validator('model_types')
    @classmethod
    def validate_model_types(cls, v):
        """Ensure at least one model type is specified"""
        if v and len(v) == 0:
            raise ValueError('At least one model type must be specified')
        return v


class WorkflowIntent(BaseModel):
    """LLM-parsed workflow requirements with intelligent analysis"""
    
    # Core Workflow Needs
    needs_data_cleaning: bool = Field(
        description="Requires data cleaning/preprocessing"
    )
    needs_feature_engineering: bool = Field(
        description="Requires feature engineering"
    )
    needs_ml_modeling: bool = Field(
        description="Requires ML model training"
    )
    
    # Intelligent Analysis Focus Areas
    data_quality_focus: bool = Field(
        description="Primary focus on data quality issues"
    )
    exploratory_analysis: bool = Field(
        description="Needs exploratory data analysis"
    )
    prediction_focus: bool = Field(
        description="Primary goal is prediction/modeling"
    )
    statistical_analysis: bool = Field(
        description="Needs statistical analysis and insights"
    )
    
    # AI-Extracted Information
    suggested_target_variable: Optional[str] = Field(
        default=None,
        description="AI-suggested target variable based on request and data"
    )
    suggested_problem_type: Optional[ProblemType] = Field(
        default=None,
        description="AI-suggested problem type (classification/regression)"
    )
    key_requirements: List[str] = Field(
        description="Key requirements extracted from user request"
    )
    complexity_level: Literal["simple", "moderate", "complex"] = Field(
        description="Assessed complexity level of the analysis request"
    )
    
    # Confidence Scores
    intent_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for intent parsing (0.0-1.0)"
    )
    target_variable_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for target variable suggestion"
    )
    
    # NEW: Prediction-specific intent fields
    needs_prediction: bool = Field(
        default=False,
        description="Request requires making predictions with existing model"
    )
    needs_model_analysis: bool = Field(
        default=False,
        description="Request requires analyzing existing model results/insights"
    )
    
    # NEW: Prediction details
    prediction_data_source: Optional[str] = Field(
        default=None,
        description="Data source for prediction (CSV URL, inline data, etc.)"
    )
    prediction_type: Optional[Literal["single_prediction", "batch_prediction", "model_analysis"]] = Field(
        default=None,
        description="Type of prediction request"
    )
    extracted_prediction_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted prediction input data from user request"
    )


class DataQualityMetrics(BaseModel):
    """Metrics from data cleaning operations"""
    
    original_shape: Dict[str, int] = Field(description="Original dataset dimensions")
    cleaned_shape: Dict[str, int] = Field(description="Cleaned dataset dimensions")
    
    missing_values_handled: int = Field(description="Number of missing values handled")
    duplicates_removed: int = Field(description="Number of duplicate rows removed")
    outliers_detected: int = Field(description="Number of outliers detected")
    outliers_removed: int = Field(description="Number of outliers removed")
    
    columns_dropped: List[str] = Field(description="List of columns that were dropped")
    data_types_converted: Dict[str, str] = Field(description="Data type conversions performed")
    
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall data quality score after cleaning"
    )


class FeatureEngineeringMetrics(BaseModel):
    """Metrics from feature engineering operations"""
    
    original_features: int = Field(description="Number of original features")
    engineered_features: int = Field(description="Number of features after engineering")
    
    new_features_created: List[str] = Field(description="List of newly created features")
    features_selected: List[str] = Field(description="List of selected features")
    features_dropped: List[str] = Field(description="List of dropped features")
    
    categorical_features_encoded: int = Field(description="Number of categorical features encoded")
    datetime_features_created: int = Field(description="Number of datetime features created")
    
    feature_importance_available: bool = Field(description="Whether feature importance scores are available")
    target_correlation_computed: bool = Field(description="Whether target correlation was computed")


class MLModelingMetrics(BaseModel):
    """Comprehensive ML modeling metrics from H2O AutoML."""
    
    # Core Model Metrics
    models_trained: int = Field(description="Number of models trained")
    best_model_type: Optional[str] = Field(description="Type of the best performing model")
    best_model_id: Optional[str] = Field(description="ID of the best performing model")
    
    # Performance Metrics
    best_model_score: Optional[float] = Field(description="Best model's primary metric score")
    cross_validation_score: Optional[float] = Field(description="Cross-validation score")
    test_set_score: Optional[float] = Field(description="Test set score if available")
    
    # Model Details
    training_time_seconds: float = Field(description="Total training time in seconds")
    model_size_mb: Optional[float] = Field(description="Model size in megabytes")
    
    # Feature Information
    features_used: List[str] = Field(description="List of features used in the final model")
    feature_importance: Optional[Dict[str, float]] = Field(description="Feature importance scores")
    
    # Experiment Tracking
    mlflow_experiment_id: Optional[str] = Field(description="MLflow experiment ID if tracking enabled")
    mlflow_run_id: Optional[str] = Field(description="MLflow run ID if tracking enabled")
    
    # NEW: Enhanced fields for Phase 1 - Rich ML Results Display
    model_path: Optional[str] = Field(default=None, description="Path to saved model file")
    leaderboard: Optional[List[Dict[str, Any]]] = Field(default=None, description="H2O AutoML leaderboard with all trained models")
    top_model_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Detailed metrics for the top performing model")
    total_models_trained: Optional[int] = Field(default=None, description="Total number of models trained in AutoML")
    training_runtime: Optional[float] = Field(default=None, description="Total training runtime in seconds")
    generated_code: Optional[str] = Field(default=None, description="AI-generated H2O training code")
    recommended_steps: Optional[str] = Field(default=None, description="AI-recommended ML methodology steps")
    workflow_summary: Optional[str] = Field(default=None, description="Summary of the ML workflow executed")
    model_architecture: Optional[str] = Field(default=None, description="Architecture of the best model")
    enhanced_feature_importance: Optional[List[Dict[str, Any]]] = Field(default=None, description="Enhanced feature importance analysis results")


class AgentExecutionResult(BaseModel):
    """Result from executing a single agent"""
    
    agent_name: str = Field(description="Name of the executed agent")
    execution_time_seconds: float = Field(description="Agent execution time")
    success: bool = Field(description="Whether the agent executed successfully")
    
    # Agent-specific metrics
    data_quality_metrics: Optional[DataQualityMetrics] = Field(default=None, description="Data cleaning metrics")
    feature_engineering_metrics: Optional[FeatureEngineeringMetrics] = Field(default=None, description="Feature engineering metrics")
    ml_modeling_metrics: Optional[MLModelingMetrics] = Field(default=None, description="ML modeling metrics")
    
    # Outputs
    output_data_path: Optional[str] = Field(default=None, description="Path to output data file")
    model_path: Optional[str] = Field(default=None, description="Path to saved model")
    artifacts_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to generated artifacts")
    
    # Logs and Messages
    log_messages: List[str] = Field(default_factory=list, description="Log messages from agent execution")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class DataAnalysisResult(BaseModel):
    """Complete structured output from data analysis workflow"""
    
    # Metadata
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique request identifier"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Processing timestamp"
    )
    total_runtime_seconds: float = Field(description="Total processing time")
    
    # Input Summary
    original_request: str = Field(description="Original user request")
    csv_url: str = Field(description="Source CSV URL")
    data_shape: Dict[str, Any] = Field(description="Original data dimensions")
    
    # Workflow Execution
    workflow_intent: WorkflowIntent = Field(description="Parsed workflow requirements")
    agents_executed: List[str] = Field(description="List of agents that were executed")
    
    # Detailed Agent Results
    agent_results: List[AgentExecutionResult] = Field(description="Detailed results from each agent")
    
    # Summary Metrics
    overall_data_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall data quality score"
    )
    feature_engineering_effectiveness: Optional[float] = Field(
        ge=0.0,
        le=1.0,
        description="Feature engineering effectiveness score"
    )
    model_performance_score: Optional[float] = Field(
        ge=0.0,
        le=1.0,
        description="ML model performance score"
    )
    
    # File Outputs
    generated_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated file paths by type"
    )
    
    # AI-Generated Insights
    key_insights: List[str] = Field(description="Key insights from the analysis")
    recommendations: List[str] = Field(description="Recommendations for next steps")
    data_story: str = Field(description="AI-generated narrative of the analysis")
    
    # Quality Assessment
    analysis_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall analysis quality score (0-1)"
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        description="Confidence level in the analysis results"
    )
    
    # Potential Issues
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about data quality or analysis limitations"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations of the analysis"
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_consistency(cls, values):
        """Validate consistency across different fields"""
        if isinstance(values, dict):
            workflow_intent = values.get('workflow_intent')
            agents_executed = values.get('agents_executed', [])
            
            if workflow_intent:
                # Check that executed agents match workflow intent
                if (hasattr(workflow_intent, 'needs_data_cleaning') and 
                    workflow_intent.needs_data_cleaning and 'data_cleaning' not in agents_executed):
                    values.setdefault('warnings', []).append(
                        "Data cleaning was needed but not executed"
                    )
                
                if (hasattr(workflow_intent, 'needs_ml_modeling') and 
                    workflow_intent.needs_ml_modeling and 'h2o_ml' not in agents_executed):
                    values.setdefault('warnings', []).append(
                        "ML modeling was needed but not executed"
                    )
        
        return values
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Don't allow extra fields


class DatasetExtractionRequest(BaseModel):
    """Schema for extracting dataset information from text using LLM structured outputs"""
    
    extracted_csv_url: str = Field(
        description="Extracted or detected CSV URL from the user's text input"
    )
    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the URL extraction (0.0-1.0)"
    )
    extraction_method: Literal["direct_url", "inferred_from_context", "none_found"] = Field(
        description="Method used to extract the URL"
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Notes about the extraction process or issues encountered"
    )
