"""
Comprehensive Parameter Mapper for Data Analysis Agents.

This module provides intelligent mapping from user requests and workflow intents
to the specific parameters required by each agent (data_cleaning_agent, 
feature_engineering_agent, h2o_ml_agent).
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from src.schemas import DataAnalysisRequest, WorkflowIntent, ModelType

logger = logging.getLogger(__name__)


class AgentParameterMapper:
    """
    Intelligent mapper that translates user preferences and workflow intents
    into comprehensive parameter sets for each data analysis agent.
    """
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize the parameter mapper.
        
        Args:
            base_output_dir: Base directory for output files
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for unique outputs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def map_data_cleaning_parameters(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        csv_url: str
    ) -> Dict[str, Any]:
        """
        Map parameters for the data cleaning agent.
        
        Args:
            request: User's data analysis request
            intent: Parsed workflow intent
            csv_url: URL to the CSV file
            
        Returns:
            Dictionary of parameters for data_cleaning_agent
        """
        # Create unique file names
        cleaned_file_name = f"cleaned_data_{self.timestamp}.csv"
        log_path = self.base_output_dir / f"cleaning_log_{self.timestamp}.txt"
        
        params = {
            # Core parameters
            "model": "gpt-4o-mini",  # Cost-effective model for cleaning
            "n_samples": None,  # Use full dataset
            "log": True,
            "log_path": str(log_path),
            "file_name": cleaned_file_name,
            "function_name": "clean_data",
            "overwrite": True,
            
            # Advanced control parameters
            "human_in_the_loop": False,  # Automated processing
            "bypass_recommended_steps": False,  # Follow recommendations
            "bypass_explain_code": intent.complexity_level == "simple",  # Skip explanations for simple requests
            "checkpointer": None,  # No checkpointing for now
            
            # Data cleaning specific parameters (from request)
            "missing_threshold": request.missing_threshold,
            "outlier_detection": request.outlier_detection,
            "duplicate_removal": request.duplicate_removal,
            
            # User instructions (enhanced from basic approach)
            "user_instructions": self._create_cleaning_instructions(request, intent, csv_url)
        }
        
        logger.info(f"Mapped data cleaning parameters: {len(params)} parameters")
        return params
    
    def map_feature_engineering_parameters(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        cleaned_data_path: str,
        target_variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Map parameters for the feature engineering agent.
        
        Args:
            request: User's data analysis request
            intent: Parsed workflow intent
            cleaned_data_path: Path to cleaned data file
            target_variable: Target variable for ML (if determined)
            
        Returns:
            Dictionary of parameters for feature_engineering_agent
        """
        # Create unique file names
        engineered_file_name = f"engineered_features_{self.timestamp}.csv"
        log_path = self.base_output_dir / f"feature_engineering_log_{self.timestamp}.txt"
        
        # Determine target variable
        final_target = target_variable or intent.suggested_target_variable or request.target_variable
        
        params = {
            # Core parameters
            "model": "gpt-4o-mini",
            "n_samples": None,
            "log": True,
            "log_path": str(log_path),
            "file_name": engineered_file_name,
            "function_name": "engineer_features",
            "overwrite": True,
            
            # Advanced control parameters
            "human_in_the_loop": False,
            "bypass_recommended_steps": False,
            "bypass_explain_code": intent.complexity_level == "simple",
            "checkpointer": None,
            
            # Feature engineering specific
            "target_variable": final_target,
            
            # User instructions (enhanced)
            "user_instructions": self._create_feature_engineering_instructions(
                request, intent, cleaned_data_path, final_target
            )
        }
        
        logger.info(f"Mapped feature engineering parameters: {len(params)} parameters, target: {final_target}")
        return params
    
    def map_h2o_ml_parameters(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        engineered_data_path: str,
        target_variable: str
    ) -> Dict[str, Any]:
        """
        Map parameters for the H2O ML agent.
        
        Args:
            request: User's data analysis request
            intent: Parsed workflow intent
            engineered_data_path: Path to engineered features file
            target_variable: Target variable for ML modeling
            
        Returns:
            Dictionary of parameters for h2o_ml_agent
        """
        # Create unique directories and paths
        model_dir = self.base_output_dir / f"models_{self.timestamp}"
        model_dir.mkdir(exist_ok=True)
        
        log_path = self.base_output_dir / f"ml_log_{self.timestamp}.txt"
        
        # Map model types from request
        h2o_model_types = self._map_model_types(request.model_types)
        
        # Determine problem type
        problem_type = self._determine_problem_type(intent.suggested_problem_type, request.problem_type)
        
        params = {
            # Core parameters
            "model": "gpt-4o-mini",
            "n_samples": None,
            "log": True,
            "log_path": str(log_path),
            "file_name": f"ml_results_{self.timestamp}.json",
            "function_name": "train_models",
            "overwrite": True,
            
            # Advanced control parameters
            "human_in_the_loop": False,
            "bypass_recommended_steps": False,
            "bypass_explain_code": intent.complexity_level == "simple",
            "checkpointer": None,
            
            # H2O ML specific parameters
            "target_variable": target_variable,
            "problem_type": problem_type,
            "model_directory": str(model_dir),
            "max_runtime_secs": request.max_runtime_seconds,
            
            # MLflow configuration
            "enable_mlflow": request.enable_mlflow,
            "mlflow_tracking_uri": "file:./mlruns",
            "mlflow_experiment_name": f"data_analysis_{self.timestamp}",
            
            # Model training configuration
            "model_types": h2o_model_types,
            "cross_validation": True,
            "cv_folds": request.cross_validation_folds,
            "early_stopping": True,
            
            # User instructions (enhanced)
            "user_instructions": self._create_ml_instructions(
                request, intent, engineered_data_path, target_variable, problem_type
            )
        }
        
        logger.info(f"Mapped H2O ML parameters: {len(params)} parameters, problem: {problem_type}")
        return params
    
    def _create_cleaning_instructions(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        csv_url: str
    ) -> str:
        """Create detailed instructions for data cleaning."""
        
        instructions = [
            f"Clean the dataset from: {csv_url}",
            f"Original user request: {request.user_request}",
            "",
            "CLEANING REQUIREMENTS:"
        ]
        
        # Add specific cleaning requirements based on user preferences
        if request.duplicate_removal:
            instructions.append("- Remove duplicate rows")
        
        if request.outlier_detection:
            instructions.append("- Detect and handle outliers appropriately")
        
        instructions.append(f"- Handle missing values (drop columns with >{request.missing_threshold*100}% missing)")
        
        # Add intent-based requirements
        if intent.data_quality_focus:
            instructions.append("- Focus on comprehensive data quality assessment")
            instructions.append("- Provide detailed quality metrics and recommendations")
        
        if intent.exploratory_analysis:
            instructions.append("- Include basic exploratory data analysis")
            instructions.append("- Generate summary statistics and data profiling")
        
        # Add key requirements from intent
        if intent.key_requirements:
            instructions.append("\nKEY REQUIREMENTS:")
            for req in intent.key_requirements:
                instructions.append(f"- {req}")
        
        instructions.extend([
            "",
            "Please clean the data thoroughly and prepare it for subsequent analysis.",
            "Ensure the output is ready for feature engineering and machine learning."
        ])
        
        return "\n".join(instructions)
    
    def _create_feature_engineering_instructions(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        cleaned_data_path: str,
        target_variable: Optional[str]
    ) -> str:
        """Create detailed instructions for feature engineering."""
        
        instructions = [
            f"Engineer features from cleaned dataset: {cleaned_data_path}",
            f"Original user request: {request.user_request}",
            "",
            "FEATURE ENGINEERING REQUIREMENTS:"
        ]
        
        # Add target variable information
        if target_variable:
            instructions.append(f"- Target variable: {target_variable}")
            if request.problem_type:
                instructions.append(f"- Problem type: {request.problem_type.value}")
            else:
                instructions.append("- Problem type: auto")
        
        # Add specific feature engineering requirements
        if request.categorical_encoding:
            instructions.append("- Encode categorical variables appropriately")
        
        if request.datetime_features:
            instructions.append("- Create datetime-based features if datetime columns exist")
        
        if request.feature_selection:
            instructions.append("- Perform intelligent feature selection")
            instructions.append("- Remove highly correlated and low-importance features")
        
        # Add intent-based requirements
        if intent.prediction_focus:
            instructions.append("- Focus on features that improve prediction accuracy")
            instructions.append("- Create interaction features if beneficial")
        
        if intent.statistical_analysis:
            instructions.append("- Include statistical feature analysis")
            instructions.append("- Provide feature importance and correlation analysis")
        
        # Add complexity-based instructions
        if intent.complexity_level == "complex":
            instructions.extend([
                "- Create advanced engineered features",
                "- Consider polynomial features and interactions",
                "- Implement domain-specific feature engineering"
            ])
        elif intent.complexity_level == "simple":
            instructions.extend([
                "- Focus on basic but effective feature engineering",
                "- Prioritize interpretability over complexity"
            ])
        
        instructions.extend([
            "",
            "Please engineer features that will maximize model performance",
            "while maintaining interpretability and avoiding overfitting."
        ])
        
        return "\n".join(instructions)
    
    def _create_ml_instructions(
        self,
        request: DataAnalysisRequest,
        intent: WorkflowIntent,
        engineered_data_path: str,
        target_variable: str,
        problem_type: str
    ) -> str:
        """Create detailed instructions for ML modeling."""
        
        instructions = [
            f"Train machine learning models using: {engineered_data_path}",
            f"Original user request: {request.user_request}",
            "",
            f"ML MODELING REQUIREMENTS:",
            f"- Target variable: {target_variable}",
            f"- Problem type: {problem_type}",
            f"- Maximum runtime: {request.max_runtime_seconds} seconds"
        ]
        
        # Add model type preferences
        if request.model_types:
            model_names = [model.value for model in request.model_types]
            instructions.append(f"- Model types to try: {', '.join(model_names)}")
        else:
            instructions.append("- Model types to try: GBM, RandomForest, GLM")
        
        # Add cross-validation requirements
        instructions.append(f"- Use {request.cross_validation_folds}-fold cross-validation")
        
        # Add MLflow requirements
        if request.enable_mlflow:
            instructions.append("- Enable MLflow experiment tracking")
            instructions.append("- Log all model metrics and parameters")
        
        # Add intent-based requirements
        if intent.prediction_focus:
            instructions.append("- Optimize for prediction accuracy")
            instructions.append("- Focus on model performance metrics")
        
        if intent.statistical_analysis:
            instructions.append("- Provide comprehensive model analysis")
            instructions.append("- Include feature importance and model interpretability")
        
        # Add complexity-based instructions
        if intent.complexity_level == "complex":
            instructions.extend([
                "- Use advanced hyperparameter tuning",
                "- Consider ensemble methods and model stacking",
                "- Implement sophisticated model validation"
            ])
        elif intent.complexity_level == "simple":
            instructions.extend([
                "- Focus on simple, interpretable models",
                "- Prioritize model explainability",
                "- Use standard hyperparameter settings"
            ])
        
        # Add key requirements
        if intent.key_requirements:
            instructions.append("\nKEY REQUIREMENTS:")
            for req in intent.key_requirements:
                if any(keyword in req.lower() for keyword in ['model', 'predict', 'accuracy']):
                    instructions.append(f"- {req}")
        
        instructions.extend([
            "",
            "Please train the best possible models for this problem,",
            "provide comprehensive evaluation metrics, and ensure reproducibility."
        ])
        
        return "\n".join(instructions)
    
    def _map_model_types(self, model_types: List[ModelType]) -> List[str]:
        """Map ModelType enums to H2O model type strings."""
        model_mapping = {
            ModelType.GBM: "GBM",
            ModelType.RANDOM_FOREST: "RandomForest", 
            ModelType.GLM: "GLM",
            ModelType.DEEP_LEARNING: "DeepLearning",
            ModelType.XG_BOOST: "XGBoost",
            ModelType.AUTO_ML: "AutoML"
        }
        
        return [model_mapping.get(model_type, model_type.value) for model_type in model_types if model_type]
    
    def _determine_problem_type(
        self,
        suggested_type: Optional[str],
        requested_type: Optional[str]
    ) -> str:
        """Determine the final problem type for ML modeling."""
        
        # Priority: suggested_type > requested_type > auto
        if suggested_type and suggested_type.lower() != "auto":
            return suggested_type.lower()
        elif requested_type and requested_type.lower() != "auto":
            return requested_type.lower()
        else:
            return "auto"
    
    def get_output_directory(self) -> Path:
        """Get the output directory for this mapping session."""
        return self.base_output_dir
    
    def get_timestamp(self) -> str:
        """Get the timestamp for this mapping session."""
        return self.timestamp 