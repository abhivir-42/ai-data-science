"""
ML Prediction Agent for making predictions with trained H2O models.

This module provides functionality to make predictions using trained H2O AutoML models
that are stored in the session after successful training.
"""

import os
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from src.schemas.data_analysis_schemas import MLModelingMetrics
from src.uagent_v2.config import UAgentConfig


class MLPredictionAgent:
    """Agent for making predictions with trained H2O models."""
    
    def __init__(self, model_metrics: MLModelingMetrics, target_variable: str, config: UAgentConfig):
        """
        Initialize the MLPredictionAgent.
        
        Args:
            model_metrics: MLModelingMetrics object containing model information
            target_variable: Target variable name used for training
            config: UAgentConfig object
        """
        self.model_metrics = model_metrics
        self.target_variable = target_variable
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._h2o_model = None
        
    def load_model(self):
        """Load the H2O model for predictions."""
        try:
            import h2o
            h2o.init()
            
            # Try to get model from H2O cluster by ID first (model in memory)
            try:
                self._h2o_model = h2o.get_model(self.model_metrics.best_model_id)
                self.logger.info(f"Loaded H2O model from cluster: {self.model_metrics.best_model_id}")
            except Exception:
                # Fallback to file loading if model path is available
                if self.model_metrics.model_path:
                    self._h2o_model = h2o.load_model(self.model_metrics.model_path)
                    self.logger.info(f"Loaded H2O model from file: {self.model_metrics.best_model_id}")
                else:
                    raise MLPredictionError(f"Model {self.model_metrics.best_model_id} not found in H2O cluster and no file path available")
            
        except ImportError:
            raise MLPredictionError("H2O is not installed. Please install h2o package: pip install h2o")
        except Exception as e:
            raise MLPredictionError(f"Failed to load model: {e}")
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single data point.
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if self._h2o_model is None:
                self.load_model()
            
            # Convert input to H2O frame
            import h2o
            df = pd.DataFrame([input_data])
            h2o_frame = h2o.H2OFrame(df)
            
            # Make prediction
            predictions = self._h2o_model.predict(h2o_frame)
            
            # Convert to results
            pred_df = predictions.as_data_frame()
            
            # Format results based on problem type  
            problem_type = self._determine_problem_type()
            if problem_type == "classification":
                return self._format_classification_result(pred_df, input_data)
            else:
                return self._format_regression_result(pred_df, input_data)
                
        except Exception as e:
            raise MLPredictionError(f"Prediction failed: {e}")
    
    def predict_batch(self, data_source: str) -> Dict[str, Any]:
        """
        Make predictions for batch data from CSV URL.
        
        Args:
            data_source: CSV URL containing batch data
            
        Returns:
            Dictionary containing batch prediction results
        """
        try:
            if self._h2o_model is None:
                self.load_model()
            
            # Load data
            import h2o
            df = pd.read_csv(data_source)
            h2o_frame = h2o.H2OFrame(df)
            
            # Make predictions
            predictions = self._h2o_model.predict(h2o_frame)
            pred_df = predictions.as_data_frame()
            
            # Combine with original data
            result_df = pd.concat([df, pred_df], axis=1)
            
            # Save results
            output_path = self._save_prediction_results(result_df)
            
            return {
                "prediction_type": "batch",
                "input_rows": len(df),
                "output_path": output_path,
                "predictions_summary": self._summarize_batch_predictions(pred_df),
                "download_link": output_path
            }
            
        except Exception as e:
            raise MLPredictionError(f"Batch prediction failed: {e}")
    
    def analyze_model(self, query: str) -> Dict[str, Any]:
        """
        Answer questions about the trained model.
        
        Args:
            query: User question about the model
            
        Returns:
            Dictionary containing model analysis results
        """
        try:
            # Use LLM to answer model-related questions
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            
            model_info = {
                "architecture": self.model_metrics.model_architecture,
                "features": self.model_metrics.features_used,
                "performance": self.model_metrics.best_model_score,
                "target": self.target_variable,
                "training_duration": self.model_metrics.training_time_seconds,
                # Rich H2O data for comprehensive analysis
                "leaderboard": self.model_metrics.leaderboard,
                "top_model_metrics": self.model_metrics.top_model_metrics,
                "total_models_trained": self.model_metrics.total_models_trained,
                "feature_importance": self.model_metrics.enhanced_feature_importance,
                "generated_code": self.model_metrics.generated_code,
                "recommended_steps": self.model_metrics.recommended_steps,
                "workflow_summary": self.model_metrics.workflow_summary
            }
            
            prompt = f"""
            Answer the user's question about this trained ML model using the comprehensive H2O AutoML data:
            
            Model Information:
            - Architecture: {model_info['architecture']}
            - Target Variable: {model_info['target']}
            - Features Used: {', '.join(model_info['features'])}
            - Performance Score: {model_info['performance']}
            - Training Duration: {model_info['training_duration']} seconds
            - Total Models Trained: {model_info['total_models_trained']}
            
            Feature Importance: {model_info['feature_importance']}
            Top Model Metrics: {model_info['top_model_metrics']}
            Leaderboard: {model_info['leaderboard']}
            Workflow Summary: {model_info['workflow_summary']}
            
            User Question: {query}
            
            Provide a helpful, accurate answer based on the comprehensive model information above.
            Use the feature importance, leaderboard, and metrics to give detailed insights.
            """
            
            response = llm.invoke(prompt)
            
            return {
                "analysis_type": "model_question",
                "question": query,
                "answer": response.content,
                "model_info": model_info
            }
            
        except Exception as e:
            raise MLPredictionError(f"Model analysis failed: {e}")
    
    def _determine_problem_type(self) -> str:
        """Determine if this is a classification or regression problem."""
        # Try to determine from model architecture
        if self.model_metrics.model_architecture:
            arch = self.model_metrics.model_architecture.lower()
            if "classification" in arch:
                return "classification"
            elif "regression" in arch:
                return "regression"
        
        # Try to determine from metrics
        if self.model_metrics.top_model_metrics:
            metrics = self.model_metrics.top_model_metrics
            # AUC suggests classification, RMSE suggests regression
            if 'auc' in str(metrics).lower() or 'logloss' in str(metrics).lower():
                return "classification"
            elif 'rmse' in str(metrics).lower() or 'mae' in str(metrics).lower():
                return "regression"
        
        # Default fallback
        return "classification"
    
    def _format_classification_result(self, pred_df: pd.DataFrame, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format classification prediction results."""
        # Get prediction and probability
        prediction = pred_df.iloc[0]['predict'] if 'predict' in pred_df.columns else pred_df.iloc[0, 0]
        
        # Get probability if available
        probability = None
        prob_cols = [col for col in pred_df.columns if col.startswith('p') and col != 'predict']
        if prob_cols:
            probability = pred_df.iloc[0][prob_cols[0]]
        
        return {
            "prediction_type": "single_prediction", 
            "target_variable": self.target_variable,
            "prediction": prediction,
            "probability": probability,
            "input_data": input_data,
            "model_architecture": self.model_metrics.model_architecture,
            "model_score": self.model_metrics.best_model_score
        }
    
    def _format_regression_result(self, pred_df: pd.DataFrame, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format regression prediction results."""
        prediction = pred_df.iloc[0]['predict'] if 'predict' in pred_df.columns else pred_df.iloc[0, 0]
        
        return {
            "prediction_type": "single_prediction",
            "target_variable": self.target_variable,
            "prediction": prediction,
            "input_data": input_data,
            "model_architecture": self.model_metrics.model_architecture,
            "model_score": self.model_metrics.best_model_score
        }
    
    def _save_prediction_results(self, result_df: pd.DataFrame) -> str:
        """Save batch prediction results to file."""
        import os
        from datetime import datetime
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_predictions_{timestamp}.csv"
        
        # Use config output directory or create default
        output_dir = getattr(self.config, 'output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        result_df.to_csv(output_path, index=False)
        
        return output_path
    
    def _summarize_batch_predictions(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for batch predictions."""
        summary = {}
        
        if 'predict' in pred_df.columns:
            predictions = pred_df['predict']
            
            # For classification
            if predictions.dtype == 'object' or predictions.nunique() < 20:
                summary['prediction_counts'] = predictions.value_counts().to_dict()
            
            # For regression
            else:
                summary['prediction_stats'] = {
                    'mean': float(predictions.mean()),
                    'std': float(predictions.std()),
                    'min': float(predictions.min()),
                    'max': float(predictions.max())
                }
        
        return summary


# Custom exception for ML prediction errors
class MLPredictionError(Exception):
    """Custom exception for ML prediction failures."""
    pass 