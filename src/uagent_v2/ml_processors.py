"""
ML processing module for the enhanced uAgent implementation.

This module contains all ML result processing and formatting functions
extracted from the original enhanced_uagent.py file for better code organization.
"""

import re
import logging
from typing import Dict, Any, List, Optional

from .config import UAgentConfig

logger = logging.getLogger(__name__)


class MLResultProcessor:
    """Comprehensive ML result processing for H2O AutoML results."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def extract_h2o_ml_results(self, agent_result) -> Dict[str, Any]:
        """Extract comprehensive ML results from H2O agent execution."""
        
        if agent_result.agent_name != "h2o_ml" or not agent_result.success:
            return {}
        
        ml_results = {}
        
        try:
            # Extract from ML modeling metrics if available
            if hasattr(agent_result, 'ml_modeling_metrics') and agent_result.ml_modeling_metrics:
                metrics = agent_result.ml_modeling_metrics
                
                # Basic metrics
                ml_results["models_trained"] = getattr(metrics, 'total_models_trained', getattr(metrics, 'models_trained', 0))
                ml_results["best_model_id"] = getattr(metrics, 'best_model_id', None)
                ml_results["model_architecture"] = getattr(metrics, 'model_architecture', getattr(metrics, 'best_model_type', None))
                ml_results["training_time"] = getattr(metrics, 'training_runtime', getattr(metrics, 'training_time_seconds', 0))
                
                # Performance metrics
                ml_results["best_score"] = getattr(metrics, 'best_model_score', None)
                ml_results["cross_validation_score"] = getattr(metrics, 'cross_validation_score', None)
                
                # Rich data
                ml_results["leaderboard"] = getattr(metrics, 'leaderboard', None)
                ml_results["top_model_metrics"] = getattr(metrics, 'top_model_metrics', {})
                ml_results["generated_code"] = getattr(metrics, 'generated_code', None)
                ml_results["recommended_steps"] = getattr(metrics, 'recommended_steps', None)
                ml_results["workflow_summary"] = getattr(metrics, 'workflow_summary', None)
                ml_results["model_path"] = getattr(metrics, 'model_path', None)
                ml_results["enhanced_feature_importance"] = getattr(metrics, 'enhanced_feature_importance', [])
                
                # Status flags
                ml_results["has_leaderboard"] = ml_results["leaderboard"] is not None and len(ml_results["leaderboard"]) > 0
                ml_results["model_saved"] = ml_results["model_path"] is not None
                
            # Fallback: try to extract from log messages
            if not ml_results.get("models_trained") and agent_result.log_messages:
                result_str = " ".join(agent_result.log_messages)
                
                # Look for model performance patterns
                auc_matches = re.findall(r'AUC[:\s]+([0-9.]+)', result_str, re.IGNORECASE)
                if auc_matches:
                    ml_results["best_auc"] = float(auc_matches[0])
                
                # Look for accuracy patterns  
                acc_matches = re.findall(r'accuracy[:\s]+([0-9.]+)', result_str, re.IGNORECASE)
                if acc_matches:
                    ml_results["accuracy"] = float(acc_matches[0])
                
                # Look for model count
                model_matches = re.findall(r'(\d+)\s+models?\s+trained', result_str, re.IGNORECASE)
                if model_matches:
                    ml_results["models_trained"] = int(model_matches[0])
            
            # Extract model path information from agent result
            if hasattr(agent_result, 'model_path') and agent_result.model_path:
                ml_results["model_path"] = agent_result.model_path
                ml_results["model_saved"] = True
            
            return ml_results
            
        except Exception as e:
            logger.warning(f"Failed to extract H2O ML results: {e}")
            return {}
    
    def format_ml_leaderboard_display(self, ml_results: Dict[str, Any], execution_time: float = 0) -> List[str]:
        """Format ML leaderboard for beautiful user display."""
        
        lines = [
            "🤖 **MACHINE LEARNING RESULTS**",
            "=" * 50,
            ""
        ]
        
        # Training summary
        if ml_results.get("models_trained", 0) > 0:
            lines.extend([
                "🏆 **MODEL TRAINING COMPLETE**:",
                f"   • Models Trained: {ml_results.get('models_trained', 'Multiple')}",
                f"   • Training Time: {ml_results.get('training_time', execution_time):.1f} seconds",
                ""
            ])
            
            # Performance metrics
            if ml_results.get("best_score"):
                lines.append(f"   • Best Model Score: {ml_results['best_score']:.4f}")
            if ml_results.get("best_auc"):
                lines.append(f"   • Best AUC Score: {ml_results['best_auc']:.4f}")
            if ml_results.get("accuracy"):
                lines.append(f"   • Accuracy: {ml_results['accuracy']:.4f}")
            if ml_results.get("cross_validation_score"):
                lines.append(f"   • Cross-Validation Score: {ml_results['cross_validation_score']:.4f}")
            
            lines.append("")
        
        # Leaderboard display
        if ml_results.get("has_leaderboard") and ml_results.get("leaderboard"):
            lines.extend([
                "🏆 **MODEL LEADERBOARD** (Top 5 Models):",
                ""
            ])
            
            leaderboard = ml_results["leaderboard"]
            for idx, model in enumerate(leaderboard[:5]):  # Show top 5 models
                rank = idx + 1
                model_id = model.get('model_id', f'Model_{rank}')
                model_name = model_id[:40] + "..." if len(model_id) > 40 else model_id
                
                # Get performance metric (try different fields)
                performance = (
                    model.get('auc', model.get('rmse', model.get('logloss', model.get('mean_residual_deviance', 0))))
                )
                
                rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                lines.append(f"   {rank_emoji} **{model_name}**")
                
                if performance:
                    metric_name = "AUC" if 'auc' in model else "RMSE" if 'rmse' in model else "Score"
                    lines.append(f"      • {metric_name}: {performance:.4f}")
                
                if rank == 1:
                    lines.append(f"      🏆 **WINNER** - This is your best model!")
                
                lines.append("")
        
        # Model architecture and details
        if ml_results.get("model_architecture") or ml_results.get("best_model_id"):
            lines.extend([
                "🎯 **BEST MODEL DETAILS**:",
            ])
            
            if ml_results.get("best_model_id"):
                lines.append(f"   • Model ID: `{ml_results['best_model_id']}`")
            if ml_results.get("model_architecture"):
                lines.append(f"   • Architecture: {ml_results['model_architecture']}")
            if ml_results.get("training_time", 0) > 0:
                lines.append(f"   • Training Time: {ml_results['training_time']:.1f} seconds")
            
            lines.append("")
        
        # ML methodology  
        lines.extend([
            "🧠 **AI METHODOLOGY APPLIED**:",
            "   • Automated algorithm selection (Random Forest, GBM, Neural Networks, etc.)",
            "   • Hyperparameter optimization for best performance",
            "   • Cross-validation to ensure model reliability",
            "   • Feature importance analysis for interpretability",
            ""
        ])
        
        # Performance summary
        lines.extend([
            "🎯 **MODEL PERFORMANCE**:",
            "   🥇 **Best Model Selected**: AutoML chose the highest-performing algorithm",
            "   📊 **Cross-Validated**: Results are validated to avoid overfitting",
            "   ⚡ **Production Ready**: Model can be used for predictions immediately",
            ""
        ])
        
        return lines
    
    def format_ml_generated_code_display(self, ml_results: Dict[str, Any]) -> List[str]:
        """Format AI-generated ML code for user display."""
        
        lines = []
        
        # Show generated code if available
        generated_code = ml_results.get("generated_code")
        if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 10:
            lines.extend([
                "💻 **AI-GENERATED MODEL CODE**:",
                "```python",
                generated_code.strip(),
                "```",
                "",
                "💡 **Usage**: Copy this code to reproduce the same model training independently!",
                ""
            ])
        else:
            # Provide template code
            model_id = ml_results.get("best_model_id", "your_model")
            lines.extend([
                "💻 **AI AGENT GENERATED CODE**:",
                "```python",
                "# H2O AutoML Training Code (Generated by AI)",
                "import h2o",
                "from h2o.automl import H2OAutoML",
                "",
                "# Initialize H2O",
                "h2o.init()",
                "",
                "# Load your data",
                "data = h2o.import_file('your_dataset.csv')",
                "",
                "# Prepare training data",
                "train, test = data.split_frame(ratios=[0.8])",
                "x = train.columns[:-1]  # All columns except target",
                "y = train.columns[-1]   # Target column",
                "",
                "# Train AutoML model",
                "aml = H2OAutoML(max_models=20, seed=42)",
                "aml.train(x=x, y=y, training_frame=train)",
                "",
                "# Get best model and make predictions",
                "best_model = aml.leader",
                "predictions = best_model.predict(test)",
                "",
                "# View leaderboard",
                "print(aml.leaderboard.head())",
                "```",
                "",
                "💡 **Usage**: Copy this code to train H2O AutoML models independently!",
                ""
            ])
        
        return lines
    
    def format_ml_workflow_summary_display(self, ml_results: Dict[str, Any]) -> List[str]:
        """Format ML workflow summary and recommendations."""
        
        lines = []
        
        # Workflow summary
        workflow_summary = ml_results.get("workflow_summary")
        if workflow_summary:
            lines.extend([
                "📚 **ML WORKFLOW SUMMARY**:",
                f"   {workflow_summary}",
                ""
            ])
        
        # Recommended steps - CLARIFIED MESSAGING
        recommended_steps = ml_results.get("recommended_steps")
        if recommended_steps:
            lines.extend([
                "📋 **ML METHODOLOGY EXECUTED**:",
                "   The following approach was automatically applied by the ML agent:",
                "",
                f"   {recommended_steps}",
                ""
            ])
        
        # Feature importance if available
        feature_importance = ml_results.get("enhanced_feature_importance", [])
        if feature_importance and len(feature_importance) > 0:
            lines.extend([
                "🎯 **FEATURE IMPORTANCE ANALYSIS**:",
                ""
            ])
            
            for feature in feature_importance[:5]:  # Show top 5 features
                feature_name = feature.get("feature", "Unknown")
                importance = feature.get("importance", 0)
                impact = feature.get("impact", "Unknown")
                lines.append(f"   • **{feature_name}**: {importance:.3f} ({impact} Impact)")
            
            if len(feature_importance) > 5:
                lines.append(f"   • ...and {len(feature_importance) - 5} more features")
            
            lines.append("")
        
        return lines 