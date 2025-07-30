"""
Prediction response formatting module for the enhanced uAgent implementation.

This module contains all prediction-related formatting functions for ML predictions,
model analysis, and batch prediction results.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import UAgentConfig

logger = logging.getLogger(__name__)


class PredictionResponseFormatter:
    """Format prediction results for user-friendly display."""
    
    def __init__(self, config: UAgentConfig):
        self.config = config
    
    def format_single_prediction(self, prediction_result: Dict[str, Any]) -> str:
        """Format single prediction result following the existing pattern."""
        try:
            lines = [
                "🔮 **PREDICTION RESULT**",
                "=" * 60,
                "",
                f"🎯 **Target Variable**: {prediction_result.get('target_variable') or 'Unknown'}",
                f"📊 **Prediction**: {prediction_result.get('prediction') or 'N/A'}",
                "",
                "─" * 40,
                ""
            ]
            
            # Add confidence/probability if available
            probability = prediction_result.get("probability")
            if probability is not None and probability != "":
                try:
                    lines.extend([
                        f"📈 **Confidence**: {float(probability):.2%}",
                        ""
                    ])
                except (ValueError, TypeError):
                    lines.extend([
                        f"📈 **Confidence**: {probability}",
                        ""
                    ])
            
            # Display input features
            lines.extend([
                "📋 **Input Features**:",
            ])
            
            input_data = prediction_result.get("input_data", {})
            if input_data is None:
                input_data = {}
            
            if input_data:
                for feature, value in input_data.items():
                    lines.append(f"   • **{feature}**: {value}")
            else:
                lines.append("   • No input features available")
            
            lines.extend([
                "",
                "─" * 40,
                "",
                "🤖 **Model Information**:",
                f"   • **Architecture**: {prediction_result.get('model_architecture') or 'Unknown'}",
                f"   • **Performance Score**: {prediction_result.get('model_score') or 'N/A'}",
                "",
                "💡 **Next Steps**:",
                "   • Try different input values to explore the model behavior",
                "   • Ask model questions: 'What features are most important?'",
                "   • Make batch predictions: 'Predict for https://example.com/newdata.csv'",
                ""
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Single prediction formatting failed: {e}", exc_info=True)
            return f"❌ **Error formatting prediction**: {str(e)}"
    
    def format_batch_prediction(self, prediction_result: Dict[str, Any]) -> str:
        """Format batch prediction results following the existing pattern."""
        try:
            lines = [
                "🔮 **BATCH PREDICTION COMPLETE**",
                "=" * 60,
                "",
                f"📊 **Processing Summary**:",
                f"   • **Rows Processed**: {prediction_result.get('input_rows', 0):,}",
                f"   • **Predictions Generated**: {prediction_result.get('input_rows', 0):,}",
                f"   • **Processing Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "─" * 40,
                ""
            ]
            
            # Add download information
            output_path = prediction_result.get('output_path')
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                lines.extend([
                    "📁 **Download Results**:",
                    f"   • **File Path**: {output_path}",
                    f"   • **File Size**: {file_size / 1024:.1f} KB",
                    "",
                ])
            
            # Add prediction summary statistics
            predictions_summary = prediction_result.get('predictions_summary', {})
            if predictions_summary is None:
                predictions_summary = {}
            if predictions_summary:
                lines.extend([
                    "📈 **Prediction Summary**:",
                ])
                
                # Handle classification counts
                if 'prediction_counts' in predictions_summary:
                    lines.append("   • **Class Distribution**:")
                    for class_label, count in predictions_summary['prediction_counts'].items():
                        percentage = (count / prediction_result.get('input_rows', 1)) * 100
                        lines.append(f"     - {class_label}: {count:,} ({percentage:.1f}%)")
                
                # Handle regression statistics
                if 'prediction_stats' in predictions_summary:
                    stats = predictions_summary['prediction_stats']
                    lines.extend([
                        "   • **Statistical Summary**:",
                        f"     - Mean: {stats.get('mean', 0):.3f}",
                        f"     - Std Dev: {stats.get('std', 0):.3f}",
                        f"     - Min: {stats.get('min', 0):.3f}",
                        f"     - Max: {stats.get('max', 0):.3f}",
                    ])
                
                lines.append("")
            
            lines.extend([
                "💡 **Next Steps**:",
                "   • Download the results file to analyze detailed predictions",
                "   • Ask questions about the model: 'What features drove these predictions?'",
                "   • Train a new model if predictions don't meet expectations",
                ""
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Batch prediction formatting failed: {e}", exc_info=True)
            return f"❌ **Error formatting batch predictions**: {str(e)}"
    
    def format_model_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """Format model analysis response following the existing pattern."""
        try:
            lines = [
                "🧠 **MODEL ANALYSIS**",
                "=" * 60,
                "",
                f"❓ **Your Question**: {analysis_result.get('question', 'Unknown')}",
                "",
                "─" * 40,
                "",
                "💡 **Analysis Result**:",
                ""
            ]
            
            # Add the AI-generated analysis
            answer = analysis_result.get('answer', 'No analysis available')
            lines.extend([
                answer,
                "",
                "─" * 40,
                ""
            ])
            
            # Add model technical details
            model_info = analysis_result.get('model_info', {})
            if model_info is None:
                model_info = {}
            if model_info:
                lines.extend([
                    "🤖 **Model Technical Details**:",
                    f"   • **Architecture**: {model_info.get('architecture', 'Unknown')}",
                    f"   • **Target Variable**: {model_info.get('target', 'Unknown')}",
                    f"   • **Performance Score**: {model_info.get('performance', 'N/A')}",
                    f"   • **Training Duration**: {model_info.get('training_duration', 'N/A')} seconds",
                    f"   • **Total Models Trained**: {model_info.get('total_models_trained', 'N/A')}",
                    "",
                ])
                
                # Add feature information if available
                features = model_info.get('features', [])
                if features is None:
                    features = []
                if features:
                    lines.extend([
                        f"📊 **Features Used**: {len(features)} features",
                        f"   • {', '.join(features[:5])}{'...' if len(features) > 5 else ''}",
                        ""
                    ])
            
            lines.extend([
                "💡 **Ask More Questions**:",
                "   • 'What features are most important for predictions?'",
                "   • 'How accurate is this model?'",
                "   • 'What were the different models trained?'",
                "   • 'Show me feature importance details'",
                ""
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Model analysis formatting failed: {e}", exc_info=True)
            return f"❌ **Error formatting model analysis**: {str(e)}"
    
    def format_no_model_response(self) -> str:
        """Format response when no trained model is available."""
        return """🚫 **No Trained Model Found**

I don't have a trained ML model in this session to make predictions.

**To get started with predictions:**

1️⃣ **First, train a model:**
   "Clean and build ML model using https://example.com/data.csv to predict target_column"

2️⃣ **Then make predictions:**
   "Predict target_column for feature1=value1, feature2=value2"

📋 **Example workflow:**
```
You: "Train a model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv to predict Survived"
Me: [Trains model and shows comprehensive results]

You: "Predict survival for Age=25, Sex=male, Pclass=3"
Me: [Uses trained model to make prediction]
```

💡 **Batch predictions:**
   "Use the model to predict for https://example.com/new_data.csv"

🧠 **Model analysis:**
   "What features are most important for predictions?"
"""
    
    def format_prediction_error_response(self, error: Exception) -> str:
        """Format error response for prediction failures."""
        return f"""🚫 **Prediction Error**

Sorry, I encountered an issue while making the prediction:

**Error Details**: {str(error)}

**Common solutions:**
1. ✅ **Check Input Format**: Ensure your input data format matches the training data
2. ✅ **Verify Features**: Make sure all required features are provided
3. ✅ **Model Session**: Check if the model session hasn't expired
4. ✅ **Retrain Model**: Try retraining if data format has changed

**Get Help:**
   • "What features does the model expect?"
   • "Train a new model using [your dataset URL]"
   • "What was the original target variable?"

💡 **Note**: Model sessions expire after {self.config.session_timeout_hours} hours
""" 