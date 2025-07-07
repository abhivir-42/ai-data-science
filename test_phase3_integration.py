"""
Phase 3 Testing: Integration & Response Formatting

This test suite validates the complete integration of ML prediction capabilities
into the enhanced uAgent, including response formatting and error handling.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig
from src.uagent_v2.prediction_formatters import PredictionResponseFormatter
from src.schemas.data_analysis_schemas import MLModelingMetrics, WorkflowIntent
from src.agents.ml_prediction_agent import MLPredictionAgent, MLPredictionError


class TestPhase3Integration:
    """Test Phase 3: Integration & Response Formatting"""
    
    def test_prediction_formatter_initialization(self):
        """Test that prediction formatter is properly initialized in enhanced uAgent."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Check that prediction formatter is initialized
        assert hasattr(enhanced_uagent, 'prediction_formatter')
        assert isinstance(enhanced_uagent.prediction_formatter, PredictionResponseFormatter)
        assert enhanced_uagent.prediction_formatter.config == config
        
        print("✅ Prediction formatter initialization test passed")
    
    def test_intent_parser_initialization(self):
        """Test that intent parser is properly initialized in enhanced uAgent."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Check that intent parser is initialized
        assert hasattr(enhanced_uagent, 'intent_parser')
        assert enhanced_uagent.intent_parser is not None
        
        print("✅ Intent parser initialization test passed")
    
    def test_prediction_request_handling_no_model(self):
        """Test handling prediction requests when no model is available."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock intent parser to return prediction request
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"age": 25, "sex": "male"}
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            result = enhanced_uagent.process_query("Predict survival for age=25, sex=male")
        
        # Should return no model response
        assert "No Trained Model Found" in result
        assert "train a model" in result.lower()
        
        print("✅ Prediction request handling (no model) test passed")
    
    def test_prediction_request_handling_with_model(self):
        """Test handling prediction requests when model is available."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock trained model in session
        mock_metrics = self._create_mock_ml_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Survived"
        
        # Mock intent parser to return prediction request
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"Age": 25, "Sex": "male", "Pclass": 3}
        
        # Mock prediction agent
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Survived",
            "prediction": 0,
            "probability": 0.23,
            "input_data": {"Age": 25, "Sex": "male", "Pclass": 3},
            "model_architecture": "AutoML",
            "model_score": 0.85
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                result = enhanced_uagent.process_query("Predict survival for Age=25, Sex=male, Pclass=3")
        
        # Should return formatted prediction result
        assert "PREDICTION RESULT" in result
        assert "Survived" in result
        assert "23.00%" in result  # probability
        assert "Age" in result
        assert "male" in result
        
        print("✅ Prediction request handling (with model) test passed")
    
    def test_model_analysis_request_handling(self):
        """Test handling model analysis requests."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock trained model in session
        mock_metrics = self._create_mock_ml_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Survived"
        
        # Mock intent parser to return model analysis request
        mock_intent = Mock()
        mock_intent.needs_prediction = False
        mock_intent.needs_model_analysis = True
        
        # Mock analysis result
        mock_analysis_result = {
            "analysis_type": "model_question",
            "question": "What features are most important?",
            "answer": "Age, Sex, and Pclass are the most important features for survival prediction.",
            "model_info": {
                "architecture": "AutoML",
                "target": "Survived",
                "performance": 0.85,
                "features": ["Age", "Sex", "Pclass", "Fare"]
            }
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'analyze_model', return_value=mock_analysis_result):
                result = enhanced_uagent.process_query("What features are most important for survival prediction?")
        
        # Should return formatted analysis result
        assert "MODEL ANALYSIS" in result
        assert "What features are most important?" in result
        assert "Age, Sex, and Pclass" in result
        assert "AutoML" in result
        
        print("✅ Model analysis request handling test passed")
    
    def test_batch_prediction_request_handling(self):
        """Test handling batch prediction requests."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock trained model in session
        mock_metrics = self._create_mock_ml_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Survived"
        
        # Mock intent parser to return batch prediction request
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "batch_prediction"
        mock_intent.prediction_data_source = "https://example.com/test_data.csv"
        
        # Mock batch prediction result
        mock_batch_result = {
            "prediction_type": "batch",
            "input_rows": 100,
            "output_path": "/tmp/batch_predictions_20240101_120000.csv",
            "predictions_summary": {
                "prediction_counts": {"0": 62, "1": 38}
            }
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_batch', return_value=mock_batch_result):
                result = enhanced_uagent.process_query("Predict for https://example.com/test_data.csv")
        
        # Should return formatted batch prediction result
        assert "BATCH PREDICTION COMPLETE" in result
        assert "100" in result  # input rows
        assert "62" in result   # prediction counts
        assert "38" in result   # prediction counts
        
        print("✅ Batch prediction request handling test passed")
    
    def test_prediction_error_handling(self):
        """Test error handling in prediction requests."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock trained model in session
        mock_metrics = self._create_mock_ml_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Survived"
        
        # Mock intent parser to return prediction request
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"Age": 25}
        
        # Mock prediction agent to raise error
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', side_effect=MLPredictionError("Model loading failed")):
                result = enhanced_uagent.process_query("Predict survival for Age=25")
        
        # Should return formatted error response
        assert "Prediction Error" in result
        assert "Model loading failed" in result
        assert "Common solutions" in result
        
        print("✅ Prediction error handling test passed")
    
    def test_intent_parser_fallback(self):
        """Test fallback behavior when intent parser fails."""
        config = UAgentConfig()
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock intent parser to raise exception
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', side_effect=Exception("Intent parsing failed")):
            with patch.object(enhanced_uagent, '_process_analysis_request', return_value="Analysis complete"):
                result = enhanced_uagent.process_query("Clean and analyze data")
        
        # Should fall back to normal analysis processing
        assert "Analysis complete" in result
        
        print("✅ Intent parser fallback test passed")
    
    def test_session_expiration_handling(self):
        """Test handling of expired model sessions."""
        config = UAgentConfig()
        config.session_timeout_hours = 1  # 1 hour timeout
        enhanced_uagent = EnhancedDataAnalysisUAgent(config)
        
        # Mock expired model session
        mock_metrics = self._create_mock_ml_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time() - 3600 - 1  # 1 hour + 1 second ago
        enhanced_uagent._last_target_variable = "Survived"
        
        # Mock intent parser to return prediction request
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"Age": 25}
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            result = enhanced_uagent.process_query("Predict survival for Age=25")
        
        # Should return no model response due to expiration
        assert "No Trained Model Found" in result
        
        print("✅ Session expiration handling test passed")
    
    def _create_mock_ml_metrics(self):
        """Create a mock MLModelingMetrics object."""
        return Mock(spec=MLModelingMetrics, 
                   model_path="/tmp/test_model", 
                   best_model_id="AutoML_1_20240101_120000",
                   model_architecture="AutoML",
                   best_model_score=0.85,
                   features_used=["Age", "Sex", "Pclass", "Fare"],
                   leaderboard=[{"model_id": "AutoML_1_20240101_120000", "auc": 0.85}],
                   top_model_metrics={"auc": 0.85, "logloss": 0.45},
                   total_models_trained=20,
                   enhanced_feature_importance=["Age: 0.35", "Sex: 0.28", "Pclass: 0.22"],
                   generated_code="# H2O AutoML model",
                   recommended_steps=["Feature engineering", "Model tuning"],
                   workflow_summary="Trained 20 models, best: AutoML")


def test_prediction_response_formatter_single():
    """Test single prediction response formatting."""
    config = UAgentConfig()
    formatter = PredictionResponseFormatter(config)
    
    prediction_result = {
        "prediction_type": "single_prediction",
        "target_variable": "Survived",
        "prediction": 0,
        "probability": 0.23,
        "input_data": {"Age": 25, "Sex": "male", "Pclass": 3},
        "model_architecture": "AutoML",
        "model_score": 0.85
    }
    
    result = formatter.format_single_prediction(prediction_result)
    
    assert "PREDICTION RESULT" in result
    assert "Survived" in result
    assert "23.00%" in result
    assert "Age" in result
    assert "male" in result
    assert "AutoML" in result
    
    print("✅ Single prediction response formatting test passed")


def test_prediction_response_formatter_batch():
    """Test batch prediction response formatting."""
    config = UAgentConfig()
    formatter = PredictionResponseFormatter(config)
    
    prediction_result = {
        "prediction_type": "batch",
        "input_rows": 100,
        "output_path": "/tmp/batch_predictions_20240101_120000.csv",
        "predictions_summary": {
            "prediction_counts": {"0": 62, "1": 38}
        }
    }
    
    # Mock file existence
    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', return_value=5120):  # 5KB file
            result = formatter.format_batch_prediction(prediction_result)
    
    assert "BATCH PREDICTION COMPLETE" in result
    assert "100" in result
    assert "62" in result
    assert "38" in result
    assert "5.0 KB" in result
    
    print("✅ Batch prediction response formatting test passed")


def test_prediction_response_formatter_analysis():
    """Test model analysis response formatting."""
    config = UAgentConfig()
    formatter = PredictionResponseFormatter(config)
    
    analysis_result = {
        "analysis_type": "model_question",
        "question": "What features are most important?",
        "answer": "Age, Sex, and Pclass are the most important features for survival prediction.",
        "model_info": {
            "architecture": "AutoML",
            "target": "Survived",
            "performance": 0.85,
            "features": ["Age", "Sex", "Pclass", "Fare"],
            "training_duration": 120,
            "total_models_trained": 20
        }
    }
    
    result = formatter.format_model_analysis(analysis_result)
    
    assert "MODEL ANALYSIS" in result
    assert "What features are most important?" in result
    assert "Age, Sex, and Pclass" in result
    assert "AutoML" in result
    assert "120 seconds" in result
    assert "20" in result
    
    print("✅ Model analysis response formatting test passed")


if __name__ == "__main__":
    print("🚀 Running Phase 3 Integration Tests...")
    print("=" * 60)
    
    # Run integration tests
    test_suite = TestPhase3Integration()
    test_suite.test_prediction_formatter_initialization()
    test_suite.test_intent_parser_initialization()
    test_suite.test_prediction_request_handling_no_model()
    test_suite.test_prediction_request_handling_with_model()
    test_suite.test_model_analysis_request_handling()
    test_suite.test_batch_prediction_request_handling()
    test_suite.test_prediction_error_handling()
    test_suite.test_intent_parser_fallback()
    test_suite.test_session_expiration_handling()
    
    # Run formatter tests
    test_prediction_response_formatter_single()
    test_prediction_response_formatter_batch()
    test_prediction_response_formatter_analysis()
    
    print("=" * 60)
    print("🎉 All Phase 3 Integration Tests Passed!")
    print("✅ Prediction request handling: WORKING")
    print("✅ Model analysis handling: WORKING")
    print("✅ Response formatting: WORKING")
    print("✅ Error handling: WORKING")
    print("✅ Session management: WORKING")
    print("\n🚀 Phase 3 Integration Complete - Ready for End-to-End Testing!") 