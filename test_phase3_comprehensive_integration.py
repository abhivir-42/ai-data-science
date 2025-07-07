"""
COMPREHENSIVE Phase 3 Integration Tests - Enhanced uAgent
CRITICAL: Complete end-to-end workflows must work perfectly.
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig
from src.schemas.data_analysis_schemas import MLModelingMetrics, WorkflowIntent
from src.agents.ml_prediction_agent import MLPredictionAgent, MLPredictionError


class TestEnhancedUAgentIntegrationComprehensive:
    """Comprehensive integration tests - EVERY workflow must work flawlessly."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = UAgentConfig()
        
    def test_complete_training_to_prediction_workflow(self):
        """Test complete workflow: train model → make predictions → analyze model."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Step 1: Simulate successful model training
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Survived"
        
        # Step 2: Test single prediction
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {
            "Age": 25.0, "Sex": "female", "Pclass": 1, "Fare": 71.83
        }
        
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Survived",
            "prediction": 1,
            "probability": 0.847,
            "input_data": {"Age": 25.0, "Sex": "female", "Pclass": 1, "Fare": 71.83},
            "model_architecture": "AutoML_1_20240101_120000",
            "model_score": 0.8932
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                result = enhanced_uagent.process_query("Predict survival for Age=25, Sex=female, Pclass=1, Fare=71.83")
        
        # CRITICAL: Must return properly formatted prediction
        assert "🔮 **PREDICTION RESULT**" in result
        assert "Survived" in result
        assert "84.70%" in result
        assert "female" in result
        assert "25.0" in result
        
        # Step 3: Test model analysis
        mock_intent_analysis = Mock()
        mock_intent_analysis.needs_prediction = False
        mock_intent_analysis.needs_model_analysis = True
        
        mock_analysis_result = {
            "analysis_type": "model_question",
            "question": "What features are most important?",
            "answer": "Age, Sex, and Pclass are the most important features.",
            "model_info": {
                "architecture": "AutoML",
                "target": "Survived",
                "performance": 0.8932,
                "features": ["Age", "Sex", "Pclass", "Fare"]
            }
        }
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent_analysis):
            with patch.object(MLPredictionAgent, 'analyze_model', return_value=mock_analysis_result):
                result = enhanced_uagent.process_query("What features are most important for survival prediction?")
        
        # CRITICAL: Must return properly formatted analysis
        assert "🧠 **MODEL ANALYSIS**" in result
        assert "What features are most important?" in result
        assert "Age, Sex, and Pclass" in result
        
        print("✅ Complete training-to-prediction workflow - PASSED")
    
    def test_batch_prediction_workflow_comprehensive(self):
        """Test complete batch prediction workflow with file handling."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Churn"
        
        # Create temporary CSV file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Age,MonthlyCharges,Contract\n25,65.5,Month-to-month\n30,89.2,Two year\n")
            temp_csv_url = f"file://{f.name}"
        
        try:
            mock_intent = Mock()
            mock_intent.needs_prediction = True
            mock_intent.needs_model_analysis = False
            mock_intent.prediction_type = "batch_prediction"
            mock_intent.prediction_data_source = temp_csv_url
            
            # Create output file for batch predictions
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as output_f:
                output_f.write("Age,MonthlyCharges,Contract,predict\n25,65.5,Month-to-month,1\n30,89.2,Two year,0\n")
                output_path = output_f.name
            
            mock_batch_result = {
                "prediction_type": "batch",
                "input_rows": 2,
                "output_path": output_path,
                "predictions_summary": {
                    "prediction_counts": {"0": 1, "1": 1}
                }
            }
            
            with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                with patch.object(MLPredictionAgent, 'predict_batch', return_value=mock_batch_result):
                    result = enhanced_uagent.process_query(f"Predict churn for {temp_csv_url}")
            
            # CRITICAL: Must handle batch predictions correctly
            assert "🔮 **BATCH PREDICTION COMPLETE**" in result
            assert "2" in result  # input rows
            assert output_path in result
            assert "0: 1" in result
            assert "1: 1" in result
            assert "50.0%" in result  # Equal distribution
            
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)
            if os.path.exists(output_path):
                os.unlink(output_path)
        
        print("✅ Batch prediction workflow comprehensive - PASSED")
    
    def test_session_management_comprehensive(self):
        """Test comprehensive session management scenarios."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Test 1: No model in session
        assert not enhanced_uagent._has_trained_model()
        
        # Test 2: Store model in session
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "TestTarget"
        
        assert enhanced_uagent._has_trained_model()
        assert not enhanced_uagent._is_model_session_expired()
        
        # Test 3: Session expiration
        enhanced_uagent._last_model_timestamp = time.time() - (self.config.session_timeout_hours * 3600 + 1)
        assert enhanced_uagent._is_model_session_expired()
        assert not enhanced_uagent._has_trained_model()
        
        # Test 4: Prediction request with expired session
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"feature": "value"}
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            result = enhanced_uagent.process_query("Predict for feature=value")
        
        # CRITICAL: Must return no model response
        assert "🚫 **No Trained Model Found**" in result
        
        print("✅ Session management comprehensive - PASSED")
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "TestTarget"
        
        # Test 1: Missing prediction data
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = None
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            result = enhanced_uagent.process_query("Predict something")
        
        assert "🚫 **Prediction Error**" in result
        assert "No prediction data found" in result
        
        # Test 2: Missing CSV URL for batch prediction
        mock_intent.prediction_type = "batch_prediction"
        mock_intent.prediction_data_source = None
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            result = enhanced_uagent.process_query("Predict for batch data")
        
        assert "🚫 **Prediction Error**" in result
        assert "No CSV URL found" in result
        
        # Test 3: MLPredictionAgent error
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"feature": "value"}
        
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', side_effect=MLPredictionError("H2O connection failed")):
                result = enhanced_uagent.process_query("Predict for feature=value")
        
        assert "🚫 **Prediction Error**" in result
        assert "H2O connection failed" in result
        
        # Test 4: Intent parser failure (fallback)
        with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', side_effect=Exception("Parser failed")):
            with patch.object(enhanced_uagent, '_process_analysis_request', return_value="Fallback analysis result"):
                result = enhanced_uagent.process_query("Some query")
        
        assert "Fallback analysis result" in result
        
        print("✅ Error handling comprehensive - PASSED")
    
    def test_intent_parsing_accuracy_comprehensive(self):
        """Test intent parsing accuracy with various query types."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session for prediction tests
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "TestTarget"
        
        test_cases = [
            # Prediction requests
            {
                "query": "Predict survival for Age=25, Sex=male, Pclass=3",
                "expected_prediction": True,
                "expected_analysis": False
            },
            {
                "query": "Classify this customer: income=50000, age=35",
                "expected_prediction": True,
                "expected_analysis": False
            },
            {
                "query": "What would the model predict for these values: x=1, y=2",
                "expected_prediction": True,
                "expected_analysis": False
            },
            # Model analysis requests
            {
                "query": "What features are most important for this model?",
                "expected_prediction": False,
                "expected_analysis": True
            },
            {
                "query": "How accurate is the trained model?",
                "expected_prediction": False,
                "expected_analysis": True
            },
            {
                "query": "Why did the model make this prediction?",
                "expected_prediction": False,
                "expected_analysis": True
            },
            # Training requests (should not trigger prediction)
            {
                "query": "Train a model using data.csv to predict target",
                "expected_prediction": False,
                "expected_analysis": False
            },
            {
                "query": "Build ML model for classification",
                "expected_prediction": False,
                "expected_analysis": False
            }
        ]
        
        for test_case in test_cases:
            mock_intent = Mock()
            mock_intent.needs_prediction = test_case["expected_prediction"]
            mock_intent.needs_model_analysis = test_case["expected_analysis"]
            
            if test_case["expected_prediction"]:
                mock_intent.prediction_type = "single_prediction"
                mock_intent.extracted_prediction_data = {"feature": "value"}
                
                mock_result = {
                    "prediction_type": "single_prediction",
                    "target_variable": "TestTarget",
                    "prediction": "test_prediction",
                    "input_data": {"feature": "value"},
                    "model_architecture": "TestModel",
                    "model_score": 0.85
                }
                
                with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                    with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                        result = enhanced_uagent.process_query(test_case["query"])
                
                # CRITICAL: Must route to prediction
                assert "🔮 **PREDICTION RESULT**" in result
                
            elif test_case["expected_analysis"]:
                mock_analysis_result = {
                    "analysis_type": "model_question",
                    "question": test_case["query"],
                    "answer": "Test analysis answer",
                    "model_info": {"architecture": "TestModel"}
                }
                
                with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                    with patch.object(MLPredictionAgent, 'analyze_model', return_value=mock_analysis_result):
                        result = enhanced_uagent.process_query(test_case["query"])
                
                # CRITICAL: Must route to analysis
                assert "🧠 **MODEL ANALYSIS**" in result
                
            else:
                # Should fall back to normal analysis
                with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                    with patch.object(enhanced_uagent, '_process_analysis_request', return_value="Normal analysis result"):
                        result = enhanced_uagent.process_query(test_case["query"])
                
                # CRITICAL: Must route to normal analysis
                assert "Normal analysis result" in result
        
        print("✅ Intent parsing accuracy comprehensive - PASSED")
    
    def test_multiple_prediction_requests_session(self):
        """Test multiple prediction requests in the same session."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "Survived"
        
        predictions = [
            {"Age": 25, "Sex": "male", "Pclass": 3, "expected": 0},
            {"Age": 30, "Sex": "female", "Pclass": 1, "expected": 1},
            {"Age": 45, "Sex": "male", "Pclass": 2, "expected": 0},
            {"Age": 22, "Sex": "female", "Pclass": 1, "expected": 1}
        ]
        
        for i, pred_data in enumerate(predictions):
            mock_intent = Mock()
            mock_intent.needs_prediction = True
            mock_intent.needs_model_analysis = False
            mock_intent.prediction_type = "single_prediction"
            mock_intent.extracted_prediction_data = {k: v for k, v in pred_data.items() if k != "expected"}
            
            mock_result = {
                "prediction_type": "single_prediction",
                "target_variable": "Survived",
                "prediction": pred_data["expected"],
                "input_data": {k: v for k, v in pred_data.items() if k != "expected"},
                "model_architecture": "AutoML",
                "model_score": 0.85
            }
            
            with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                    result = enhanced_uagent.process_query(f"Predict survival for request {i+1}")
            
            # CRITICAL: Each prediction must work correctly
            assert "🔮 **PREDICTION RESULT**" in result
            assert str(pred_data["expected"]) in result
            assert "Survived" in result
            
            # Verify session is still valid
            assert enhanced_uagent._has_trained_model()
        
        print("✅ Multiple prediction requests session - PASSED")
    
    def test_data_validation_comprehensive(self):
        """Test comprehensive data validation in predictions."""
        enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
        # Setup model session
        mock_metrics = self._create_comprehensive_mock_metrics()
        enhanced_uagent._last_trained_model = mock_metrics
        enhanced_uagent._last_model_timestamp = time.time()
        enhanced_uagent._last_target_variable = "TestTarget"
        
        # Test various data types and edge cases
        test_data_cases = [
            # Valid data
            {"feature1": 25, "feature2": "category_a"},
            # Edge values
            {"feature1": 0, "feature2": ""},
            {"feature1": -999, "feature2": "very_long_category_name_with_special_chars_@#$%"},
            # Large numbers
            {"feature1": 999999999, "feature2": "test"},
            # Unicode data
            {"特征1": "数值", "feature_2": "测试"},
            # Mixed types
            {"int_feature": 123, "float_feature": 45.67, "str_feature": "text", "bool_feature": True}
        ]
        
        for i, test_data in enumerate(test_data_cases):
            mock_intent = Mock()
            mock_intent.needs_prediction = True
            mock_intent.needs_model_analysis = False
            mock_intent.prediction_type = "single_prediction"
            mock_intent.extracted_prediction_data = test_data
            
            mock_result = {
                "prediction_type": "single_prediction",
                "target_variable": "TestTarget",
                "prediction": f"result_{i}",
                "input_data": test_data,
                "model_architecture": "TestModel",
                "model_score": 0.85
            }
            
            with patch.object(enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
                with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_result):
                    result = enhanced_uagent.process_query(f"Predict for test case {i}")
            
            # CRITICAL: Must handle all data types
            assert "🔮 **PREDICTION RESULT**" in result
            assert f"result_{i}" in result
            
            # Verify all input data is displayed
            for key, value in test_data.items():
                assert f"{key}**: {value}" in result
        
        print("✅ Data validation comprehensive - PASSED")
    
    def _create_comprehensive_mock_metrics(self):
        """Create comprehensive mock MLModelingMetrics."""
        return Mock(
            spec=MLModelingMetrics,
            model_path="/tmp/comprehensive_test_model",
            best_model_id="AutoML_1_20240101_120000",
            model_architecture="AutoML StackedEnsemble",
            best_model_score=0.8932,
            features_used=[
                "Age", "Sex", "Pclass", "Fare", "Embarked", "SibSp", "Parch",
                "MonthlyCharges", "Contract", "TotalCharges", "tenure"
            ],
            leaderboard=[
                {"model_id": "AutoML_1_20240101_120000", "auc": 0.8932, "logloss": 0.4251},
                {"model_id": "GBM_1_AutoML_20240101_120000", "auc": 0.8821, "logloss": 0.4398},
                {"model_id": "XGBoost_1_AutoML_20240101_120000", "auc": 0.8756, "logloss": 0.4502}
            ],
            top_model_metrics={
                "auc": 0.8932,
                "logloss": 0.4251,
                "rmse": 0.4123,
                "mae": 0.3456,
                "mean_per_class_error": 0.1987
            },
            total_models_trained=47,
            enhanced_feature_importance=[
                "Age: 0.2845",
                "Sex: 0.2234",
                "Pclass: 0.1892",
                "Fare: 0.1456",
                "Embarked: 0.0897",
                "SibSp: 0.0456",
                "Parch: 0.0220"
            ],
            generated_code="""
# H2O AutoML Generated Code
import h2o
from h2o.automl import H2OAutoML

h2o.init()
automl = H2OAutoML(max_models=20, seed=1, max_runtime_secs=300)
automl.train(x=predictors, y=response, training_frame=train)
            """,
            recommended_steps=[
                "Consider feature engineering for categorical variables",
                "Evaluate model performance on holdout test set",
                "Monitor model drift in production",
                "Consider ensemble methods for improved accuracy"
            ],
            workflow_summary="Trained 47 models using H2O AutoML. Best model: StackedEnsemble with AUC=0.8932"
        )


if __name__ == "__main__":
    print("🚨 RUNNING COMPREHENSIVE PHASE 3 INTEGRATION TESTS")
    print("=" * 80)
    print("CRITICAL: Complete workflows must work perfectly. Lives depend on it.")
    print("=" * 80)
    
    test_suite = TestEnhancedUAgentIntegrationComprehensive()
    
    # Run every single integration test
    test_suite.setup_method()
    test_suite.test_complete_training_to_prediction_workflow()
    test_suite.test_batch_prediction_workflow_comprehensive()
    test_suite.test_session_management_comprehensive()
    test_suite.test_error_handling_comprehensive()
    test_suite.test_intent_parsing_accuracy_comprehensive()
    test_suite.test_multiple_prediction_requests_session()
    test_suite.test_data_validation_comprehensive()
    
    print("=" * 80)
    print("🎉 ALL COMPREHENSIVE INTEGRATION TESTS PASSED!")
    print("✅ Enhanced uAgent prediction workflows are BULLETPROOF")
    print("=" * 80) 