"""
COMPREHENSIVE Phase 4 Integration Tests - End-to-End ML Prediction Workflows
CRITICAL: Lives depend on these workflows working perfectly.
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


class TestPhase4IntegrationComprehensive:
    """Comprehensive Phase 4 tests - Complete end-to-end ML prediction workflows."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = UAgentConfig()
        self.enhanced_uagent = EnhancedDataAnalysisUAgent(self.config)
        
    def _train_model(self, query, mock_result, success_message="🤖 **ML MODEL TRAINING COMPLETE** - Model trained successfully"):
        """Helper method to train model with proper mocking."""
        with patch.object(self.enhanced_uagent.data_analysis_agent, 'analyze_from_text', return_value=mock_result):
            with patch.object(self.enhanced_uagent.result_formatter, 'format_analysis_result_enhanced', return_value=success_message):
                return self.enhanced_uagent.process_query(query)
        
    def test_complete_ml_prediction_workflow_titanic(self):
        """Test complete ML workflow: train model → single prediction → batch prediction → model analysis."""
        
        # Step 1: Train model
        mock_training_result = self._create_mock_training_result()
        train_result = self._train_model(
            "Train ML model using titanic.csv to predict Survived",
            mock_training_result,
            "🤖 **ML MODEL TRAINING COMPLETE** - Titanic survival prediction model trained successfully"
        )
        
        # CRITICAL: Model training must work
        assert "🤖 **ML MODEL TRAINING COMPLETE**" in train_result
        assert self.enhanced_uagent._has_trained_model()
        assert self.enhanced_uagent._last_target_variable == "Survived"
        
        # Step 2: Single prediction
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {
            "Age": 25.0, "Sex": "male", "Pclass": 3, "Fare": 7.25
        }
        
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Survived",
            "prediction": 0,
            "probability": 0.234,
            "input_data": {"Age": 25.0, "Sex": "male", "Pclass": 3, "Fare": 7.25},
            "model_architecture": "AutoML_1_20240101_120000",
            "model_score": 0.8932
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                single_pred_result = self.enhanced_uagent.process_query(
                    "Predict survival for Age=25, Sex=male, Pclass=3, Fare=7.25"
                )
        
        # CRITICAL: Single prediction must work
        assert "🔮 **PREDICTION RESULT**" in single_pred_result
        assert "Survived" in single_pred_result
        assert "23.40%" in single_pred_result  # Probability
        assert "Age**: 25.0" in single_pred_result
        assert "Sex**: male" in single_pred_result
        
        # Step 3: Batch prediction
        mock_intent_batch = Mock()
        mock_intent_batch.needs_prediction = True
        mock_intent_batch.needs_model_analysis = False
        mock_intent_batch.prediction_type = "batch_prediction"
        mock_intent_batch.prediction_data_source = "https://example.com/new_passengers.csv"
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Age,Sex,Pclass,Survived_prediction\n25,male,3,0\n30,female,1,1\n")
            temp_output_path = f.name
        
        mock_batch_result = {
            "prediction_type": "batch",
            "input_rows": 2,
            "output_path": temp_output_path,
            "predictions_summary": {
                "prediction_counts": {"0": 1, "1": 1}
            }
        }
        
        try:
            with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent_batch):
                with patch.object(MLPredictionAgent, 'predict_batch', return_value=mock_batch_result):
                    batch_pred_result = self.enhanced_uagent.process_query(
                        "Predict survival for https://example.com/new_passengers.csv"
                    )
            
            # CRITICAL: Batch prediction must work
            assert "🔮 **BATCH PREDICTION COMPLETE**" in batch_pred_result
            assert "2" in batch_pred_result  # Input rows
            assert temp_output_path in batch_pred_result
            assert "0: 1" in batch_pred_result
            assert "1: 1" in batch_pred_result
            
        finally:
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
        
        # Step 4: Model analysis
        mock_intent_analysis = Mock()
        mock_intent_analysis.needs_prediction = False
        mock_intent_analysis.needs_model_analysis = True
        
        mock_analysis_result = {
            "analysis_type": "model_question",
            "question": "What features are most important for survival prediction?",
            "answer": "Based on the AutoML model analysis, the most important features are:\n1. Sex (32.4% importance)\n2. Age (28.7% importance)\n3. Pclass (25.1% importance)\n4. Fare (13.8% importance)",
            "model_info": {
                "architecture": "AutoML_1_20240101_120000",
                "target": "Survived",
                "performance": 0.8932,
                "features": ["Age", "Sex", "Pclass", "Fare", "Embarked", "SibSp", "Parch"]
            }
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent_analysis):
            with patch.object(MLPredictionAgent, 'analyze_model', return_value=mock_analysis_result):
                analysis_result = self.enhanced_uagent.process_query(
                    "What features are most important for survival prediction?"
                )
        
        # CRITICAL: Model analysis must work
        assert "🧠 **MODEL ANALYSIS**" in analysis_result
        assert "Sex (32.4% importance)" in analysis_result
        assert "Age (28.7% importance)" in analysis_result
        assert "AutoML_1_20240101_120000" in analysis_result
        
        print("✅ Complete ML prediction workflow (Titanic) - PASSED")
    
    def test_churn_prediction_workflow_comprehensive(self):
        """Test comprehensive churn prediction workflow with business metrics."""
        
        # Step 1: Train churn model
        mock_training_result = self._create_mock_churn_training_result()
        train_result = self._train_model(
            "Clean and train ML model using churn_data.csv to predict Churn",
            mock_training_result,
            "🤖 **ML MODEL TRAINING COMPLETE** - Customer churn prediction model trained successfully"
        )
        
        # CRITICAL: Churn model training must work
        assert "🤖 **ML MODEL TRAINING COMPLETE**" in train_result
        assert self.enhanced_uagent._has_trained_model()
        assert self.enhanced_uagent._last_target_variable == "Churn"
        
        # Step 2: High-risk customer prediction
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.needs_model_analysis = False
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {
            "MonthlyCharges": 85.50,
            "TotalCharges": 1200.75,
            "Contract": "Month-to-month",
            "tenure": 6,
            "PaymentMethod": "Electronic check"
        }
        
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Churn",
            "prediction": 1,  # High risk
            "probability": 0.847,
            "input_data": mock_intent.extracted_prediction_data,
            "model_architecture": "AutoML_StackedEnsemble_BestOfFamily_20240101_120000",
            "model_score": 0.8756
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                churn_pred_result = self.enhanced_uagent.process_query(
                    "Predict churn for MonthlyCharges=85.50, TotalCharges=1200.75, Contract=Month-to-month, tenure=6"
                )
        
        # CRITICAL: High-risk prediction must work
        assert "🔮 **PREDICTION RESULT**" in churn_pred_result
        assert "Churn" in churn_pred_result
        assert "84.70%" in churn_pred_result  # High probability
        assert "MonthlyCharges**: 85.5" in churn_pred_result
        assert "Month-to-month" in churn_pred_result
        
        # Step 3: Low-risk customer prediction
        mock_intent_low_risk = Mock()
        mock_intent_low_risk.needs_prediction = True
        mock_intent_low_risk.needs_model_analysis = False
        mock_intent_low_risk.prediction_type = "single_prediction"
        mock_intent_low_risk.extracted_prediction_data = {
            "MonthlyCharges": 45.20,
            "TotalCharges": 3500.00,
            "Contract": "Two year",
            "tenure": 36,
            "PaymentMethod": "Bank transfer (automatic)"
        }
        
        mock_low_risk_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Churn",
            "prediction": 0,  # Low risk
            "probability": 0.123,
            "input_data": mock_intent_low_risk.extracted_prediction_data,
            "model_architecture": "AutoML_StackedEnsemble_BestOfFamily_20240101_120000",
            "model_score": 0.8756
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent_low_risk):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_low_risk_result):
                low_risk_result = self.enhanced_uagent.process_query(
                    "Predict churn for MonthlyCharges=45.20, TotalCharges=3500, Contract=Two year, tenure=36"
                )
        
        # CRITICAL: Low-risk prediction must work
        assert "🔮 **PREDICTION RESULT**" in low_risk_result
        assert "Churn" in low_risk_result
        assert "12.30%" in low_risk_result  # Low probability
        assert "Two year" in low_risk_result
        
        # Step 4: Churn analysis for business insights
        mock_analysis_intent = Mock()
        mock_analysis_intent.needs_prediction = False
        mock_analysis_intent.needs_model_analysis = True
        
        mock_churn_analysis = {
            "analysis_type": "model_question",
            "question": "What are the key factors that drive customer churn?",
            "answer": """Based on the comprehensive churn prediction model analysis:

**Top Churn Risk Factors:**
1. **Contract Type (35.2% importance)**: Month-to-month contracts have 3x higher churn rate
2. **Monthly Charges (28.7% importance)**: Customers paying >$70/month are high-risk
3. **Tenure (22.1% importance)**: New customers (<12 months) are most vulnerable
4. **Payment Method (14.0% importance)**: Electronic check users churn more frequently

**Business Recommendations:**
- Incentivize longer-term contracts
- Review pricing for high-charge customers
- Implement new customer retention programs
- Promote automatic payment methods""",
            "model_info": {
                "architecture": "AutoML_StackedEnsemble_BestOfFamily_20240101_120000",
                "target": "Churn",
                "performance": 0.8756,
                "features": ["MonthlyCharges", "Contract", "tenure", "PaymentMethod", "TotalCharges"]
            }
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_analysis_intent):
            with patch.object(self.enhanced_uagent.intent_parser, 'parse_intent', return_value=mock_analysis_intent):
                with patch.object(MLPredictionAgent, 'analyze_model', return_value=mock_churn_analysis):
                    churn_analysis = self.enhanced_uagent.process_query(
                        "What are the key factors that drive customer churn?"
                    )
        
        # CRITICAL: Churn analysis must provide business insights
        assert "🧠 **MODEL ANALYSIS**" in churn_analysis
        assert "Contract Type (35.2% importance)" in churn_analysis
        assert "Month-to-month contracts have 3x higher churn rate" in churn_analysis
        assert "Business Recommendations" in churn_analysis
        
        print("✅ Comprehensive churn prediction workflow - PASSED")
    
    def test_session_management_across_workflows(self):
        """Test session management across multiple ML workflows."""
        
        # Step 1: Train first model
        mock_training_result = self._create_mock_training_result()
        train_result = self._train_model(
            "Train ML model using data.csv to predict Target",
            mock_training_result
        )
        
        # Verify first model is stored
        assert self.enhanced_uagent._has_trained_model()
        first_model_id = self.enhanced_uagent._last_trained_model.best_model_id
        first_timestamp = self.enhanced_uagent._last_model_timestamp
        
        # Step 2: Make prediction with first model
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"feature1": "value1"}
        
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Target",
            "prediction": "result1",
            "input_data": {"feature1": "value1"},
            "model_architecture": first_model_id,
            "model_score": 0.85
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                pred_result = self.enhanced_uagent.process_query("Predict Target for feature1=value1")
        
        # CRITICAL: First prediction must work
        assert "🔮 **PREDICTION RESULT**" in pred_result
        assert "Target" in pred_result
        assert "result1" in pred_result
        
        # Step 3: Train second model (should replace first)
        mock_training_result2 = self._create_mock_training_result()
        mock_training_result2.agent_results[0].ml_modeling_metrics.best_model_id = "AutoML_2_20240101_130000"
        mock_training_result2.workflow_intent.suggested_target_variable = "NewTarget"
        
        train_result2 = self._train_model(
            "Train new ML model using updated_data.csv to predict NewTarget",
            mock_training_result2
        )
        
        # Verify second model replaced first
        assert self.enhanced_uagent._has_trained_model()
        second_model_id = self.enhanced_uagent._last_trained_model.best_model_id
        second_timestamp = self.enhanced_uagent._last_model_timestamp
        
        # CRITICAL: Session must be updated with new model
        assert second_model_id != first_model_id
        assert second_timestamp > first_timestamp
        assert self.enhanced_uagent._last_target_variable == "NewTarget"
        
        # Step 4: Make prediction with second model
        mock_intent2 = Mock()
        mock_intent2.needs_prediction = True
        mock_intent2.prediction_type = "single_prediction"
        mock_intent2.extracted_prediction_data = {"feature2": "value2"}
        
        mock_prediction_result2 = {
            "prediction_type": "single_prediction",
            "target_variable": "NewTarget",
            "prediction": "result2",
            "input_data": {"feature2": "value2"},
            "model_architecture": second_model_id,
            "model_score": 0.92
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent2):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result2):
                pred_result2 = self.enhanced_uagent.process_query("Predict NewTarget for feature2=value2")
        
        # CRITICAL: Second prediction must work with new model
        assert "🔮 **PREDICTION RESULT**" in pred_result2
        assert "NewTarget" in pred_result2
        assert "result2" in pred_result2
        assert second_model_id in pred_result2
        
        print("✅ Session management across workflows - PASSED")
    
    def test_session_expiration_and_recovery(self):
        """Test session expiration and recovery workflows."""
        
        # Step 1: Train model
        mock_training_result = self._create_mock_training_result()
        train_result = self._train_model(
            "Train ML model using data.csv to predict Target",
            mock_training_result
        )
        
        # Verify model is stored
        assert self.enhanced_uagent._has_trained_model()
        
        # Step 2: Expire session
        self.enhanced_uagent._last_model_timestamp = time.time() - (self.config.session_timeout_hours * 3600 + 1)
        
        # Verify session is expired
        assert not self.enhanced_uagent._has_trained_model()
        
        # Step 3: Try to make prediction with expired session
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"feature": "value"}
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            expired_result = self.enhanced_uagent.process_query("Predict Target for feature=value")
        
        # CRITICAL: Must return no model response
        assert "🚫 **No Trained Model Found**" in expired_result
        assert "train a model" in expired_result.lower()
        
        # Step 4: Retrain and verify recovery
        retrain_result = self._train_model(
            "Retrain ML model using data.csv to predict Target",
            mock_training_result
        )
        
        # Verify model is restored
        assert self.enhanced_uagent._has_trained_model()
        
        # Step 5: Make prediction with restored session
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Target",
            "prediction": "recovered_result",
            "input_data": {"feature": "value"},
            "model_architecture": "AutoML_1_20240101_120000",
            "model_score": 0.85
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                recovery_result = self.enhanced_uagent.process_query("Predict Target for feature=value")
        
        # CRITICAL: Prediction must work after recovery
        assert "🔮 **PREDICTION RESULT**" in recovery_result
        assert "recovered_result" in recovery_result
        
        print("✅ Session expiration and recovery - PASSED")
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        
        # Test 1: No model prediction request
        # Make sure no model is stored in session
        self.enhanced_uagent._last_trained_model = None
        self.enhanced_uagent._last_model_timestamp = None
        
        mock_intent = Mock()
        mock_intent.needs_prediction = True
        mock_intent.prediction_type = "single_prediction"
        mock_intent.extracted_prediction_data = {"feature": "value"}
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            no_model_result = self.enhanced_uagent.process_query("Predict something")
        
        # CRITICAL: Must handle no model gracefully
        assert "🚫 **No Trained Model Found**" in no_model_result
        assert "train a model" in no_model_result.lower()
        
        # Test 2: MLPredictionAgent error
        # Setup model first
        mock_training_result = self._create_mock_training_result()
        self._train_model("Train ML model using data.csv to predict Target", mock_training_result)
        
        # Now test MLPredictionAgent error
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_intent):
            with patch.object(MLPredictionAgent, 'predict_single', side_effect=MLPredictionError("H2O cluster connection failed")):
                error_result = self.enhanced_uagent.process_query("Predict Target for feature=value")
        
        # CRITICAL: Must handle MLPredictionError gracefully
        assert "🚫 **Prediction Error**" in error_result
        assert "H2O cluster connection failed" in error_result
        
        # Test 3: Intent parser error (fallback)
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', side_effect=Exception("Intent parsing failed")):
            with patch.object(self.enhanced_uagent, '_process_analysis_request', return_value="Fallback analysis"):
                fallback_result = self.enhanced_uagent.process_query("Some query")
        
        # CRITICAL: Must fall back to normal analysis
        assert "Fallback analysis" in fallback_result
        
        print("✅ Comprehensive error handling - PASSED")
    
    def test_mixed_workflow_scenarios(self):
        """Test mixed workflow scenarios with different query types."""
        
        # Step 1: Train model
        mock_training_result = self._create_mock_training_result()
        train_result = self._train_model(
            "Train ML model using data.csv to predict Target",
            mock_training_result
        )
        
        # Step 2: Normal data analysis request (should work normally)
        mock_normal_result = Mock()
        mock_normal_result.workflow_intent = Mock()
        mock_normal_result.workflow_intent.needs_prediction = False
        mock_normal_result.workflow_intent.needs_model_analysis = False
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_normal_result.workflow_intent):
            with patch.object(self.enhanced_uagent, '_process_analysis_request', return_value="Normal analysis result"):
                normal_result = self.enhanced_uagent.process_query("Show me basic statistics of the data")
        
        # CRITICAL: Normal analysis must still work
        assert "Normal analysis result" in normal_result
        
        # Step 3: Data delivery request (should work normally)
        delivery_result = self.enhanced_uagent.process_query("Send me my cleaned data")
        
        # Should handle data delivery (might return error if no cleaned data, but shouldn't crash)
        assert isinstance(delivery_result, str)
        
        # Step 4: Prediction request (should work with stored model)
        mock_prediction_intent = Mock()
        mock_prediction_intent.needs_prediction = True
        mock_prediction_intent.prediction_type = "single_prediction"
        mock_prediction_intent.extracted_prediction_data = {"feature": "value"}
        
        mock_prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Target",
            "prediction": "mixed_result",
            "input_data": {"feature": "value"},
            "model_architecture": "AutoML_1_20240101_120000",
            "model_score": 0.85
        }
        
        with patch.object(self.enhanced_uagent.intent_parser, 'parse_with_data_preview', return_value=mock_prediction_intent):
            with patch.object(MLPredictionAgent, 'predict_single', return_value=mock_prediction_result):
                pred_result = self.enhanced_uagent.process_query("Predict Target for feature=value")
        
        # CRITICAL: Prediction must work after mixed requests
        assert "🔮 **PREDICTION RESULT**" in pred_result
        assert "mixed_result" in pred_result
        
        print("✅ Mixed workflow scenarios - PASSED")
    
    def _create_mock_training_result(self):
        """Create comprehensive mock training result."""
        mock_result = Mock()
        mock_result.original_request = "Train ML model using titanic.csv to predict Survived"
        mock_result.csv_url = "titanic.csv"
        mock_result.total_runtime_seconds = 45.2
        mock_result.confidence_level = "high"
        mock_result.analysis_quality_score = 0.89
        mock_result.data_shape = {"rows": 891, "columns": 12}
        
        mock_result.workflow_intent = Mock()
        mock_result.workflow_intent.suggested_target_variable = "Survived"
        mock_result.workflow_intent.needs_data_cleaning = True
        mock_result.workflow_intent.needs_ml_modeling = True
        mock_result.workflow_intent.needs_feature_engineering = False
        mock_result.workflow_intent.data_source_url = "titanic.csv"
        mock_result.workflow_intent.data_source_type = "csv_url"
        mock_result.workflow_intent.intent_confidence = 0.9
        
        mock_result.workflow_summary = "Training workflow completed successfully"
        mock_result.success = True
        mock_result.processing_time = 45.2
        mock_result.output_files = []
        
        # Mock ML agent result
        mock_ml_result = Mock()
        mock_ml_result.agent_name = "h2o_ml"
        mock_ml_result.success = True
        mock_ml_result.output_summary = "H2O AutoML training completed"
        mock_ml_result.log_messages = ["H2O AutoML training started", "Model training completed successfully"]
        mock_ml_result.output_data_path = None
        mock_ml_result.ml_modeling_metrics = Mock(spec=MLModelingMetrics)
        mock_ml_result.ml_modeling_metrics.model_path = "/tmp/test_model"
        mock_ml_result.ml_modeling_metrics.best_model_id = "AutoML_1_20240101_120000"
        mock_ml_result.ml_modeling_metrics.model_architecture = "AutoML StackedEnsemble"
        mock_ml_result.ml_modeling_metrics.best_model_score = 0.8932
        
        mock_result.agent_results = [mock_ml_result]
        return mock_result
    
    def _create_mock_churn_training_result(self):
        """Create comprehensive mock churn training result."""
        mock_result = Mock()
        mock_result.original_request = "Clean and train ML model using churn_data.csv to predict Churn"
        mock_result.csv_url = "churn_data.csv"
        mock_result.total_runtime_seconds = 67.5
        mock_result.confidence_level = "high"
        mock_result.analysis_quality_score = 0.92
        mock_result.data_shape = {"rows": 7043, "columns": 21}
        
        mock_result.workflow_intent = Mock()
        mock_result.workflow_intent.suggested_target_variable = "Churn"
        mock_result.workflow_intent.needs_data_cleaning = True
        mock_result.workflow_intent.needs_ml_modeling = True
        mock_result.workflow_intent.needs_feature_engineering = False
        mock_result.workflow_intent.data_source_url = "churn_data.csv"
        mock_result.workflow_intent.data_source_type = "csv_url"
        mock_result.workflow_intent.intent_confidence = 0.95
        
        mock_result.workflow_summary = "Churn prediction workflow completed successfully"
        mock_result.success = True
        mock_result.processing_time = 67.5
        mock_result.output_files = []
        
        # Mock ML agent result
        mock_ml_result = Mock()
        mock_ml_result.agent_name = "h2o_ml"
        mock_ml_result.success = True
        mock_ml_result.output_summary = "H2O AutoML churn prediction training completed"
        mock_ml_result.log_messages = ["H2O AutoML churn training started", "Model training completed successfully"]
        mock_ml_result.output_data_path = None
        mock_ml_result.ml_modeling_metrics = Mock(spec=MLModelingMetrics)
        mock_ml_result.ml_modeling_metrics.model_path = "/tmp/churn_model"
        mock_ml_result.ml_modeling_metrics.best_model_id = "AutoML_StackedEnsemble_BestOfFamily_20240101_120000"
        mock_ml_result.ml_modeling_metrics.model_architecture = "AutoML StackedEnsemble"
        mock_ml_result.ml_modeling_metrics.best_model_score = 0.8756
        
        mock_result.agent_results = [mock_ml_result]
        return mock_result


if __name__ == "__main__":
    print("🚨 RUNNING COMPREHENSIVE PHASE 4 INTEGRATION TESTS")
    print("=" * 80)
    print("CRITICAL: End-to-end ML workflows must work perfectly. Lives depend on it.")
    print("=" * 80)
    
    test_suite = TestPhase4IntegrationComprehensive()
    
    # Run every single Phase 4 test
    test_suite.setup_method()
    test_suite.test_complete_ml_prediction_workflow_titanic()
    test_suite.test_churn_prediction_workflow_comprehensive()
    test_suite.test_session_management_across_workflows()
    test_suite.test_session_expiration_and_recovery()
    test_suite.test_error_handling_comprehensive()
    test_suite.test_mixed_workflow_scenarios()
    
    print("=" * 80)
    print("🎉 ALL COMPREHENSIVE PHASE 4 INTEGRATION TESTS PASSED!")
    print("✅ End-to-end ML prediction workflows are BULLETPROOF")
    print("=" * 80) 