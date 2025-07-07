"""
COMPREHENSIVE Phase 3 Unit Tests - Prediction Formatters
CRITICAL: Every function must work perfectly. Someone's life depends on it.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.uagent_v2.config import UAgentConfig
from src.uagent_v2.prediction_formatters import PredictionResponseFormatter


class TestPredictionFormatterComprehensive:
    """Comprehensive tests for PredictionResponseFormatter - EVERY scenario must work."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = UAgentConfig()
        self.formatter = PredictionResponseFormatter(self.config)
    
    def test_single_prediction_formatting_complete(self):
        """Test single prediction formatting with ALL possible fields."""
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Survived", 
            "prediction": 1,
            "probability": 0.847,
            "input_data": {
                "Age": 25.0,
                "Sex": "female",
                "Pclass": 1,
                "Fare": 71.83,
                "Embarked": "C",
                "SibSp": 0,
                "Parch": 0
            },
            "model_architecture": "AutoML_1_20240101_120000",
            "model_score": 0.8932
        }
        
        result = self.formatter.format_single_prediction(prediction_result)
        
        # CRITICAL: Every field must be present
        assert "🔮 **PREDICTION RESULT**" in result
        assert "Survived" in result
        assert "84.70%" in result  # probability formatted correctly
        assert "Age**: 25.0" in result
        assert "Sex**: female" in result
        assert "Pclass**: 1" in result
        assert "Fare**: 71.83" in result
        assert "Embarked**: C" in result
        assert "SibSp**: 0" in result
        assert "Parch**: 0" in result
        assert "AutoML_1_20240101_120000" in result
        assert "0.8932" in result
        
        # CRITICAL: Structure must be correct
        assert "📋 **Input Features**:" in result
        assert "🤖 **Model Information**:" in result
        assert "💡 **Next Steps**:" in result
        
        print("✅ Single prediction formatting (complete) - PASSED")
    
    def test_single_prediction_formatting_minimal(self):
        """Test single prediction formatting with minimal required fields."""
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Price",
            "prediction": 123456.78,
            "input_data": {"feature1": "value1"},
            "model_architecture": "LinearRegression",
            "model_score": 0.75
        }
        
        result = self.formatter.format_single_prediction(prediction_result)
        
        # CRITICAL: Must handle minimal data gracefully
        assert "Price" in result
        assert "123456.78" in result
        assert "feature1**: value1" in result
        assert "LinearRegression" in result
        assert "0.75" in result
        assert "📈 **Confidence**:" not in result  # No probability provided
        
        print("✅ Single prediction formatting (minimal) - PASSED")
    
    def test_single_prediction_formatting_edge_cases(self):
        """Test single prediction formatting with edge case values."""
        # Test with extreme values
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Edge_Case_Target",
            "prediction": -999999.999,
            "probability": 0.000001,  # Very low probability
            "input_data": {
                "negative_feature": -123.45,
                "zero_feature": 0,
                "large_feature": 999999999,
                "string_with_special_chars": "test@#$%^&*()",
                "unicode_feature": "测试数据",
                "boolean_feature": True,
                "none_feature": None
            },
            "model_architecture": "Very_Long_Model_Name_With_Special_Characters_@#$%",
            "model_score": 0.999999
        }
        
        result = self.formatter.format_single_prediction(prediction_result)
        
        # CRITICAL: Must handle all edge cases without errors
        assert "Edge_Case_Target" in result
        assert "-999999.999" in result
        assert "0.00%" in result  # Very low probability
        assert "negative_feature**: -123.45" in result
        assert "zero_feature**: 0" in result
        assert "large_feature**: 999999999" in result
        assert "test@#$%^&*()" in result
        assert "测试数据" in result
        assert "boolean_feature**: True" in result
        assert "none_feature**: None" in result
        assert "Very_Long_Model_Name_With_Special_Characters_@#$%" in result
        assert "0.999999" in result
        
        print("✅ Single prediction formatting (edge cases) - PASSED")
    
    def test_batch_prediction_formatting_complete(self):
        """Test batch prediction formatting with complete data."""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Age,Sex,Survived\n25,male,0\n30,female,1\n")
            temp_file = f.name
        
        try:
            prediction_result = {
                "prediction_type": "batch",
                "input_rows": 1000,
                "output_path": temp_file,
                "predictions_summary": {
                    "prediction_counts": {"0": 623, "1": 377},
                    "additional_stats": {"accuracy": 0.85}
                }
            }
            
            result = self.formatter.format_batch_prediction(prediction_result)
            
            # CRITICAL: All batch information must be present
            assert "🔮 **BATCH PREDICTION COMPLETE**" in result
            assert "1,000" in result  # Comma formatting for thousands
            assert temp_file in result
            assert "📈 **Prediction Summary**:" in result
            assert "0: 623" in result
            assert "1: 377" in result
            assert "62.3%" in result  # Percentage calculation
            assert "37.7%" in result
            
        finally:
            os.unlink(temp_file)
        
        print("✅ Batch prediction formatting (complete) - PASSED")
    
    def test_batch_prediction_formatting_regression(self):
        """Test batch prediction formatting with regression statistics."""
        prediction_result = {
            "prediction_type": "batch",
            "input_rows": 500,
            "output_path": "/tmp/predictions.csv",
            "predictions_summary": {
                "prediction_stats": {
                    "mean": 123456.789,
                    "std": 54321.123,
                    "min": -999.999,
                    "max": 999999.999
                }
            }
        }
        
        with patch('os.path.exists', return_value=False):  # File doesn't exist
            result = self.formatter.format_batch_prediction(prediction_result)
        
        # CRITICAL: Regression stats must be formatted correctly
        assert "500" in result
        assert "123456.789" in result
        assert "54321.123" in result
        assert "-999.999" in result
        assert "999999.999" in result
        assert "**Statistical Summary**:" in result
        
        print("✅ Batch prediction formatting (regression) - PASSED")
    
    def test_model_analysis_formatting_comprehensive(self):
        """Test model analysis formatting with comprehensive model info."""
        analysis_result = {
            "analysis_type": "model_question",
            "question": "What are the most important features for predicting customer churn?",
            "answer": """Based on the comprehensive analysis of the trained AutoML model, the most important features for predicting customer churn are:

1. **Monthly Charges (35.2% importance)**: Higher monthly charges strongly correlate with increased churn probability.
2. **Contract Type (28.7% importance)**: Month-to-month contracts have significantly higher churn rates.
3. **Total Charges (22.1% importance)**: Customers with lower total charges are more likely to churn.
4. **Tenure (18.9% importance)**: Newer customers have higher churn probability.

The model achieved 87.3% accuracy with these features.""",
            "model_info": {
                "architecture": "AutoML_StackedEnsemble_BestOfFamily_20240101_120000",
                "target": "Churn",
                "performance": 0.8734,
                "training_duration": 1847,
                "total_models_trained": 47,
                "features": [
                    "MonthlyCharges", "Contract", "TotalCharges", "tenure", 
                    "PaymentMethod", "InternetService", "gender", "SeniorCitizen",
                    "Partner", "Dependents", "PhoneService", "MultipleLines",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"
                ]
            }
        }
        
        result = self.formatter.format_model_analysis(analysis_result)
        
        # CRITICAL: All analysis components must be present
        assert "🧠 **MODEL ANALYSIS**" in result
        assert "What are the most important features for predicting customer churn?" in result
        assert "Monthly Charges (35.2% importance)" in result
        assert "87.3% accuracy" in result
        assert "AutoML_StackedEnsemble_BestOfFamily_20240101_120000" in result
        assert "Churn" in result
        assert "0.8734" in result
        assert "1847 seconds" in result
        assert "47" in result
        assert "19 features" in result  # Feature count
        assert "MonthlyCharges, Contract, TotalCharges, tenure, PaymentMethod" in result
        
        print("✅ Model analysis formatting (comprehensive) - PASSED")
    
    def test_no_model_response_formatting(self):
        """Test no model response formatting."""
        result = self.formatter.format_no_model_response()
        
        # CRITICAL: Must provide clear guidance
        assert "🚫 **No Trained Model Found**" in result
        assert "train a model" in result.lower()
        assert "predict" in result.lower()
        assert "example workflow" in result.lower()
        assert "titanic.csv" in result
        assert "batch predictions" in result.lower()
        assert "model analysis" in result.lower()
        
        print("✅ No model response formatting - PASSED")
    
    def test_prediction_error_response_formatting(self):
        """Test prediction error response formatting."""
        test_error = Exception("H2O cluster connection failed: timeout after 30 seconds")
        
        result = self.formatter.format_prediction_error_response(test_error)
        
        # CRITICAL: Error must be clearly communicated with solutions
        assert "🚫 **Prediction Error**" in result
        assert "H2O cluster connection failed: timeout after 30 seconds" in result
        assert "Common solutions:" in result
        assert "Check Input Format" in result
        assert "Verify Features" in result
        assert "Model Session" in result
        assert "Retrain Model" in result
        assert f"{self.config.session_timeout_hours} hours" in result
        
        print("✅ Prediction error response formatting - PASSED")
    
    def test_formatting_with_none_values(self):
        """Test formatting handles None values gracefully."""
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": None,
            "prediction": None,
            "probability": None,
            "input_data": None,
            "model_architecture": None,
            "model_score": None
        }
        
        # CRITICAL: Must not crash with None values
        result = self.formatter.format_single_prediction(prediction_result)
        
        assert "🔮 **PREDICTION RESULT**" in result
        assert "Unknown" in result or "N/A" in result
        
        print("✅ Formatting with None values - PASSED")
    
    def test_formatting_with_empty_data(self):
        """Test formatting handles empty data structures."""
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "",
            "prediction": "",
            "input_data": {},
            "model_architecture": "",
            "model_score": ""
        }
        
        # CRITICAL: Must handle empty data gracefully
        result = self.formatter.format_single_prediction(prediction_result)
        
        assert "🔮 **PREDICTION RESULT**" in result
        assert "📋 **Input Features**:" in result
        
        print("✅ Formatting with empty data - PASSED")
    
    def test_unicode_and_special_characters(self):
        """Test formatting handles Unicode and special characters."""
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "目标变量_测试",
            "prediction": "分类结果_A",
            "input_data": {
                "特征1": "数值123",
                "feature_2": "café_résumé",
                "特殊符号": "@#$%^&*()_+-=[]{}|;':\",./<>?",
                "emoji_test": "🎯🔮📊💡"
            },
            "model_architecture": "模型架构_测试",
            "model_score": 0.95
        }
        
        # CRITICAL: Must handle international characters and emojis
        result = self.formatter.format_single_prediction(prediction_result)
        
        assert "目标变量_测试" in result
        assert "分类结果_A" in result
        assert "特征1**: 数值123" in result
        assert "café_résumé" in result
        assert "@#$%^&*()_+-=[]{}|;':\",./<>?" in result
        assert "🎯🔮📊💡" in result
        assert "模型架构_测试" in result
        
        print("✅ Unicode and special characters - PASSED")
    
    def test_large_data_handling(self):
        """Test formatting handles large datasets and values."""
        # Large input data
        large_input_data = {f"feature_{i}": f"value_{i}" for i in range(1000)}
        
        prediction_result = {
            "prediction_type": "single_prediction",
            "target_variable": "Large_Dataset_Target",
            "prediction": 999999999999.123456789,
            "input_data": large_input_data,
            "model_architecture": "A" * 1000,  # Very long model name
            "model_score": 0.999999999999
        }
        
        # CRITICAL: Must handle large data without performance issues
        result = self.formatter.format_single_prediction(prediction_result)
        
        assert "Large_Dataset_Target" in result
        assert "999999999999" in result  # Check for significant digits (Python may truncate decimals)
        assert "feature_1**: value_1" in result
        assert "feature_999**: value_999" in result
        assert len(result) > 1000  # Should contain substantial content
        
        print("✅ Large data handling - PASSED")


if __name__ == "__main__":
    print("🚨 RUNNING COMPREHENSIVE PHASE 3 UNIT TESTS")
    print("=" * 80)
    print("CRITICAL: Every test must pass. Someone's life depends on it.")
    print("=" * 80)
    
    test_suite = TestPredictionFormatterComprehensive()
    
    # Run every single test
    test_suite.setup_method()
    test_suite.test_single_prediction_formatting_complete()
    test_suite.test_single_prediction_formatting_minimal()
    test_suite.test_single_prediction_formatting_edge_cases()
    test_suite.test_batch_prediction_formatting_complete()
    test_suite.test_batch_prediction_formatting_regression()
    test_suite.test_model_analysis_formatting_comprehensive()
    test_suite.test_no_model_response_formatting()
    test_suite.test_prediction_error_response_formatting()
    test_suite.test_formatting_with_none_values()
    test_suite.test_formatting_with_empty_data()
    test_suite.test_unicode_and_special_characters()
    test_suite.test_large_data_handling()
    
    print("=" * 80)
    print("🎉 ALL COMPREHENSIVE UNIT TESTS PASSED!")
    print("✅ Prediction formatters are BULLETPROOF")
    print("=" * 80) 