"""
Enhanced Intent Parser Test Suite
Tests the improved prediction pattern recognition and context awareness
"""

import pytest
import asyncio
from src.parsers.intent_parser import DataAnalysisIntentParser
from src.schemas import WorkflowIntent

class TestEnhancedIntentParser:
    """Test suite for enhanced intent parser improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DataAnalysisIntentParser()
        
        # Sample dataset info
        self.sample_data_info = {
            "shape": "(244, 7)",
            "columns": ["total_bill", "tip", "sex", "smoker", "day", "time", "size"],
            "dtypes": {"total_bill": "float64", "tip": "float64", "sex": "object"},
            "sample": "total_bill,tip,sex,smoker,day,time,size\n16.99,1.01,Female,No,Sun,Dinner,2",
            "has_trained_model": True,
            "target_variable": "tip"
        }
        
        self.no_model_data_info = {
            "shape": "(244, 7)",
            "columns": ["total_bill", "tip", "sex", "smoker", "day", "time", "size"],
            "dtypes": {"total_bill": "float64", "tip": "float64", "sex": "object"},
            "sample": "total_bill,tip,sex,smoker,day,time,size\n16.99,1.01,Female,No,Sun,Dinner,2",
            "has_trained_model": False,
            "target_variable": "Unknown"
        }
    
    def test_prediction_with_specific_values(self):
        """Test prediction detection with specific parameter values."""
        test_cases = [
            "What would be the tip for total_bill=25.0, size=2?",
            "Predict tip for total_bill=35, size=4, time=Dinner",
            "Calculate tip for bill=$50, party of 3, Saturday dinner",
            "Estimate tip amount for $40 bill, 2 people, lunch",
            "What's the predicted tip for total_bill=28.5, size=3, day=Fri?"
        ]
        
        for request in test_cases:
            result = self.parser.parse_intent(
                user_request=request,
                csv_url="https://example.com/tips.csv",
                data_info=self.sample_data_info
            )
            
            assert result.needs_prediction == True, f"Failed for: {request}"
            assert result.needs_ml_modeling == False, f"Should not trigger training for: {request}"
            assert result.prediction_type == "single_prediction", f"Wrong prediction type for: {request}"
            assert result.intent_confidence >= 0.8, f"Low confidence for clear prediction: {request}"
    
    def test_prediction_question_formats(self):
        """Test various question formats for predictions."""
        test_cases = [
            "What would be the house price for rooms=6, age=50?",
            "What's the MPG for cylinders=4, horsepower=85, weight=2500?",
            "For a customer with tenure=12, MonthlyCharges=70, will they churn?",
            "What grade would a student get with studytime=3, failures=0?",
            "What salary for 5 years experience, Master's degree?"
        ]
        
        for request in test_cases:
            result = self.parser.parse_intent(
                user_request=request,
                csv_url="https://example.com/data.csv",
                data_info=self.sample_data_info
            )
            
            assert result.needs_prediction == True, f"Failed to detect prediction for: {request}"
            assert result.needs_ml_modeling == False, f"Should not trigger training for: {request}"
    
    def test_context_aware_prediction_detection(self):
        """Test context-aware prediction detection based on model existence."""
        prediction_request = "What would be the tip for total_bill=25.0, size=2?"
        
        # Test with existing model
        result_with_model = self.parser.parse_intent(
            user_request=prediction_request,
            csv_url="https://example.com/tips.csv",
            data_info=self.sample_data_info  # has_trained_model=True
        )
        
        # Test without existing model
        result_without_model = self.parser.parse_intent(
            user_request=prediction_request,
            csv_url="https://example.com/tips.csv",
            data_info=self.no_model_data_info  # has_trained_model=False
        )
        
        # Both should detect prediction, but confidence should be higher with existing model
        assert result_with_model.needs_prediction == True
        assert result_without_model.needs_prediction == True
        assert result_with_model.intent_confidence >= result_without_model.intent_confidence
    
    def test_training_vs_prediction_distinction(self):
        """Test clear distinction between training and prediction requests."""
        training_requests = [
            "Train a model to predict tip",
            "Build ML model using tips dataset",
            "Create a regression model for tip prediction",
            "Develop machine learning algorithm for tips"
        ]
        
        prediction_requests = [
            "Predict tip for total_bill=25.0, size=2",
            "What would be the tip for $35 bill, 4 people?",
            "Calculate tip for total_bill=40, size=3",
            "Estimate tip amount for bill=$50, party of 2"
        ]
        
        # Test training requests
        for request in training_requests:
            result = self.parser.parse_intent(
                user_request=request,
                csv_url="https://example.com/tips.csv",
                data_info=self.sample_data_info
            )
            
            assert result.needs_ml_modeling == True, f"Failed to detect training for: {request}"
            assert result.needs_prediction == False, f"Should not trigger prediction for: {request}"
        
        # Test prediction requests
        for request in prediction_requests:
            result = self.parser.parse_intent(
                user_request=request,
                csv_url="https://example.com/tips.csv",
                data_info=self.sample_data_info
            )
            
            assert result.needs_prediction == True, f"Failed to detect prediction for: {request}"
            assert result.needs_ml_modeling == False, f"Should not trigger training for: {request}"
    
    def test_model_analysis_detection(self):
        """Test model analysis intent detection."""
        analysis_requests = [
            "What features are most important?",
            "How accurate is the model?",
            "Analyze model performance",
            "Feature importance analysis",
            "Model evaluation metrics",
            "What drives the predictions?",
            "How good is the model?",
            "Model insights and interpretation"
        ]
        
        for request in analysis_requests:
            result = self.parser.parse_intent(
                user_request=request,
                csv_url="https://example.com/tips.csv",
                data_info=self.sample_data_info
            )
            
            assert result.needs_model_analysis == True, f"Failed to detect analysis for: {request}"
            assert result.needs_prediction == False, f"Should not trigger prediction for: {request}"
            assert result.needs_ml_modeling == False, f"Should not trigger training for: {request}"
    
    def test_batch_prediction_detection(self):
        """Test batch prediction detection."""
        batch_requests = [
            "Use the model to predict for new data",
            "Classify this CSV: https://example.com/new_data.csv",
            "Make predictions using https://example.com/test_data.csv",
            "Predict for batch of customers",
            "Apply model to new dataset"
        ]
        
        for request in batch_requests:
            result = self.parser.parse_intent(
                user_request=request,
                csv_url="https://example.com/tips.csv",
                data_info=self.sample_data_info
            )
            
            assert result.needs_prediction == True, f"Failed to detect batch prediction for: {request}"
            assert result.prediction_type == "batch_prediction", f"Wrong prediction type for: {request}"
    
    def test_prediction_data_extraction(self):
        """Test enhanced prediction data extraction."""
        request = "What would be the tip for total_bill=25.0, size=2, time=Dinner?"
        
        result = self.parser.parse_intent(
            user_request=request,
            csv_url="https://example.com/tips.csv",
            data_info=self.sample_data_info
        )
        
        assert result.needs_prediction == True
        assert result.extracted_prediction_data is not None
        # Should extract the parameter values
        assert "total_bill" in str(result.extracted_prediction_data)
        assert "25.0" in str(result.extracted_prediction_data)
    
    def test_ambiguity_resolution(self):
        """Test ambiguity resolution logic."""
        # Ambiguous request that could be training or prediction
        ambiguous_request = "I want to work with tip data for total_bill=25.0"
        
        # With existing model, should favor prediction
        result_with_model = self.parser.parse_intent(
            user_request=ambiguous_request,
            csv_url="https://example.com/tips.csv",
            data_info=self.sample_data_info
        )
        
        # Without existing model, might favor training
        result_without_model = self.parser.parse_intent(
            user_request=ambiguous_request,
            csv_url="https://example.com/tips.csv",
            data_info=self.no_model_data_info
        )
        
        # Check that context influences decision
        assert result_with_model.intent_confidence >= 0.3
        assert result_without_model.intent_confidence >= 0.3
    
    def test_confidence_scoring(self):
        """Test confidence scoring for different request types."""
        # Very clear prediction request
        clear_request = "Predict tip for total_bill=25.0, size=2"
        clear_result = self.parser.parse_intent(
            user_request=clear_request,
            csv_url="https://example.com/tips.csv",
            data_info=self.sample_data_info
        )
        
        # Ambiguous request
        ambiguous_request = "What about the tip data?"
        ambiguous_result = self.parser.parse_intent(
            user_request=ambiguous_request,
            csv_url="https://example.com/tips.csv",
            data_info=self.sample_data_info
        )
        
        # Clear request should have higher confidence
        assert clear_result.intent_confidence > ambiguous_result.intent_confidence
        assert clear_result.intent_confidence >= 0.7
        assert ambiguous_result.intent_confidence <= 0.6


def test_real_world_scenarios():
    """Test with real-world business scenarios."""
    parser = DataAnalysisIntentParser()
    
    # Real estate scenario
    real_estate_data = {
        "shape": "(506, 14)",
        "columns": ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"],
        "dtypes": {"medv": "float64", "rm": "float64", "age": "float64"},
        "sample": "Sample data...",
        "has_trained_model": True,
        "target_variable": "medv"
    }
    
    result = parser.parse_intent(
        user_request="What would be the house price for rooms=6, age=50, distance=3.5?",
        csv_url="https://example.com/housing.csv",
        data_info=real_estate_data
    )
    
    assert result.needs_prediction == True
    assert result.needs_ml_modeling == False
    assert result.prediction_type == "single_prediction"
    assert result.intent_confidence >= 0.8


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestEnhancedIntentParser()
    test_suite.setup_method()
    
    print("🧪 TESTING ENHANCED INTENT PARSER")
    print("=" * 50)
    
    # Test prediction detection
    try:
        test_suite.test_prediction_with_specific_values()
        print("✅ Prediction with specific values - PASSED")
    except Exception as e:
        print(f"❌ Prediction with specific values - FAILED: {e}")
    
    # Test context awareness
    try:
        test_suite.test_context_aware_prediction_detection()
        print("✅ Context-aware prediction detection - PASSED")
    except Exception as e:
        print(f"❌ Context-aware prediction detection - FAILED: {e}")
    
    # Test training vs prediction
    try:
        test_suite.test_training_vs_prediction_distinction()
        print("✅ Training vs prediction distinction - PASSED")
    except Exception as e:
        print(f"❌ Training vs prediction distinction - FAILED: {e}")
    
    # Test real-world scenarios
    try:
        test_real_world_scenarios()
        print("✅ Real-world scenarios - PASSED")
    except Exception as e:
        print(f"❌ Real-world scenarios - FAILED: {e}")
    
    print("\n🚀 Enhanced intent parser ready for testing!") 