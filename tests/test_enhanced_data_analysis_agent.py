"""
Integration tests for the Enhanced Data Analysis Agent.

This module tests the complete workflow orchestration and structured outputs
of the new data analysis agent.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.schemas import DataAnalysisRequest, ModelType, ProblemType, WorkflowIntent


class TestEnhancedDataAnalysisAgent(unittest.TestCase):
    """Test cases for the Enhanced Data Analysis Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.agent = DataAnalysisAgent(output_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertIsNotNone(self.agent.intent_parser)
        self.assertIsNotNone(self.agent.parameter_mapper)
        self.assertIsNotNone(self.agent.data_cleaning_agent)
        self.assertIsNotNone(self.agent.feature_engineering_agent)
        self.assertIsNotNone(self.agent.h2o_ml_agent)
    
    def test_request_creation(self):
        """Test request creation and validation."""
        # Valid request
        request = self.agent._create_request(
            "https://example.com/data.csv",
            "Clean this data and build a model",
            model_types=[ModelType.GBM, ModelType.RANDOM_FOREST]
        )
        
        self.assertIsInstance(request, DataAnalysisRequest)
        self.assertEqual(request.csv_url, "https://example.com/data.csv")
        self.assertEqual(len(request.model_types), 2)
        self.assertIsNotNone(self.agent.current_request_id)
    
    def test_intent_parser_fallback(self):
        """Test intent parser fallback mechanism."""
        # Test fallback intent creation
        fallback_intent = self.agent.intent_parser._create_fallback_intent(
            "Clean this dataset and build a machine learning model"
        )
        
        self.assertIsInstance(fallback_intent, WorkflowIntent)
        self.assertTrue(fallback_intent.needs_data_cleaning)
        self.assertTrue(fallback_intent.needs_ml_modeling)
        self.assertEqual(fallback_intent.intent_confidence, 0.3)
    
    def test_parameter_mapping(self):
        """Test parameter mapping functionality."""
        request = DataAnalysisRequest(
            csv_url="https://example.com/data.csv",
            user_request="Clean and analyze this data"
        )
        
        intent = WorkflowIntent(
            needs_data_cleaning=True,
            needs_feature_engineering=True,
            needs_ml_modeling=True,
            data_quality_focus=True,
            exploratory_analysis=False,
            prediction_focus=True,
            statistical_analysis=False,
            key_requirements=["clean data", "build model"],
            complexity_level="moderate",
            intent_confidence=0.9
        )
        
        # Test data cleaning parameters
        cleaning_params = self.agent.parameter_mapper.map_data_cleaning_parameters(
            request, intent, "https://example.com/data.csv"
        )
        
        self.assertIn("model", cleaning_params)
        self.assertIn("user_instructions", cleaning_params)
        self.assertIn("log", cleaning_params)
        self.assertEqual(len(cleaning_params), 12)  # Expected number of parameters
        
        # Test feature engineering parameters
        fe_params = self.agent.parameter_mapper.map_feature_engineering_parameters(
            request, intent, "cleaned_data.csv", "target"
        )
        
        self.assertIn("target_variable", fe_params)
        self.assertEqual(fe_params["target_variable"], "target")
        self.assertEqual(len(fe_params), 13)  # Expected number of parameters
        
        # Test ML parameters
        ml_params = self.agent.parameter_mapper.map_h2o_ml_parameters(
            request, intent, "engineered_data.csv", "target"
        )
        
        self.assertIn("target_variable", ml_params)
        self.assertIn("model_types", ml_params)
        self.assertIn("enable_mlflow", ml_params)
        self.assertEqual(len(ml_params), 23)  # Expected number of parameters
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with invalid URL (should create error result)
        result = self.agent.analyze(
            csv_url="https://invalid-url-that-does-not-exist.com/data.csv",
            user_request="Analyze this data"
        )
        
        # Should return a DataAnalysisResult even on error
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence_level, "low")
        self.assertGreater(len(result.warnings), 0)
        self.assertGreater(len(result.limitations), 0)
    
    def test_target_variable_detection(self):
        """Test target variable auto-detection."""
        # This would normally read a CSV, but we'll test the logic
        target = self.agent._auto_detect_target_variable("non-existent-file.csv")
        # Should return None for non-existent file
        self.assertIsNone(target)
    
    def test_confidence_level_calculation(self):
        """Test confidence level calculation."""
        from src.schemas import AgentExecutionResult
        
        # Test high confidence
        successful_results = [
            AgentExecutionResult(
                agent_name="test_agent",
                execution_time_seconds=1.0,
                success=True
            )
        ]
        
        intent = WorkflowIntent(
            needs_data_cleaning=True,
            needs_feature_engineering=False,
            needs_ml_modeling=False,
            data_quality_focus=True,
            exploratory_analysis=False,
            prediction_focus=False,
            statistical_analysis=False,
            key_requirements=["test"],
            complexity_level="simple",
            intent_confidence=0.9
        )
        
        confidence = self.agent._determine_confidence_level(successful_results, intent)
        self.assertEqual(confidence, "high")
        
        # Test low confidence
        failed_results = [
            AgentExecutionResult(
                agent_name="test_agent",
                execution_time_seconds=1.0,
                success=False,
                error_message="Test error"
            )
        ]
        
        intent.intent_confidence = 0.2
        confidence = self.agent._determine_confidence_level(failed_results, intent)
        self.assertEqual(confidence, "low")


class TestSchemaValidation(unittest.TestCase):
    """Test cases for schema validation."""
    
    def test_data_analysis_request_validation(self):
        """Test DataAnalysisRequest validation."""
        # Valid request
        request = DataAnalysisRequest(
            csv_url="https://example.com/data.csv",
            user_request="This is a valid request with sufficient length"
        )
        self.assertEqual(request.csv_url, "https://example.com/data.csv")
        
        # Invalid URL
        with self.assertRaises(Exception):  # ValidationError is a subclass of Exception
            DataAnalysisRequest(
                csv_url="not-a-valid-url",
                user_request="Valid request"
            )
        
        # Test that empty model types are handled (should use defaults)
        request = DataAnalysisRequest(
            csv_url="https://example.com/data.csv",
            user_request="Valid request"
        )
        self.assertIsNotNone(request.model_types)  # Should have default values
    
    def test_workflow_intent_validation(self):
        """Test WorkflowIntent validation."""
        intent = WorkflowIntent(
            needs_data_cleaning=True,
            needs_feature_engineering=False,
            needs_ml_modeling=True,
            data_quality_focus=True,
            exploratory_analysis=False,
            prediction_focus=True,
            statistical_analysis=False,
            key_requirements=["clean", "model"],
            complexity_level="moderate",
            intent_confidence=0.8
        )
        
        self.assertTrue(intent.needs_data_cleaning)
        self.assertEqual(intent.complexity_level, "moderate")
        self.assertEqual(intent.intent_confidence, 0.8)


if __name__ == "__main__":
    unittest.main() 