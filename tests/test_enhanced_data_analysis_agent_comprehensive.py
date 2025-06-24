"""
Comprehensive Test Suite for Enhanced Data Analysis Agent

This module implements the full pytest test suite as originally planned in Step 7
of the implementation plan, providing thorough validation of all components.
"""

import pytest
import asyncio
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.schemas import (
    DataAnalysisRequest, 
    WorkflowIntent, 
    DataAnalysisResult,
    ProblemType, 
    ModelType
)
from src.parsers import DataAnalysisIntentParser
from src.mappers import AgentParameterMapper


class TestDataAnalysisAgent:
    """Comprehensive test suite for enhanced data analysis agent"""
    
    @pytest.fixture
    def agent(self):
        """Create test agent instance"""
        return DataAnalysisAgent(
            output_dir="test_outputs",
            intent_parser_model="gpt-4o-mini"
        )
    
    @pytest.fixture
    def sample_titanic_url(self):
        """Sample Titanic dataset URL for testing"""
        return "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    @pytest.fixture
    def sample_cleaning_request(self, sample_titanic_url):
        """Basic data cleaning request"""
        return DataAnalysisRequest(
            csv_url=sample_titanic_url,
            user_request="Clean this dataset and handle missing values, remove duplicates"
        )
    
    @pytest.fixture
    def sample_ml_request(self, sample_titanic_url):
        """Full ML pipeline request"""
        return DataAnalysisRequest(
            csv_url=sample_titanic_url,
            user_request="Build a machine learning model to predict passenger survival",
            target_variable="Survived",
            problem_type=ProblemType.CLASSIFICATION,
            model_types=[ModelType.GBM, ModelType.GLM]
        )

    # ============================================================================
    # BASIC FUNCTIONALITY TESTS
    # ============================================================================
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly with all components"""
        assert isinstance(agent, DataAnalysisAgent)
        assert agent.intent_parser is not None
        assert agent.parameter_mapper is not None
        assert agent.data_cleaning_agent is not None
        assert agent.feature_engineering_agent is not None
        assert agent.h2o_ml_agent is not None
        assert agent.output_dir.exists()
    
    def test_request_creation_valid(self, agent, sample_titanic_url):
        """Test valid request creation and validation"""
        request = agent._create_request(
            csv_url=sample_titanic_url,
            user_request="Test analysis request with sufficient length",
            target_variable="Survived"
        )
        
        assert isinstance(request, DataAnalysisRequest)
        assert request.csv_url == sample_titanic_url
        assert request.target_variable == "Survived"
        assert request.problem_type == ProblemType.AUTO
    
    def test_request_creation_invalid_url(self, agent):
        """Test request creation with invalid URL"""
        with pytest.raises(Exception):  # Should raise validation error
            agent._create_request(
                csv_url="not-a-valid-url",
                user_request="Test request"
            )
    
    def test_request_creation_short_request(self, agent, sample_titanic_url):
        """Test request creation with too short user request"""
        with pytest.raises(Exception):  # Should raise validation error
            agent._create_request(
                csv_url=sample_titanic_url,
                user_request="short"  # Too short
            )

    # ============================================================================
    # INTENT PARSING TESTS
    # ============================================================================
    
    def test_intent_parsing_cleaning_focus(self, agent, sample_cleaning_request):
        """Test intent parsing for data cleaning focused requests"""
        intent = agent._parse_intent_sync(sample_cleaning_request)
        
        assert isinstance(intent, WorkflowIntent)
        assert intent.needs_data_cleaning == True
        assert intent.data_quality_focus == True
        assert intent.intent_confidence > 0.0
    
    def test_intent_parsing_ml_focus(self, agent, sample_ml_request):
        """Test intent parsing for ML focused requests"""
        intent = agent._parse_intent_sync(sample_ml_request)
        
        assert isinstance(intent, WorkflowIntent)
        assert intent.needs_ml_modeling == True
        assert intent.prediction_focus == True
        assert intent.suggested_target_variable is not None
        assert intent.intent_confidence > 0.0

    # ============================================================================
    # WORKFLOW EXECUTION TESTS
    # ============================================================================
    
    @pytest.mark.slow
    def test_simple_cleaning_workflow(self, agent, sample_cleaning_request):
        """Test basic data cleaning workflow execution"""
        result = agent.analyze(
            csv_url=sample_cleaning_request.csv_url,
            user_request=sample_cleaning_request.user_request
        )
        
        assert isinstance(result, DataAnalysisResult)
        assert result.workflow_intent.needs_data_cleaning == True
        assert 'data_cleaning' in result.agents_executed
        assert result.analysis_quality_score > 0.0
        assert len(result.key_insights) > 0
        assert len(result.recommendations) > 0
    
    @pytest.mark.slow  
    def test_full_ml_pipeline(self, agent, sample_ml_request):
        """Test complete ML workflow execution"""
        result = agent.analyze(
            csv_url=sample_ml_request.csv_url,
            user_request=sample_ml_request.user_request,
            target_variable="Survived",
            max_runtime_seconds=300  # Shorter for testing
        )
        
        assert isinstance(result, DataAnalysisResult)
        assert result.workflow_intent.needs_data_cleaning == True
        assert result.workflow_intent.needs_feature_engineering == True
        assert result.workflow_intent.needs_ml_modeling == True
        assert len(result.agents_executed) >= 2  # At least cleaning + one other
        assert result.overall_data_quality_score > 0.0
        assert result.analysis_quality_score > 0.0

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================
    
    def test_invalid_url_handling(self, agent):
        """Test graceful handling of invalid URLs"""
        result = agent.analyze(
            csv_url="https://invalid-url-that-does-not-exist.com/data.csv",
            user_request="Analyze this non-existent dataset"
        )
        
        assert isinstance(result, DataAnalysisResult)
        assert result.confidence_level == "low"
        assert len(result.warnings) > 0
        assert len(result.limitations) > 0
        assert result.analysis_quality_score == 0.0
    
    def test_malformed_csv_handling(self, agent):
        """Test handling of malformed CSV data"""
        # This would require creating a mock malformed CSV URL
        # For now, we'll test the error result structure
        error_result = agent._create_error_result(
            csv_url="https://example.com/bad.csv",
            user_request="Test request",
            error_message="CSV parsing failed"
        )
        
        assert isinstance(error_result, DataAnalysisResult)
        assert error_result.confidence_level == "low"
        assert "CSV parsing failed" in error_result.warnings
        assert error_result.feature_engineering_effectiveness == 0.0
        assert error_result.model_performance_score == 0.0

    # ============================================================================
    # PARAMETER MAPPING TESTS
    # ============================================================================
    
    def test_parameter_mapping_data_cleaning(self, agent, sample_cleaning_request):
        """Test parameter mapping for data cleaning agent"""
        intent = agent._parse_intent_sync(sample_cleaning_request)
        
        params = agent.parameter_mapper.map_data_cleaning_parameters(
            sample_cleaning_request, intent, "test_data.csv"
        )
        
        assert isinstance(params, dict)
        assert "user_instructions" in params
        assert "missing_threshold" in params
        assert "file_name" in params
        assert "log_path" in params
    
    def test_parameter_mapping_feature_engineering(self, agent, sample_ml_request):
        """Test parameter mapping for feature engineering agent"""
        intent = agent._parse_intent_sync(sample_ml_request)
        
        params = agent.parameter_mapper.map_feature_engineering_parameters(
            sample_ml_request, intent, "test_data.csv", "Survived"
        )
        
        assert isinstance(params, dict)
        assert "user_instructions" in params
        assert "target_variable" in params
        assert "file_name" in params
        assert "log_path" in params
    
    def test_parameter_mapping_h2o_ml(self, agent, sample_ml_request):
        """Test parameter mapping for H2O ML agent"""
        intent = agent._parse_intent_sync(sample_ml_request)
        
        params = agent.parameter_mapper.map_h2o_ml_parameters(
            sample_ml_request, intent, "test_data.csv", "Survived"
        )
        
        assert isinstance(params, dict)
        assert "user_instructions" in params
        assert "target_variable" in params
        assert "model_directory" in params
        assert "enable_mlflow" in params

    # ============================================================================
    # STRUCTURED OUTPUT VALIDATION TESTS
    # ============================================================================
    
    def test_result_schema_validation(self, agent, sample_cleaning_request):
        """Test that results conform to structured schema"""
        result = agent.analyze(
            csv_url=sample_cleaning_request.csv_url,
            user_request=sample_cleaning_request.user_request
        )
        
        # Test all required fields are present
        assert hasattr(result, 'request_id')
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'total_runtime_seconds')
        assert hasattr(result, 'workflow_intent')
        assert hasattr(result, 'agents_executed')
        assert hasattr(result, 'analysis_quality_score')
        assert hasattr(result, 'confidence_level')
        
        # Test field types
        assert isinstance(result.request_id, str)
        assert isinstance(result.total_runtime_seconds, float)
        assert isinstance(result.agents_executed, list)
        assert isinstance(result.key_insights, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.generated_files, dict)
    
    def test_confidence_level_calculation(self, agent):
        """Test confidence level calculation logic"""
        from src.schemas import AgentExecutionResult
        
        # Test high confidence (all agents successful)
        high_confidence_results = [
            AgentExecutionResult(agent_name="test1", execution_time_seconds=1.0, success=True),
            AgentExecutionResult(agent_name="test2", execution_time_seconds=1.0, success=True),
            AgentExecutionResult(agent_name="test3", execution_time_seconds=1.0, success=True)
        ]
        
        intent = WorkflowIntent(
            needs_data_cleaning=True,
            needs_feature_engineering=True, 
            needs_ml_modeling=True,
            data_quality_focus=True,
            exploratory_analysis=True,
            prediction_focus=True,
            statistical_analysis=True,
            key_requirements=["test"],
            complexity_level="simple",
            intent_confidence=0.9
        )
        
        confidence = agent._determine_confidence_level(high_confidence_results, intent)
        assert confidence == "high"
        
        # Test low confidence (all agents failed)
        low_confidence_results = [
            AgentExecutionResult(agent_name="test1", execution_time_seconds=1.0, success=False),
            AgentExecutionResult(agent_name="test2", execution_time_seconds=1.0, success=False)
        ]
        
        confidence = agent._determine_confidence_level(low_confidence_results, intent)
        assert confidence == "low"

    # ============================================================================
    # INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.integration
    def test_end_to_end_titanic_analysis(self, agent, sample_titanic_url):
        """End-to-end integration test with Titanic dataset"""
        result = agent.analyze(
            csv_url=sample_titanic_url,
            user_request="Perform comprehensive analysis to predict survival, including data cleaning, feature engineering, and model training",
            target_variable="Survived",
            problem_type=ProblemType.CLASSIFICATION,
            max_runtime_seconds=300
        )
        
        # Validate comprehensive workflow execution
        assert len(result.agents_executed) >= 2
        assert result.workflow_intent.intent_confidence > 0.5
        assert result.analysis_quality_score > 0.3
        assert result.overall_data_quality_score > 0.0
        assert len(result.key_insights) >= 2
        assert len(result.recommendations) >= 2
        assert result.data_story is not None
        assert len(result.data_story) > 50  # Meaningful story length

    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    @pytest.mark.performance
    def test_analysis_performance_timing(self, agent, sample_cleaning_request):
        """Test that analysis completes within reasonable time"""
        import time
        
        start_time = time.time()
        result = agent.analyze(
            csv_url=sample_cleaning_request.csv_url,
            user_request=sample_cleaning_request.user_request,
            max_runtime_seconds=120  # 2 minute limit for testing
        )
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 180  # 3 minutes max for test
        assert result.total_runtime_seconds > 0
        assert result.total_runtime_seconds < 180


# ============================================================================
# COMPONENT-SPECIFIC TESTS
# ============================================================================

class TestIntentParser:
    """Test the intent parser component specifically"""
    
    @pytest.fixture
    def parser(self):
        return DataAnalysisIntentParser(model_name="gpt-4o-mini")
    
    def test_intent_parser_initialization(self, parser):
        """Test intent parser initializes correctly"""
        assert parser is not None
        assert parser.model_name == "gpt-4o-mini"
    
    def test_parse_cleaning_intent(self, parser):
        """Test parsing data cleaning specific intents"""
        intent = parser.parse_with_data_preview(
            "Clean this dataset and handle missing values",
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
        
        assert intent.needs_data_cleaning == True
        assert intent.data_quality_focus == True


class TestParameterMapper:
    """Test the parameter mapper component specifically"""
    
    @pytest.fixture
    def mapper(self):
        return AgentParameterMapper(base_output_dir="test_outputs")
    
    def test_mapper_initialization(self, mapper):
        """Test parameter mapper initializes correctly"""
        assert mapper is not None
        assert str(mapper.base_output_dir) == "test_outputs"
    
    def test_get_timestamp(self, mapper):
        """Test timestamp generation"""
        timestamp = mapper.get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


# ============================================================================
# TEST UTILITIES
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_outputs():
    """Clean up test outputs after all tests complete"""
    yield
    
    # Cleanup test directories
    import shutil
    test_dirs = ["test_outputs", "loan_analysis_outputs", "demo_outputs", "demo_outputs_advanced", "demo_outputs_error"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
            except:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"]) 