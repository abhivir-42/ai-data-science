import pytest
from src.schemas.data_analysis_schemas import MLModelingMetrics
from src.agents.ml_prediction_agent import MLPredictionAgent, MLPredictionError
from src.uagent_v2.config import UAgentConfig

def test_ml_prediction_agent_initialization():
    """Test MLPredictionAgent initializes with mock MLModelingMetrics"""
    # Create mock MLModelingMetrics with all required fields
    mock_metrics = MLModelingMetrics(
        models_trained=5,
        best_model_type="GBM",
        best_model_id="GBM_model_123",
        best_model_score=0.85,
        cross_validation_score=0.82,
        test_set_score=0.81,
        training_time_seconds=120.0,
        model_size_mb=15.2,
        features_used=["age", "sex", "pclass"],
        feature_importance={"age": 0.45, "sex": 0.35, "pclass": 0.20},
        mlflow_experiment_id="exp_123",
        mlflow_run_id="run_456",
        model_path="/tmp/fake_model.zip",
        leaderboard=[{"model_id": "GBM_model_123", "auc": 0.85}],
        top_model_metrics={"auc": 0.85, "logloss": 0.42},
        total_models_trained=5,
        model_architecture="Gradient Boosting Machine (GBM)"
    )
    
    config = UAgentConfig.from_env()
    prediction_agent = MLPredictionAgent(mock_metrics, "survived", config)
    
    # Test initialization
    assert prediction_agent.model_metrics == mock_metrics
    assert prediction_agent.target_variable == "survived"
    assert prediction_agent.config == config
    assert prediction_agent._h2o_model is None  # Not loaded yet
    
    print("✅ MLPredictionAgent initialization working correctly")

def test_problem_type_detection():
    """Test _determine_problem_type method"""
    # Classification mock
    classification_metrics = MLModelingMetrics(
        models_trained=1,
        best_model_type="GBM",
        best_model_id="GBM_1",
        best_model_score=0.85,
        cross_validation_score=0.82,
        test_set_score=0.81,
        training_time_seconds=60.0,
        model_size_mb=10.0,
        features_used=["age"],
        feature_importance={"age": 1.0},
        mlflow_experiment_id="exp_1",
        mlflow_run_id="run_1",
        model_architecture="Classification Model",
        top_model_metrics={"auc": 0.85, "logloss": 0.42}
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(classification_metrics, "survived", config)
    
    assert agent._determine_problem_type() == "classification"
    print("✅ Problem type detection working correctly")

def test_regression_problem_type_detection():
    """Test _determine_problem_type method for regression"""
    # Regression mock
    regression_metrics = MLModelingMetrics(
        models_trained=1,
        best_model_type="GBM",
        best_model_id="GBM_reg_1",
        best_model_score=0.92,
        cross_validation_score=0.89,
        test_set_score=0.88,
        training_time_seconds=60.0,
        model_size_mb=8.5,
        features_used=["age"],
        feature_importance={"age": 1.0},
        mlflow_experiment_id="exp_reg_1",
        mlflow_run_id="run_reg_1",
        model_architecture="Regression Model",
        top_model_metrics={"rmse": 2.5, "mae": 1.8}
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(regression_metrics, "price", config)
    
    assert agent._determine_problem_type() == "regression"
    print("✅ Regression problem type detection working correctly")

def test_prediction_agent_methods_exist():
    """Test that all required methods exist and are callable"""
    mock_metrics = MLModelingMetrics(
        models_trained=1,
        best_model_type="Test",
        best_model_id="test_1",
        best_model_score=0.8,
        cross_validation_score=0.75,
        test_set_score=0.78,
        training_time_seconds=60.0,
        model_size_mb=5.0,
        features_used=["age"],
        feature_importance={"age": 1.0},
        mlflow_experiment_id="test_exp",
        mlflow_run_id="test_run",
        model_architecture="Test Model"
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(mock_metrics, "target", config)
    
    # Check all methods exist
    assert hasattr(agent, 'predict_single'), "Missing predict_single method"
    assert hasattr(agent, 'predict_batch'), "Missing predict_batch method"
    assert hasattr(agent, 'analyze_model'), "Missing analyze_model method"
    assert hasattr(agent, 'load_model'), "Missing load_model method"
    
    # Check methods are callable
    assert callable(agent.predict_single), "predict_single should be callable"
    assert callable(agent.predict_batch), "predict_batch should be callable"
    assert callable(agent.analyze_model), "analyze_model should be callable"
    assert callable(agent.load_model), "load_model should be callable"
    
    print("✅ All prediction agent methods exist and are callable")

def test_format_classification_result():
    """Test _format_classification_result method"""
    import pandas as pd
    
    mock_metrics = MLModelingMetrics(
        models_trained=1,
        best_model_type="GBM",
        best_model_id="cls_test_1",
        best_model_score=0.85,
        cross_validation_score=0.82,
        test_set_score=0.83,
        training_time_seconds=60.0,
        model_size_mb=7.0,
        features_used=["age"],
        feature_importance={"age": 1.0},
        mlflow_experiment_id="cls_exp",
        mlflow_run_id="cls_run",
        model_architecture="Classification Model"
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(mock_metrics, "survived", config)
    
    # Mock prediction DataFrame
    pred_df = pd.DataFrame({
        'predict': [1],
        'p0': [0.3],
        'p1': [0.7]
    })
    
    input_data = {"age": 25, "sex": "male"}
    result = agent._format_classification_result(pred_df, input_data)
    
    # Verify result structure
    assert result["prediction_type"] == "single_prediction"
    assert result["target_variable"] == "survived"
    assert result["prediction"] == 1
    assert result["probability"] == 0.3  # p0 column
    assert result["input_data"] == input_data
    
    print("✅ Classification result formatting working correctly")

def test_format_regression_result():
    """Test _format_regression_result method"""
    import pandas as pd
    
    mock_metrics = MLModelingMetrics(
        models_trained=1,
        best_model_type="GBM",
        best_model_id="reg_test_1",
        best_model_score=0.92,
        cross_validation_score=0.89,
        test_set_score=0.90,
        training_time_seconds=60.0,
        model_size_mb=6.5,
        features_used=["age"],
        feature_importance={"age": 1.0},
        mlflow_experiment_id="reg_exp",
        mlflow_run_id="reg_run",
        model_architecture="Regression Model"
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(mock_metrics, "price", config)
    
    # Mock prediction DataFrame
    pred_df = pd.DataFrame({
        'predict': [25000.5]
    })
    
    input_data = {"age": 25, "mileage": 50000}
    result = agent._format_regression_result(pred_df, input_data)
    
    # Verify result structure
    assert result["prediction_type"] == "single_prediction"
    assert result["target_variable"] == "price"
    assert result["prediction"] == 25000.5
    assert result["input_data"] == input_data
    
    print("✅ Regression result formatting working correctly")

def test_batch_prediction_summary():
    """Test _summarize_batch_predictions method"""
    import pandas as pd
    
    mock_metrics = MLModelingMetrics(
        models_trained=1,
        best_model_type="GBM",
        best_model_id="batch_test_1",
        best_model_score=0.88,
        cross_validation_score=0.85,
        test_set_score=0.86,
        training_time_seconds=60.0,
        model_size_mb=4.0,
        features_used=["age"],
        feature_importance={"age": 1.0},
        mlflow_experiment_id="batch_exp",
        mlflow_run_id="batch_run"
    )
    
    config = UAgentConfig.from_env()
    agent = MLPredictionAgent(mock_metrics, "target", config)
    
    # Test classification summary
    pred_df_cls = pd.DataFrame({
        'predict': ['A', 'B', 'A', 'A', 'B']
    })
    
    summary_cls = agent._summarize_batch_predictions(pred_df_cls)
    assert 'prediction_counts' in summary_cls
    assert summary_cls['prediction_counts']['A'] == 3
    assert summary_cls['prediction_counts']['B'] == 2
    
    # Test regression summary (with more unique values to ensure it's treated as regression)
    pred_df_reg = pd.DataFrame({
        'predict': [10.5, 15.2, 8.9, 12.1, 11.3, 9.8, 13.7, 14.2, 16.1, 7.5, 
                   18.3, 19.2, 6.7, 20.1, 21.4, 5.8, 22.9, 23.1, 24.7, 25.3, 
                   26.8, 27.2, 28.5, 29.1, 30.2]  # 25 unique values > 20
    })
    
    summary_reg = agent._summarize_batch_predictions(pred_df_reg)
    assert 'prediction_stats' in summary_reg
    assert 'mean' in summary_reg['prediction_stats']
    assert 'std' in summary_reg['prediction_stats']
    
    print("✅ Batch prediction summary working correctly")

if __name__ == "__main__":
    test_ml_prediction_agent_initialization()
    test_problem_type_detection()
    test_regression_problem_type_detection()
    test_prediction_agent_methods_exist()
    test_format_classification_result()
    test_format_regression_result()
    test_batch_prediction_summary()
    print("🎉 Phase 2A: MLPredictionAgent tests PASSED") 