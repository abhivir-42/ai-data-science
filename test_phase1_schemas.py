import pytest
from src.schemas.data_analysis_schemas import WorkflowIntent
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig

def test_workflow_intent_prediction_fields():
    """Test new prediction fields in WorkflowIntent schema"""
    intent = WorkflowIntent(
        needs_data_cleaning=True,
        needs_feature_engineering=False, 
        needs_ml_modeling=True,
        data_quality_focus=True,
        exploratory_analysis=False,
        prediction_focus=True,
        statistical_analysis=False,
        key_requirements=["predict survival"],
        complexity_level="moderate",
        intent_confidence=0.8,
        
        # NEW FIELDS TO TEST
        needs_prediction=True,
        needs_model_analysis=False,
        prediction_data_source="inline",
        prediction_type="single_prediction",
        extracted_prediction_data={"age": 25, "sex": "male", "pclass": 3}
    )
    
    # Test new fields
    assert intent.needs_prediction == True
    assert intent.needs_model_analysis == False
    assert intent.prediction_type == "single_prediction"
    assert intent.extracted_prediction_data == {"age": 25, "sex": "male", "pclass": 3}
    
    print("✅ WorkflowIntent prediction fields working correctly")

def test_workflow_intent_prediction_types():
    """Test all prediction types are valid"""
    valid_types = ["single_prediction", "batch_prediction", "model_analysis"]
    
    for pred_type in valid_types:
        intent = WorkflowIntent(
            needs_data_cleaning=False,
            needs_feature_engineering=False,
            needs_ml_modeling=False,
            data_quality_focus=False,
            exploratory_analysis=False,
            prediction_focus=True,
            statistical_analysis=False,
            key_requirements=["test"],
            complexity_level="simple",
            intent_confidence=0.9,
            needs_prediction=True,
            prediction_type=pred_type
        )
        assert intent.prediction_type == pred_type
    
    print("✅ All prediction types validated")

def test_workflow_intent_defaults():
    """Test that prediction fields have correct defaults"""
    intent = WorkflowIntent(
        needs_data_cleaning=False,
        needs_feature_engineering=False,
        needs_ml_modeling=False,
        data_quality_focus=False,
        exploratory_analysis=False,
        prediction_focus=False,
        statistical_analysis=False,
        key_requirements=["test"],
        complexity_level="simple",
        intent_confidence=0.9
    )
    
    # Test defaults
    assert intent.needs_prediction == False
    assert intent.needs_model_analysis == False
    assert intent.prediction_data_source is None
    assert intent.prediction_type is None
    assert intent.extracted_prediction_data is None
    
    print("✅ Prediction field defaults working correctly")

if __name__ == "__main__":
    test_workflow_intent_prediction_fields()
    test_workflow_intent_prediction_types()
    test_workflow_intent_defaults()
    print("🎉 Phase 1A: Schema validation tests PASSED") 