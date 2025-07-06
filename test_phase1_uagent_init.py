import pytest
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
from src.uagent_v2.config import UAgentConfig

def test_enhanced_uagent_ml_session_variables():
    """Test enhanced uAgent initializes ML session variables correctly"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Check new ML session variables exist
    assert hasattr(agent, '_last_trained_model'), "Missing _last_trained_model"
    assert hasattr(agent, '_last_model_timestamp'), "Missing _last_model_timestamp"
    assert hasattr(agent, '_last_training_result'), "Missing _last_training_result"
    assert hasattr(agent, '_last_target_variable'), "Missing _last_target_variable"
    
    # Check they're properly initialized as None
    assert agent._last_trained_model is None
    assert agent._last_model_timestamp is None
    assert agent._last_training_result is None
    assert agent._last_target_variable is None
    
    print("✅ Enhanced uAgent ML session variables initialized correctly")

def test_enhanced_uagent_has_trained_model_method():
    """Test _has_trained_model method exists and works"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Should return False when no model is stored
    assert agent._has_trained_model() == False
    
    print("✅ _has_trained_model method working correctly")

def test_enhanced_uagent_model_session_expired_method():
    """Test _is_model_session_expired method exists and works"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Should return True when no model timestamp is set
    assert agent._is_model_session_expired() == True
    
    print("✅ _is_model_session_expired method working correctly")

def test_enhanced_uagent_extract_target_variable_method():
    """Test _extract_target_variable method exists"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Method should exist and be callable
    assert hasattr(agent, '_extract_target_variable'), "Missing _extract_target_variable method"
    assert callable(agent._extract_target_variable), "_extract_target_variable should be callable"
    
    print("✅ _extract_target_variable method exists and callable")

def test_enhanced_uagent_store_ml_model_method():
    """Test _store_ml_model_if_available method exists"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Method should exist and be callable
    assert hasattr(agent, '_store_ml_model_if_available'), "Missing _store_ml_model_if_available method"
    assert callable(agent._store_ml_model_if_available), "_store_ml_model_if_available should be callable"
    
    print("✅ _store_ml_model_if_available method exists and callable")

def test_enhanced_uagent_original_functionality_preserved():
    """Test that original functionality is preserved"""
    config = UAgentConfig.from_env()
    agent = EnhancedDataAnalysisUAgent(config)
    
    # Check original session management still exists
    assert hasattr(agent, '_last_cleaned_data'), "Missing original _last_cleaned_data"
    assert hasattr(agent, '_last_processed_timestamp'), "Missing original _last_processed_timestamp"
    
    # Check original methods still exist
    assert hasattr(agent, '_is_session_expired'), "Missing original _is_session_expired"
    assert hasattr(agent, 'cleanup_session'), "Missing original cleanup_session"
    assert hasattr(agent, 'process_query'), "Missing original process_query"
    
    print("✅ Original functionality preserved")

if __name__ == "__main__":
    test_enhanced_uagent_ml_session_variables()
    test_enhanced_uagent_has_trained_model_method()
    test_enhanced_uagent_model_session_expired_method()
    test_enhanced_uagent_extract_target_variable_method()
    test_enhanced_uagent_store_ml_model_method()
    test_enhanced_uagent_original_functionality_preserved()
    print("🎉 Phase 1B: Enhanced uAgent initialization tests PASSED") 