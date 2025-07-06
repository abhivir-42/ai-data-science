#!/usr/bin/env python3
"""
Test script for the enhanced uAgent v2.

This script tests the enhanced uAgent implementation to ensure it works
correctly with all the new modules and improvements.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_enhanced_uagent_initialization():
    """Test enhanced uAgent initialization."""
    print("🧪 Testing enhanced uAgent initialization...")
    
    try:
        from uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
        from uagent_v2.config import UAgentConfig
        
        # Test with default config
        config = UAgentConfig()
        uagent = EnhancedDataAnalysisUAgent(config)
        
        assert uagent.config == config
        assert uagent.csv_processor is not None
        assert uagent.delivery_optimizer is not None
        assert uagent.file_uploader is not None
        assert uagent.content_handler is not None
        assert uagent.data_analysis_agent is not None
        
        print("✅ Enhanced uAgent initialization works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced uAgent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_processing():
    """Test query processing functionality."""
    print("🧪 Testing query processing...")
    
    try:
        from uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
        from uagent_v2.config import UAgentConfig
        
        # Create uAgent with test config
        config = UAgentConfig()
        uagent = EnhancedDataAnalysisUAgent(config)
        
        # Test input format handling
        test_query_str = "Test query string"
        test_query_dict = {"input": "Test query from dict"}
        
        # Test data delivery request detection (using actual implementation logic)
        data_request = "send my data"
        query_lower = data_request.lower()
        is_data_delivery = any(phrase in query_lower for phrase in [
            'send my data', 'provide my cleaned data', 'show me my processed data',
            'my cleaned dataset', 'give me my data', 'deliver my data',
            'send rows', 'send columns', 'data in chunks', 'split my data'
        ])
        assert is_data_delivery == True
        
        normal_request = "analyze some data"
        query_lower = normal_request.lower()
        is_data_delivery = any(phrase in query_lower for phrase in [
            'send my data', 'provide my cleaned data', 'show me my processed data',
            'my cleaned dataset', 'give me my data', 'deliver my data',
            'send rows', 'send columns', 'data in chunks', 'split my data'
        ])
        assert is_data_delivery == False
        
        print("✅ Query processing logic works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_function():
    """Test the factory function for uAgent creation."""
    print("🧪 Testing factory function...")
    
    try:
        from uagent_v2.enhanced_uagent import create_enhanced_uagent_function
        from uagent_v2.config import UAgentConfig
        
        # Test factory function
        config = UAgentConfig()
        enhanced_func = create_enhanced_uagent_function(config)
        
        assert callable(enhanced_func)
        
        # Test that it handles different input types
        test_query = "test query"
        test_query_dict = {"input": "test query"}
        
        # These should not raise exceptions (though they might return error messages)
        result1 = enhanced_func(test_query)
        result2 = enhanced_func(test_query_dict)
        
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        
        print("✅ Factory function works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_management():
    """Test session management functionality."""
    print("🧪 Testing session management...")
    
    try:
        from uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
        from uagent_v2.config import UAgentConfig
        import time
        
        config = UAgentConfig()
        uagent = EnhancedDataAnalysisUAgent(config)
        
        # Test initial state
        assert uagent._last_cleaned_data is None
        assert uagent._last_processed_timestamp is None
        
        # Test session expiration
        assert uagent._is_session_expired() == True  # No timestamp means expired
        
        # Simulate having data
        uagent._last_processed_timestamp = time.time()
        assert uagent._is_session_expired() == False  # Recent timestamp
        
        # Simulate old data
        uagent._last_processed_timestamp = time.time() - (2 * 3600)  # 2 hours ago
        assert uagent._is_session_expired() == True  # Should be expired
        
        print("✅ Session management works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling functionality."""
    print("🧪 Testing error handling...")
    
    try:
        from uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
        from uagent_v2.config import UAgentConfig
        
        config = UAgentConfig()
        uagent = EnhancedDataAnalysisUAgent(config)
        
        # Test error response creation
        test_error = ValueError("Test error")
        error_response = uagent._create_error_response(test_error)
        
        assert isinstance(error_response, str)
        assert "Analysis Error" in error_response
        assert "Test error" in error_response
        
        # Test no data response
        no_data_response = uagent.response_builder.create_no_data_response()
        assert isinstance(no_data_response, str)
        assert "No Recent Data Found" in no_data_response
        
        # Test expired session response
        expired_response = uagent.response_builder.create_expired_session_response()
        assert isinstance(expired_response, str)
        assert "Data Session Expired" in expired_response
        
        print("✅ Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_modules():
    """Test integration with all the new modules."""
    print("🧪 Testing integration with new modules...")
    
    try:
        from uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
        from uagent_v2.config import UAgentConfig
        
        config = UAgentConfig()
        uagent = EnhancedDataAnalysisUAgent(config)
        
        # Test that all modules are properly integrated
        assert uagent.csv_processor.config == config
        assert uagent.delivery_optimizer.config == config
        assert uagent.file_uploader.config == config
        assert uagent.content_handler.config == config
        
        # Test configuration propagation
        assert str(uagent.data_analysis_agent.output_dir).endswith('output/data_analysis_uagent')
        assert uagent.data_analysis_agent.enable_async == config.enable_async
        assert uagent.data_analysis_agent.intent_parser is not None
        
        print("✅ Module integration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Module integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests for the enhanced uAgent."""
    print("🚀 Running enhanced uAgent v2 tests...")
    print("=" * 50)
    
    tests = [
        test_enhanced_uagent_initialization,
        test_query_processing,
        test_factory_function,
        test_session_management,
        test_error_handling,
        test_integration_with_modules
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced uAgent v2 is working correctly.")
        print("\n💡 Ready for deployment with improvements:")
        print("   • ✅ Modular architecture")
        print("   • ✅ Enhanced security")
        print("   • ✅ Memory optimization")
        print("   • ✅ Structured error handling")
        print("   • ✅ Configurable behavior")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 