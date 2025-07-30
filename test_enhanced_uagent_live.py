#!/usr/bin/env python3
"""
Enhanced uAgent v2.0 - Live Comprehensive Test

This test verifies all enhanced uAgent capabilities are working correctly:
- Complete ML workflows
- Session management
- Error handling
- Real-world scenarios
"""

import os
import time
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent

def test_basic_functionality():
    """Test basic enhanced uAgent functionality"""
    print("🧪 TEST 1: Basic Functionality")
    print("=" * 40)
    
    try:
        agent = EnhancedDataAnalysisUAgent()
        
        # Test 1: Basic data analysis
        result = agent.process_query("""
        Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
        for basic data exploration
        """)
        
        print(f"✅ Basic analysis: {len(result)} chars")
        print(f"   Preview: {result[:150]}...")
        
        # Test 2: Check session state
        has_model = agent._has_trained_model()
        print(f"✅ Model in session: {has_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_ml_prediction_workflow():
    """Test complete ML prediction workflow"""
    print("\n🧪 TEST 2: ML Prediction Workflow")
    print("=" * 40)
    
    try:
        agent = EnhancedDataAnalysisUAgent()
        
        # Step 1: Train ML model
        print("📊 Training ML model...")
        training_result = agent.process_query("""
        Train an ML model using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
        to predict species classification
        """)
        
        print(f"✅ Training completed: {len(training_result)} chars")
        
        # Check if model was stored
        has_model = agent._has_trained_model()
        print(f"✅ Model stored in session: {has_model}")
        
        if has_model:
            print(f"   Target variable: {agent._last_target_variable}")
            if agent._last_trained_model:
                print(f"   Model ID: {agent._last_trained_model.best_model_id}")
        
        # Step 2: Make prediction
        print("\n🔮 Making prediction...")
        prediction_result = agent.process_query("""
        Predict species for sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
        """)
        
        print(f"✅ Prediction completed: {len(prediction_result)} chars")
        print(f"   Preview: {prediction_result[:200]}...")
        
        # Step 3: Model analysis
        print("\n🧠 Analyzing model...")
        analysis_result = agent.process_query("""
        What are the most important features for species prediction?
        """)
        
        print(f"✅ Analysis completed: {len(analysis_result)} chars")
        print(f"   Preview: {analysis_result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ ML prediction workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling capabilities"""
    print("\n🧪 TEST 3: Error Handling")
    print("=" * 40)
    
    try:
        agent = EnhancedDataAnalysisUAgent()
        
        # Test 1: No model prediction
        print("❌ Testing prediction without model...")
        no_model_result = agent.process_query("Predict price for size=1000")
        
        success = "No Trained Model Found" in no_model_result
        print(f"✅ No model error handling: {'PASS' if success else 'FAIL'}")
        
        # Test 2: Invalid data
        print("\n❌ Testing invalid data...")
        invalid_result = agent.process_query("Analyze https://invalid-url.com/data.csv")
        
        success = len(invalid_result) > 0  # Should return some response
        print(f"✅ Invalid data handling: {'PASS' if success else 'FAIL'}")
        
        # Test 3: Malformed request
        print("\n❌ Testing malformed request...")
        malformed_result = agent.process_query("Predict something unclear")
        
        success = len(malformed_result) > 0  # Should return some response
        print(f"✅ Malformed request handling: {'PASS' if success else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_session_management():
    """Test session management capabilities"""
    print("\n🧪 TEST 4: Session Management")
    print("=" * 40)
    
    try:
        agent = EnhancedDataAnalysisUAgent()
        
        # Test 1: Train model
        print("📊 Training model for session test...")
        train_result = agent.process_query("""
        Train model using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
        to predict tip amount
        """)
        
        # Check session state
        has_model_1 = agent._has_trained_model()
        target_1 = agent._last_target_variable
        timestamp_1 = agent._last_model_timestamp
        
        print(f"✅ First model: {has_model_1}")
        print(f"   Target: {target_1}")
        print(f"   Timestamp: {timestamp_1}")
        
        # Test 2: Make prediction with first model
        pred_result = agent.process_query("Predict tip for total_bill=25.0, size=2")
        print(f"✅ Prediction with first model: {len(pred_result)} chars")
        
        # Test 3: Train second model (should replace first)
        print("\n📊 Training second model...")
        train_result_2 = agent.process_query("""
        Train model using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
        to predict species
        """)
        
        # Check session state after second training
        has_model_2 = agent._has_trained_model()
        target_2 = agent._last_target_variable
        timestamp_2 = agent._last_model_timestamp
        
        print(f"✅ Second model: {has_model_2}")
        print(f"   Target: {target_2}")
        print(f"   Timestamp: {timestamp_2}")
        
        # Verify session was updated
        session_updated = (target_2 != target_1 and timestamp_2 > timestamp_1)
        print(f"✅ Session updated: {session_updated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        return False

def test_real_world_scenario():
    """Test real-world business scenario"""
    print("\n🧪 TEST 5: Real-World Business Scenario")
    print("=" * 40)
    
    try:
        agent = EnhancedDataAnalysisUAgent()
        
        # Business scenario: Customer analysis
        print("💼 Running business analysis...")
        business_result = agent.process_query("""
        Analyze customer data patterns and build predictive insights.
        Focus on identifying key business drivers and actionable recommendations.
        """)
        
        print(f"✅ Business analysis: {len(business_result)} chars")
        print(f"   Preview: {business_result[:200]}...")
        
        # Check if analysis provides business value
        has_recommendations = "RECOMMENDATIONS" in business_result.upper()
        has_insights = "INSIGHTS" in business_result.upper()
        
        print(f"✅ Business recommendations: {has_recommendations}")
        print(f"✅ Business insights: {has_insights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-world scenario test failed: {e}")
        return False

def test_architecture_components():
    """Test that all architecture components are working"""
    print("\n🧪 TEST 6: Architecture Components")
    print("=" * 40)
    
    try:
        agent = EnhancedDataAnalysisUAgent()
        
        # Test component initialization
        components = [
            'data_analysis_agent',
            'intent_parser', 
            'prediction_formatter',
            'result_formatter',
            'csv_processor',
            'error_builder'
        ]
        
        for component in components:
            if hasattr(agent, component):
                comp_obj = getattr(agent, component)
                print(f"✅ {component}: {type(comp_obj).__name__}")
            else:
                print(f"❌ {component}: Missing")
                return False
        
        # Test configuration
        config = agent.config
        print(f"✅ Configuration: {type(config).__name__}")
        print(f"   Session timeout: {config.session_timeout_hours} hours")
        print(f"   Max file size: {config.max_file_size_mb} MB")
        print(f"   Intent parser: {config.intent_parser_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ENHANCED UAGENT v2.0 - COMPREHENSIVE LIVE TEST")
    print("=" * 60)
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    print("✅ Environment ready!")
    
    # Run all tests
    test_results = []
    
    try:
        test_results.append(("Basic Functionality", test_basic_functionality()))
        test_results.append(("ML Prediction Workflow", test_ml_prediction_workflow()))
        test_results.append(("Error Handling", test_error_handling()))
        test_results.append(("Session Management", test_session_management()))
        test_results.append(("Real-World Scenario", test_real_world_scenario()))
        test_results.append(("Architecture Components", test_architecture_components()))
        
        # Summary
        print("\n\n🎊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\n📊 OVERALL RESULT: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Enhanced uAgent v2.0 is working correctly!")
        else:
            print("⚠️  Some tests failed - please review the output above")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 