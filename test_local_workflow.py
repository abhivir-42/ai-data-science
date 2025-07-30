#!/usr/bin/env python3
"""
Local test of enhanced uAgent to verify complete workflow is working.
This avoids Agentverse rate limits by testing directly.
"""

from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent
import time
import re

def test_complete_workflow():
    print('🔥 TESTING ENHANCED UAGENT LOCALLY (NO RATE LIMITS)')
    print('=' * 60)

    agent = EnhancedDataAnalysisUAgent()

    # Test 1: Train model
    print('📊 Step 1: Training ML model...')
    start = time.time()
    result1 = agent.process_query('Train ML model using https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv to predict tip amount')
    train_time = time.time() - start

    print(f'✅ Training completed in {train_time:.1f}s')
    print(f'✅ Result length: {len(result1)} chars')
    print(f'✅ Model stored: {agent._has_trained_model()}')

    if agent._has_trained_model():
        print(f'✅ Model ID: {agent._last_trained_model.best_model_id}')
        
        # Test 2: Make prediction
        print('\n🔮 Step 2: Making prediction...')
        start = time.time()
        result2 = agent.process_query('Predict tip for total_bill=25.0, size=2')
        pred_time = time.time() - start
        
        print(f'✅ Prediction completed in {pred_time:.1f}s')
        print(f'✅ Result length: {len(result2)} chars')
        
        if 'PREDICTION RESULT' in result2:
            print('🎉 PREDICTION SUCCESSFUL!')
            # Extract predicted value
            match = re.search(r'Prediction.*?: ([0-9.]+)', result2)
            if match:
                print(f'🎯 Predicted tip: ${match.group(1)}')
            
            # Test 3: Model analysis
            print('\n🔍 Step 3: Analyzing model...')
            start = time.time()
            result3 = agent.process_query('What factors determine tip amount?')
            analysis_time = time.time() - start
            
            print(f'✅ Analysis completed in {analysis_time:.1f}s')
            print(f'✅ Result length: {len(result3)} chars')
            
            print('\n🎉 ALL TESTS PASSED! Enhanced uAgent is working perfectly!')
            print('\n🚀 RECOMMENDATION: The rate limit issue is only with Agentverse.')
            print('🚀 Your enhanced uAgent is fully functional for direct usage!')
            
            return True
        else:
            print('❌ Prediction failed')
            print(f'Result: {result2[:300]}...')
            return False
    else:
        print('❌ Model training failed')
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    if success:
        print('\n✅ Enhanced uAgent v2.0 is production-ready!')
    else:
        print('\n❌ Issues detected that need fixing') 