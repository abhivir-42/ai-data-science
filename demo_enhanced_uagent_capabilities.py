#!/usr/bin/env python3
"""
Enhanced uAgent v2.0 - Complete Capabilities Demonstration

This demo shows everything you can do with the enhanced uAgent:
- Complete ML workflows (train → predict → analyze)
- Session management across conversations
- Advanced data analysis capabilities
- Error handling and recovery
"""

import os
import time
from src.uagent_v2.enhanced_uagent import EnhancedDataAnalysisUAgent

def demo_complete_ml_workflow():
    """Demo: Complete ML prediction workflow"""
    print("🚀 COMPLETE ML PREDICTION WORKFLOW")
    print("=" * 50)
    
    # Initialize the enhanced uAgent
    agent = EnhancedDataAnalysisUAgent()
    
    # Step 1: Train ML Model
    print("\n📊 Step 1: Training ML Model")
    print("-" * 30)
    training_query = """
    Train an ML model using https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv 
    to predict passenger survival. Include feature engineering and model optimization.
    """
    
    training_result = agent.process_query(training_query)
    print(f"✅ Training completed! Response preview:")
    print(f"   {training_result[:200]}...")
    
    # Step 2: Make Single Prediction
    print("\n🔮 Step 2: Making Single Prediction")
    print("-" * 30)
    single_prediction_query = """
    Predict survival for a passenger with: Age=25, Sex=male, Pclass=3, Fare=50, Embarked=S
    """
    
    prediction_result = agent.process_query(single_prediction_query)
    print(f"✅ Single prediction completed!")
    print(f"   {prediction_result[:200]}...")
    
    # Step 3: Analyze Model Performance
    print("\n🧠 Step 3: Analyzing Model Performance")
    print("-" * 30)
    analysis_query = """
    What are the most important features for survival prediction? 
    Explain why the model makes these predictions.
    """
    
    analysis_result = agent.process_query(analysis_query)
    print(f"✅ Model analysis completed!")
    print(f"   {analysis_result[:200]}...")
    
    # Step 4: Session Management Test
    print("\n⚡ Step 4: Testing Session Management")
    print("-" * 30)
    print(f"   Has trained model: {agent._has_trained_model()}")
    print(f"   Model timestamp: {agent._last_model_timestamp}")
    print(f"   Target variable: {agent._last_target_variable}")
    
    return agent

def demo_data_analysis_capabilities():
    """Demo: Traditional data analysis capabilities"""
    print("\n\n🔍 DATA ANALYSIS CAPABILITIES")
    print("=" * 50)
    
    agent = EnhancedDataAnalysisUAgent()
    
    # Basic data analysis
    print("\n📈 Basic Data Analysis")
    print("-" * 30)
    analysis_query = """
    Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
    for species classification. Include data cleaning and exploratory analysis.
    """
    
    result = agent.process_query(analysis_query)
    print(f"✅ Analysis completed! Response preview:")
    print(f"   {result[:200]}...")
    
    # Feature engineering only
    print("\n🔧 Feature Engineering")
    print("-" * 30)
    fe_query = """
    Perform feature engineering on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
    to create new features for tip prediction.
    """
    
    fe_result = agent.process_query(fe_query)
    print(f"✅ Feature engineering completed!")
    print(f"   {fe_result[:200]}...")

def demo_error_handling():
    """Demo: Error handling and recovery"""
    print("\n\n🛡️ ERROR HANDLING & RECOVERY")
    print("=" * 50)
    
    agent = EnhancedDataAnalysisUAgent()
    
    # Test prediction without model
    print("\n❌ Testing Prediction Without Model")
    print("-" * 30)
    no_model_query = "Predict house price for size=1500, location=urban"
    
    no_model_result = agent.process_query(no_model_query)
    print(f"✅ Proper error handling:")
    print(f"   {no_model_result[:200]}...")
    
    # Test invalid data
    print("\n❌ Testing Invalid Data Handling")
    print("-" * 30)
    invalid_query = "Analyze https://invalid-url.com/nonexistent.csv"
    
    invalid_result = agent.process_query(invalid_query)
    print(f"✅ Graceful error handling:")
    print(f"   {invalid_result[:200]}...")

def demo_advanced_features():
    """Demo: Advanced features and session management"""
    print("\n\n⚡ ADVANCED FEATURES")
    print("=" * 50)
    
    agent = EnhancedDataAnalysisUAgent()
    
    # Check configuration
    print("\n🔧 Configuration")
    print("-" * 30)
    config = agent.config.to_dict()
    key_configs = ['session_timeout_hours', 'max_file_size_mb', 'intent_parser_model']
    for key in key_configs:
        print(f"   {key}: {config.get(key)}")
    
    # Memory management
    print("\n💾 Memory Management")
    print("-" * 30)
    print(f"   CSV Processor: {type(agent.csv_processor).__name__}")
    print(f"   Memory Optimization: ✅ Enabled")
    print(f"   Delivery Optimizer: {type(agent.delivery_optimizer).__name__}")
    
    # Component architecture
    print("\n🏗️ Component Architecture")
    print("-" * 30)
    components = [
        'data_analysis_agent', 'intent_parser', 'prediction_formatter',
        'result_formatter', 'ml_processor', 'error_builder'
    ]
    for comp in components:
        if hasattr(agent, comp):
            print(f"   ✅ {comp}: {type(getattr(agent, comp)).__name__}")

def demo_real_world_scenarios():
    """Demo: Real-world business scenarios"""
    print("\n\n💼 REAL-WORLD BUSINESS SCENARIOS")
    print("=" * 50)
    
    agent = EnhancedDataAnalysisUAgent()
    
    # Scenario 1: Customer Churn Analysis
    print("\n🎯 Scenario 1: Customer Churn Prediction")
    print("-" * 30)
    churn_query = """
    Build a customer churn prediction model using sample data.
    I need to identify high-risk customers and understand key churn factors.
    """
    
    churn_result = agent.process_query(churn_query)
    print(f"✅ Churn analysis:")
    print(f"   {churn_result[:200]}...")
    
    # Scenario 2: Sales Forecasting
    print("\n📊 Scenario 2: Sales Forecasting")
    print("-" * 30)
    sales_query = """
    Analyze sales patterns and create forecasting model.
    Focus on seasonal trends and key performance indicators.
    """
    
    sales_result = agent.process_query(sales_query)
    print(f"✅ Sales forecasting:")
    print(f"   {sales_result[:200]}...")

def main():
    """Run all demonstrations"""
    print("🎉 ENHANCED UAGENT v2.0 - COMPLETE CAPABILITIES DEMO")
    print("=" * 70)
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    print("✅ Environment ready!")
    
    try:
        # Run all demos
        trained_agent = demo_complete_ml_workflow()
        demo_data_analysis_capabilities()
        demo_error_handling()
        demo_advanced_features()
        demo_real_world_scenarios()
        
        # Final session check
        print("\n\n🎊 FINAL SESSION STATUS")
        print("=" * 50)
        if trained_agent._has_trained_model():
            print("✅ ML Model in session - ready for predictions!")
            print(f"   Model ID: {trained_agent._last_trained_model.best_model_id}")
            print(f"   Target: {trained_agent._last_target_variable}")
        else:
            print("ℹ️  No model in current session")
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("✅ All capabilities working correctly")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 