"""
Enhanced Data Analysis Agent Demo

This demo showcases the new structured output data analysis agent that replaces
the current supervisor_agent.py with sophisticated workflow orchestration.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.schemas import DataAnalysisRequest, ModelType, ProblemType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic usage of the enhanced data analysis agent."""
    
    print("🚀 Enhanced Data Analysis Agent Demo")
    print("=" * 50)
    
    # Initialize the enhanced agent
    agent = DataAnalysisAgent(
        output_dir="demo_outputs",
        intent_parser_model="gpt-4o-mini"
    )
    
    print(f"✅ Agent initialized with output directory: {agent.output_dir}")
    
    # Example 1: Basic analysis request
    print("\n📊 Example 1: Basic Data Analysis")
    print("-" * 30)
    
    csv_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    user_request = "Clean this Titanic dataset and build a machine learning model to predict passenger survival"
    
    try:
        result = agent.analyze(
            csv_url=csv_url,
            user_request=user_request
        )
        
        print(f"✅ Analysis completed!")
        print(f"📝 Request ID: {result.request_id}")
        print(f"⏱️  Total runtime: {result.total_runtime_seconds:.2f} seconds")
        print(f"🎯 Workflow intent confidence: {result.workflow_intent.intent_confidence:.2f}")
        print(f"🔧 Agents executed: {', '.join(result.agents_executed)}")
        print(f"📈 Analysis quality score: {result.analysis_quality_score:.2f}")
        print(f"🎖️  Confidence level: {result.confidence_level}")
        
        print(f"\n💡 Key Insights:")
        for insight in result.key_insights:
            print(f"   • {insight}")
        
        print(f"\n🎯 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")
        
        print(f"\n📖 Data Story:")
        print(f"   {result.data_story}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   • {warning}")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


def demo_advanced_usage():
    """Demonstrate advanced usage with custom parameters."""
    
    print("\n\n🔬 Example 2: Advanced Analysis with Custom Parameters")
    print("-" * 50)
    
    agent = DataAnalysisAgent(output_dir="demo_outputs_advanced")
    
    # Advanced request with custom parameters
    csv_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    user_request = "Perform comprehensive data analysis focusing on feature engineering and advanced modeling"
    
    try:
        result = agent.analyze(
            csv_url=csv_url,
            user_request=user_request,
            target_variable="Survived",
            problem_type=ProblemType.CLASSIFICATION,
            model_types=[ModelType.GBM, ModelType.RANDOM_FOREST, ModelType.XG_BOOST],
            max_runtime_seconds=600,
            missing_threshold=0.3,
            cross_validation_folds=10,
            enable_mlflow=True
        )
        
        print(f"✅ Advanced analysis completed!")
        print(f"🎯 Target variable: {result.workflow_intent.suggested_target_variable}")
        print(f"🤖 Model types used: {', '.join([m.value for m in [ModelType.GBM, ModelType.RANDOM_FOREST, ModelType.XG_BOOST]])}")
        print(f"📊 Data quality score: {result.overall_data_quality_score:.2f}")
        
        if result.feature_engineering_effectiveness:
            print(f"⚙️  Feature engineering effectiveness: {result.feature_engineering_effectiveness:.2f}")
        
        if result.model_performance_score:
            print(f"🎯 Model performance score: {result.model_performance_score:.2f}")
        
        print(f"\n📁 Generated files:")
        for file_type, file_path in result.generated_files.items():
            print(f"   • {file_type}: {file_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        return None


def demo_error_handling():
    """Demonstrate error handling with invalid inputs."""
    
    print("\n\n🛡️  Example 3: Error Handling Demo")
    print("-" * 35)
    
    agent = DataAnalysisAgent(output_dir="demo_outputs_error")
    
    # Test with invalid URL
    try:
        result = agent.analyze(
            csv_url="https://invalid-url-that-does-not-exist.com/data.csv",
            user_request="Analyze this non-existent dataset"
        )
        
        print(f"🛡️  Error handling result:")
        print(f"   • Confidence level: {result.confidence_level}")
        print(f"   • Analysis quality: {result.analysis_quality_score:.2f}")
        print(f"   • Warnings: {len(result.warnings)}")
        print(f"   • Limitations: {len(result.limitations)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return None


def demo_schema_validation():
    """Demonstrate schema validation and structured outputs."""
    
    print("\n\n📋 Example 4: Schema Validation Demo")
    print("-" * 40)
    
    try:
        # Test invalid CSV URL
        print("Testing invalid CSV URL validation...")
        try:
            request = DataAnalysisRequest(
                csv_url="not-a-valid-url",
                user_request="Test request"
            )
            print("❌ Should have failed validation!")
        except Exception as e:
            print(f"✅ Correctly caught validation error: {e}")
        
        # Test valid request
        print("\nTesting valid request creation...")
        request = DataAnalysisRequest(
            csv_url="https://example.com/data.csv",
            user_request="This is a valid request with sufficient length",
            model_types=[ModelType.GBM, ModelType.GLM],
            cross_validation_folds=5
        )
        print(f"✅ Valid request created: {request.problem_type.value}")
        
        # Test schema serialization
        print("\nTesting schema serialization...")
        request_dict = request.model_dump()
        print(f"✅ Request serialized: {len(request_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        return False


def demo_comparison_with_current():
    """Compare the enhanced agent with the current supervisor approach."""
    
    print("\n\n⚖️  Comparison: Enhanced vs Current Supervisor Agent")
    print("=" * 60)
    
    print("🔥 ENHANCED DATA ANALYSIS AGENT:")
    print("   ✅ Structured input validation (15+ parameters)")
    print("   ✅ LLM-powered intent parsing with confidence scores")
    print("   ✅ Comprehensive parameter mapping (48 total parameters)")
    print("   ✅ Rich structured outputs with metrics and insights")
    print("   ✅ Intelligent target variable detection")
    print("   ✅ Robust error handling and fallback mechanisms")
    print("   ✅ Async/sync support for scalability")
    print("   ✅ MLflow integration and experiment tracking")
    
    print("\n❌ CURRENT SUPERVISOR AGENT:")
    print("   ❌ Basic keyword matching for intent parsing")
    print("   ❌ Only passes user_instructions to agents")
    print("   ❌ Simple string concatenation for outputs")
    print("   ❌ Fragile target variable extraction")
    print("   ❌ Limited error handling")
    print("   ❌ No structured validation or schemas")
    print("   ❌ No confidence scoring or quality assessment")
    print("   ❌ No comprehensive parameter utilization")


def main():
    """Run all demo examples."""
    
    print("🎯 ENHANCED DATA ANALYSIS AGENT - COMPREHENSIVE DEMO")
    print("=" * 65)
    print("This demo showcases the new structured output data analysis agent")
    print("that replaces supervisor_agent.py with sophisticated orchestration.")
    print()
    
    # Run demos
    basic_result = demo_basic_usage()
    advanced_result = demo_advanced_usage()
    error_result = demo_error_handling()
    schema_valid = demo_schema_validation()
    demo_comparison_with_current()
    
    # Summary
    print("\n\n🎉 DEMO SUMMARY")
    print("=" * 20)
    
    demos_run = [
        ("Basic Usage", basic_result is not None),
        ("Advanced Usage", advanced_result is not None),
        ("Error Handling", error_result is not None),
        ("Schema Validation", schema_valid)
    ]
    
    for demo_name, success in demos_run:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {demo_name}: {status}")
    
    success_count = sum(1 for _, success in demos_run if success)
    print(f"\n🎯 Overall: {success_count}/{len(demos_run)} demos successful")
    
    if success_count == len(demos_run):
        print("🎉 All demos completed successfully!")
        print("🚀 Enhanced Data Analysis Agent is ready for production!")
    else:
        print("⚠️  Some demos failed - please check the implementation")


if __name__ == "__main__":
    main() 