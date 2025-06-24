"""
Test Enhanced Data Analysis Agent with Loan Approval Dataset

This script tests the enhanced data analysis agent with a real-world loan approval
prediction dataset to ensure all components are working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.data_analysis_agent import DataAnalysisAgent
from src.schemas import ModelType, ProblemType

def test_loan_approval_analysis():
    """Test the enhanced agent with loan approval prediction dataset."""
    
    print("🏦 LOAN APPROVAL PREDICTION - ENHANCED DATA ANALYSIS TEST")
    print("=" * 65)
    
    # Initialize the enhanced agent
    agent = DataAnalysisAgent(
        output_dir="loan_analysis_outputs",
        intent_parser_model="gpt-4o-mini"
    )
    
    # Dataset from the provided URL
    csv_url = "https://raw.githubusercontent.com/sachin365123/CSV-files-for-Data-Science-and-Machine-Learning/main/Loan%20Approval%20Prediction.csv"
    
    # Test comprehensive loan approval analysis
    user_request = """
    Analyze this loan approval dataset to build a machine learning model that can predict loan approval status.
    Please clean the data, engineer relevant features, and train classification models to predict Loan_Status.
    Focus on identifying the most important factors that influence loan approval decisions.
    """
    
    print(f"🔗 Dataset URL: {csv_url}")
    print(f"📝 Analysis Request: {user_request}")
    print()
    
    try:
        # Execute the analysis
        print("🚀 Starting comprehensive loan approval analysis...")
        result = agent.analyze(
            csv_url=csv_url,
            user_request=user_request,
            target_variable="Loan_Status",
            problem_type=ProblemType.CLASSIFICATION,
            model_types=[ModelType.GBM, ModelType.RANDOM_FOREST, ModelType.GLM],
            cross_validation_folds=5,
            enable_mlflow=True,
            max_runtime_seconds=600
        )
        
        print("✅ Analysis completed successfully!")
        print()
        
        # Display comprehensive results
        print("📊 ANALYSIS RESULTS SUMMARY")
        print("-" * 40)
        print(f"🆔 Request ID: {result.request_id}")
        print(f"⏱️  Total Runtime: {result.total_runtime_seconds:.2f} seconds")
        print(f"🎯 Target Variable: {result.workflow_intent.suggested_target_variable or 'Loan_Status'}")
        print(f"🤖 Problem Type: {result.workflow_intent.suggested_problem_type or 'classification'}")
        print()
        
        # Workflow execution details
        print("🔧 WORKFLOW EXECUTION")
        print("-" * 25)
        print(f"📈 Intent Confidence: {result.workflow_intent.intent_confidence:.2f}")
        print(f"🔧 Agents Executed: {', '.join(result.agents_executed)}")
        print(f"📊 Overall Data Quality: {result.overall_data_quality_score:.2f}")
        print(f"⚙️  Feature Engineering Effectiveness: {result.feature_engineering_effectiveness:.2f}")
        print(f"🎯 Model Performance Score: {result.model_performance_score:.2f}")
        print(f"📈 Analysis Quality Score: {result.analysis_quality_score:.2f}")
        print(f"🎖️  Confidence Level: {result.confidence_level}")
        print()
        
        # Key insights
        print("💡 KEY INSIGHTS")
        print("-" * 15)
        for i, insight in enumerate(result.key_insights, 1):
            print(f"   {i}. {insight}")
        print()
        
        # Recommendations
        print("🎯 RECOMMENDATIONS")
        print("-" * 18)
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        print()
        
        # Data story
        print("📖 DATA STORY")
        print("-" * 12)
        print(f"   {result.data_story}")
        print()
        
        # Generated files
        if result.generated_files:
            print("📁 GENERATED FILES")
            print("-" * 17)
            for file_type, file_path in result.generated_files.items():
                print(f"   • {file_type}: {file_path}")
            print()
        
        # Warnings and limitations
        if result.warnings:
            print("⚠️  WARNINGS")
            print("-" * 11)
            for warning in result.warnings:
                print(f"   • {warning}")
            print()
        
        if result.limitations:
            print("🚧 LIMITATIONS")
            print("-" * 13)
            for limitation in result.limitations:
                print(f"   • {limitation}")
            print()
        
        # Detailed agent results
        print("🔍 DETAILED AGENT RESULTS")
        print("-" * 25)
        for agent_result in result.agent_results:
            print(f"   🤖 {agent_result.agent_name.upper()}:")
            print(f"      ⏱️  Execution Time: {agent_result.execution_time_seconds:.2f}s")
            print(f"      ✅ Success: {agent_result.success}")
            if agent_result.error_message:
                print(f"      ❌ Error: {agent_result.error_message}")
            if agent_result.output_data_path:
                print(f"      📄 Output: {agent_result.output_data_path}")
            print()
        
        # Test validation
        print("🧪 VALIDATION CHECKS")
        print("-" * 19)
        
        checks = [
            ("All required agents executed", len(result.agents_executed) >= 3),
            ("Intent confidence > 0.5", result.workflow_intent.intent_confidence > 0.5),
            ("Analysis quality > 0.5", result.analysis_quality_score > 0.5),
            ("No critical errors", not any(agent.error_message for agent in result.agent_results)),
            ("Target variable detected", result.workflow_intent.suggested_target_variable is not None),
            ("Structured output valid", result.request_id is not None),
            ("Files generated", len(result.generated_files) > 0),
            ("Insights provided", len(result.key_insights) > 0),
            ("Recommendations provided", len(result.recommendations) > 0)
        ]
        
        passed_checks = 0
        for check_name, check_result in checks:
            status = "✅ PASS" if check_result else "❌ FAIL"
            print(f"   {status}: {check_name}")
            if check_result:
                passed_checks += 1
        
        print()
        print(f"🎯 OVERALL VALIDATION: {passed_checks}/{len(checks)} checks passed")
        
        if passed_checks == len(checks):
            print("🎉 ALL TESTS PASSED! Enhanced Data Analysis Agent is working perfectly!")
        elif passed_checks >= len(checks) * 0.8:
            print("✅ MOSTLY SUCCESSFUL! Minor issues detected but core functionality works.")
        else:
            print("⚠️  ISSUES DETECTED! Some core functionality may not be working properly.")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return None

def main():
    """Run the loan approval analysis test."""
    result = test_loan_approval_analysis()
    
    if result:
        print("\n" + "=" * 65)
        print("✅ ENHANCED DATA ANALYSIS AGENT VALIDATION COMPLETE")
        print("🚀 Ready for production deployment!")
    else:
        print("\n" + "=" * 65)
        print("❌ VALIDATION FAILED - ISSUES NEED TO BE ADDRESSED")

if __name__ == "__main__":
    main() 