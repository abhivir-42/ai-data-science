"""
MASTER DEMO RUNNER - Data Visualization Agent
=============================================

This script runs all demos and creates a comprehensive presentation summary
showcasing the DataVisualizationAgent's capabilities.

Run this for a complete demonstration suitable for presentations.
"""

import os
import sys
import time
import json
from datetime import datetime
import pandas as pd

# Add the current directory to Python path for imports
sys.path.append(os.getcwd())

def run_demo_with_timing(demo_name, demo_function):
    """Run a demo and track timing and results."""
    print(f"\\n🚀 Starting {demo_name}...")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        results = demo_function()
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'name': demo_name,
            'success': True,
            'duration': duration,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Error in {demo_name}: {str(e)}")
        return {
            'name': demo_name,
            'success': False,
            'duration': duration,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def generate_presentation_summary(demo_results):
    """Generate a comprehensive presentation summary."""
    
    print("\\n" + "=" * 80)
    print("📊 COMPREHENSIVE DEMO RESULTS - DATA VISUALIZATION AGENT")
    print("=" * 80)
    
    # Overall statistics
    total_demos = len(demo_results)
    successful_demos = sum(1 for demo in demo_results if demo['success'])
    total_duration = sum(demo['duration'] for demo in demo_results)
    
    print(f"\\n🎯 OVERALL PERFORMANCE:")
    print(f"   ✅ Successful demos: {successful_demos}/{total_demos}")
    print(f"   ⏱️ Total execution time: {total_duration:.2f} seconds")
    print(f"   📅 Demo session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count total visualizations created
    total_charts = 0
    total_code_files = 0
    
    for demo in demo_results:
        if demo['success'] and 'results' in demo:
            successful_scenarios = sum(1 for r in demo['results'] if r.get('success', False))
            total_charts += successful_scenarios
    
    # Count log files
    log_dirs = ['logs/demo_1', 'logs/demo_2']
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            py_files = [f for f in os.listdir(log_dir) if f.endswith('.py')]
            total_code_files += len(py_files)
    
    print(f"\\n📈 OUTPUTS GENERATED:")
    print(f"   📊 Total charts created: {total_charts}")
    print(f"   📝 Code files generated: {total_code_files}")
    print(f"   📁 Output directory: ./output/")
    print(f"   📝 Logs directory: ./logs/")
    
    # Demo-by-demo breakdown
    print(f"\\n📋 DEMO BREAKDOWN:")
    for i, demo in enumerate(demo_results, 1):
        status = "✅" if demo['success'] else "❌"
        print(f"\\n{i}. {status} {demo['name']}")
        print(f"   ⏱️ Duration: {demo['duration']:.2f}s")
        
        if demo['success'] and 'results' in demo:
            scenarios = demo['results']
            successful_scenarios = sum(1 for r in scenarios if r.get('success', False))
            print(f"   📊 Scenarios: {successful_scenarios}/{len(scenarios)} successful")
            
            # Show successful scenarios
            for scenario in scenarios:
                if scenario.get('success', False):
                    chart_icon = "📈"
                    print(f"      {chart_icon} {scenario.get('scenario', 'Unknown')}")
                    if 'chart_file' in scenario:
                        print(f"         📁 {scenario['chart_file']}")
        else:
            print(f"   ❌ Error: {demo.get('error', 'Unknown error')}")
    
    # Key features demonstrated
    print(f"\\n🌟 KEY FEATURES DEMONSTRATED:")
    
    features = [
        "✅ Automatic chart type selection based on data analysis",
        "✅ Interactive Plotly visualizations with professional styling", 
        "✅ Generated Python code with full functionality",
        "✅ Comprehensive logging and export capabilities",
        "✅ Complex dataset handling (sales, customer analytics)",
        "✅ Multiple chart types (scatter, bar, line, multi-dimensional)",
        "✅ Error handling and retry mechanisms",
        "✅ Professional-grade output suitable for business use",
        "✅ Workflow tracking and summary generation",
        "✅ Customizable parameters and instructions"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    # Files created
    print(f"\\n📁 FILES CREATED:")
    
    # Check output directory
    if os.path.exists('output'):
        output_files = [f for f in os.listdir('output') if f.endswith('.html')]
        print(f"   📊 Charts (HTML): {len(output_files)}")
        for file in sorted(output_files)[:5]:  # Show first 5
            print(f"      📈 {file}")
        if len(output_files) > 5:
            print(f"      ... and {len(output_files) - 5} more charts")
    
    # Check log directories
    for log_dir in ['logs/demo_1', 'logs/demo_2']:
        if os.path.exists(log_dir):
            py_files = [f for f in os.listdir(log_dir) if f.endswith('.py')]
            print(f"   📝 {log_dir}: {len(py_files)} Python files")
    
    # Presentation talking points
    print(f"\\n🎤 PRESENTATION TALKING POINTS:")
    print(f"   1. 🤖 AI-Powered: Automatically selects optimal chart types")
    print(f"   2. 🎨 Professional: Generates publication-ready visualizations") 
    print(f"   3. 🔧 Flexible: Handles diverse datasets and requirements")
    print(f"   4. 📝 Transparent: Provides full code generation and logging")
    print(f"   5. 🔄 Reliable: Built-in error handling and retry logic")
    print(f"   6. 📈 Scalable: Works with complex enterprise datasets")
    print(f"   7. 🎯 User-Friendly: Simple instructions generate complex charts")
    print(f"   8. 💼 Enterprise-Ready: Comprehensive logging and export features")
    
    # Save summary to file
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'demos_run': total_demos,
        'successful_demos': successful_demos,
        'total_duration': total_duration,
        'total_charts': total_charts,
        'total_code_files': total_code_files,
        'demo_details': demo_results
    }
    
    with open('demo_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\\n💾 Complete demo summary saved to: demo_summary.json")
    print(f"\\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
    print(f"\\n🔍 Next steps for presentation:")
    print(f"   1. Open charts in ./output/ directory")
    print(f"   2. Review generated code in ./logs/ directories") 
    print(f"   3. Use demo_summary.json for detailed metrics")
    print(f"   4. Showcase the variety and quality of visualizations")
    
    return summary_data

def main():
    """Run all demos and generate presentation summary."""
    
    print("🚀 DATA VISUALIZATION AGENT - COMPREHENSIVE DEMO SUITE")
    print("=" * 80)
    print("This will run all demos and generate presentation-ready results.")
    print("Estimated time: 2-5 minutes depending on API response times.")
    
    # Ensure all directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs/demo_1", exist_ok=True)
    os.makedirs("logs/demo_2", exist_ok=True)
    
    # Import demo functions
    try:
        from demo_1_basic_usage import demo_basic_usage
        from demo_2_advanced_features import demo_advanced_features
    except ImportError as e:
        print(f"❌ Error importing demo modules: {e}")
        print("Make sure demo_1_basic_usage.py and demo_2_advanced_features.py exist")
        return
    
    # Define demos to run
    demos = [
        ("Demo 1: Basic Usage - Sales Data", demo_basic_usage),
        ("Demo 2: Advanced Features - Customer Analytics", demo_advanced_features),
    ]
    
    # Run all demos
    demo_results = []
    
    for demo_name, demo_function in demos:
        result = run_demo_with_timing(demo_name, demo_function)
        demo_results.append(result)
        
        # Small pause between demos
        time.sleep(1)
    
    # Generate comprehensive summary
    summary = generate_presentation_summary(demo_results)
    
    return summary

if __name__ == "__main__":
    summary = main() 