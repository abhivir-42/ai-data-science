#!/usr/bin/env python3
"""
Data Analysis uAgent Implementation

Following the Fetch.ai LangGraph adapter example pattern.
This wraps the DataAnalysisAgent as a uAgent for deployment on ASI:One.

Unlike the supervisor_uagent.py, this leverages the full structured output
capabilities of the enhanced DataAnalysisAgent with proper schema validation
and intelligent parameter mapping.
"""

import os
import time
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from uagents_adapter import LangchainRegisterTool, cleanup_uagent
from src.agents.data_analysis_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

# Set API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_TOKEN = os.environ.get("AGENTVERSE_API_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

if not API_TOKEN:
    print("Warning: AGENTVERSE_API_TOKEN not set - will register locally only")

# Initialize the enhanced data analysis agent
data_analysis_agent = DataAnalysisAgent(
    output_dir="output/data_analysis_uagent/",
    intent_parser_model="gpt-4o-mini",
    enable_async=False  # Use synchronous mode for uAgent compatibility
)

# Dataset URL mappings for convenience (optional fallback)
DATASET_URLS = {
    'iris': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
    'titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
    'wine': 'https://raw.githubusercontent.com/plotly/datasets/master/wine_data.csv',
    'boston': 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
    'diabetes': 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',
    'tips': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
    'flights': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv'
}

def detect_dataset_url(query_text):
    """Simple dataset detection for convenience - fallback only."""
    query_lower = query_text.lower()
    
    for dataset_name, url in DATASET_URLS.items():
        if dataset_name in query_lower:
            return url
    
    return None

def data_analysis_agent_func(query):
    """
    Enhanced data analysis agent function that leverages the full DataAnalysisAgent capabilities.
    
    This wrapper:
    - Handles input format conversion
    - Leverages DataAnalysisAgent's structured input/output 
    - Uses intelligent intent parsing and parameter mapping
    - Returns comprehensive structured analysis results
    - Supports both simple text and structured dict inputs
    """
    # Handle input if it's a dict with 'input' key (standard uAgent pattern)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    try:
        csv_url = None
        user_request = None
        additional_params = {}
        
        # Parse the query to extract parameters
        if isinstance(query, str):
            user_request = query
            
            # Try to detect dataset from query if no URL provided
            detected_url = detect_dataset_url(query)
            if detected_url:
                csv_url = detected_url
                print(f"🔍 Auto-detected dataset URL: {csv_url}")
            
        elif isinstance(query, dict):
            # Extract parameters from structured input
            user_request = query.get('user_request', query.get('query', ''))
            csv_url = query.get('csv_url')
            
            # Extract additional DataAnalysisRequest parameters
            additional_params = {
                'target_variable': query.get('target_variable'),
                'problem_type': query.get('problem_type'),
                'max_runtime_minutes': query.get('max_runtime_minutes', 30),
                'enable_advanced_features': query.get('enable_advanced_features', True),
                'quality_threshold': query.get('quality_threshold', 0.8),
                'performance_vs_speed': query.get('performance_vs_speed', 'balanced')
            }
            
            # Remove None values
            additional_params = {k: v for k, v in additional_params.items() if v is not None}
            
            # If no CSV URL provided, try to detect from request
            if not csv_url and user_request:
                detected_url = detect_dataset_url(user_request)
                if detected_url:
                    csv_url = detected_url
                    print(f"🔍 Auto-detected dataset URL: {csv_url}")
                    
        else:
            return """
🚫 **Input Format Error**

I need either:
1. A simple text request like: "Analyze the iris dataset for classification"
2. A structured request like: {
     "csv_url": "your-url", 
     "user_request": "your-task",
     "target_variable": "optional",
     "problem_type": "classification|regression"
   }

**Supported datasets**: iris, titanic, wine, boston, diabetes, tips, flights
"""
        
        # Validate required parameters
        if not user_request:
            return """
🚫 **Missing Request**

Please tell me what you'd like me to do! For example:
- "Analyze the iris dataset for species classification"
- "Clean and engineer features for the titanic dataset"
- "Build a regression model to predict wine quality"
- "Perform complete ML pipeline on the flights dataset"
"""
        
        if not csv_url:
            return """
🚫 **Dataset Required**

Please provide either:
- A known dataset name (iris, titanic, wine, boston, diabetes, tips, flights)
- A direct CSV URL in the request
- A structured input with 'csv_url' field

Example: "Analyze the iris dataset for classification"
Or: {"csv_url": "https://your-url.com/data.csv", "user_request": "classify the data"}
"""
        
        # Process the request using the enhanced DataAnalysisAgent
        print(f"🚀 Processing enhanced data analysis request:")
        print(f"   📊 Dataset URL: {csv_url}")
        print(f"   📝 Request: {user_request}")
        if additional_params:
            print(f"   ⚙️  Additional params: {additional_params}")
        
        # Call the DataAnalysisAgent with structured parameters
        result = data_analysis_agent.analyze(
            csv_url=csv_url,
            user_request=user_request,
            **additional_params
        )
        
        # Format the structured result for display
        return format_analysis_result(result)
        
    except Exception as e:
        error_msg = f"""
🚫 **Analysis Error**

Sorry, I encountered an issue during analysis: {str(e)}

**Common solutions:**
1. Check if the dataset URL is accessible
2. Ensure your request is clear and specific
3. Try specifying the target variable explicitly for ML tasks
4. Check that the dataset format is valid CSV

**Need help?** Try: "Analyze the iris dataset for species classification"
"""
        print(f"❌ Error in data analysis agent: {str(e)}")
        return error_msg

def format_analysis_result(result) -> str:
    """Format the DataAnalysisResult into a comprehensive string report."""
    
    if not result:
        return "❌ No analysis result received"
    
    try:
        # Build comprehensive report
        lines = [
            "🎉 **DATA ANALYSIS COMPLETE**",
            "=" * 50,
            "",
            f"📊 **Dataset**: {result.csv_url}",
            f"📝 **Request**: {result.original_request}",
            f"📏 **Data Shape**: {result.data_shape.get('rows', 'unknown')} rows × {result.data_shape.get('columns', 'unknown')} columns",
            f"⏱️  **Runtime**: {result.total_runtime_seconds:.2f} seconds",
            f"🎯 **Confidence**: {result.confidence_level.upper()}",
            f"⭐ **Quality Score**: {result.analysis_quality_score:.2f}/1.0",
            ""
        ]
        
        # Workflow information
        if result.workflow_intent:
            lines.extend([
                "🔄 **WORKFLOW EXECUTED**:",
                f"   • Data Cleaning: {'✅' if result.workflow_intent.needs_data_cleaning else '❌'}",
                f"   • Feature Engineering: {'✅' if result.workflow_intent.needs_feature_engineering else '❌'}",
                f"   • ML Modeling: {'✅' if result.workflow_intent.needs_ml_modeling else '❌'}",
                f"   • Intent Confidence: {result.workflow_intent.intent_confidence:.2f}",
                ""
            ])
        
        # Agents executed
        if result.agents_executed:
            lines.extend([
                "🤖 **AGENTS EXECUTED**:",
                *[f"   • {agent.replace('_', ' ').title()}" for agent in result.agents_executed],
                ""
            ])
        
        # Performance metrics
        metrics_added = False
        if result.overall_data_quality_score is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Data Quality: {result.overall_data_quality_score:.2f}/1.0")
        
        if result.feature_engineering_effectiveness is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Feature Engineering: {result.feature_engineering_effectiveness:.2f}/1.0")
        
        if result.model_performance_score is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Model Performance: {result.model_performance_score:.2f}/1.0")
        
        if metrics_added:
            lines.append("")
        
        # Key insights
        if result.key_insights:
            lines.extend([
                "💡 **KEY INSIGHTS**:",
                *[f"   • {insight}" for insight in result.key_insights],
                ""
            ])
        
        # Recommendations
        if result.recommendations:
            lines.extend([
                "🎯 **RECOMMENDATIONS**:",
                *[f"   • {rec}" for rec in result.recommendations],
                ""
            ])
        
        # Generated files
        if result.generated_files:
            lines.extend([
                "📁 **GENERATED FILES**:",
                *[f"   • {name}: {path}" for name, path in result.generated_files.items()],
                ""
            ])
        
        # Warnings
        if result.warnings:
            lines.extend([
                "⚠️  **WARNINGS**:",
                *[f"   • {warning}" for warning in result.warnings],
                ""
            ])
        
        # Data story (AI narrative)
        if result.data_story:
            lines.extend([
                "📖 **ANALYSIS NARRATIVE**:",
                result.data_story,
                ""
            ])
        
        # Limitations
        if result.limitations:
            lines.extend([
                "⚠️  **LIMITATIONS**:",
                *[f"   • {limitation}" for limitation in result.limitations],
                ""
            ])
        
        lines.extend([
            "=" * 50,
            "✅ **Analysis completed successfully!**"
        ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Error formatting result: {str(e)}\n\nRaw result: {str(result)}"

# Register the enhanced data analysis agent via uAgent
tool = LangchainRegisterTool()

print("🚀 Registering enhanced data analysis uAgent...")

agent_info = tool.invoke(
    {
        "agent_obj": data_analysis_agent_func,
        "name": "enhanced_data_analysis",
        "port": 8102,
        "description": "Enhanced data analysis agent with structured outputs, intelligent intent parsing, and comprehensive ML pipeline orchestration from remote CSV files",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)

print(f"✅ Registration result: {agent_info}")
print(f"📊 Result type: {type(agent_info)}")

# Extract address info for display
if isinstance(agent_info, dict):
    agent_address = agent_info.get('agent_address', 'Unknown')
    agent_port = agent_info.get('agent_port', '8102')
elif isinstance(agent_info, str):
    agent_address = "Check logs above for actual address"
    agent_port = "8102"
else:
    agent_address = "Unknown"
    agent_port = "8102"

# Keep the agent alive
if __name__ == "__main__":
    try:
        print("\n🎉 ENHANCED DATA ANALYSIS UAGENT IS RUNNING!")
        print("=" * 60)
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print("\n📋 Usage:")
        print("Send a message with:")
        print('• Simple string: "Analyze the iris dataset for classification"')
        print('• Structured dict: {')
        print('    "csv_url": "https://your-url.com/data.csv",')
        print('    "user_request": "Build a classification model",')
        print('    "target_variable": "species",')
        print('    "problem_type": "classification"')
        print('  }')
        print("\n🔗 Supported datasets: iris, titanic, wine, boston, diabetes, tips, flights")
        print("🎯 Enhanced features: Structured outputs, intelligent parsing, comprehensive reports")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced data analysis agent...")
        cleanup_uagent("enhanced_data_analysis")
        print("✅ Agent stopped.") 