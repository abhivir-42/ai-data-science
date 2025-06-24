#!/usr/bin/env python3
"""
Data Analysis uAgent Implementation

Following EXACTLY the Fetch.ai LangGraph adapter example:
https://innovationlab.fetch.ai/resources/docs/examples/adapters/langgraph-adapter-example

This wraps the DataAnalysisAgent as a uAgent for deployment on ASI:One.
The wrapper is minimal and leverages the full intelligence of DataAnalysisAgent.
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

# Set API keys (exactly like the example)
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
    enable_async=False  # Keep synchronous for uAgent stability, but optimized with multi-threading
)

def data_analysis_agent_func(query):
    """
    Enhanced data analysis agent function following the LangGraph adapter pattern.
    
    This wrapper:
    - Handles input format conversion (exactly like LangGraph example)
    - Directly invokes DataAnalysisAgent.analyze_from_text()
    - Returns formatted results
    - Leverages all DataAnalysisAgent intelligence without duplication
    
    The DataAnalysisAgent intelligently:
    - Extracts CSV URLs from text using LLM structured outputs
    - Parses workflow intent to determine which agents to run
    - Executes only the needed agents (cleaning, feature engineering, ML)
    - Returns comprehensive structured results
    """
    # Handle input if it's a dict with 'input' key (EXACT pattern from example)
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    try:
        # Direct invocation of the underlying DataAnalysisAgent
        # This uses LLM structured outputs to extract CSV URLs and parse intent
        result = data_analysis_agent.analyze_from_text(query)
        
        # Format the structured result for uAgent compatibility
        return format_analysis_result(result)
        
    except Exception as e:
        error_msg = f"""
🚫 **Analysis Error**

Sorry, I encountered an issue: {str(e)}

**Common solutions:**
1. Include a direct CSV URL in your request (e.g., https://example.com/data.csv)
2. Be specific about what analysis you want
3. Example: "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"

**Need help?** Try: "Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv for species classification"
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
        
        if hasattr(result, 'feature_engineering_effectiveness') and result.feature_engineering_effectiveness is not None:
            if not metrics_added:
                lines.extend(["📈 **PERFORMANCE METRICS**:", ""])
                metrics_added = True
            lines.append(f"   • Feature Engineering: {result.feature_engineering_effectiveness:.2f}/1.0")
        
        if hasattr(result, 'model_performance_score') and result.model_performance_score is not None:
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

# Register the DataAnalysisAgent via uAgent (EXACT pattern from LangGraph example)
tool = LangchainRegisterTool()

print("🚀 Registering enhanced data analysis uAgent...")

agent_info = tool.invoke(
    {
        "agent_obj": data_analysis_agent_func,  # Pass the function
        "name": "enhanced_data_analysis",
        "port": 8102,
        "description": "Enhanced data analysis agent with LLM-powered CSV URL extraction, intelligent workflow orchestration, and comprehensive ML pipeline automation",
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
    # If it's a string, extract from logs
    agent_address = "Check logs above for actual address"
    agent_port = "8102"
else:
    agent_address = "Unknown"
    agent_port = "8102"

# Keep the agent alive (EXACT pattern from example)
if __name__ == "__main__":
    try:
        print("\n🎉 ENHANCED DATA ANALYSIS UAGENT IS RUNNING!")
        print("=" * 60)
        print(f"🔗 Agent address: {agent_address}")
        print(f"🌐 Port: {agent_port}")
        print(f"🎯 Inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agent_port}&address={agent_address}")
        print("\n📋 Usage:")
        print("Send a message with a CSV URL and analysis request:")
        print('- "Clean and analyze https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv for survival prediction"')
        print('- "Perform feature engineering on https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"')
        print('- "Build ML model using https://example.com/your-data.csv to predict target_column"')
        print("\n🎯 The agent uses AI to:")
        print("• Extract CSV URLs from your text using LLM structured outputs")
        print("• Parse your intent to determine which analysis steps to run")
        print("• Execute only the needed agents (cleaning, feature engineering, ML)")
        print("• Return comprehensive structured results")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced data analysis agent...")
        cleanup_uagent("enhanced_data_analysis")
        print("✅ Agent stopped.") 